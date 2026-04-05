"""
GameAnalyzer — top-level orchestrator for post-game PGN analysis.

Iterates through each half-move in a PGN, runs engine analysis, assembles
context, and generates LLM explanations as a continuous narrative conversation.

Smart filtering decides which moves get full LLM commentary:
  - Inaccuracies, mistakes, blunders → always explained
  - Critical positions → always explained
  - Quiet/good moves → skipped, but every 4th consecutive player move
    without analysis is guaranteed commentary

For open-source (single-user) use, analysis runs as an async generator
and results are streamed via SSE.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import chess
import chess.pgn

from analysis.context import AssembledContext, ContextAssembler
from analysis.eco import ECOLookup
from analysis.priority import PriorityResult, PriorityTier, classify
from chess_engine.service import CandidateMove, EngineService
from explanation.generator import ExplanationGenerator, SkillLevel
from explanation.templates.game_review import (
    build_game_review_system_prompt,
    build_move_user_message,
)


def _parse_elo(value: str | None) -> int | None:
    """Parse an Elo string from PGN headers. Returns None if missing or invalid."""
    if value is None:
        return None
    try:
        elo = int(value)
        return elo if elo > 0 else None
    except (ValueError, TypeError):
        return None


@dataclass
class MoveAnalysis:
    move_number: int
    color: str               # "white" or "black"
    context: AssembledContext
    priority: PriorityResult
    explanation: Optional[str]   # None if move was skipped


class GameAnalyzer:
    def __init__(
        self,
        engine: EngineService,
        assembler: Optional[ContextAssembler] = None,
        explainer: Optional[ExplanationGenerator] = None,
    ):
        self._engine = engine
        self._assembler = assembler or ContextAssembler()
        self._explainer = explainer or ExplanationGenerator()
        self._eco = self._assembler._eco

    async def analyze_pgn(
        self,
        pgn_text: str,
        skill_level: SkillLevel = "intermediate",
        player_color: str = "white",
        analysis_depth: int = 16,
        multipv: int = 3,
        api_key: str | None = None,
        player_elo: int | None = None,
        opponent_elo: int | None = None,
    ) -> AsyncIterator[MoveAnalysis]:
        """Async generator — yields MoveAnalysis for each analyzed position."""
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return

        # Extract Elo from PGN headers (fallback to explicit params)
        headers = game.headers
        white_elo = _parse_elo(headers.get("WhiteElo")) or player_elo
        black_elo = _parse_elo(headers.get("BlackElo")) or opponent_elo

        # Assign per-side Elo based on player_color
        if player_color == "white":
            p_elo = white_elo or player_elo
            o_elo = black_elo or opponent_elo
        else:
            p_elo = black_elo or player_elo
            o_elo = white_elo or opponent_elo

        board = game.board()
        followup_depth = max(analysis_depth - 4, 10)

        # Narrative conversation state — stored on instance so the view
        # can call generate_summary() after iteration completes.
        self._system_prompt = build_game_review_system_prompt(player_color, skill_level)
        self._conversation: list[dict] = []
        # Track how many consecutive player moves went without LLM analysis
        player_moves_without_analysis = 0
        prev_context: AssembledContext | None = None

        for node in game.mainline():
            move = node.move
            color = "white" if board.turn == chess.WHITE else "black"
            full_move_number = board.fullmove_number
            played_on = board.copy()

            # Skip forced/trivial positions
            if self._should_skip(board, move):
                board.push(move)
                continue

            # Use the correct Elo for the side to move
            move_player_elo = white_elo if color == "white" else black_elo
            move_opponent_elo = black_elo if color == "white" else white_elo

            # Engine analysis on the position BEFORE the move
            engine_result = await self._engine.analyze(
                played_on.fen(), depth=analysis_depth, multipv=multipv
            )

            # If the played move isn't in the multi-PV candidates, evaluate
            # the position AFTER the move and negate the score.
            if not any(c.move == move for c in engine_result.candidates):
                post_board = played_on.copy()
                post_board.push(move)
                post_result = await self._engine.analyze(
                    post_board.fen(), depth=followup_depth, multipv=1
                )
                if post_result.candidates:
                    post_best = post_result.best
                    played_cp = -post_best.score_cp if post_best.score_cp is not None else None
                    played_mate = -post_best.mate_in if post_best.mate_in is not None else None
                    engine_result.candidates.append(CandidateMove(
                        move=move,
                        score_cp=played_cp,
                        mate_in=played_mate,
                        pv=[move],
                    ))

            context = self._assembler.assemble(
                played_on, engine_result, move,
                player_elo=move_player_elo,
                opponent_elo=move_opponent_elo,
                prev_context=prev_context,
            )
            priority = classify(engine_result, board=played_on)
            prev_context = context

            # Smart filtering: decide if this move gets LLM commentary
            is_player_move = (color == player_color)
            needs_explanation = self._needs_explanation(
                context, priority, is_player_move, player_moves_without_analysis
            )

            explanation = None
            if needs_explanation:
                explanation = await self._explainer.generate_narrative(
                    context, priority, full_move_number, color,
                    self._conversation, self._system_prompt, api_key=api_key,
                )
                if is_player_move:
                    player_moves_without_analysis = 0
            else:
                # Add data-only entry to conversation for continuity
                user_msg = build_move_user_message(
                    context, priority, full_move_number, color,
                )
                self._conversation.append({"role": "user", "content": user_msg})
                self._conversation.append({
                    "role": "assistant",
                    "content": "Solid move, no issues here.",
                })
                if is_player_move:
                    player_moves_without_analysis += 1

            board.push(move)

            yield MoveAnalysis(
                move_number=full_move_number,
                color=color,
                context=context,
                priority=priority,
                explanation=explanation,
            )

    async def generate_summary(self, api_key: str | None = None) -> str:
        """Generate a final game summary. Call after analyze_pgn iteration completes."""
        return await self._explainer.generate_game_summary(
            self._conversation, self._system_prompt, api_key=api_key,
        )

    @staticmethod
    def _needs_explanation(
        context: AssembledContext,
        priority: PriorityResult,
        is_player_move: bool,
        player_moves_without_analysis: int,
    ) -> bool:
        """Decide if a move warrants full LLM commentary."""
        label = context.cp_loss_label
        # Always explain special classifications
        if label in ("brilliant", "great", "miss"):
            return True
        # Always explain inaccuracies, mistakes, and blunders
        if label in ("inaccuracy", "mistake", "blunder"):
            return True
        # Always explain critical positions
        if priority.tier == PriorityTier.CRITICAL:
            return True
        # Guarantee analysis every 4th consecutive player move without it
        if is_player_move and player_moves_without_analysis >= 3:
            return True
        return False

    @staticmethod
    def _should_skip(board: chess.Board, move: chess.Move) -> bool:
        """
        Skip positions that don't need full engine analysis to save time.

        Only skips truly forced moves (one legal option). Recaptures and
        book moves are analyzed normally since the player had other choices.
        """
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 1:
            return True
        return False
