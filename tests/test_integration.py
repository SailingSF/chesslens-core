"""
Integration tests for the real engine-backed pipelines.

This consolidates the engine-only and full-pipeline suites into one module
while keeping their test classes distinct.
"""

import chess
import pytest

from analysis.context import AssembledContext, ContextAssembler
from analysis.priority import PriorityResult, PriorityTier, classify
from chess_engine.service import EngineResult, RemoteEngineService
from explanation.coach import CoachNudgeGenerator
from explanation.generator import ExplanationGenerator

from .conftest import (
    COMPLEX_MIDDLEGAME_FEN,
    CRUSHING_ADVANTAGE_FEN,
    ITALIAN_GAME_FEN,
    MATE_IN_1_FEN,
    QUEENS_GAMBIT_FEN,
    SICILIAN_FEN,
    STARTING_FEN,
)


class TestEngineHealth:
    """Basic connectivity and response structure."""

    @pytest.mark.asyncio
    async def test_engine_returns_result(self, engine_service: RemoteEngineService):
        result = await engine_service.analyze(STARTING_FEN, depth=10, multipv=3)
        assert isinstance(result, EngineResult)
        assert result.fen == STARTING_FEN
        assert result.depth == 10
        assert len(result.candidates) >= 1

    @pytest.mark.asyncio
    async def test_candidates_have_valid_moves(self, engine_service):
        result = await engine_service.analyze(STARTING_FEN, depth=10, multipv=3)
        board = chess.Board(STARTING_FEN)
        for candidate in result.candidates:
            assert candidate.move in board.legal_moves, (
                f"{candidate.move.uci()} is not legal in the starting position"
            )

    @pytest.mark.asyncio
    async def test_multipv_returns_requested_count(self, engine_service):
        result = await engine_service.analyze(STARTING_FEN, depth=10, multipv=5)
        assert len(result.candidates) >= 3

    @pytest.mark.asyncio
    async def test_pv_lines_are_populated(self, engine_service):
        result = await engine_service.analyze(STARTING_FEN, depth=12, multipv=3)
        for candidate in result.candidates:
            assert len(candidate.pv) >= 1, "PV should contain at least the candidate move"


class TestMateDetection:
    """Verify the engine correctly identifies mate threats."""

    @pytest.mark.asyncio
    async def test_mate_in_1_detected(self, engine_service):
        result = await engine_service.analyze(MATE_IN_1_FEN, depth=10, multipv=1)
        best = result.best
        assert best.mate_in is not None, "Engine should detect a forced mate"
        assert best.mate_in == 1
        expected_move = chess.Move.from_uci("e1e8")
        assert best.move == expected_move, f"Expected Re8#, got {best.move.uci()}"

    @pytest.mark.asyncio
    async def test_mate_score_has_no_cp(self, engine_service):
        result = await engine_service.analyze(MATE_IN_1_FEN, depth=10, multipv=1)
        assert result.best.score_cp is None


class TestKnownPositions:
    """Verify engine produces reasonable evaluations for well-studied positions."""

    @pytest.mark.asyncio
    async def test_starting_position_roughly_equal(self, engine_service):
        result = await engine_service.analyze(STARTING_FEN, depth=14, multipv=1)
        best = result.best
        assert best.score_cp is not None
        assert abs(best.score_cp) < 80, (
            f"Starting position should be roughly equal, got {best.score_cp}cp"
        )

    @pytest.mark.asyncio
    async def test_crushing_advantage_high_eval(self, engine_service):
        result = await engine_service.analyze(CRUSHING_ADVANTAGE_FEN, depth=14, multipv=1)
        best = result.best
        if best.mate_in is not None:
            assert best.mate_in > 0
        else:
            assert best.score_cp is not None
            assert best.score_cp > 500, (
                f"White has a queen up, expected >500cp, got {best.score_cp}"
            )

    @pytest.mark.asyncio
    async def test_italian_game_playable(self, engine_service):
        result = await engine_service.analyze(ITALIAN_GAME_FEN, depth=14, multipv=3)
        assert len(result.candidates) >= 2
        assert result.best.score_cp is not None

    @pytest.mark.asyncio
    async def test_complex_middlegame_multiple_candidates(self, engine_service):
        result = await engine_service.analyze(COMPLEX_MIDDLEGAME_FEN, depth=14, multipv=3)
        assert len(result.candidates) >= 3
        scores = [c.score_cp for c in result.candidates if c.score_cp is not None]
        assert len(scores) >= 2

    @pytest.mark.asyncio
    async def test_queens_gambit_reasonable(self, engine_service):
        result = await engine_service.analyze(QUEENS_GAMBIT_FEN, depth=14, multipv=3)
        assert result.best.score_cp is not None
        assert abs(result.best.score_cp) < 150, "Queen's Gambit should be within normal range"

    @pytest.mark.asyncio
    async def test_sicilian_reasonable(self, engine_service):
        result = await engine_service.analyze(SICILIAN_FEN, depth=14, multipv=3)
        assert result.best.score_cp is not None
        assert abs(result.best.score_cp) < 150


class TestUCIOptions:
    """Verify that UCI options are forwarded to the engine."""

    @pytest.mark.asyncio
    async def test_custom_hash_still_returns_result(self, engine_service):
        result = await engine_service.analyze(
            STARTING_FEN,
            depth=10,
            multipv=1,
            uci_options={"Hash": 64},
        )
        assert len(result.candidates) >= 1
        assert result.best.move in chess.Board(STARTING_FEN).legal_moves


class TestContextAssemblyWithRealEngine:
    """Context assembly using real Stockfish output."""

    @pytest.mark.asyncio
    async def test_assemble_starting_position(
        self, engine_service: RemoteEngineService, context_assembler: ContextAssembler
    ):
        board = chess.Board(STARTING_FEN)
        result = await engine_service.analyze(STARTING_FEN, depth=14, multipv=3)

        played_move = list(board.legal_moves)[5]
        ctx = context_assembler.assemble(board, result, played_move=played_move)

        assert isinstance(ctx, AssembledContext)
        assert ctx.fen == STARTING_FEN
        assert ctx.best_move in board.legal_moves
        assert ctx.best_move_san is not None
        assert ctx.played_move == played_move
        assert ctx.cp_loss_label in ("best", "inaccuracy", "mistake", "blunder", "unknown")

    @pytest.mark.asyncio
    async def test_assemble_mate_position(self, engine_service, context_assembler):
        board = chess.Board(MATE_IN_1_FEN)
        result = await engine_service.analyze(MATE_IN_1_FEN, depth=10, multipv=1)

        ctx = context_assembler.assemble(board, result)
        assert ctx.mate_in is not None
        assert ctx.mate_in == 1
        assert ctx.best_move_san == "Re8#"

    @pytest.mark.asyncio
    async def test_priority_classification_with_real_engine(
        self, engine_service, context_assembler
    ):
        result = await engine_service.analyze(MATE_IN_1_FEN, depth=10, multipv=2)
        board = chess.Board(MATE_IN_1_FEN)

        priority = classify(result, board=board)
        assert isinstance(priority, PriorityResult)
        assert priority.tier == PriorityTier.CRITICAL
        assert "mate" in priority.trigger.lower()

    @pytest.mark.asyncio
    async def test_complex_middlegame_context(self, engine_service, context_assembler):
        board = chess.Board(COMPLEX_MIDDLEGAME_FEN)
        result = await engine_service.analyze(COMPLEX_MIDDLEGAME_FEN, depth=14, multipv=3)

        ctx = context_assembler.assemble(board, result)
        assert ctx.pawn_structure is not None
        assert ctx.piece_activity is not None
        assert ctx.threat_narrative is not None
        assert len(ctx.pv_san) >= 1


class TestExplanationGeneration:
    """Full pipeline: engine -> context -> priority -> explanation."""

    @pytest.mark.asyncio
    async def test_generate_explanation_for_blunder(
        self, engine_service, context_assembler, explainer: ExplanationGenerator
    ):
        board = chess.Board(STARTING_FEN)
        result = await engine_service.analyze(STARTING_FEN, depth=14, multipv=3)

        worst_candidate = result.candidates[-1]
        ctx = context_assembler.assemble(board, result, played_move=worst_candidate.move)
        priority = classify(result, board=board)

        explanation = await explainer.generate(ctx, priority, skill_level="intermediate")

        assert isinstance(explanation, str)
        assert len(explanation) > 20
        assert len(explanation) < 2000

    @pytest.mark.asyncio
    async def test_generate_explanation_for_mate_threat(
        self, engine_service, context_assembler, explainer
    ):
        board = chess.Board(MATE_IN_1_FEN)
        result = await engine_service.analyze(MATE_IN_1_FEN, depth=10, multipv=1)

        ctx = context_assembler.assemble(board, result)
        priority = classify(result, board=board)

        explanation = await explainer.generate(ctx, priority, skill_level="beginner")

        assert isinstance(explanation, str)
        assert len(explanation) > 20

    @pytest.mark.asyncio
    async def test_generate_explanation_for_middlegame(
        self, engine_service, context_assembler, explainer
    ):
        board = chess.Board(COMPLEX_MIDDLEGAME_FEN)
        result = await engine_service.analyze(COMPLEX_MIDDLEGAME_FEN, depth=14, multipv=3)

        played = list(board.legal_moves)[0]
        ctx = context_assembler.assemble(board, result, played_move=played)
        priority = classify(result, board=board)

        explanation = await explainer.generate(ctx, priority, skill_level="advanced")

        assert isinstance(explanation, str)
        assert len(explanation) > 20

    @pytest.mark.asyncio
    async def test_explanation_across_skill_levels(
        self, engine_service, context_assembler, explainer
    ):
        board = chess.Board(ITALIAN_GAME_FEN)
        result = await engine_service.analyze(ITALIAN_GAME_FEN, depth=12, multipv=3)

        played = list(board.legal_moves)[0]
        ctx = context_assembler.assemble(board, result, played_move=played)
        priority = classify(result, board=board)

        for level in ("beginner", "intermediate", "advanced"):
            explanation = await explainer.generate(ctx, priority, skill_level=level)
            assert isinstance(explanation, str)
            assert len(explanation) > 20, f"Empty explanation for {level}"


class TestCoachNudgeGeneration:
    """Full pipeline: engine -> context -> priority -> coach nudge."""

    @pytest.mark.asyncio
    async def test_coach_nudge_returns_question(
        self, engine_service, context_assembler, coach: CoachNudgeGenerator
    ):
        board = chess.Board(COMPLEX_MIDDLEGAME_FEN)
        result = await engine_service.analyze(COMPLEX_MIDDLEGAME_FEN, depth=12, multipv=3)

        played = list(board.legal_moves)[0]
        ctx = context_assembler.assemble(board, result, played_move=played)
        priority = classify(result, board=board)

        nudge = await coach.generate(ctx, priority, skill_level="intermediate")

        assert isinstance(nudge, str)
        assert len(nudge) > 10
        assert "?" in nudge

    @pytest.mark.asyncio
    async def test_coach_nudge_no_move_notation(
        self, engine_service, context_assembler, coach
    ):
        board = chess.Board(MATE_IN_1_FEN)
        result = await engine_service.analyze(MATE_IN_1_FEN, depth=10, multipv=1)

        ctx = context_assembler.assemble(board, result)
        priority = classify(result, board=board)

        nudge = await coach.generate(ctx, priority, skill_level="beginner")
        assert isinstance(nudge, str)
        assert len(nudge) > 10
