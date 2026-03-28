"""
ContextAssembler — enriches raw engine output with named chess features.

Accepts an EngineResult + chess.Board, returns an AssembledContext containing:
  - opening name / ECO code
  - theory deviation flag
  - tactical patterns detected
  - pawn structure summary
  - piece activity signals
  - EP-based move classification (win probability change)
  - special move flags (brilliant, great, miss)
  - threat narrative (for coaching prompts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import chess

from analysis.eco import ECOLookup
from analysis.expected_points import (
    StockfishNativeWDLProvider,
    classify_ep_loss,
    resolve_provider,
)
from analysis.patterns import detect_tactics
from analysis.pawns import analyze_pawns
from analysis.special_moves import detect_brilliant, detect_great, detect_miss
from chess_engine.service import CandidateMove, EngineResult
from config.classification import ClassificationConfig


PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


@dataclass
class AssembledContext:
    fen: str
    played_move: Optional[chess.Move]
    played_move_san: Optional[str]

    # Engine data
    best_move: chess.Move
    best_move_san: str
    best_move_cp: Optional[int]
    mate_in: Optional[int]
    played_move_cp_loss: Optional[int]   # positive = worse for side to move (raw cp, unchanged)
    cp_loss_label: str                   # now EP-derived: best / excellent / good / inaccuracy / mistake / blunder
    pv_san: list[str]                    # principal variation in SAN notation

    # Expected Points data
    win_pct_before: Optional[float] = None   # win probability at best eval (0.0–1.0)
    win_pct_after: Optional[float] = None    # win probability at played move eval (0.0–1.0)
    ep_loss: Optional[float] = None          # win_pct_before - win_pct_after
    player_elo: Optional[int] = None         # Elo used for curve adjustment
    opponent_elo: Optional[int] = None       # stored for context
    wdl_source: str = "sigmoid"              # "sigmoid" or "native"

    # Special classification flags
    is_brilliant: bool = False
    is_great: bool = False
    is_miss: bool = False

    # Opening
    eco_code: Optional[str] = None
    opening_name: Optional[str] = None
    theory_deviation: bool = False

    # Tactics & structure
    tactical_patterns: list[str] = field(default_factory=list)
    pawn_structure: dict = field(default_factory=dict)
    piece_activity: dict = field(default_factory=dict)

    # Coaching
    threat_narrative: str = ""


# Large centipawn value used when converting mate-in-N to centipawns.
_MATE_CP_BASE = 10000


def _mate_to_cp(mate_in: Optional[int]) -> Optional[int]:
    """Convert mate-in-N to a centipawn equivalent. Returns None if mate_in is None."""
    if mate_in is None:
        return None
    if mate_in > 0:
        return _MATE_CP_BASE - mate_in * 10
    return -_MATE_CP_BASE - mate_in * 10


# Legacy cp-loss classifier — kept for backward compatibility but no longer
# used as the primary classification method. EP-based classification is primary.
def _classify_cp_loss(cp_loss: Optional[int]) -> str:
    if cp_loss is None:
        return "unknown"
    if cp_loss == 0:
        return "best"
    if cp_loss <= 10:
        return "excellent"
    if cp_loss <= 30:
        return "good"
    if cp_loss < 100:
        return "inaccuracy"
    if cp_loss < 300:
        return "mistake"
    return "blunder"


class ContextAssembler:
    def __init__(
        self,
        eco_lookup: Optional[ECOLookup] = None,
        classification_config: Optional[ClassificationConfig] = None,
    ):
        self._eco = eco_lookup or ECOLookup()
        self._config = classification_config or ClassificationConfig()

    def assemble(
        self,
        board: chess.Board,
        engine_result: EngineResult,
        played_move: Optional[chess.Move] = None,
        player_elo: Optional[int] = None,
        opponent_elo: Optional[int] = None,
        prev_context: Optional[AssembledContext] = None,
    ) -> AssembledContext:
        best = engine_result.best
        best_move_san = board.san(best.move)
        pv_san = self._pv_to_san(board, best.pv)

        # --- Raw centipawn loss (unchanged from original) ---
        cp_loss = None
        played_move_san = None
        played_candidate = None
        if played_move is not None:
            played_move_san = board.san(played_move)
            played_candidate = self._find_candidate(engine_result, played_move)
            played_cp = self._cp_for_move(engine_result, played_move)
            best_cp = best.score_cp if best.score_cp is not None else _mate_to_cp(best.mate_in)
            if played_cp is not None and best_cp is not None:
                cp_loss = abs(best_cp - played_cp)

        # --- EP-based classification ---
        provider = resolve_provider(engine_result, self._config, player_elo=player_elo)
        wdl_source = "native" if isinstance(provider, StockfishNativeWDLProvider) else "sigmoid"

        best_win_pct = provider.get_win_pct(
            cp=best.score_cp, mate_in=best.mate_in, elo=player_elo,
            wdl_win=best.wdl_win, wdl_draw=best.wdl_draw, wdl_loss=best.wdl_loss,
        )

        played_win_pct = None
        if played_candidate is not None:
            played_win_pct = provider.get_win_pct(
                cp=played_candidate.score_cp, mate_in=played_candidate.mate_in,
                elo=player_elo,
                wdl_win=played_candidate.wdl_win, wdl_draw=played_candidate.wdl_draw,
                wdl_loss=played_candidate.wdl_loss,
            )

        ep_loss_val = None
        if best_win_pct is not None and played_win_pct is not None:
            # Win probabilities are from white's perspective (score.white()).
            # For white moves: losing EP means best_wp > played_wp (eval dropped).
            # For black moves: losing EP means played_wp > best_wp (white got better,
            # i.e. black got worse). Equivalently: black's wp = 1 - white's wp,
            # so black's EP loss = (1-best_wp) - (1-played_wp) = played_wp - best_wp.
            if board.turn == chess.WHITE:
                ep_loss_val = max(0.0, best_win_pct - played_win_pct)
            else:
                ep_loss_val = max(0.0, played_win_pct - best_win_pct)

        # Primary label from EP (replaces old cp-based classification)
        label = classify_ep_loss(ep_loss_val, self._config)

        # --- Special classification checks ---
        is_brilliant = False
        is_great = False
        is_miss = False

        if played_move is not None and ep_loss_val is not None and best_win_pct is not None and played_win_pct is not None:
            side = board.turn

            # Check Brilliant
            board_after = board.copy()
            board_after.push(played_move)
            is_brilliant = detect_brilliant(
                board_before=board, board_after=board_after, move=played_move,
                ep_loss=ep_loss_val, win_pct_before=best_win_pct,
                win_pct_after=played_win_pct, best_pv=best.pv, best_mate_in=best.mate_in,
                side=side, elo=player_elo, config=self._config,
            )

            # Check Great Move
            is_great = detect_great(
                ep_loss=ep_loss_val, win_pct_before=best_win_pct,
                win_pct_after=played_win_pct,
                candidates=engine_result.candidates,
                provider=provider, elo=player_elo, config=self._config,
                prev_context=prev_context,
            )

            # Check Miss (requires previous move context)
            if prev_context is not None:
                is_miss = detect_miss(
                    prev_context=prev_context, best_win_pct=best_win_pct,
                    played_win_pct=played_win_pct, ep_loss=ep_loss_val,
                    elo=player_elo, config=self._config,
                )

            # Override label for special classifications
            if is_brilliant:
                label = "brilliant"
            elif is_great and label in ("best", "excellent", "good"):
                label = "great"
            elif is_miss and label in ("best", "excellent", "good"):
                label = "miss"

        # Opening
        eco_code, opening_name = self._eco.lookup(board)
        theory_deviation = self._detect_deviation(board, played_move, eco_code)

        # Tactics, pawns, pieces
        tactical_patterns = detect_tactics(board)
        pawn_structure = analyze_pawns(board)
        piece_activity = self._piece_activity(board)
        threat_narrative = self._build_threat_narrative(board)

        return AssembledContext(
            fen=engine_result.fen,
            played_move=played_move,
            played_move_san=played_move_san,
            best_move=best.move,
            best_move_san=best_move_san,
            best_move_cp=best.score_cp,
            mate_in=best.mate_in,
            played_move_cp_loss=cp_loss,
            cp_loss_label=label,
            pv_san=pv_san,
            win_pct_before=best_win_pct,
            win_pct_after=played_win_pct,
            ep_loss=ep_loss_val,
            player_elo=player_elo,
            opponent_elo=opponent_elo,
            wdl_source=wdl_source,
            is_brilliant=is_brilliant,
            is_great=is_great,
            is_miss=is_miss,
            eco_code=eco_code,
            opening_name=opening_name,
            theory_deviation=theory_deviation,
            tactical_patterns=tactical_patterns,
            pawn_structure=pawn_structure,
            piece_activity=piece_activity,
            threat_narrative=threat_narrative,
        )

    def _find_candidate(self, result: EngineResult, move: chess.Move) -> Optional[CandidateMove]:
        """Find the CandidateMove entry for the played move."""
        for candidate in result.candidates:
            if candidate.move == move:
                return candidate
        return None

    def _cp_for_move(self, result: EngineResult, move: chess.Move) -> Optional[int]:
        for candidate in result.candidates:
            if candidate.move == move:
                if candidate.score_cp is not None:
                    return candidate.score_cp
                return _mate_to_cp(candidate.mate_in)
        return None

    def _pv_to_san(self, board: chess.Board, pv: list[chess.Move]) -> list[str]:
        san_list = []
        temp = board.copy()
        for move in pv[:6]:
            try:
                san_list.append(temp.san(move))
                temp.push(move)
            except Exception:
                break
        return san_list

    def _detect_deviation(
        self,
        board: chess.Board,
        played_move: Optional[chess.Move],
        eco_code: Optional[str],
    ) -> bool:
        """
        Check if the played move deviates from known opening theory.

        Pseudocode for full implementation:
          1. If eco_code is None, we're outside book territory → not a deviation
          2. Look up the ECO book continuation for this position
             (requires an extended eco_data.json with move sequences, or a polyglot book)
          3. If the played move matches the book continuation → no deviation
          4. If the played move does NOT match → deviation detected
          5. Also flag deviation if the position was in-book on the previous move
             but the current position is no longer recognized
        """
        if played_move is None or eco_code is None:
            return False

        # Check if the position AFTER the played move is still in the book.
        # If the current position is in the book but the resulting position is not,
        # the played move is a deviation from theory.
        temp = board.copy()
        temp.push(played_move)
        next_eco, _ = self._eco.lookup(temp)
        if next_eco is None and eco_code is not None:
            # Was in book, now out of book → deviation
            return True

        return False

    def _piece_activity(self, board: chess.Board) -> dict:
        """
        Extract piece activity signals from the position.

        Returns dict with keys: bishop_pair, rooks_on_open_files,
        knight_outposts, king_safety_score.
        """
        result = {}

        for color_name, color in [("white", chess.WHITE), ("black", chess.BLACK)]:
            bishops = board.pieces(chess.BISHOP, color)
            rooks = board.pieces(chess.ROOK, color)

            # Bishop pair: having two bishops is a positional advantage
            result[f"{color_name}_bishop_pair"] = len(bishops) >= 2

            # Rooks on open files: a file with no pawns of either color
            rooks_on_open = []
            for rook_sq in rooks:
                f = chess.square_file(rook_sq)
                file_mask = chess.BB_FILES[f]
                white_pawns_on_file = board.pieces(chess.PAWN, chess.WHITE) & file_mask
                black_pawns_on_file = board.pieces(chess.PAWN, chess.BLACK) & file_mask
                if not white_pawns_on_file and not black_pawns_on_file:
                    rooks_on_open.append(chess.square_name(rook_sq))
            result[f"{color_name}_rooks_open_files"] = rooks_on_open

            knights = board.pieces(chess.KNIGHT, color)
            outposts = []
            for knight_sq in knights:
                rank = chess.square_rank(knight_sq)
                if color == chess.WHITE and 3 <= rank <= 5:
                    outposts.append(chess.square_name(knight_sq))
                elif color == chess.BLACK and 2 <= rank <= 4:
                    outposts.append(chess.square_name(knight_sq))
            result[f"{color_name}_knight_outposts"] = outposts

            king_sq = board.king(color)
            if king_sq is not None:
                enemy = not color
                king_zone = chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]) | chess.SquareSet.from_square(king_sq)
                attacker_count = 0
                for sq in king_zone:
                    attackers = board.attackers(enemy, sq)
                    attacker_count += len(attackers)
                result[f"{color_name}_king_danger"] = attacker_count
            else:
                result[f"{color_name}_king_danger"] = 0

        return result

    def _build_threat_narrative(self, board: chess.Board) -> str:
        """
        Build a human-readable description of threats in the position.

        Uses python-chess attack maps to name which pieces are attacked
        and which squares are controlled. Formatted for use in coaching
        nudge prompts.
        """
        side = board.turn
        enemy = not side
        side_name = "White" if side == chess.WHITE else "Black"
        enemy_name = "Black" if side == chess.WHITE else "White"

        threats = []

        # Find enemy pieces that are attacked by the side to move
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.color != enemy:
                continue

            attackers = board.attackers(side, square)
            if not attackers:
                continue

            piece_name = PIECE_NAMES.get(piece.piece_type, "piece")
            square_name = chess.square_name(square)

            # Check if the attacked piece is defended
            defenders = board.attackers(enemy, square)
            if defenders:
                threats.append(
                    f"{enemy_name}'s {piece_name} on {square_name} is attacked and defended"
                )
            else:
                threats.append(
                    f"{enemy_name}'s {piece_name} on {square_name} is attacked and undefended"
                )

        # Find hanging pieces (undefended pieces of the side to move attacked by enemy)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.color != side:
                continue

            enemy_attackers = board.attackers(enemy, square)
            if not enemy_attackers:
                continue

            own_defenders = board.attackers(side, square)
            if not own_defenders:
                piece_name = PIECE_NAMES.get(piece.piece_type, "piece")
                square_name = chess.square_name(square)
                threats.append(
                    f"{side_name}'s {piece_name} on {square_name} is hanging (undefended)"
                )

        if not threats:
            return "No immediate threats detected."

        return "; ".join(threats[:5])  # Cap at 5 to keep prompt concise
