"""
Special move detection — Brilliant (!!), Great (!), and Miss.

These classifications use logic beyond EP thresholds. They are evaluated after
the base EP classification and can override it.

- Brilliant: queen/rook sacrifice that is the best move, leading to decisive advantage
- Great: turning move (losing→equal), seizing move (equal→winning), or only move
- Miss: player failed to capitalize on opponent's mistake
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import chess

from config.classification import ClassificationConfig

if TYPE_CHECKING:
    from analysis.context import AssembledContext
    from analysis.expected_points import WDLProvider
    from chess_engine.service import CandidateMove, EngineResult

# Piece values for material calculation
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def _material_balance(board: chess.Board, color: chess.Color) -> int:
    """Sum of piece values for one side."""
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        total += len(board.pieces(piece_type, color)) * value
    return total


def _is_heavy_piece_sacrifice(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
    side: chess.Color,
    elo: Optional[int],
    config: ClassificationConfig,
) -> bool:
    """
    Determine if the move constitutes a heavy piece sacrifice.

    At Elo < 1200, minor piece sacrifices (3+ points) also qualify.
    At all other Elo levels, the sacrifice must cost at least 5 points (rook value).
    Recaptures of equal or greater value are excluded.
    """
    mat_before = _material_balance(board_before, side)
    mat_after = _material_balance(board_after, side)
    mat_delta = mat_before - mat_after  # positive = lost material

    if elo is not None and elo < config.brilliant_low_elo_threshold:
        min_sacrifice = config.brilliant_low_elo_material_sacrifice
    else:
        min_sacrifice = config.brilliant_min_material_sacrifice

    if mat_delta < min_sacrifice:
        return False

    # Exclude recaptures: if we captured a piece worth >= what we lost, it's a trade
    if board_before.is_capture(move):
        captured_piece = board_before.piece_at(move.to_square)
        if captured_piece:
            captured_value = PIECE_VALUES.get(captured_piece.piece_type, 0)
            if captured_value >= mat_delta:
                return False

    return True


def _material_recovered_in_pv(
    board: chess.Board,
    pv: list[chess.Move],
    side: chess.Color,
    depth: int = 6,
) -> bool:
    """
    Check if sacrificed material is recovered within `depth` half-moves of the PV.
    If so, it's a tactical exchange, not a lasting sacrifice.
    """
    initial_balance = _material_balance(board, side)
    temp = board.copy()
    for move in pv[:depth]:
        try:
            temp.push(move)
        except Exception:
            break
        if _material_balance(temp, side) >= initial_balance:
            return True
    return False


def _pv_has_checkmate(board: chess.Board, pv: list[chess.Move], max_depth: int = 6) -> bool:
    """
    Walk the PV and check if any resulting position is checkmate.

    We check a few positions deep into the PV to catch mates that arise
    from the sacrifice, even if the player doesn't play perfectly thereafter.
    """
    temp = board.copy()
    for move in pv[:max_depth]:
        try:
            temp.push(move)
        except Exception:
            break
        if temp.is_checkmate():
            return True
    return False


def detect_brilliant(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
    ep_loss: float,
    win_pct_before: float,
    win_pct_after: float,
    best_pv: list[chess.Move],
    best_mate_in: Optional[int],
    side: chess.Color,
    elo: Optional[int] = None,
    config: Optional[ClassificationConfig] = None,
) -> bool:
    """
    Returns True if the move qualifies as Brilliant (!!).

    Brilliant = queen or rook sacrifice that is the best move, in a non-winning
    position, leading to a decisive advantage or mate, with material not recovered
    in the near-term PV.
    """
    c = config or ClassificationConfig()

    # Elo-dependent thresholds
    if elo is not None and elo < c.brilliant_low_elo_max_win_pct_threshold:
        max_win_before = c.brilliant_low_elo_max_win_pct_before
    else:
        max_win_before = c.brilliant_max_win_pct_before

    # Condition 1: move is best or near-best
    if ep_loss > c.brilliant_ep_tolerance:
        return False

    # Condition 2: heavy piece sacrifice (or minor piece at low Elo)
    if not _is_heavy_piece_sacrifice(board_before, board_after, move, side, elo, c):
        return False

    # Condition 3: not losing after the sacrifice
    if win_pct_after < c.brilliant_min_win_pct_after:
        return False

    # Condition 4: not already completely winning before the sacrifice
    if win_pct_before > max_win_before:
        return False

    # Condition 5: material not recovered quickly in PV
    if _material_recovered_in_pv(board_after, best_pv, side, depth=c.brilliant_pv_depth_check):
        return False

    # Condition 6: sacrifice leads to something decisive
    # Check for mate in engine result or in PV positions
    has_mate = (best_mate_in is not None and best_mate_in > 0)
    if not has_mate:
        has_mate = _pv_has_checkmate(board_after, best_pv, max_depth=c.brilliant_pv_depth_check)
    if not has_mate and win_pct_after < c.brilliant_decisive_win_pct:
        return False

    return True


def _elo_thresholds(elo: Optional[int]) -> dict:
    """
    Return Elo-dependent thresholds for Great Move detection.

    Thresholds are linearly interpolated between anchor points.
    """
    if elo is None:
        elo = 1500  # default to intermediate

    # Anchor points from PRD: 1000, 1500, 2000+
    if elo >= 2000:
        return {
            "losing": 0.35,
            "equal_lower": 0.40,
            "equal_upper": 0.60,
            "equal_threshold": 0.45,
            "winning": 0.65,
        }
    elif elo >= 1500:
        t = (elo - 1500) / 500.0
        return {
            "losing": 0.30 + t * 0.05,
            "equal_lower": 0.35 + t * 0.05,
            "equal_upper": 0.65 - t * 0.05,
            "equal_threshold": 0.40 + t * 0.05,
            "winning": 0.70 - t * 0.05,
        }
    else:
        t = max(0.0, (elo - 1000) / 500.0)
        return {
            "losing": 0.25 + t * 0.05,
            "equal_lower": 0.30 + t * 0.05,
            "equal_upper": 0.70 - t * 0.05,
            "equal_threshold": 0.35 + t * 0.05,
            "winning": 0.75 - t * 0.05,
        }


def _is_recapture(board: chess.Board, move: chess.Move) -> bool:
    """
    Check if a move is a recapture — capturing on the same square where the
    opponent just captured. Recaptures are routine and should not qualify as "great".
    """
    if not board.is_capture(move):
        return False
    # Check if the opponent's last move was a capture on the same square
    if len(board.move_stack) == 0:
        return False
    prev_move = board.move_stack[-1]
    return prev_move.to_square == move.to_square


def detect_great(
    ep_loss: float,
    win_pct_before: float,
    win_pct_after: float,
    candidates: list[CandidateMove],
    provider: WDLProvider,
    board: Optional[chess.Board] = None,
    move: Optional[chess.Move] = None,
    elo: Optional[int] = None,
    config: Optional[ClassificationConfig] = None,
    prev_context: Optional[AssembledContext] = None,
    is_engine_top: bool = False,
    candidate_gap_cp: Optional[int] = None,
) -> bool:
    """
    Returns True if the move qualifies as a Great Move (!).

    Based on chess.com data analysis, "great" moves are characterized by:
      - The played move is ALWAYS the engine's #1 choice
      - The gap between the best and 2nd-best move is large (median ~269cp)
      - The move is essentially the "only good move" in the position
      - The move is NOT a recapture or forced/obvious response

    Triggers (in priority order):
      A. Candidate gap: engine's top move with a large gap to alternatives.
         This is the primary trigger — chess.com great moves have 100% top-move
         rate and huge candidate gaps.
      B. Capitalization: opponent's previous move was a non-trivial mistake
         and the player found the best response with a significant gap.
    """
    c = config or ClassificationConfig()

    # Must be the engine's top choice (or very near-best) for any "great"
    if not is_engine_top or ep_loss > c.great_ep_tolerance:
        return False

    # Filter: recaptures are routine — large candidate gap is just because
    # not recapturing is terrible, not because the move is hard to find.
    if board is not None and move is not None and _is_recapture(board, move):
        return False

    # Filter: in already heavily-won positions, finding the "best" move is
    # less impressive — chess.com rarely awards "great" when WP > 0.90.
    if win_pct_before > c.great_max_win_pct_before or win_pct_before < c.great_min_win_pct_before:
        return False

    # A. Candidate gap: the played move is uniquely strong — all alternatives
    #    are significantly worse. This is the primary "great" signal.
    if candidate_gap_cp is not None and candidate_gap_cp >= c.great_min_candidate_gap_cp:
        return True

    capitalization_gap_cp = max(
        1,
        int(round(c.great_min_candidate_gap_cp * c.great_capitalization_gap_scale)),
    )

    # B. Capitalization with gap: opponent made a moderate mistake and the
    #    player found the best response, but only if there's a meaningful gap
    #    to alternatives (otherwise it's just a routine best move).
    if prev_context is not None and candidate_gap_cp is not None:
        prev_ep_loss = prev_context.ep_loss
        if (prev_ep_loss is not None
                and prev_ep_loss >= c.great_capitalization_min_ep_loss
                and prev_ep_loss < c.great_capitalization_max_ep_loss
                and candidate_gap_cp >= capitalization_gap_cp):
            return True

    return False


def detect_miss(
    prev_context: AssembledContext,
    best_win_pct: float,
    played_win_pct: float,
    ep_loss: float,
    side: Optional[chess.Color] = None,
    elo: Optional[int] = None,
    config: Optional[ClassificationConfig] = None,
) -> bool:
    """
    Returns True if the player missed capitalizing on opponent's mistake.

    Miss = opponent's previous move was a mistake/blunder (EP loss >= threshold),
    and the player's move loses significant expected points (ep_loss >= threshold).
    The key idea: the opponent gave away advantage, and the player failed to
    take it — like a "great" move that wasn't chosen.

    Unlike inaccuracy/mistake/blunder, miss is contextual: the same EP loss
    that would be an inaccuracy in a normal position becomes a miss when the
    opponent just blundered.
    """
    c = config or ClassificationConfig()

    # Condition 1: opponent's previous move was a mistake or blunder
    prev_ep_loss = prev_context.ep_loss
    if prev_ep_loss is None or prev_ep_loss < c.miss_opponent_ep_loss_threshold:
        return False

    # Condition 2: player's move loses meaningful expected points
    # (the ep_loss is already side-corrected, so this works for both colors)
    if ep_loss < c.miss_min_ep_loss:
        return False

    return True
