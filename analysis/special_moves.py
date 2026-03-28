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
    max_win_before = 0.90 if (elo is not None and elo < 1500) else c.brilliant_max_win_pct_before

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


def detect_great(
    ep_loss: float,
    win_pct_before: float,
    win_pct_after: float,
    candidates: list[CandidateMove],
    provider: WDLProvider,
    elo: Optional[int] = None,
    config: Optional[ClassificationConfig] = None,
    prev_context: Optional[AssembledContext] = None,
) -> bool:
    """
    Returns True if the move qualifies as a Great Move (!).

    Great = one of:
      A. Capitalization: opponent's previous move was a mistake/blunder,
         and the player found the best (or near-best) response to exploit it.
         This is the most common "great" trigger on chess.com.
      B. Turning move: position swings from losing to equal or better.
      C. Seizing move: position swings from equal to winning.
      D. Only move: single non-losing candidate, and the player found it.
    """
    c = config or ClassificationConfig()
    th = _elo_thresholds(elo)

    # Must be the best or near-best move for any "great" classification
    if ep_loss > c.great_ep_tolerance:
        return False

    # A. Capitalization: opponent made a mistake and player punished it.
    #    Only fires for moderate mistakes (not massive blunders where exploitation
    #    is trivial). Exception: forced mate in the response is always "great".
    if prev_context is not None:
        prev_ep_loss = prev_context.ep_loss
        if prev_ep_loss is not None and prev_ep_loss >= c.great_capitalization_min_ep_loss:
            has_mate_response = (
                len(candidates) > 0
                and candidates[0].mate_in is not None
                and candidates[0].mate_in > 0
            )
            if has_mate_response or prev_ep_loss < c.great_capitalization_max_ep_loss:
                return True

    # B. Turning move: losing → equal or better
    if (win_pct_before < th["losing"]
            and win_pct_after >= th["equal_threshold"]):
        return True

    # C. Seizing move: equal → winning
    if (th["equal_lower"] <= win_pct_before <= th["equal_upper"]
            and win_pct_after > th["winning"]):
        return True

    # D. Only move: exactly one candidate within great_only_move_ep_gap of best
    if len(candidates) >= 2:
        best_wp = provider.get_win_pct(
            cp=candidates[0].score_cp, mate_in=candidates[0].mate_in, elo=elo,
            wdl_win=candidates[0].wdl_win, wdl_draw=candidates[0].wdl_draw,
            wdl_loss=candidates[0].wdl_loss,
        )
        if best_wp is not None:
            close_count = 0
            for cand in candidates:
                cand_wp = provider.get_win_pct(
                    cp=cand.score_cp, mate_in=cand.mate_in, elo=elo,
                    wdl_win=cand.wdl_win, wdl_draw=cand.wdl_draw,
                    wdl_loss=cand.wdl_loss,
                )
                if cand_wp is not None and (best_wp - cand_wp) <= c.great_only_move_ep_gap:
                    close_count += 1
            # Only one candidate is close to best → only move
            if close_count == 1:
                return True

    return False


def detect_miss(
    prev_context: AssembledContext,
    best_win_pct: float,
    played_win_pct: float,
    ep_loss: float,
    elo: Optional[int] = None,
    config: Optional[ClassificationConfig] = None,
) -> bool:
    """
    Returns True if the player missed capitalizing on opponent's mistake.

    Miss = opponent's previous move was a mistake/blunder (EP loss >= 0.10),
    the best reply would reach a winning position, and the player's actual
    reply does not reach it. Does not override worse labels (inaccuracy+).
    """
    c = config or ClassificationConfig()
    th = _elo_thresholds(elo)

    # Condition 1: opponent's previous move was a mistake or blunder
    prev_ep_loss = prev_context.ep_loss
    if prev_ep_loss is None or prev_ep_loss < c.miss_opponent_ep_loss_threshold:
        return False

    # Condition 2: best reply would reach a winning position
    if best_win_pct < th["winning"]:
        return False

    # Condition 3: player's actual move does NOT reach a winning position
    if played_win_pct >= th["winning"]:
        return False

    # Condition 4: the played move is not itself a blunder (label it blunder instead)
    if ep_loss >= c.ep_mistake:
        return False

    return True
