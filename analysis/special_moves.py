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


def _static_exchange_eval(
    board: chess.Board,
    to_square: chess.Square,
    attacker_square: chess.Square,
) -> int:
    """
    Standard swap-list SEE. Returns the net material gain, in pawns, for the
    side initiating the capture on `to_square` with the piece on
    `attacker_square`, assuming both sides play optimal recaptures using their
    smallest available attacker.

    Positive = attacker ends up ahead; 0 = clean trade; negative = losing
    capture. Pins and X-ray attacks through the pushed attacker are not
    modeled — this is a one-square exchange heuristic, not a full tactical
    search.
    """
    target = board.piece_at(to_square)
    attacker = board.piece_at(attacker_square)
    if target is None or attacker is None:
        return 0

    gains = [PIECE_VALUES.get(target.piece_type, 0)]
    on_square_value = PIECE_VALUES.get(attacker.piece_type, 0)

    temp = board.copy(stack=False)
    initial_move = chess.Move(attacker_square, to_square)
    if attacker.piece_type == chess.PAWN and chess.square_rank(to_square) in (0, 7):
        initial_move = chess.Move(attacker_square, to_square, promotion=chess.QUEEN)
    if initial_move not in temp.legal_moves:
        return 0
    temp.push(initial_move)

    while True:
        side_to_capture = temp.turn
        attackers = temp.attackers(side_to_capture, to_square)
        if not attackers:
            break
        smallest_sq = None
        smallest_val = 10
        for sq in attackers:
            piece = temp.piece_at(sq)
            if piece is None:
                continue
            val = PIECE_VALUES.get(piece.piece_type, 0)
            if val < smallest_val:
                smallest_val = val
                smallest_sq = sq
        if smallest_sq is None:
            break
        recapture = chess.Move(smallest_sq, to_square)
        piece = temp.piece_at(smallest_sq)
        if piece.piece_type == chess.PAWN and chess.square_rank(to_square) in (0, 7):
            recapture = chess.Move(smallest_sq, to_square, promotion=chess.QUEEN)
        if recapture not in temp.legal_moves:
            break
        gains.append(on_square_value - gains[-1])
        on_square_value = smallest_val
        temp.push(recapture)

    for i in range(len(gains) - 1, 0, -1):
        gains[i - 1] = -max(-gains[i - 1], gains[i])
    return gains[0]


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


def is_concrete_blunder(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
    played_candidate: Optional[CandidateMove],
    side: chess.Color,
    ep_loss: Optional[float] = None,
    config: Optional[ClassificationConfig] = None,
) -> bool:
    """
    Chess.com's blunder requires concrete damage, not just a large EP drop.
    A move is a concrete blunder if it:
      1. Allows forced mate (opponent has mate after this move)
      2. Loses material on the move (hangs a piece, losing trade)
      3. Leaves material hanging — opponent has a capture whose static
         exchange evaluation nets at least `blunder_min_hanging_see` pawns
      4. Has a large EP loss combined with any concrete threat (SEE>=1 or
         EP >= blunder_pure_ep_floor — the latter covers multi-move tactical
         sequences beyond the shallow exchange check)

    The SEE check in gate 3 replaces an earlier 1-ply "is this piece defended?"
    test that over-fired on undefended pawns and on captures that walked into
    recapture sequences losing for the attacker.
    """
    c = config or ClassificationConfig()

    # Gate 1: allows forced mate
    if played_candidate is not None and played_candidate.mate_in is not None:
        if played_candidate.mate_in < 0:
            return True

    # Gate 2: loses material on the move itself
    mat_before = _material_balance(board_before, side)
    mat_after = _material_balance(board_after, side)
    opp_mat_before = _material_balance(board_before, not side)
    opp_mat_after = _material_balance(board_after, not side)

    # Net material swing: if our relative material dropped, we lost something
    net_before = mat_before - opp_mat_before
    net_after = mat_after - opp_mat_after
    if net_after < net_before:
        return True

    # Gate 3: leaves material hanging (SEE-verified)
    best_see = 0
    for opp_move in board_after.legal_moves:
        if not board_after.is_capture(opp_move):
            continue
        if board_after.is_en_passant(opp_move):
            see_value = PIECE_VALUES[chess.PAWN]
        else:
            see_value = _static_exchange_eval(
                board_after, opp_move.to_square, opp_move.from_square
            )
        if see_value > best_see:
            best_see = see_value

    if best_see >= c.blunder_min_hanging_see:
        return True

    # Gate 4: large EP loss — require at least some concrete material signal
    # to avoid flagging pure-positional collapses as blunders. The
    # `blunder_pure_ep_floor` override keeps truly catastrophic drops (which
    # almost always hide multi-move tactics) in the blunder bucket.
    if ep_loss is not None:
        if ep_loss >= c.blunder_ep_floor and best_see >= 1:
            return True
        if ep_loss >= c.blunder_pure_ep_floor:
            return True

    return False


def detect_brilliant(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
    ep_loss: float,
    win_pct_before: float,
    win_pct_after: float,
    played_pv: list[chess.Move],
    played_mate_in: Optional[int],
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
    mover_win_pct_before = _white_pov_to_side_win_pct(win_pct_before, side)
    mover_win_pct_after = _white_pov_to_side_win_pct(win_pct_after, side)

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
    if mover_win_pct_after < c.brilliant_min_win_pct_after:
        return False

    # Condition 4: not already completely winning before the sacrifice
    if mover_win_pct_before > max_win_before:
        return False

    # Condition 5: material not recovered quickly in PV
    if _material_recovered_in_pv(board_after, played_pv, side, depth=c.brilliant_pv_depth_check):
        return False

    # Condition 6: sacrifice leads to something decisive
    # Check for mate in engine result or in PV positions
    has_mate = (played_mate_in is not None and played_mate_in > 0)
    if not has_mate:
        has_mate = _pv_has_checkmate(board_after, played_pv, max_depth=c.brilliant_pv_depth_check)
    if not has_mate and mover_win_pct_after < c.brilliant_decisive_win_pct:
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
    if prev_move.to_square != move.to_square:
        return False
    # The previous move must also have been a capture (not just a move to that square).
    # We need the board state before the previous move to check this.
    temp = board.copy()
    temp.pop()  # undo opponent's last move
    return temp.is_capture(prev_move)


def _is_response_capture_same_square(board: chess.Board, move: chess.Move) -> bool:
    """Return True when capturing the piece that just moved onto that square."""
    return (
        board.is_capture(move)
        and len(board.move_stack) > 0
        and board.move_stack[-1].to_square == move.to_square
    )


def _white_pov_to_side_win_pct(
    win_pct: float,
    side: Optional[chess.Color],
) -> float:
    """Convert white-perspective win probability to the mover's perspective."""
    if side == chess.BLACK:
        return 1.0 - win_pct
    return win_pct


def detect_great(
    ep_loss: float,
    win_pct_before: float,
    win_pct_after: float,
    candidates: list[CandidateMove],
    provider: WDLProvider,
    board: Optional[chess.Board] = None,
    move: Optional[chess.Move] = None,
    side: Optional[chess.Color] = None,
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

    `win_pct_before` and `win_pct_after` are expected to be in white's
    perspective, matching engine output. Threshold checks are applied from the
    mover's perspective after normalization.

    Triggers (in priority order):
      A. Candidate gap: engine's top move with a large gap to alternatives.
         This is the primary trigger — chess.com great moves have 100% top-move
         rate and huge candidate gaps.
      B. Capitalization: opponent's previous move was a non-trivial mistake
         and the player found the best response with a significant gap.
      E. Mate-finding: the best move leads to forced mate and alternatives don't.
    """
    c = config or ClassificationConfig()
    side_to_move = side if side is not None else (board.turn if board is not None else None)
    mover_win_pct_before = _white_pov_to_side_win_pct(win_pct_before, side_to_move)

    # Must be the engine's top choice (or very near-best) for any "great"
    if not is_engine_top or ep_loss > c.great_ep_tolerance:
        return False

    # Filter: recaptures are routine — large candidate gap is just because
    # not recapturing is terrible, not because the move is hard to find.
    if board is not None and move is not None and _is_recapture(board, move):
        return False

    # Filter: moves played while in check with very few legal options are
    # forced, not great — the gap comes from having no real choice.
    # When more options exist, the move can still be great (finding the
    # one good response among several check-escaping moves).
    if board is not None and board.is_check():
        legal_count = board.legal_moves.count()
        if legal_count <= c.great_max_forced_check_moves:
            return False

    # E. Mate-finding move: the best move leads to forced mate with a
    #    significant gap to alternatives. These are great regardless of
    #    the current win probability (even in winning positions).
    if (candidate_gap_cp is not None
            and candidate_gap_cp >= c.great_min_candidate_gap_cp
            and len(candidates) >= 1
            and candidates[0].mate_in is not None
            and candidates[0].mate_in > 0):
        # Best move leads to mate — check that alternatives don't also mate
        if len(candidates) < 2 or candidates[1].mate_in is None:
            return True

    # A. Candidate gap: the played move is uniquely strong — all alternatives
    #    are significantly worse. This is the primary "great" signal.
    #    Only fires in non-decisive positions (chess.com rarely awards "great"
    #    when WP > 0.90 or WP < 0.10).
    if (candidate_gap_cp is not None
            and candidate_gap_cp >= c.great_min_candidate_gap_cp
            and c.great_min_win_pct_before <= mover_win_pct_before <= c.great_max_win_pct_before):
        return True

    capitalization_gap_cp = max(
        1,
        int(round(c.great_min_candidate_gap_cp * c.great_capitalization_gap_scale)),
    )

    # B. Capitalization with gap: opponent made a moderate mistake and the
    #    player found the best response, but only if there's a meaningful gap
    #    to alternatives (otherwise it's just a routine best move).
    #    Response captures (capturing a piece that just moved to that square)
    #    need a higher gap since they are more obvious.
    if (prev_context is not None
            and candidate_gap_cp is not None
            and c.great_min_win_pct_before <= mover_win_pct_before <= c.great_max_win_pct_before):
        prev_ep_loss = prev_context.ep_loss
        effective_cap_gap = capitalization_gap_cp
        if board is not None and move is not None and _is_response_capture_same_square(board, move):
            # Capture responds to opponent's last move on the same square —
            # more obvious, require 1.5x the normal capitalization gap.
            effective_cap_gap = max(
                effective_cap_gap,
                int(round(capitalization_gap_cp * 1.5)),
            )
        if (prev_ep_loss is not None
                and prev_ep_loss >= c.great_capitalization_min_ep_loss
                and prev_ep_loss < c.great_capitalization_max_ep_loss
                and candidate_gap_cp >= effective_cap_gap):
            return True

    # B2. Strong capitalization after a clear mistake/blunder. These are often
    #     marked "great" even when the mover was already somewhat better before
    #     the punishment, provided the reply is still uniquely strong.
    if prev_context is not None and candidate_gap_cp is not None:
        prev_ep_loss = prev_context.ep_loss
        prev_label = prev_context.cp_loss_label
        effective_gap = c.great_post_blunder_min_gap_cp
        if board is not None and move is not None and _is_response_capture_same_square(board, move):
            effective_gap = max(effective_gap, int(round(effective_gap * 1.5)))
        if (
            prev_ep_loss is not None
            and prev_ep_loss >= c.great_post_blunder_min_prev_ep_loss
            and prev_label in {"mistake", "blunder", "miss"}
            and candidate_gap_cp >= effective_gap
            and mover_win_pct_before <= c.great_post_blunder_max_win_pct_before
        ):
            return True

    return False


def _best_move_wins_material(
    board: chess.Board,
    pv: list[chess.Move],
    side: chess.Color,
    depth: int = 4,
) -> bool:
    """
    Check if following the best PV results in material gain within `depth` half-moves.
    Uses relative material balance (our material minus opponent material) so that
    exchanges don't count — only net gains.
    """
    initial = _material_balance(board, side) - _material_balance(board, not side)
    temp = board.copy()
    for move in pv[:depth]:
        try:
            temp.push(move)
        except Exception:
            break
        current = _material_balance(temp, side) - _material_balance(temp, not side)
        if current > initial:
            return True
    return False


def _has_concrete_opportunity(
    best_candidate: Optional[CandidateMove],
    board_before: Optional[chess.Board],
    mover_best_win_pct: float,
    side: Optional[chess.Color],
    config: ClassificationConfig,
) -> bool:
    """
    True if the best move leads to a concrete tactical gain: forced mate,
    material win inside the PV, or a win-probability threshold crossing.
    """
    if best_candidate is not None:
        if best_candidate.mate_in is not None and best_candidate.mate_in > 0:
            return True

        if board_before is not None and best_candidate.pv:
            temp = board_before.copy()
            try:
                temp.push(best_candidate.move)
                if _best_move_wins_material(
                    temp, best_candidate.pv[1:], side or chess.WHITE,
                    depth=config.miss_best_wins_material_depth,
                ):
                    return True
            except Exception:
                pass

    return mover_best_win_pct >= config.miss_best_win_pct_threshold


def _best_move_is_forcing(
    best_candidate: CandidateMove,
    board_before: chess.Board,
) -> bool:
    """
    True if the best move is a check, capture, or leads to forced mate —
    the tactical-character signal used by Trigger C.
    """
    if best_candidate.mate_in is not None and best_candidate.mate_in > 0:
        return True
    if board_before.is_capture(best_candidate.move):
        return True
    temp = board_before.copy()
    try:
        temp.push(best_candidate.move)
    except Exception:
        return False
    return temp.is_check()


def detect_miss(
    prev_context: Optional[AssembledContext],
    best_win_pct: float,
    played_win_pct: float,
    ep_loss: float,
    best_candidate: Optional[CandidateMove] = None,
    board_before: Optional[chess.Board] = None,
    side: Optional[chess.Color] = None,
    elo: Optional[int] = None,
    config: Optional[ClassificationConfig] = None,
    candidate_gap_cp: Optional[int] = None,
) -> bool:
    """
    Returns True if the player missed capitalizing on a concrete opportunity.

    Two parallel triggers qualify as a miss — both require the player to have
    lost meaningful EP and a concrete opportunity to have existed:

    Trigger A (opponent-blunder): opponent's previous move had significant EP
    loss (or was labeled mistake/blunder/miss) and the best reply achieves a
    concrete gain.

    Trigger C (direct tactic missed): a significantly better move existed —
    large candidate gap between best and 2nd-best, best move is forcing
    (check/capture/mate), and the player wasn't already losing. Captures the
    pattern where the played move is "fine" in isolation but a decisive tactic
    was available. Fires independently of Trigger A's opponent-blunder gate.
    """
    c = config or ClassificationConfig()
    mover_best_win_pct = _white_pov_to_side_win_pct(best_win_pct, side)

    # Shared gate: player's move loses meaningful EP
    if ep_loss < c.miss_min_ep_loss:
        return False

    # Shared gate: best move leads to a concrete gain (mate, material, or WP threshold)
    if not _has_concrete_opportunity(
        best_candidate, board_before, mover_best_win_pct, side, c,
    ):
        return False

    # Trigger A: opponent's previous move created the opportunity
    trigger_a = False
    if prev_context is not None:
        prev_ep_loss = prev_context.ep_loss
        prev_label = prev_context.cp_loss_label
        if (
            (prev_ep_loss is not None and prev_ep_loss >= c.miss_opponent_ep_loss_threshold)
            or prev_label in {"mistake", "blunder", "miss"}
        ):
            trigger_a = True

    if trigger_a:
        return True

    # Trigger C: direct tactic missed — large candidate gap + forcing best move
    # Fires without an opponent-blunder context. Gate on mover not already
    # losing so we don't punish futile positions.
    if (
        best_candidate is not None
        and board_before is not None
        and candidate_gap_cp is not None
        and candidate_gap_cp >= c.miss_direct_tactic_min_gap_cp
        and mover_best_win_pct >= c.miss_direct_tactic_min_mover_wp
        and _best_move_is_forcing(best_candidate, board_before)
    ):
        return True

    return False
