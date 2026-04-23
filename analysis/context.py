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
    draw_sensitivity_factor,
    pv_end_win_pct,
    resolve_provider,
)
from analysis.patterns import detect_tactics
from analysis.pawns import analyze_pawns
from analysis.special_moves import detect_brilliant, detect_great, detect_miss, is_concrete_blunder
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
    win_pct_before: Optional[float] = None   # white-perspective win probability at best eval (0.0-1.0)
    win_pct_after: Optional[float] = None    # white-perspective win probability at played eval (0.0-1.0)
    ep_loss: Optional[float] = None          # expected points lost by side to move
    player_elo: Optional[int] = None         # Elo used for curve adjustment
    opponent_elo: Optional[int] = None       # stored for context
    wdl_source: str = "sigmoid"              # "sigmoid" or "native"

    # Candidate analysis
    is_engine_top_move: bool = False         # played move == engine's #1 candidate
    candidate_gap_cp: Optional[int] = None   # centipawn gap between 1st and 2nd best candidate
    candidate_gap_wp: Optional[float] = None # win probability gap between 1st and 2nd best

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

        # --- Raw centipawn loss ---
        cp_loss = None
        played_move_san = None
        played_candidate = None
        played_move_rank = None
        if played_move is not None:
            played_move_san = board.san(played_move)
            played_candidate = self._find_candidate(engine_result, played_move)
            played_move_rank = self._candidate_rank(engine_result, played_move)
            played_cp = self._cp_for_move(engine_result, played_move)
            best_cp = best.score_cp if best.score_cp is not None else _mate_to_cp(best.mate_in)
            if played_cp is not None and best_cp is not None:
                cp_loss = abs(best_cp - played_cp)

        # --- Candidate analysis: is this the engine's top pick? ---
        is_engine_top = False
        candidate_gap_cp = None
        candidate_gap_wp = None
        if played_move is not None:
            is_engine_top = (played_move == best.move)

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

        # Candidate gap: how much better is the best move than the 2nd best?
        # This measures move uniqueness — large gap means this was the "only good move".
        if len(engine_result.candidates) >= 2:
            second = engine_result.candidates[1]
            second_cp = second.score_cp if second.score_cp is not None else _mate_to_cp(second.mate_in)
            best_cp_raw = best.score_cp if best.score_cp is not None else _mate_to_cp(best.mate_in)
            if best_cp_raw is not None and second_cp is not None:
                candidate_gap_cp = abs(best_cp_raw - second_cp)

            second_wp = provider.get_win_pct(
                cp=second.score_cp, mate_in=second.mate_in, elo=player_elo,
                wdl_win=second.wdl_win, wdl_draw=second.wdl_draw,
                wdl_loss=second.wdl_loss,
            )
            if best_win_pct is not None and second_wp is not None:
                if board.turn == chess.WHITE:
                    candidate_gap_wp = max(0.0, best_win_pct - second_wp)
                else:
                    candidate_gap_wp = max(0.0, second_wp - best_win_pct)

        ep_loss_val = None
        if best_win_pct is not None and played_win_pct is not None:
            if board.turn == chess.WHITE:
                ep_loss_val = max(0.0, best_win_pct - played_win_pct)
            else:
                ep_loss_val = max(0.0, played_win_pct - best_win_pct)

        # --- PV-end EP loss refinement (Theme 4) ---
        # The root eval for a non-top candidate only looks one move deep at
        # best; a shallow eval of +20cp can hide an 80cp collapse four plies
        # later. When both the best and played candidates carry a `pv_end`
        # re-eval (populated by the engine layer when `pv_end_nodes` is set),
        # recompute a stricter EP loss. For top-1/2/3 moves we compare both
        # PV endpoints, but for out-of-top-window plays (the searchmoves exact
        # eval stored on `EngineResult.played_move`) the played move's own PV
        # drift is the cleaner signal: use the root best eval against the
        # played PV endpoint to avoid letting best-move drift dilute the lift.
        is_rank4_plus = played_move_rank is not None and played_move_rank >= 4
        rank4_future_ep_loss = None
        rank4_future_lift = None
        if (
            self._config.pv_end_ep_loss_enabled
            and played_candidate is not None
            and ep_loss_val is not None
        ):
            played_wp_pv_end = pv_end_win_pct(
                played_candidate, provider, player_elo, self._config
            )
            min_root_ep = (
                self._config.pv_end_min_ep_loss_rank4
                if is_rank4_plus
                else self._config.pv_end_min_ep_loss
            )
            min_lift = (
                self._config.pv_end_min_lift_rank4
                if is_rank4_plus
                else self._config.pv_end_min_lift
            )
            ep_loss_pv_end = None
            if played_wp_pv_end is not None:
                if is_rank4_plus:
                    if board.turn == chess.WHITE:
                        rank4_future_ep_loss = max(0.0, best_win_pct - played_wp_pv_end)
                    else:
                        rank4_future_ep_loss = max(0.0, played_wp_pv_end - best_win_pct)
                    rank4_future_lift = rank4_future_ep_loss - ep_loss_val
            if ep_loss_val > min_root_ep and played_wp_pv_end is not None:
                if is_rank4_plus:
                    ep_loss_pv_end = rank4_future_ep_loss
                else:
                    best_wp_pv_end = pv_end_win_pct(best, provider, player_elo, self._config)
                    if best_wp_pv_end is not None:
                        if board.turn == chess.WHITE:
                            ep_loss_pv_end = max(0.0, best_wp_pv_end - played_wp_pv_end)
                        else:
                            ep_loss_pv_end = max(0.0, played_wp_pv_end - best_wp_pv_end)
            # Only override when the endpoint reveals materially worse drift
            # than the root eval. Small positive deltas (~5cp) are noise from
            # the shallow endpoint search and should not re-bucket the move.
            if (
                ep_loss_pv_end is not None
                and ep_loss_pv_end - ep_loss_val > min_lift
            ):
                ep_loss_val = max(ep_loss_val, ep_loss_pv_end)

        # --- Draw-sensitivity boost (Theme 2) ---
        # In structurally drawn positions (high `wdl_draw` on the best
        # candidate) chess.com penalises the same EP loss more harshly than
        # in decisive ones. The published EP thresholds are position-agnostic
        # so we match chess.com's behaviour by scaling the *classification*
        # EP loss in draw-heavy regimes before bucketing. The underlying
        # `ep_loss` we store on the context remains the raw/refined value so
        # special-move detectors and downstream analysis still operate on the
        # true expected-score drop rather than the draw-context adjustment.
        classification_ep_loss = ep_loss_val
        if (
            classification_ep_loss is not None
            and classification_ep_loss > self._config.draw_boost_min_ep_loss
        ):
            classification_ep_loss *= draw_sensitivity_factor(best.wdl_draw, self._config)

        # --- Base classification ---
        # Step 1: classify by EP loss thresholds
        # At low Elo, tighten the "excellent" threshold — the sigmoid curve
        # is flatter so small EP losses are less meaningful. Chess.com uses
        # a higher bar for "excellent" at lower Elo.
        effective_config = self._config
        if player_elo is not None and player_elo < self._config.excellent_elo_threshold:
            from dataclasses import replace as _replace
            effective_config = _replace(
                self._config,
                ep_excellent=self._config.ep_excellent_low_elo,
            )
        label = classify_ep_loss(classification_ep_loss, effective_config)

        # Step 2: if the played move is the engine's exact top choice AND
        # there's a meaningful gap to alternatives, promote to "best".
        # When multiple moves are nearly equal (small candidate gap),
        # being engine #1 is noise — leave as "excellent".
        if is_engine_top and label in ("best", "excellent"):
            min_gap = self._config.best_promotion_min_gap_cp
            if candidate_gap_cp is not None and candidate_gap_cp >= min_gap:
                label = "best"
            elif candidate_gap_cp is None:
                # No candidate gap info (single-PV analysis) — promote anyway
                label = "best"
        elif label == "best" and not is_engine_top and candidate_gap_cp is not None:
            in_gap_band = (
                self._config.best_non_top_excellent_min_gap_cp
                <= candidate_gap_cp
                < self._config.best_non_top_excellent_max_gap_cp
            )
            # Engine-equivalent: very small cp_loss means the move is practically
            # as good as #1; keep it labeled "best" to match chess.com.
            cp_loss_meaningful = (
                cp_loss is None
                or cp_loss >= self._config.best_non_top_excellent_min_cp_loss
            )
            if in_gap_band and cp_loss_meaningful:
                label = "excellent"

        # Step 3: rank-4+ exact-eval severity override.
        # These are the out-of-top-window plays where the shallow root eval most
        # often underrates the eventual damage. Use the played move's PV-end only
        # to bump one severity bucket, keeping the override narrow and product-
        # focused on good/inaccuracy/mistake recall rather than broad relabeling.
        if (
            is_rank4_plus
            and rank4_future_ep_loss is not None
            and rank4_future_lift is not None
        ):
            if (
                label == "good"
                and best.wdl_draw is not None
                and best.wdl_draw >= self._config.rank4_good_to_inaccuracy_min_draw_permille
                and rank4_future_ep_loss >= self._config.rank4_good_to_inaccuracy_min_future_ep
                and rank4_future_lift >= self._config.rank4_good_to_inaccuracy_min_lift
            ):
                label = "inaccuracy"
            elif (
                label == "inaccuracy"
                and rank4_future_ep_loss >= self._config.rank4_inaccuracy_to_mistake_min_future_ep
                and rank4_future_lift >= self._config.rank4_inaccuracy_to_mistake_min_lift
            ):
                label = "mistake"

        # --- Blunder concrete damage gate ---
        # Chess.com requires blunders to involve material loss or forced mate.
        # Positional collapses with large EP loss are demoted to "mistake".
        if label == "blunder" and played_move is not None and played_candidate is not None:
            board_after_gate = board.copy()
            board_after_gate.push(played_move)
            if not is_concrete_blunder(
                board,
                board_after_gate,
                played_move,
                played_candidate,
                board.turn,
                ep_loss=ep_loss_val,
                config=self._config,
            ):
                label = "mistake"

        # --- Special classification checks ---
        is_brilliant = False
        is_great = False
        is_miss = False

        if played_move is not None and ep_loss_val is not None and best_win_pct is not None and played_win_pct is not None:
            side = board.turn

            # Check Brilliant (sacrifice-based, unchanged)
            board_after = board.copy()
            board_after.push(played_move)
            is_brilliant = detect_brilliant(
                board_before=board,
                board_after=board_after,
                move=played_move,
                ep_loss=ep_loss_val,
                win_pct_before=best_win_pct,
                win_pct_after=played_win_pct,
                played_pv=played_candidate.pv[1:] if played_candidate is not None else [],
                played_mate_in=played_candidate.mate_in if played_candidate is not None else None,
                side=side,
                elo=player_elo,
                config=self._config,
            )

            # Check Great Move — driven by candidate gap + filters
            is_great = detect_great(
                ep_loss=ep_loss_val, win_pct_before=best_win_pct,
                win_pct_after=played_win_pct,
                candidates=engine_result.candidates,
                provider=provider, board=board, move=played_move,
                side=side,
                elo=player_elo, config=self._config,
                prev_context=prev_context,
                is_engine_top=is_engine_top,
                candidate_gap_cp=candidate_gap_cp,
            )

            # Check Miss — Trigger A needs prev_context, Trigger C does not.
            is_miss = detect_miss(
                prev_context=prev_context, best_win_pct=best_win_pct,
                played_win_pct=played_win_pct, ep_loss=ep_loss_val,
                best_candidate=best, board_before=board,
                side=side, elo=player_elo, config=self._config,
                candidate_gap_cp=candidate_gap_cp,
            )

            # Override label for special classifications
            # Priority: brilliant > great > miss > base label
            # Miss overrides inaccuracy/mistake too — the "missed opportunity"
            # context is more informative than the raw severity label.
            if is_brilliant:
                label = "brilliant"
            elif is_great and label in ("best", "excellent", "good"):
                label = "great"
            elif is_miss:
                if label == "blunder":
                    # Miss overrides blunder only when the opponent's
                    # previous mistake was very large — the "missed
                    # opportunity" context is strong enough to override
                    # the blunder classification.
                    if (prev_context is not None
                            and (
                                (prev_context.ep_loss is not None
                                 and prev_context.ep_loss >= self._config.miss_blunder_override_min_prev_ep)
                                or prev_context.cp_loss_label in ("mistake", "blunder", "miss")
                            )):
                        label = "miss"
                else:
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
            is_engine_top_move=is_engine_top,
            candidate_gap_cp=candidate_gap_cp,
            candidate_gap_wp=candidate_gap_wp,
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
        if result.played_move is not None and result.played_move.move == move:
            return result.played_move
        return None

    def _candidate_rank(self, result: EngineResult, move: chess.Move) -> Optional[int]:
        """Return the played move's rank, or a lower bound for out-of-window evals."""
        for idx, candidate in enumerate(result.candidates, start=1):
            if candidate.move == move:
                return idx
        if result.played_move is not None and result.played_move.move == move:
            # searchmoves exact evals are only populated when the move fell
            # outside the multipv window, so treat them as rank-(N+1)+
            return len(result.candidates) + 1
        return None

    def _cp_for_move(self, result: EngineResult, move: chess.Move) -> Optional[int]:
        for candidate in result.candidates:
            if candidate.move == move:
                if candidate.score_cp is not None:
                    return candidate.score_cp
                return _mate_to_cp(candidate.mate_in)
        if result.played_move is not None and result.played_move.move == move:
            if result.played_move.score_cp is not None:
                return result.played_move.score_cp
            return _mate_to_cp(result.played_move.mate_in)
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
