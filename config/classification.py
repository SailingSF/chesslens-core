"""
ClassificationConfig — tunable parameters for the EP-based move classification system.

All thresholds, sigmoid constants, and special move detection parameters live here
so they can be overridden per-call for A/B testing or per-mode tuning.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClassificationConfig:
    """Tunable parameters for the EP classification system."""

    # --- Sigmoid WDL provider ---
    # Base steepness constant for cp → win probability sigmoid.
    # Calibrated against chess.com Classification V2 using SF16.1 1M-node
    # analysis across 8 games (352 base-severity moves): k=0.0026 gives
    # 73.9% base-label accuracy vs 71.6% at the old Lichess default (0.00368).
    k_base: float = 0.00368

    # Elo scaling curve: scaling_factor = floor + range * sigmoid(steepness * (elo - midpoint))
    #
    # Chess.com applies Elo-dependent harshness: the same EP loss earns different
    # labels at different ratings. At 40cp loss in a draw-heavy position the
    # classification shifts from 76% "good" at sub-1000 ratings to 63%
    # "inaccuracy" at 1900+. To reproduce that behaviour we scale the sigmoid
    # steepness (k = k_base * scaling_factor), which amplifies ep_loss at high
    # Elo and compresses it at low Elo.
    #
    # Tuning (Theme 5, 796-game review corpus, ~46k comparable moves):
    #   baseline (floor=0.85, range=0.2, steep=0.003, midpoint=850)  → 70.67%
    #   tuned   (floor=0.85, range=0.6, steep=0.008, midpoint=1500)  → 72.72%
    # +2.05pp overall, +4.7pp at Elo≥1900, +6.2pp inaccuracy recall, +6.0pp
    # mistake recall; low-Elo buckets preserved (≤0.1pp drift). Floor stays at
    # 0.85 — any lower value crashed low-Elo accuracy sharply in the sweep.
    elo_scale_floor: float = 0.85
    elo_scale_range: float = 0.6
    elo_scale_steepness: float = 0.008
    elo_scale_midpoint: float = 1500.0

    # --- EP thresholds — pinned to chess.com's published table ---
    # https://support.chess.com/en/articles/8572705-how-are-moves-classified
    # These should NOT be optimizer-tuned; use the optimizer to validate
    # contextual rules, not to drift thresholds away from published values.
    ep_best: float = 0.00
    ep_excellent: float = 0.02
    ep_good: float = 0.05
    ep_inaccuracy: float = 0.1
    ep_mistake: float = 0.2
    # anything above ep_mistake = blunder (but see blunder gate below)

    # --- Elo-dependent excellent threshold ---
    # At low Elo (below threshold), use a tighter excellent threshold.
    # The flatter sigmoid curve means small EP losses are less meaningful,
    # so chess.com requires moves to be closer to optimal for "excellent".
    excellent_elo_threshold: int = 750
    ep_excellent_low_elo: float = 0.012

    # --- Best move promotion ---
    # Minimum candidate gap (cp) to promote engine-top move from "excellent"
    # to "best". When alternatives are nearly as good (small gap), being the
    # engine's #1 pick is noise, not skill — leave as "excellent".
    best_promotion_min_gap_cp: int = 15
    # If a move is not the engine's exact top choice but stays in the same
    # zero-EP bucket, chess.com often prefers "excellent" over "best" when the
    # candidate gap is small but non-zero. Treat this narrow near-tie band as
    # excellent rather than best.
    best_non_top_excellent_min_gap_cp: int = 2
    best_non_top_excellent_max_gap_cp: int = 30
    # Engine-equivalent safeguard: don't demote non-top bests to excellent when
    # the raw centipawn loss is below this value, even if the candidate gap
    # falls in the demotion band. Drill-down on chess.com labels shows a
    # large pocket of rank-2 moves with cp_loss ~7 and gap ~6 that CC still
    # calls "best" — the engine considers them equivalent to the top pick.
    best_non_top_excellent_min_cp_loss: int = 5

    # --- Brilliant detection ---
    brilliant_ep_tolerance: float = 0.02     # how close to best move
    brilliant_min_win_pct_after: float = 0.35
    brilliant_max_win_pct_before: float = 0.85
    brilliant_decisive_win_pct: float = 0.60  # sacrifice must lead to this or mate
    brilliant_pv_depth_check: int = 6         # half-moves to check material recovery
    brilliant_min_material_sacrifice: int = 5  # rook value (default)
    brilliant_low_elo_material_sacrifice: int = 3  # minor piece (for Elo < 1200)
    brilliant_low_elo_threshold: int = 1200
    # At lower Elo, chess.com is a bit more permissive about awarding "brilliant"
    # in already-good positions, but still not in completely won ones.
    brilliant_low_elo_max_win_pct_before: float = 0.9
    brilliant_low_elo_max_win_pct_threshold: int = 1500

    # --- Great move detection ---
    great_ep_tolerance: float = 0.02
    great_only_move_ep_gap: float = 0.05
    # Candidate gap threshold: the played move must be the engine's #1 pick
    # AND the gap to the 2nd-best move must exceed this (in centipawns).
    # chess.com data shows great moves have median gap of ~269cp.
    great_min_candidate_gap_cp: int = 250
    # Capitalization: opponent's prev EP loss must be in this range.
    # Too small = routine best move. Too large (blunder) = exploitation is trivial.
    great_capitalization_min_ep_loss: float = 0.06
    great_capitalization_max_ep_loss: float = 0.20
    # Candidate-gap relaxation for capitalization cases. Default 0.5 preserves
    # the existing heuristic that these can qualify with about half the normal
    # "only move" gap.
    great_capitalization_gap_scale: float = 0.54
    # "Great" should not fire in already decisive positions for either side.
    # (applies to Trigger A candidate-gap only; transition triggers have own bounds)
    great_min_win_pct_before: float = 0.15
    great_max_win_pct_before: float = 0.82
    # After a clear opponent mistake/blunder, chess.com still awards "great"
    # for the strongest punishment even when the mover was already somewhat better.
    great_post_blunder_min_prev_ep_loss: float = 0.20
    great_post_blunder_min_gap_cp: int = 100
    great_post_blunder_max_win_pct_before: float = 0.90
    # When in check with this many or fewer legal moves, the response is
    # forced and should not qualify as "great" regardless of candidate gap.
    great_max_forced_check_moves: int = 3

    # --- Concrete blunder gate ---
    # Gate 3 (hanging piece): minimum static-exchange value of opponent's best
    # capture after the played move. Must be >= 1 so that only captures that
    # actually net material fire the gate — the earlier 1-ply "is defended?"
    # check over-fired on undefended pawns and on captures that walked into
    # losing recapture sequences.
    blunder_min_hanging_see: int = 1
    # Gate 4 (EP floor): if the player lost at least this much EP, we still
    # require at least SEE>=1 before calling the move a concrete blunder.
    # Below this floor, moves go to mistake/miss regardless of EP.
    blunder_ep_floor: float = 0.35
    # Pure-EP override: a drop this large on its own is enough evidence of a
    # multi-move tactical sequence to qualify as a blunder without any
    # one-ply material signal.
    blunder_pure_ep_floor: float = 0.40

    # --- Miss detection ---
    # Chess.com's miss requires a concrete missed opportunity, not just EP loss.
    # Opponent must have created a real chance (tactical/material), best reply must
    # achieve a concrete gain, and the player must have failed to capitalize.
    miss_opponent_ep_loss_threshold: float = 0.1  # opponent's prev EP loss
    miss_min_ep_loss: float = 0.1                 # player's own EP loss
    miss_blunder_override_min_prev_ep: float = 0.30  # opponent's prev EP loss required for miss to override blunder
    miss_best_wins_material_depth: int = 4         # half-moves to check material gain in best PV
    miss_best_win_pct_threshold: float = 0.70      # best reply must reach this for "clearly winning" gate

    # --- Miss Trigger C: direct tactic missed ---
    # Fires when a significantly better move existed (large candidate gap + forcing
    # character) regardless of whether the opponent's prior move was clearly bad.
    # Captures positions where the player made a "reasonable" move while a
    # concrete tactic was available — classic chess.com "miss" pattern.
    miss_direct_tactic_min_gap_cp: int = 250       # gap between best and 2nd-best candidate
    miss_direct_tactic_min_mover_wp: float = 0.15  # don't fire when already losing

    # --- Draw-sensitivity boost (ML-vs-core analysis, Theme 2) ---
    # Chess.com penalises moderate EP loss more harshly in near-drawn positions
    # than in decisive ones: a 50cp slip that turns a draw into a loss is an
    # "inaccuracy" but the same cp loss while you're already +3 is "good".
    # The 7-class ML-vs-core diff (796 games, ~46k moves) showed the three
    # biggest confusion pairs on the "ML-right / core-wrong" slice all had
    # `best_wdl_draw` around 530‰ vs 340‰ on the control — core was
    # under-penalising EP loss whenever the game was structurally drawn.
    # We multiply the computed ep_loss by a factor that ramps linearly from
    # 1.0 at `draw_boost_min_permille` to `draw_boost_max_factor` at 1000‰.
    # Set `draw_boost_max_factor = 1.0` to disable the adjustment entirely.
    draw_boost_min_permille: int = 500
    draw_boost_max_factor: float = 1.30
    draw_boost_min_ep_loss: float = 0.05

    # --- PV-endpoint EP loss (ML-vs-core analysis, Theme 4) ---
    # The raw root eval of a non-top candidate is shallow — a 5-ply drift on
    # the candidate's PV can reveal that what looked like a 20cp inaccuracy
    # is really an 80cp mistake. When both the best and played candidates
    # carry a `pv_end` re-evaluation, we compute a second ep_loss using the
    # endpoint scores and take the stricter of the two. This lifts core on
    # the rank-2/3/4 plays where it is weakest (~25-45% accuracy vs ~87%
    # on rank-1 plays in the same corpus).
    # `pv_end_min_pushed` gates out PVs that only managed 0-1 plies before
    # reaching a terminal or illegal state — those drifts are unreliable.
    # Set `pv_end_ep_loss_enabled = False` to disable the adjustment.
    pv_end_ep_loss_enabled: bool = True
    pv_end_min_pushed: int = 2
    # Below this root ep_loss the pv_end lift is skipped: best / excellent /
    # good plays (root ep_loss ≤ ep_good) should not be demoted just because
    # their PV drifts a few centipawns further than the best candidate's PV.
    # Chess.com's "good" label is assigned from the root eval without PV
    # drift adjustment, so overriding via pv_end regresses the good bucket
    # even more than it lifts mistake/blunder. Full-corpus A/B tests on this
    # corpus showed ungated lift migrated ~1.1k excellent plays into good
    # and a 0.02-gated lift regressed `good` by 8pp; gating at 0.05 preserves
    # best / excellent / good while still surfacing hidden drift on moves the
    # root eval already flags as inaccuracy-or-worse.
    pv_end_min_ep_loss: float = 0.05
    # The pv_end lift is applied only when the endpoint ep_loss exceeds the
    # root ep_loss by at least this much. Small drifts (a few cp) are noise
    # from the shallow endpoint search and should not re-bucket a move. The
    # ML-vs-core signal came from plays whose pv_end revealed material
    # collapses (hundreds of cp), so we require at least a half-bucket delta
    # before overriding the root label. Without this guard a systematic
    # ~2-5cp drift shifts chess.com's inaccuracy buckets down to mistake.
    pv_end_min_lift: float = 0.05
    # Rank-4+ plays get a separate targeted severity override below, but the
    # generic PV-end replacement keeps the same conservative defaults as top-3.
    pv_end_min_ep_loss_rank4: float = 0.05
    pv_end_min_lift_rank4: float = 0.05

    # --- Rank-4+ future-line severity override ---
    # These exact-eval out-of-top-window plays are where core most often
    # under-calls `good` / `inaccuracy` / `mistake`. Rather than lowering the
    # global EP thresholds, use the played move's own PV endpoint to bump only
    # the next severity bucket when the deeper line clearly collapses.
    rank4_good_to_inaccuracy_min_future_ep: float = 0.065
    rank4_good_to_inaccuracy_min_lift: float = 0.02
    rank4_good_to_inaccuracy_min_draw_permille: int = 500
    rank4_inaccuracy_to_mistake_min_future_ep: float = 0.11
    rank4_inaccuracy_to_mistake_min_lift: float = 0.03

