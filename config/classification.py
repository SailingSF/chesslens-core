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
    # Optimized against the larger external review corpus
    # (760 games / ~45k comparable moves) using per-side Elos and SF16.1
    # node-based multi-PV caches.
    elo_scale_floor: float = 0.80
    elo_scale_range: float = 0.25
    elo_scale_steepness: float = 0.008
    elo_scale_midpoint: float = 1250.0

    # --- EP thresholds — pinned to chess.com's published table ---
    # https://support.chess.com/en/articles/8572705-how-are-moves-classified
    # These should NOT be optimizer-tuned; use the optimizer to validate
    # contextual rules, not to drift thresholds away from published values.
    ep_best: float = 0.00
    ep_excellent: float = 0.02
    ep_good: float = 0.05
    ep_inaccuracy: float = 0.10
    ep_mistake: float = 0.20
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
    brilliant_low_elo_max_win_pct_before: float = 0.90
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
    # Trigger C — Defensive equalizer: only saving move from a losing position.
    great_defensive_losing_threshold: float = 0.35   # win_pct_before must be below this
    great_defensive_equal_threshold: float = 0.40     # win_pct_after must reach this
    # Trigger D — Seizing move: breakthrough from balanced to clearly winning.
    great_seizing_balanced_lower: float = 0.35
    great_seizing_balanced_upper: float = 0.65
    great_seizing_winning_threshold: float = 0.70     # win_pct_after must reach this
    # When in check with this many or fewer legal moves, the response is
    # forced and should not qualify as "great" regardless of candidate gap.
    great_max_forced_check_moves: int = 3
    # Minimum candidate gap for transition triggers (relaxed vs Trigger A's 200cp).
    great_transition_min_gap_cp: int = 50

    # --- Miss detection ---
    # Chess.com's miss requires a concrete missed opportunity, not just EP loss.
    # Opponent must have created a real chance (tactical/material), best reply must
    # achieve a concrete gain, and the player must have failed to capitalize.
    miss_opponent_ep_loss_threshold: float = 0.08  # opponent's prev EP loss
    miss_min_ep_loss: float = 0.08                 # player's own EP loss
    miss_blunder_override_min_prev_ep: float = 0.30  # opponent's prev EP loss required for miss to override blunder
    miss_best_wins_material_depth: int = 4         # half-moves to check material gain in best PV
    miss_best_win_pct_threshold: float = 0.70      # best reply must reach this for "clearly winning" gate
