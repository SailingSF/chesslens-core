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

    # --- Sigmoid WDL provider (fallback) ---
    # Base steepness constant (Lichess default, calibrated on 2300+ games)
    k_base: float = 0.00368208

    # Elo scaling curve: scaling_factor = floor + range * sigmoid(steepness * (elo - midpoint))
    # Optimized via 4-phase grid search (43849 runs) against chess.com Classification V2
    # using per-side Elos (365–1616) and SF16.1 depth-20 multipv-3 caches. 66.5% accuracy.
    elo_scale_floor: float = 0.80
    elo_scale_range: float = 0.20
    elo_scale_steepness: float = 0.005
    elo_scale_midpoint: float = 1700.0

    # --- EP thresholds — calibrated against chess.com Classification V2 ---
    ep_best: float = 0.00
    ep_excellent: float = 0.02
    ep_good: float = 0.04
    ep_inaccuracy: float = 0.097
    ep_mistake: float = 0.26
    # anything above ep_mistake = blunder

    # --- Elo-dependent excellent threshold ---
    # At low Elo (below threshold), use a tighter excellent threshold.
    # The flatter sigmoid curve means small EP losses are less meaningful,
    # so chess.com requires moves to be closer to optimal for "excellent".
    excellent_elo_threshold: int = 1200
    ep_excellent_low_elo: float = 0.0125

    # --- Best move promotion ---
    # Minimum candidate gap (cp) to promote engine-top move from "excellent"
    # to "best". When alternatives are nearly as good (small gap), being the
    # engine's #1 pick is noise, not skill — leave as "excellent".
    best_promotion_min_gap_cp: int = 15

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
    great_min_candidate_gap_cp: int = 200
    # Capitalization: opponent's prev EP loss must be in this range.
    # Too small = routine best move. Too large (blunder) = exploitation is trivial.
    great_capitalization_min_ep_loss: float = 0.06
    great_capitalization_max_ep_loss: float = 0.20
    # Candidate-gap relaxation for capitalization cases. Default 0.5 preserves
    # the existing heuristic that these can qualify with about half the normal
    # "only move" gap.
    great_capitalization_gap_scale: float = 0.5
    # "Great" should not fire in already decisive positions for either side.
    great_min_win_pct_before: float = 0.10
    great_max_win_pct_before: float = 0.90

    # --- Miss detection ---
    miss_opponent_ep_loss_threshold: float = 0.04  # opponent's prev move EP loss
    miss_min_ep_loss: float = 0.20                 # player's own EP loss to qualify
