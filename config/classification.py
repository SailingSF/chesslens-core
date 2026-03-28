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
    # Optimized via grid search (1125 runs) against chess.com Classification V2 at Elo 820.
    # High floor (0.75) ensures even low-rated players show meaningful win probability
    # differences for large eval swings. Narrow range (0.15) keeps the curve stable
    # across Elo bands.
    elo_scale_floor: float = 0.75
    elo_scale_range: float = 0.15
    elo_scale_steepness: float = 0.003
    elo_scale_midpoint: float = 1500.0

    # --- EP thresholds — calibrated against chess.com Classification V2 ---
    ep_best: float = 0.00
    ep_excellent: float = 0.015
    ep_good: float = 0.05
    ep_inaccuracy: float = 0.07
    ep_mistake: float = 0.22
    # anything above ep_mistake = blunder

    # --- Brilliant detection ---
    brilliant_ep_tolerance: float = 0.02     # how close to best move
    brilliant_min_win_pct_after: float = 0.35
    brilliant_max_win_pct_before: float = 0.85
    brilliant_decisive_win_pct: float = 0.60  # sacrifice must lead to this or mate
    brilliant_pv_depth_check: int = 6         # half-moves to check material recovery
    brilliant_min_material_sacrifice: int = 5  # rook value (default)
    brilliant_low_elo_material_sacrifice: int = 3  # minor piece (for Elo < 1200)
    brilliant_low_elo_threshold: int = 1200

    # --- Great move detection ---
    great_ep_tolerance: float = 0.02
    great_only_move_ep_gap: float = 0.05
    # Capitalization: opponent's prev EP loss must be in this range.
    # Too small = routine best move. Too large (blunder) = exploitation is trivial.
    great_capitalization_min_ep_loss: float = 0.06
    great_capitalization_max_ep_loss: float = 0.20

    # --- Miss detection ---
    miss_opponent_ep_loss_threshold: float = 0.10
