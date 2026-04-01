"""
Expected Points — WDL providers and EP-based move classification.

Converts engine evaluations into win probability (Expected Points) using either
Stockfish 16+ native WDL output or a logistic sigmoid fallback. Classifies moves
by EP loss rather than raw centipawn loss.

Provider selection is automatic: if the engine returned WDL permille values,
the native provider is used. Otherwise, the sigmoid fallback is used.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

from config.classification import ClassificationConfig


class WDLProvider(ABC):
    """Abstract interface for converting engine output to win probability."""

    @abstractmethod
    def get_win_pct(
        self,
        cp: Optional[int] = None,
        mate_in: Optional[int] = None,
        elo: Optional[int] = None,
        wdl_win: Optional[int] = None,
        wdl_draw: Optional[int] = None,
        wdl_loss: Optional[int] = None,
    ) -> Optional[float]:
        """Return win probability as a float between 0.0 and 1.0."""
        ...


class StockfishNativeWDLProvider(WDLProvider):
    """
    Uses Stockfish 16+ native WDL permille values.

    Ignores Elo adjustment — the engine's WDL already reflects engine-vs-engine
    win rates. For human-calibrated probabilities, the sigmoid provider with Elo
    adjustment is more appropriate.
    """

    def get_win_pct(
        self,
        cp: Optional[int] = None,
        mate_in: Optional[int] = None,
        elo: Optional[int] = None,
        wdl_win: Optional[int] = None,
        wdl_draw: Optional[int] = None,
        wdl_loss: Optional[int] = None,
    ) -> Optional[float]:
        if mate_in is not None:
            return 1.0 if mate_in > 0 else 0.0
        if wdl_win is not None and wdl_draw is not None:
            # Expected score: win + 0.5 * draw (standard chess scoring)
            return (wdl_win + 0.5 * wdl_draw) / 1000.0
        return None  # signal to fall back to sigmoid


class SigmoidWDLProvider(WDLProvider):
    """
    Logistic sigmoid mapping from centipawns to win probability.
    Supports Elo-adjusted steepness. Works with any UCI engine.
    """

    def __init__(self, config: Optional[ClassificationConfig] = None):
        self._config = config or ClassificationConfig()

    def _elo_scaling_factor(self, elo: Optional[int]) -> float:
        if elo is None:
            return 1.0
        c = self._config
        return c.elo_scale_floor + c.elo_scale_range * (
            1.0 / (1.0 + math.exp(-c.elo_scale_steepness * (elo - c.elo_scale_midpoint)))
        )

    def get_win_pct(
        self,
        cp: Optional[int] = None,
        mate_in: Optional[int] = None,
        elo: Optional[int] = None,
        wdl_win: Optional[int] = None,
        wdl_draw: Optional[int] = None,
        wdl_loss: Optional[int] = None,
    ) -> Optional[float]:
        if mate_in is not None:
            return 1.0 if mate_in > 0 else 0.0
        if cp is None:
            return None
        k = self._config.k_base * self._elo_scaling_factor(elo)
        return 0.5 + 0.5 * (2.0 / (1.0 + math.exp(-k * cp)) - 1.0)


def resolve_provider(
    engine_result,
    config: Optional[ClassificationConfig] = None,
    player_elo: Optional[int] = None,
) -> WDLProvider:
    """
    Auto-select the best available WDL provider based on engine output.

    When player_elo is provided, always uses the Elo-calibrated sigmoid —
    Stockfish native WDL is engine-vs-engine (perfect play), not calibrated
    for human conversion rates. An 800-rated player with +4 pawns does NOT
    have a 99% win rate, but native WDL would say so.

    Without player_elo, prefers native WDL if available (engine accuracy).
    """
    if player_elo is not None:
        return SigmoidWDLProvider(config)
    best = engine_result.best
    if (best.wdl_win is not None and best.wdl_draw is not None
            and best.wdl_loss is not None):
        return StockfishNativeWDLProvider()
    return SigmoidWDLProvider(config)


def classify_ep_loss(ep_loss: Optional[float], config: Optional[ClassificationConfig] = None) -> str:
    """
    Classify a move based on expected points lost.
    Returns one of: best, excellent, good, inaccuracy, mistake, blunder, unknown.
    """
    if ep_loss is None:
        return "unknown"
    c = config or ClassificationConfig()
    if ep_loss <= c.ep_best:
        return "best"
    if ep_loss <= c.ep_excellent:
        return "excellent"
    if ep_loss <= c.ep_good:
        return "good"
    if ep_loss < c.ep_inaccuracy:
        return "inaccuracy"
    if ep_loss < c.ep_mistake:
        return "mistake"
    return "blunder"
