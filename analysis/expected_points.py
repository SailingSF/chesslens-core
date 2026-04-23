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


def draw_sensitivity_factor(
    wdl_draw_permille: Optional[int],
    config: Optional[ClassificationConfig] = None,
) -> float:
    """Multiplier applied to ep_loss in near-drawn positions.

    Chess.com labels the same EP loss more harshly in structurally drawn
    positions than in decisive ones — a 50cp slip is an "inaccuracy" when
    a draw is on the line but merely "good" when you're already winning by
    a piece. The chesslens-core EP thresholds are pinned to chess.com's
    published table, so the draw-dependence is applied by scaling the
    effective ep_loss up in draw-heavy positions before it is bucketed.

    The factor ramps linearly from 1.0 at `draw_boost_min_permille` to
    `draw_boost_max_factor` at the 1000‰ ceiling. Positions below the
    threshold, or ones where no WDL is available, are unaffected.
    """
    c = config or ClassificationConfig()
    if wdl_draw_permille is None or c.draw_boost_max_factor <= 1.0:
        return 1.0
    if wdl_draw_permille <= c.draw_boost_min_permille:
        return 1.0
    ramp_width = max(1, 1000 - c.draw_boost_min_permille)
    ratio = min(1.0, (wdl_draw_permille - c.draw_boost_min_permille) / ramp_width)
    return 1.0 + ratio * (c.draw_boost_max_factor - 1.0)


def pv_end_win_pct(
    candidate,
    provider: WDLProvider,
    elo: Optional[int] = None,
    config: Optional[ClassificationConfig] = None,
) -> Optional[float]:
    """Convert a candidate's PV-endpoint eval to a white-POV win probability.

    The endpoint is the position reached after pushing the candidate's PV
    on a board copy and re-searching. Both the root eval and the PV-end
    eval are stored in white-POV by the engine layer, so a straight
    provider call is valid. Returns None when:

      - `pv_end` is absent (the caller didn't request the re-eval),
      - the PV resolved to fewer than `pv_end_min_pushed` plies (too
        shallow to trust as a drift signal), or
      - the PV reached a terminal state the provider can't score
        (e.g. stalemate with mate_in=None and cp=0 would falsely read as
        a draw even when one side is overwhelmingly winning — we let the
        root eval handle those).

    Terminal mates (`terminal=="checkmate"`) are honored via `mate_in`.
    """
    c = config or ClassificationConfig()
    pv_end = getattr(candidate, "pv_end", None)
    if pv_end is None:
        return None
    if pv_end.pushed < c.pv_end_min_pushed:
        return None
    if pv_end.terminal is not None:
        # Synthesized terminal evals are degraded: stalemate/draw carry
        # `cp=0` which mis-represents drawn-but-winning positions, and
        # checkmate carries `mate_in=0` which has lost the who-mated-whom
        # sign. The root eval already assigns a decisive score in these
        # cases, so we defer to it rather than guessing parity here.
        return None
    return provider.get_win_pct(
        cp=pv_end.score_cp,
        mate_in=pv_end.mate_in,
        elo=elo,
    )
