"""
PriorityClassifier — derives CRITICAL / TACTICAL / STRATEGIC tier from engine output.

Pure logic, no LLM, no I/O. Input: multi-PV EngineResult + board. Output: PriorityTier.

Tier rules:
  CRITICAL  — best move is 200cp+ better than 2nd best, OR mate threat exists in PV
  TACTICAL  — score delta 50–200cp between best and 2nd best; PV has captures within 3 moves
  STRATEGIC — all top-3 moves within 50cp of each other
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import chess

from chess_engine.service import EngineResult


class PriorityTier(str, Enum):
    CRITICAL = "CRITICAL"
    TACTICAL = "TACTICAL"
    STRATEGIC = "STRATEGIC"


@dataclass
class PriorityResult:
    tier: PriorityTier
    trigger: str          # human-readable explanation of why this tier was assigned
    score_delta: Optional[int]   # cp delta between best and 2nd best (None if only 1 candidate)


def classify(result: EngineResult, board: Optional[chess.Board] = None) -> PriorityResult:
    """
    Derive the priority tier from engine output.

    If board is provided, capture detection in the PV is accurate.
    Without board, captures cannot be detected and the classifier
    falls back to score-delta-only TACTICAL triggers.
    """
    candidates = result.candidates
    if not candidates:
        return PriorityResult(tier=PriorityTier.STRATEGIC, trigger="no candidates", score_delta=None)

    best = candidates[0]

    # CRITICAL: mate threat
    if best.mate_in is not None and best.mate_in > 0:
        return PriorityResult(
            tier=PriorityTier.CRITICAL,
            trigger=f"forced mate in {best.mate_in}",
            score_delta=None,
        )

    if len(candidates) < 2:
        return PriorityResult(tier=PriorityTier.STRATEGIC, trigger="single candidate", score_delta=None)

    second = candidates[1]
    if best.score_cp is None or second.score_cp is None:
        return PriorityResult(tier=PriorityTier.STRATEGIC, trigger="score unavailable", score_delta=None)

    delta = abs(best.score_cp - second.score_cp)

    # CRITICAL: huge score gap
    if delta >= 200:
        return PriorityResult(
            tier=PriorityTier.CRITICAL,
            trigger=f"only move — {delta}cp gap to 2nd best",
            score_delta=delta,
        )

    # TACTICAL: moderate gap with captures in PV
    if delta >= 50 and board is not None and _pv_has_captures(board, best.pv[:3]):
        return PriorityResult(
            tier=PriorityTier.TACTICAL,
            trigger=f"sharp position — {delta}cp gap, captures in PV",
            score_delta=delta,
        )

    if delta >= 50:
        return PriorityResult(
            tier=PriorityTier.TACTICAL,
            trigger=f"concrete position — {delta}cp gap",
            score_delta=delta,
        )

    # STRATEGIC: all moves close
    return PriorityResult(
        tier=PriorityTier.STRATEGIC,
        trigger=f"stable position — top moves within {delta}cp",
        score_delta=delta,
    )


def _pv_has_captures(board: chess.Board, pv_moves: list[chess.Move]) -> bool:
    """Check if any of the first N PV moves is a capture, using the actual board."""
    temp = board.copy()
    for move in pv_moves:
        try:
            if temp.is_capture(move):
                return True
            temp.push(move)
        except Exception:
            break
    return False
