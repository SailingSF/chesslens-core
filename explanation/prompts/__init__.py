"""
Loader for chat prompt text and the Stockfish tool schema (chat.yaml).

All prompt strings live in chat.yaml — these helpers just read and fill them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

from analysis.context import AssembledContext

_CHAT_YAML = Path(__file__).parent / "chat.yaml"


@lru_cache(maxsize=1)
def load_chat_prompts() -> dict:
    """Parse chat.yaml once and cache it."""
    with _CHAT_YAML.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def analyze_position_tool() -> dict:
    """Return the provider-agnostic tool definition (name/description/schema)."""
    return load_chat_prompts()["tool"]


def build_chat_system_prompt(skill_level: str, position_context: str) -> str:
    """Fill the system prompt with the skill level and the position analysis."""
    return load_chat_prompts()["system_prompt"].format(
        skill_level=skill_level,
        position_context=position_context,
    )


def _format_eval(cp: Optional[int], mate_in: Optional[int]) -> str:
    if mate_in is not None:
        return f"mate in {abs(mate_in)} for {'White' if mate_in > 0 else 'Black'}"
    if cp is None:
        return "unknown"
    return f"{cp / 100:+.2f} (white's perspective)"


def _format_candidate_lines(candidates: list[dict]) -> str:
    if not candidates:
        return "  (none)"
    lines = []
    for i, c in enumerate(candidates, 1):
        ev = _format_eval(c.get("cp"), c.get("mate_in"))
        pv = " ".join(c.get("pv") or [])
        lines.append(f"  {i}. {c['move']} [{ev}] — {pv}")
    return "\n".join(lines)


def format_position_context(
    context: AssembledContext,
    candidates: list[dict],
    board,
) -> str:
    """Render the engine-analysis block embedded in the system prompt."""
    opening = (
        f"{context.opening_name} ({context.eco_code})"
        if context.opening_name
        else "unknown"
    )
    pawns = []
    for color in ("white", "black"):
        data = (context.pawn_structure or {}).get(color, {})
        for label in ("isolated", "doubled", "passed"):
            squares = data.get(label, [])
            if squares:
                pawns.append(f"{color} {label}: {', '.join(squares)}")
    pawn_line = "; ".join(pawns) if pawns else "normal"

    return load_chat_prompts()["position_context_template"].format(
        fen=context.fen,
        side_to_move="White" if board.turn else "Black",
        opening=opening,
        best_move=context.best_move_san,
        evaluation=_format_eval(context.best_move_cp, context.mate_in),
        candidates=_format_candidate_lines(candidates),
        tactics=", ".join(context.tactical_patterns) if context.tactical_patterns else "none",
        pawns=pawn_line,
        threats=context.threat_narrative or "none",
    )
