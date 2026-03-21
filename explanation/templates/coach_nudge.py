"""
Coach nudge prompt templates.

Strict constraints enforced in prompt:
  - Name threats and attacked squares
  - Ask the player a question about how they want to respond
  - NEVER suggest or hint at a specific move
  - Output: single sentence
"""

from __future__ import annotations

from analysis.context import AssembledContext
from analysis.priority import PriorityResult

_SKILL_NUDGE_INSTRUCTIONS = {
    "beginner": "Name the specific piece under attack and ask what the player wants to do about it.",
    "intermediate": "Name the tactical pattern and key squares, then ask the player to consider the tradeoff.",
    "advanced": "Reference positional imbalances and ask how the player wants to manage the tension.",
}


def build_coach_nudge_prompt(
    context: AssembledContext,
    priority: PriorityResult,
    skill_level: str,
) -> str:
    skill_instruction = _SKILL_NUDGE_INSTRUCTIONS.get(
        skill_level, _SKILL_NUDGE_INSTRUCTIONS["intermediate"]
    )

    threat_section = (
        f"Threat narrative: {context.threat_narrative}"
        if context.threat_narrative
        else "Threat narrative: (none computed)"
    )

    tactics_line = (
        f"Tactical patterns: {', '.join(context.tactical_patterns)}"
        if context.tactical_patterns
        else "No tactical patterns detected."
    )

    return f"""You are a chess coach providing a live coaching nudge during a game.

STRICT RULES:
1. Do NOT suggest, name, or hint at any specific chess move.
2. Do NOT use move notation (e.g. Nf3, Bxe5, e4).
3. Name what is happening on the board (threats, attacked pieces, controlled squares).
4. End with a question asking how the player wants to respond.
5. Output exactly ONE sentence.

Position (FEN): {context.fen}
Move just played: {context.played_move_san or 'N/A'}
{threat_section}
{tactics_line}
Priority: {priority.tier.value} — {priority.trigger}

Skill level: {skill_level}
{skill_instruction}

Write your single-sentence coaching nudge now:"""
