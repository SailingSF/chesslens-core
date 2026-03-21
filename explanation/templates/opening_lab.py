"""
Opening lab prompt templates.

Fired when the player deviates from book theory. Explains:
  - What the book continuation achieves
  - Why the player's move is suboptimal in this opening context
  - What positional themes the opening is designed to produce
"""

from __future__ import annotations

from analysis.context import AssembledContext
from analysis.priority import PriorityResult


def build_opening_lab_prompt(
    context: AssembledContext,
    priority: PriorityResult,
    skill_level: str,
    book_move_san: str,
    opening_name: str,
    eco_code: str,
) -> str:
    return f"""You are a chess opening coach. The player has deviated from book theory.

Opening: {opening_name} ({eco_code})
Position (FEN): {context.fen}

Player's move: {context.played_move_san}
Book continuation: {book_move_san}

Your task:
1. Explain what the book move ({book_move_san}) achieves in the context of the {opening_name}.
2. Explain specifically why the player's move ({context.played_move_san}) is suboptimal here.
3. Name the positional themes this opening is designed to produce.

Skill level: {skill_level}
Write 2-3 sentences. Reference the opening name and variation. Do not suggest specific follow-up moves beyond the book move listed above."""
