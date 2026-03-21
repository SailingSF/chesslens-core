"""
Game review prompt templates.

Two modes:
  1. Narrative (conversation thread) — system prompt + per-move user messages.
     Used by GameAnalyzer for full-game review with conversational continuity.
  2. Standalone — single self-contained prompt per move.
     Used by PositionExplorer for one-off position analysis.
"""

from __future__ import annotations

from analysis.context import AssembledContext
from analysis.priority import PriorityResult, PriorityTier

_SKILL_INSTRUCTIONS = {
    "beginner": (
        "Use plain language. No jargon. Name the specific piece at risk and what happens "
        "if the player doesn't act. Keep it concrete and immediate."
    ),
    "intermediate": (
        "Reference named tactical and strategic themes. Light theory is fine. "
        "Name the pattern and explain why it matters here."
    ),
    "advanced": (
        "Be strategic and specific. Reference outposts, pawn tension, piece coordination, "
        "and engine continuations. Assume fluency with chess terminology."
    ),
}


# ---------------------------------------------------------------------------
# Narrative mode: system prompt + per-move user messages + summary request
# ---------------------------------------------------------------------------


def build_game_review_system_prompt(player_color: str, skill_level: str) -> str:
    """Build the system prompt for a narrative game review conversation."""
    skill_instruction = _SKILL_INSTRUCTIONS.get(skill_level, _SKILL_INSTRUCTIONS["intermediate"])
    opponent_color = "black" if player_color == "white" else "white"

    return f"""You are a chess coach reviewing a completed game. The player played **{player_color}**. \
When discussing {player_color}'s moves, address them as "you". Refer to {opponent_color}'s moves in third person \
(e.g. "your opponent played…").

For each move I send you, I will include the following structured data from the engine:

- **move**: Move number, color, and SAN notation (e.g. "15. white: Nxe5")
- **cp_loss** / **cp_loss_label**: How many centipawns worse the played move was compared to the engine's best move, \
and its classification (best, excellent, good, inaccuracy, mistake, blunder)
- **best_move**: The engine's preferred move in SAN notation
- **pv**: The engine's best continuation (principal variation), up to 6 moves
- **tactical_patterns**: Any detected tactical motifs (forks, pins, skewers, back-rank weakness, etc.)
- **pawn_structure**: Notable pawn features — isolated, doubled, or passed pawns
- **priority_tier** / **trigger**: How urgent this position is (CRITICAL / TACTICAL / STRATEGIC) and why
- **threat_narrative**: Which pieces are attacked, hanging, or under pressure

Your job is to **narrate, not calculate**. All chess facts (best moves, scores, tactics) come from the engine. \
Use only the data provided.

Reply with **2-3 sentences** of commentary in markdown. When a move connects thematically to earlier play \
(e.g. a pawn weakness created several moves ago now becomes exploitable), reference it. \
For good moves, briefly acknowledge what was right. For mistakes and blunders, explain what went wrong and why.

{skill_instruction}"""


def build_move_user_message(
    context: AssembledContext,
    priority: PriorityResult,
    move_number: int,
    color: str,
) -> str:
    """Build a structured user message for one move in the narrative conversation."""
    tactics_line = (
        f"tactical_patterns: {', '.join(context.tactical_patterns)}"
        if context.tactical_patterns
        else "tactical_patterns: none"
    )

    pawn_line = _format_pawn_structure(context.pawn_structure)

    pv_line = (
        f"pv: {' '.join(context.pv_san[:6])}"
        if context.pv_san
        else "pv: N/A"
    )

    return f"""move: {move_number}. {color}: {context.played_move_san or 'N/A'}
cp_loss: {context.played_move_cp_loss or 0} ({context.cp_loss_label})
best_move: {context.best_move_san}
{pv_line}
{tactics_line}
{pawn_line}
priority_tier: {priority.tier.value} — {priority.trigger}
threat_narrative: {context.threat_narrative}"""


def build_game_summary_request() -> str:
    """Build the final user message requesting a game summary."""
    return (
        "The game is over. Provide a **4-5 sentence summary** of the entire game. "
        "Cover the key turning points, the player's strengths and weaknesses, "
        "and one concrete area to improve. Use markdown formatting."
    )


def build_game_review_prompt(
    context: AssembledContext,
    priority: PriorityResult,
    skill_level: str,
) -> str:
    skill_instruction = _SKILL_INSTRUCTIONS.get(skill_level, _SKILL_INSTRUCTIONS["intermediate"])

    opening_line = (
        f"Opening: {context.opening_name} ({context.eco_code})"
        if context.opening_name
        else "Opening: unknown"
    )
    deviation_note = " — theory deviation" if context.theory_deviation else ""

    tactics_line = (
        f"Tactical patterns detected: {', '.join(context.tactical_patterns)}"
        if context.tactical_patterns
        else "No tactical patterns detected."
    )

    pawn_line = _format_pawn_structure(context.pawn_structure)

    pv_line = (
        f"Engine best continuation: {' '.join(context.pv_san[:6])}"
        if context.pv_san
        else ""
    )

    if priority.tier == PriorityTier.CRITICAL:
        focus = "Focus on the immediate threat or forced sequence. Be direct — this is urgent."
    elif priority.tier == PriorityTier.TACTICAL:
        focus = "Focus on the concrete calculation — what the sharp position demands move-by-move."
    else:
        focus = "Focus on long-term imbalances — pawn structure, piece activity, and planning."

    return f"""You are a chess coach explaining a position to a player. Your job is to narrate — not calculate.

Position (FEN): {context.fen}
{opening_line}{deviation_note}

Move played: {context.played_move_san or 'N/A'} ({context.cp_loss_label}, {context.played_move_cp_loss or 0}cp loss)
Engine best move: {context.best_move_san}
{pv_line}

{tactics_line}
{pawn_line}

Priority tier: {priority.tier.value} — {priority.trigger}

Skill level: {skill_level}
{skill_instruction}

{focus}

Write a 2-3 sentence explanation of this position. Do not suggest specific moves beyond what is listed above. Do not evaluate the position yourself — use only the data provided."""


def _format_pawn_structure(pawn_structure: dict) -> str:
    lines = []
    for color in ("white", "black"):
        data = pawn_structure.get(color, {})
        parts = []
        for label in ("isolated", "doubled", "passed"):
            squares = data.get(label, [])
            if squares:
                parts.append(f"{label}: {', '.join(squares)}")
        if parts:
            lines.append(f"{color.capitalize()} pawns — {'; '.join(parts)}")
    return "\n".join(lines) if lines else "Pawn structure: normal"
