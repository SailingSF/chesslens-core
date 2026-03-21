"""
CoachNudgeGenerator — generates a coaching question during live games.

Rules enforced at the prompt level:
  - Names the threat / attacked square / tactical pattern
  - Asks the player how they want to respond
  - NEVER suggests or hints at a specific move
  - Returns a single sentence

Uses AsyncAnthropic to avoid blocking the event loop.
"""

from __future__ import annotations

import logging
import re
from typing import Literal, Optional

import anthropic
from anthropic._exceptions import OverloadedError
from django.conf import settings

from analysis.context import AssembledContext
from analysis.priority import PriorityResult
from explanation.retry import retry_overloaded
from explanation.templates.coach_nudge import build_coach_nudge_prompt

logger = logging.getLogger(__name__)

SkillLevel = Literal["beginner", "intermediate", "advanced"]

# Move notation patterns to reject from coach nudges:
#   Piece moves: Nf3, Bxe5, Qd1, Rae1, R1e2
#   Pawn moves: e4, d5, exd5, e8=Q
#   Castling: O-O, O-O-O
_MOVE_NOTATION_RE = re.compile(
    r"\b[NBRQK][a-h1-8]?x?[a-h][1-8]\b"    # piece moves
    r"|\b[a-h]x[a-h][1-8]\b"                 # pawn captures (exd5)
    r"|\b[a-h][1-8]=[NBRQ]\b"                # pawn promotion (e8=Q)
    r"|\bO-O(-O)?\b"                          # castling
)


class CoachNudgeGenerator:
    def __init__(self, client: anthropic.AsyncAnthropic | None = None):
        self._client = client or anthropic.AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
        )
        self._per_key_clients: dict[str, anthropic.AsyncAnthropic] = {}

    def _get_client(self, api_key: str | None = None) -> anthropic.AsyncAnthropic:
        """Return a cached client for the given key, or the default client."""
        if not api_key:
            return self._client
        if api_key not in self._per_key_clients:
            self._per_key_clients[api_key] = anthropic.AsyncAnthropic(
                api_key=api_key,
            )
        return self._per_key_clients[api_key]

    async def generate(
        self,
        context: AssembledContext,
        priority: PriorityResult,
        skill_level: SkillLevel = "intermediate",
        *,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 128,
        api_key: str | None = None,
    ) -> str:
        prompt = build_coach_nudge_prompt(context, priority, skill_level)
        client = self._get_client(api_key)
        try:
            message = await retry_overloaded(
                lambda: client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
            )
        except OverloadedError:
            logger.warning("Anthropic API overloaded after all retries, skipping coach nudge")
            return ""
        text = message.content[0].text.strip()
        self._validate_no_move_notation(text)
        return text

    def _validate_no_move_notation(self, text: str) -> None:
        """Raise if the response accidentally contains move notation."""
        match = _MOVE_NOTATION_RE.search(text)
        if match:
            raise ValueError(
                f"Coach nudge contains move notation '{match.group()}' "
                f"— prompt constraint violated: {text!r}"
            )


# Singleton factory
_coach: Optional[CoachNudgeGenerator] = None


def get_coach() -> CoachNudgeGenerator:
    global _coach
    if _coach is None:
        _coach = CoachNudgeGenerator()
    return _coach
