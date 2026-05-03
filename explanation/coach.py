"""
CoachNudgeGenerator — generates a coaching question during live games.

Rules enforced at the prompt level:
  - Names the threat / attacked square / tactical pattern
  - Asks the player how they want to respond
  - NEVER suggests or hints at a specific move
  - Returns a single sentence

Supports Anthropic and OpenAI providers via LLMConfig.
"""

from __future__ import annotations

import logging
import re
from typing import Literal, Optional

import anthropic
import openai
from anthropic._exceptions import OverloadedError
from django.conf import settings

from analysis.context import AssembledContext
from analysis.priority import PriorityResult
from explanation.providers import DEFAULT_ANTHROPIC_MODEL, LLMConfig
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
    def __init__(
        self,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
    ):
        self._anthropic_default = anthropic_client or anthropic.AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
        )
        self._anthropic_clients: dict[str, anthropic.AsyncAnthropic] = {}
        self._openai_clients: dict[str, openai.AsyncOpenAI] = {}

    # ------------------------------------------------------------------
    # Client helpers
    # ------------------------------------------------------------------

    def _get_anthropic_client(self, api_key: str | None) -> anthropic.AsyncAnthropic:
        if not api_key:
            return self._anthropic_default
        if api_key not in self._anthropic_clients:
            self._anthropic_clients[api_key] = anthropic.AsyncAnthropic(api_key=api_key)
        return self._anthropic_clients[api_key]

    def _get_openai_client(self, api_key: str | None) -> openai.AsyncOpenAI:
        key = api_key or getattr(settings, "OPENAI_API_KEY", "")
        if key not in self._openai_clients:
            self._openai_clients[key] = openai.AsyncOpenAI(api_key=key)
        return self._openai_clients[key]

    # ------------------------------------------------------------------
    # Low-level call helpers
    # ------------------------------------------------------------------

    async def _call_anthropic(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int,
        api_key: str | None,
    ) -> str:
        client = self._get_anthropic_client(api_key)
        try:
            message = await retry_overloaded(
                lambda: client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            return message.content[0].text.strip()
        except OverloadedError:
            logger.warning("Anthropic API overloaded after all retries, skipping coach nudge")
            return ""

    async def _call_openai(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int,
        api_key: str | None,
        reasoning_effort: str | None,
    ) -> str:
        client = self._get_openai_client(api_key)
        kwargs: dict = dict(
            model=model,
            input=prompt,
            max_output_tokens=max_tokens,
        )
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        try:
            response = await client.responses.create(**kwargs)
            return response.output_text.strip()
        except openai.RateLimitError:
            logger.warning("OpenAI rate limit hit, skipping coach nudge")
            return ""
        except openai.APIStatusError as exc:
            logger.warning("OpenAI API error %s, skipping coach nudge", exc.status_code)
            return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        context: AssembledContext,
        priority: PriorityResult,
        skill_level: SkillLevel = "intermediate",
        *,
        llm_config: LLMConfig | None = None,
        # Legacy params
        model: str = DEFAULT_ANTHROPIC_MODEL,
        max_tokens: int = 128,
        api_key: str | None = None,
    ) -> str:
        cfg = llm_config or LLMConfig(model=model, api_key=api_key)
        prompt = build_coach_nudge_prompt(context, priority, skill_level)

        if cfg.is_openai:
            text = await self._call_openai(
                prompt,
                model=cfg.model,
                max_tokens=max_tokens,
                api_key=cfg.api_key,
                reasoning_effort=cfg.reasoning_effort,
            )
        else:
            text = await self._call_anthropic(
                prompt,
                model=cfg.model,
                max_tokens=max_tokens,
                api_key=cfg.api_key,
            )

        if text:
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
