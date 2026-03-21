"""
ExplanationGenerator — calls Claude API to narrate a chess position.

Accepts AssembledContext + PriorityResult + skill level.
Returns a 2-3 sentence explanation string.

The LLM never calculates — it only narrates. All chess facts in the prompt
come from the engine and context assembly layers.

Uses AsyncAnthropic to avoid blocking the event loop in Django Channels
and async Django views.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import anthropic
from anthropic._exceptions import OverloadedError
from django.conf import settings

from analysis.context import AssembledContext
from analysis.priority import PriorityResult, PriorityTier
from explanation.retry import retry_overloaded
from explanation.templates.game_review import (
    build_game_review_prompt,
    build_game_summary_request,
    build_move_user_message,
)

logger = logging.getLogger(__name__)

SkillLevel = Literal["beginner", "intermediate", "advanced"]


class ExplanationGenerator:
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
        max_tokens: int = 256,
        api_key: str | None = None,
    ) -> str:
        """Standalone (stateless) explanation for a single position."""
        prompt = build_game_review_prompt(context, priority, skill_level)
        client = self._get_client(api_key)
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
            logger.warning("Anthropic API overloaded after all retries, skipping explanation")
            return ""

    async def generate_narrative(
        self,
        context: AssembledContext,
        priority: PriorityResult,
        move_number: int,
        color: str,
        conversation: list[dict],
        system_prompt: str,
        *,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 256,
        api_key: str | None = None,
    ) -> str:
        """
        Generate commentary for one move within a narrative conversation.

        Appends user and assistant messages to *conversation* in-place so
        subsequent calls see the full history.
        """
        user_msg = build_move_user_message(context, priority, move_number, color)
        conversation.append({"role": "user", "content": user_msg})

        client = self._get_client(api_key)
        try:
            message = await retry_overloaded(
                lambda: client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=conversation,
                )
            )
            text = message.content[0].text.strip()
        except OverloadedError:
            logger.warning("Anthropic API overloaded after all retries, skipping explanation")
            text = ""

        # Always append an assistant message to keep the conversation coherent.
        conversation.append({
            "role": "assistant",
            "content": text or "[Analysis unavailable]",
        })
        return text

    async def generate_game_summary(
        self,
        conversation: list[dict],
        system_prompt: str,
        *,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 512,
        api_key: str | None = None,
    ) -> str:
        """Generate a 4-5 sentence game summary at the end of the conversation."""
        summary_msg = build_game_summary_request()
        conversation.append({"role": "user", "content": summary_msg})

        client = self._get_client(api_key)
        try:
            message = await retry_overloaded(
                lambda: client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=conversation,
                )
            )
            return message.content[0].text.strip()
        except OverloadedError:
            logger.warning("Anthropic API overloaded after all retries, skipping summary")
            return ""


# Singleton factory
_explainer: Optional[ExplanationGenerator] = None


def get_explainer() -> ExplanationGenerator:
    global _explainer
    if _explainer is None:
        _explainer = ExplanationGenerator()
    return _explainer
