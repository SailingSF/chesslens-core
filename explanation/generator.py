"""
ExplanationGenerator — calls the LLM to narrate a chess position.

Accepts AssembledContext + PriorityResult + skill level.
Returns a 2-3 sentence explanation string.

The LLM never calculates — it only narrates. All chess facts in the prompt
come from the engine and context assembly layers.

Supports two providers:
  - Anthropic (Claude) — model names starting with "claude-"
  - OpenAI — all other model names, accessed via the Responses API

Uses async clients throughout to avoid blocking the event loop.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import anthropic
import openai
from anthropic._exceptions import OverloadedError
from django.conf import settings

from analysis.context import AssembledContext
from analysis.priority import PriorityResult, PriorityTier
from explanation.providers import (
    ANTHROPIC_EFFORT_MODELS,
    DEFAULT_ANTHROPIC_MODEL,
    LLMConfig,
)
from explanation.retry import retry_overloaded
from explanation.templates.game_review import (
    build_game_review_prompt,
    build_game_review_system_prompt,
    build_game_summary_request,
    build_move_user_message,
)

logger = logging.getLogger(__name__)

SkillLevel = Literal["beginner", "intermediate", "advanced"]


class ExplanationGenerator:
    def __init__(
        self,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
    ):
        self._anthropic_default = anthropic_client or anthropic.AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
        )
        # Per-key client caches: keyed by api_key string
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

    # Public accessors so other modules (e.g. the chat agent) can reuse the
    # same per-API-key client cache instead of constructing their own.
    def get_anthropic_client(self, api_key: str | None) -> anthropic.AsyncAnthropic:
        return self._get_anthropic_client(api_key)

    def get_openai_client(self, api_key: str | None) -> openai.AsyncOpenAI:
        return self._get_openai_client(api_key)

    # ------------------------------------------------------------------
    # Low-level call helpers
    # ------------------------------------------------------------------

    async def _call_anthropic(
        self,
        messages: list[dict],
        *,
        model: str,
        max_tokens: int,
        system: str | None = None,
        api_key: str | None = None,
        effort: str | None = None,
    ) -> str:
        client = self._get_anthropic_client(api_key)
        kwargs: dict = dict(model=model, max_tokens=max_tokens, messages=messages)
        if system:
            kwargs["system"] = system
        # Apply reasoning effort only to models that support output_config.effort
        # (e.g. Opus/Sonnet); Haiku and older models would reject it.
        if effort and model in ANTHROPIC_EFFORT_MODELS:
            kwargs["output_config"] = {"effort": effort}
        try:
            message = await retry_overloaded(lambda: client.messages.create(**kwargs))
            # Guard against a response with no text block (e.g. a refusal).
            return next((b.text for b in message.content if b.type == "text"), "").strip()
        except OverloadedError:
            logger.warning("Anthropic API overloaded after all retries, skipping explanation")
            return ""

    async def _call_openai(
        self,
        messages: list[dict],
        *,
        model: str,
        max_tokens: int,
        system: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        client = self._get_openai_client(api_key)
        # Prepend system message if provided
        input_messages: list[dict] = []
        if system:
            input_messages.append({"role": "system", "content": system})
        input_messages.extend(messages)

        kwargs: dict = dict(
            model=model,
            input=input_messages,
            max_output_tokens=max_tokens,
        )
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        try:
            response = await client.responses.create(**kwargs)
            return response.output_text.strip()
        except openai.RateLimitError:
            logger.warning("OpenAI rate limit hit, skipping explanation")
            return ""
        except openai.APIStatusError as exc:
            logger.warning("OpenAI API error %s, skipping explanation", exc.status_code)
            return ""

    async def _call_llm(
        self,
        messages: list[dict],
        *,
        llm_config: LLMConfig,
        max_tokens: int,
        system: str | None = None,
    ) -> str:
        if llm_config.is_openai:
            return await self._call_openai(
                messages,
                model=llm_config.model,
                max_tokens=max_tokens,
                system=system,
                api_key=llm_config.api_key,
                reasoning_effort=llm_config.reasoning_effort,
            )
        return await self._call_anthropic(
            messages,
            model=llm_config.model,
            max_tokens=max_tokens,
            system=system,
            api_key=llm_config.api_key,
            effort=llm_config.reasoning_effort,
        )

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
        # Legacy params — used when llm_config is not provided
        model: str = DEFAULT_ANTHROPIC_MODEL,
        max_tokens: int = 256,
        api_key: str | None = None,
    ) -> str:
        """Standalone (stateless) explanation for a single position."""
        cfg = llm_config or LLMConfig(model=model, api_key=api_key)
        prompt = build_game_review_prompt(context, priority, skill_level)
        return await self._call_llm(
            [{"role": "user", "content": prompt}],
            llm_config=cfg,
            max_tokens=max_tokens,
        )

    async def generate_narrative(
        self,
        context: AssembledContext,
        priority: PriorityResult,
        move_number: int,
        color: str,
        conversation: list[dict],
        system_prompt: str,
        *,
        llm_config: LLMConfig | None = None,
        # Legacy params
        model: str = DEFAULT_ANTHROPIC_MODEL,
        max_tokens: int = 256,
        api_key: str | None = None,
    ) -> str:
        """Generate commentary for one move within a narrative conversation.

        Appends user and assistant messages to *conversation* in-place so
        subsequent calls see the full history.
        """
        cfg = llm_config or LLMConfig(model=model, api_key=api_key)
        user_msg = build_move_user_message(context, priority, move_number, color)
        conversation.append({"role": "user", "content": user_msg})

        text = await self._call_llm(
            conversation,
            llm_config=cfg,
            max_tokens=max_tokens,
            system=system_prompt,
        )

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
        llm_config: LLMConfig | None = None,
        # Legacy params
        model: str = DEFAULT_ANTHROPIC_MODEL,
        max_tokens: int = 512,
        api_key: str | None = None,
    ) -> str:
        """Generate a 4-5 sentence game summary at the end of the conversation."""
        cfg = llm_config or LLMConfig(model=model, api_key=api_key)
        summary_msg = build_game_summary_request()
        conversation.append({"role": "user", "content": summary_msg})
        return await self._call_llm(
            conversation,
            llm_config=cfg,
            max_tokens=max_tokens,
            system=system_prompt,
        )


# Singleton factory
_explainer: Optional[ExplanationGenerator] = None


def get_explainer() -> ExplanationGenerator:
    global _explainer
    if _explainer is None:
        _explainer = ExplanationGenerator()
    return _explainer
