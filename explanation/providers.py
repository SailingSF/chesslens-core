"""
LLM provider abstraction for chesslens-core.

Supports Anthropic (Claude) and OpenAI (via the Responses API).
Provider is determined by the model name: claude-* → Anthropic, all else → OpenAI.

OpenAI reasoning models (gpt-5.5, o3, o4-mini, o3-mini) accept a
reasoning_effort parameter ("low", "medium", "high"). The default
when no effort is specified is "low".
"""

from __future__ import annotations

from dataclasses import dataclass, field

# OpenAI models that support the reasoning parameter in the Responses API
OPENAI_REASONING_MODELS: frozenset[str] = frozenset({
    "gpt-5.5",
    "o3",
    "o3-mini",
    "o4-mini",
})

# All recognised OpenAI models (non-exhaustive — unknown names are still
# forwarded; this list is used only for default-model resolution).
OPENAI_MODELS: frozenset[str] = frozenset({
    "gpt-5.5",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "o3",
    "o3-mini",
    "o4-mini",
}) | OPENAI_REASONING_MODELS

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_OPENAI_REASONING_EFFORT = "low"

ReasoningEffort = str  # "low" | "medium" | "high"


@dataclass
class LLMConfig:
    """Provider-agnostic LLM configuration for a single request."""

    model: str = DEFAULT_ANTHROPIC_MODEL
    api_key: str | None = None
    # Only meaningful for OpenAI reasoning models; None means "not applicable".
    reasoning_effort: str | None = None

    @property
    def provider(self) -> str:
        return "anthropic" if self.model.startswith("claude-") else "openai"

    @property
    def is_openai(self) -> bool:
        return self.provider == "openai"

    @property
    def uses_reasoning(self) -> bool:
        return self.is_openai and self.model in OPENAI_REASONING_MODELS


def make_llm_config(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
) -> LLMConfig:
    """Build an LLMConfig from request parameters, applying sensible defaults.

    Rules:
    - If provider is "openai" and no model given → gpt-5.5
    - If provider is "anthropic" and no model given → claude-sonnet-4-6
    - If neither is given, default to Anthropic
    - For OpenAI reasoning models without an explicit effort → "low"
    """
    if model is None:
        model = DEFAULT_OPENAI_MODEL if provider == "openai" else DEFAULT_ANTHROPIC_MODEL

    # Auto-apply low reasoning effort for OpenAI reasoning models when
    # no explicit effort was requested.
    if reasoning_effort is None and model in OPENAI_REASONING_MODELS:
        reasoning_effort = DEFAULT_OPENAI_REASONING_EFFORT

    return LLMConfig(
        model=model,
        api_key=api_key,
        reasoning_effort=reasoning_effort,
    )
