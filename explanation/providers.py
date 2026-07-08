"""
LLM provider abstraction for chesslens-core.

Supports Anthropic (Claude) and OpenAI (via the Responses API).
Provider is determined by the model name: claude-* → Anthropic, all else → OpenAI.

OpenAI reasoning models (gpt-5.5, o3, o4-mini, o3-mini) accept a
reasoning_effort parameter ("low", "medium", "high"). The default
when no effort is specified is "low".
"""

from __future__ import annotations

from dataclasses import dataclass

from config import llm_models

# Model registry lives in config/llm_models.py (single source of truth for the
# UI picker and these defaults). Effort/reasoning support is derived from it so
# the two never drift.

# OpenAI models that support the Responses `reasoning.effort` parameter.
OPENAI_REASONING_MODELS: frozenset[str] = frozenset(llm_models.openai_reasoning_slugs())

# Anthropic models that support `output_config.effort`.
ANTHROPIC_EFFORT_MODELS: frozenset[str] = frozenset(llm_models.anthropic_effort_slugs())

DEFAULT_ANTHROPIC_MODEL = llm_models.DEFAULT_ANTHROPIC_MODEL
DEFAULT_OPENAI_MODEL = llm_models.DEFAULT_OPENAI_MODEL
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

    @property
    def supports_effort(self) -> bool:
        """Whether this model accepts the effort/reasoning-effort setting
        (OpenAI reasoning models or Anthropic effort-capable models)."""
        if self.is_openai:
            return self.model in OPENAI_REASONING_MODELS
        return self.model in ANTHROPIC_EFFORT_MODELS


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
