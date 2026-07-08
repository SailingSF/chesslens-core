"""
LLM model registry — single source of truth for the model picker.

This drives the provider / model / effort dropdowns on the home page (via
template context) and the backend model defaults (via explanation/providers.py).

Adding or removing a model
--------------------------
To make a model available in the picker, add a ``ModelOption`` to
``ANTHROPIC_MODELS`` or ``OPENAI_MODELS`` below — ``slug`` is the exact provider
model id. It appears the next time the app is loaded; no other code changes are
needed. For example, to offer Anthropic's most capable model:

    ModelOption("claude-fable-5", "Claude Fable 5", supports_effort=True)

You can also add slugs *without editing this file* via environment variables
(comma-separated) — handy for a model a provider ships after this release:

    CHESSLENS_EXTRA_ANTHROPIC_MODELS="claude-fable-5,claude-opus-4-9"
    CHESSLENS_EXTRA_OPENAI_MODELS="gpt-6,gpt-6-mini"

Provider is inferred from the slug: ids starting with ``claude-`` are Anthropic,
everything else is OpenAI (matching explanation/providers.py).

Reasoning effort
----------------
Models marked ``supports_effort=True`` accept the effort dropdown
(low / medium / high). For Anthropic that maps to ``output_config.effort``; for
OpenAI reasoning models to the Responses ``reasoning.effort`` param. Models
without effort support ignore the setting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelOption:
    slug: str          # exact provider model id sent to the API
    label: str         # human-friendly name shown in the dropdown
    supports_effort: bool = False


# --- Anthropic (Claude) ---
# The DEFAULT is a Sonnet-tier model on purpose: a game review makes one LLM
# call per key move, so Sonnet is the cost/quality sweet spot for high-volume
# narration. Pick Opus in the UI for the deepest analysis.
ANTHROPIC_MODELS: list[ModelOption] = [
    ModelOption("claude-sonnet-5",   "Claude Sonnet 5",   supports_effort=True),
    ModelOption("claude-opus-4-8",   "Claude Opus 4.8",   supports_effort=True),
    ModelOption("claude-opus-4-7",   "Claude Opus 4.7",   supports_effort=True),
    ModelOption("claude-sonnet-4-6", "Claude Sonnet 4.6", supports_effort=True),
    ModelOption("claude-haiku-4-5",  "Claude Haiku 4.5"),  # Haiku has no effort control
]

# --- OpenAI ---
OPENAI_MODELS: list[ModelOption] = [
    ModelOption("gpt-5.5",      "GPT-5.5 (reasoning)",  supports_effort=True),
    ModelOption("gpt-4.1",      "GPT-4.1"),
    ModelOption("gpt-4.1-mini", "GPT-4.1 Mini"),
    ModelOption("gpt-4o",       "GPT-4o"),
    ModelOption("gpt-4o-mini",  "GPT-4o Mini"),
    ModelOption("o3",           "o3 (reasoning)",       supports_effort=True),
    ModelOption("o4-mini",      "o4-mini (reasoning)",  supports_effort=True),
]

# --- Defaults (also drive the backend fallback in explanation/providers.py) ---
DEFAULT_PROVIDER = "anthropic"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OPENAI_MODEL = "gpt-5.5"

# Reasoning effort — the shared low/medium/high subset both providers accept.
# (Anthropic also has xhigh/max, but keeping the list provider-agnostic avoids
# sending an OpenAI model an effort it would reject.)
REASONING_EFFORTS: list[str] = ["low", "medium", "high"]
DEFAULT_REASONING_EFFORT = "medium"


def _extra_models(env_var: str) -> list[ModelOption]:
    """Parse a comma-separated env override into ModelOptions (label = slug)."""
    raw = os.environ.get(env_var, "")
    return [ModelOption(slug, slug) for slug in (s.strip() for s in raw.split(",")) if slug]


def anthropic_models() -> list[ModelOption]:
    return ANTHROPIC_MODELS + _extra_models("CHESSLENS_EXTRA_ANTHROPIC_MODELS")


def openai_models() -> list[ModelOption]:
    return OPENAI_MODELS + _extra_models("CHESSLENS_EXTRA_OPENAI_MODELS")


def openai_reasoning_slugs() -> set[str]:
    return {m.slug for m in openai_models() if m.supports_effort}


def anthropic_effort_slugs() -> set[str]:
    return {m.slug for m in anthropic_models() if m.supports_effort}


def effort_capable_slugs() -> set[str]:
    """All model slugs (either provider) that accept the effort dropdown."""
    return anthropic_effort_slugs() | openai_reasoning_slugs()


def get_ui_config() -> dict:
    """JSON-serialisable config consumed by the home-page model picker."""

    def dump(models: list[ModelOption]) -> list[dict]:
        return [
            {"slug": m.slug, "label": m.label, "supports_effort": m.supports_effort}
            for m in models
        ]

    return {
        "default_provider": DEFAULT_PROVIDER,
        "providers": {
            "anthropic": {
                "label": "Anthropic",
                "default_model": DEFAULT_ANTHROPIC_MODEL,
                "models": dump(anthropic_models()),
            },
            "openai": {
                "label": "OpenAI",
                "default_model": DEFAULT_OPENAI_MODEL,
                "models": dump(openai_models()),
            },
        },
        "efforts": REASONING_EFFORTS,
        "default_effort": DEFAULT_REASONING_EFFORT,
    }
