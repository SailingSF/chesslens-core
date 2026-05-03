"""
Middleware to allow per-request LLM provider configuration.

Open-source users paste their own API key in the UI. Headers supported:

  X-Anthropic-Key     — Anthropic API key override
  X-OpenAI-Key        — OpenAI API key override
  X-LLM-Provider      — "anthropic" or "openai"
  X-LLM-Model         — model name (e.g. "gpt-5.5", "claude-sonnet-4-6")
  X-LLM-Reasoning     — reasoning effort for OpenAI reasoning models:
                         "low" | "medium" | "high"

The middleware attaches these to the request object so that API views can
build an LLMConfig without re-parsing headers.
"""


class LLMMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def _attach(self, request):
        h = request.headers
        if key := h.get("X-Anthropic-Key", ""):
            request.anthropic_api_key = key
        if key := h.get("X-OpenAI-Key", ""):
            request.openai_api_key = key
        if provider := h.get("X-LLM-Provider", ""):
            request.llm_provider = provider
        if model := h.get("X-LLM-Model", ""):
            request.llm_model = model
        if effort := h.get("X-LLM-Reasoning", ""):
            request.llm_reasoning_effort = effort

    def __call__(self, request):
        self._attach(request)
        return self.get_response(request)

    async def __acall__(self, request):
        self._attach(request)
        return await self.get_response(request)


# Keep the old name as an alias so existing settings that reference
# ui.middleware.AnthropicKeyMiddleware continue to work.
AnthropicKeyMiddleware = LLMMiddleware
