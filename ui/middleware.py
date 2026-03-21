"""
Middleware to allow per-request Anthropic API keys.

Open-source users paste their own API key in the UI. The frontend sends it
via the X-Anthropic-Key header. This middleware attaches it to the request
so that the explanation layer can use it instead of the server-wide key.
"""


class AnthropicKeyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        key = request.headers.get("X-Anthropic-Key", "")
        if key:
            request.anthropic_api_key = key
        return self.get_response(request)

    async def __acall__(self, request):
        key = request.headers.get("X-Anthropic-Key", "")
        if key:
            request.anthropic_api_key = key
        return await self.get_response(request)
