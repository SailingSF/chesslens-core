"""
Async client for the chess.com public API (https://api.chess.com/pub).

No authentication is required, but chess.com asks clients to identify
themselves via User-Agent and to keep requests serial (parallel requests
get rate-limited with 429).
"""

from __future__ import annotations

import asyncio

import httpx

DEFAULT_USER_AGENT = "ChessLens/1.0 (+https://github.com/SailingSF/chesslens-core)"

# Seconds to sleep after each 429 before retrying; length = retry budget.
RETRY_BACKOFF = (2.0, 5.0, 10.0)


class ChessComError(Exception):
    """Base error; message is safe to surface to the UI."""


class PlayerNotFoundError(ChessComError):
    """The chess.com API returned 404 for this player."""


class RateLimitedError(ChessComError):
    """Still rate-limited (429) after exhausting retries."""


class ChessComClient:
    BASE_URL = "https://api.chess.com/pub"

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        user_agent: str = DEFAULT_USER_AGENT,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": user_agent},
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_archives(self, username: str) -> list[str]:
        """Monthly archive keys ('YYYY/MM') for a player, oldest first."""
        data = await self._get_json(f"{self.BASE_URL}/player/{username}/games/archives")
        archives = []
        for url in data.get("archives", []):
            parts = url.rstrip("/").split("/")
            if len(parts) >= 2:
                archives.append(f"{parts[-2]}/{parts[-1]}")
        return archives

    async def get_archive_games(self, username: str, archive: str) -> list[dict]:
        """Raw game dicts for one monthly archive key ('YYYY/MM')."""
        data = await self._get_json(f"{self.BASE_URL}/player/{username}/games/{archive}")
        return data.get("games", [])

    async def _get_json(self, url: str) -> dict:
        for attempt, backoff in enumerate((*RETRY_BACKOFF, None)):
            try:
                response = await self._client.get(url)
            except httpx.HTTPError as e:
                raise ChessComError(f"Could not reach chess.com: {e}") from e

            if response.status_code == 404:
                raise PlayerNotFoundError("Player not found on chess.com")
            if response.status_code == 429:
                if backoff is None:
                    raise RateLimitedError(
                        "Rate-limited by chess.com — wait a minute and try again"
                    )
                await asyncio.sleep(backoff)
                continue
            if response.status_code >= 400:
                raise ChessComError(
                    f"chess.com API error (HTTP {response.status_code})"
                )
            try:
                return response.json()
            except ValueError as e:
                raise ChessComError("Invalid JSON from chess.com") from e
        raise RateLimitedError("Rate-limited by chess.com")  # pragma: no cover
