"""
Orchestrates a chess.com game import: fetch monthly archives serially,
write one .pgn file per game, and keep index.json (the metadata/idempotency
ledger) updated after every archive so an interrupted import resumes.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from typing import AsyncIterator

from chesscom import storage
from chesscom.client import ChessComClient

# One lock per username: safe on the app's single ASGI event loop, prevents
# two concurrent imports from clobbering the same index.json.
_import_locks: dict[str, asyncio.Lock] = {}


def _lock_for(username: str) -> asyncio.Lock:
    return _import_locks.setdefault(username, asyncio.Lock())


def _current_month_key() -> str:
    now = datetime.now(timezone.utc)
    return f"{now.year:04d}/{now.month:02d}"


def _game_id(game: dict) -> str:
    uuid = game.get("uuid")
    if uuid:
        return uuid
    return hashlib.sha1(game.get("url", "").encode()).hexdigest()[:12]


def _build_record(game: dict, username_lower: str, game_id: str) -> dict:
    white = game.get("white", {})
    black = game.get("black", {})
    player_color = (
        "white" if white.get("username", "").lower() == username_lower else "black"
    )
    player, opponent = (white, black) if player_color == "white" else (black, white)

    end_time = game.get("end_time")
    end_date = (
        datetime.fromtimestamp(end_time, tz=timezone.utc).strftime("%Y-%m-%d")
        if end_time
        else "unknown-date"
    )
    filename = (
        f"{end_date}"
        f"_{storage.sanitize_component(white.get('username', 'white'))}"
        f"_vs_{storage.sanitize_component(black.get('username', 'black'))}"
        f"_{storage.sanitize_component(game_id)[:8]}.pgn"
    )
    return {
        "id": game_id,
        "filename": filename,
        "url": game.get("url", ""),
        "end_time": end_time,
        "end_date": end_date,
        "time_class": game.get("time_class", ""),
        "time_control": game.get("time_control", ""),
        "rated": game.get("rated", False),
        "rules": game.get("rules", ""),
        "white": white.get("username", ""),
        "black": black.get("username", ""),
        "white_rating": white.get("rating"),
        "black_rating": black.get("rating"),
        "player_color": player_color,
        "opponent": opponent.get("username", ""),
        "opponent_rating": opponent.get("rating"),
        "result": storage.result_bucket(player.get("result", "")),
    }


class GameImporter:
    def __init__(self, username: str, *, client: ChessComClient | None = None):
        self.username = username.lower()
        self.client = client or ChessComClient()

    async def aclose(self) -> None:
        await self.client.aclose()

    async def fetch_archives(self) -> list[str]:
        """Pre-flight: raises PlayerNotFoundError / RateLimitedError / ChessComError."""
        return await self.client.get_archives(self.username)

    async def run(self, archives: list[str]) -> AsyncIterator[dict]:
        """Import each archive serially, yielding one progress dict per archive
        and a final done dict. May raise ChessComError mid-stream; everything
        saved before the failure stays on disk and a re-run resumes."""
        async with _lock_for(self.username):
            index = await asyncio.to_thread(storage.load_index, self.username)
            current_month = _current_month_key()
            total_imported = 0
            total_skipped = 0

            for i, archive in enumerate(archives):
                # Completed past months never change; the current month can
                # still grow, so always re-fetch it (dedupe handles overlap).
                if archive in index["archives_completed"] and archive != current_month:
                    yield {
                        "archive": archive,
                        "archive_index": i + 1,
                        "total_archives": len(archives),
                        "cached": True,
                        "total_imported": total_imported,
                    }
                    continue

                games = await self.client.get_archive_games(self.username, archive)
                to_write: list[tuple[str, str]] = []
                skipped = 0
                for game in games:
                    game_id = _game_id(game)
                    if game_id in index["games"]:
                        continue
                    pgn = game.get("pgn")
                    if not pgn or game.get("rules") != "chess":
                        skipped += 1
                        continue
                    record = _build_record(game, self.username, game_id)
                    index["games"][game_id] = record
                    to_write.append((record["filename"], pgn))

                if archive not in index["archives_completed"]:
                    index["archives_completed"].append(archive)
                await asyncio.to_thread(self._save_archive, to_write, index)

                total_imported += len(to_write)
                total_skipped += skipped
                yield {
                    "archive": archive,
                    "archive_index": i + 1,
                    "total_archives": len(archives),
                    "imported": len(to_write),
                    "skipped": skipped,
                    "total_imported": total_imported,
                }

            yield {
                "done": True,
                "total_imported": total_imported,
                "total_skipped": total_skipped,
                "total_games": len(index["games"]),
            }

    def _save_archive(self, to_write: list[tuple[str, str]], index: dict) -> None:
        for filename, pgn in to_write:
            storage.write_pgn(self.username, filename, pgn)
        storage.save_index(self.username, index)
