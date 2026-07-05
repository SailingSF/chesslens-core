"""
Local storage for imported chess.com games.

Layout (rooted at settings.IMPORTED_GAMES_DIR):

    imported_games/
      {username_lower}/
        index.json          # metadata + idempotency ledger
        games/
          2024-01-15_white_vs_black_ab12cd34.pgn

All functions are synchronous; async callers must wrap them in
``asyncio.to_thread`` so file I/O never blocks the ASGI event loop.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from django.conf import settings

# Chess.com usernames are limited to letters, digits, underscore and hyphen.
# Validating before any path join also rules out path traversal.
USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,50}$")

INDEX_FILENAME = "index.json"
GAMES_SUBDIR = "games"

# Chess.com per-side result codes -> win/loss/draw buckets
_LOSS_CODES = {"checkmated", "timeout", "resigned", "lose", "abandoned"}
_DRAW_CODES = {
    "agreed", "repetition", "stalemate", "insufficient",
    "50move", "timevsinsufficient",
}


def sanitize_component(s: str) -> str:
    """Make a string safe to embed in a filename."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:40]


def result_bucket(result_code: str) -> str:
    """Map a chess.com per-side result code to win/loss/draw/other."""
    if result_code == "win":
        return "win"
    if result_code in _LOSS_CODES:
        return "loss"
    if result_code in _DRAW_CODES:
        return "draw"
    return "other"


def player_dir(username: str) -> Path:
    """Directory for one player's imports. Raises ValueError on bad usernames."""
    if not USERNAME_RE.match(username):
        raise ValueError(f"Invalid chess.com username: {username!r}")
    return Path(settings.IMPORTED_GAMES_DIR) / username.lower()


def _fresh_index(username: str) -> dict:
    return {
        "username": username.lower(),
        "updated_at": None,
        "archives_completed": [],
        "games": {},
    }


def load_index(username: str) -> dict:
    """Load a player's index.json; returns a fresh skeleton if missing/corrupt."""
    path = player_dir(username) / INDEX_FILENAME
    try:
        with open(path, encoding="utf-8") as f:
            index = json.load(f)
        if not isinstance(index.get("games"), dict):
            raise ValueError("malformed index")
        index.setdefault("archives_completed", [])
        return index
    except (OSError, ValueError):
        return _fresh_index(username)


def save_index(username: str, index: dict) -> None:
    """Atomically write a player's index.json."""
    directory = player_dir(username)
    directory.mkdir(parents=True, exist_ok=True)
    index["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = directory / (INDEX_FILENAME + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    tmp.replace(directory / INDEX_FILENAME)


def write_pgn(username: str, filename: str, pgn: str) -> None:
    games_dir = player_dir(username) / GAMES_SUBDIR
    games_dir.mkdir(parents=True, exist_ok=True)
    (games_dir / filename).write_text(pgn, encoding="utf-8")


def list_players() -> list[dict]:
    """All players with imported games: [{username, game_count, updated_at}]."""
    root = Path(settings.IMPORTED_GAMES_DIR)
    if not root.is_dir():
        return []
    players = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not (child / INDEX_FILENAME).is_file():
            continue
        if not USERNAME_RE.match(child.name):
            continue
        index = load_index(child.name)
        players.append({
            "username": index["username"],
            "game_count": len(index["games"]),
            "updated_at": index.get("updated_at"),
        })
    return players


def list_games(username: str) -> list[dict]:
    """A player's imported game records, most recent first."""
    index = load_index(username)
    games = list(index["games"].values())
    games.sort(key=lambda g: g.get("end_time") or 0, reverse=True)
    return games


def read_pgn(username: str, game_id: str) -> str | None:
    """Read one game's PGN. The filename comes from the index lookup only,
    never from the caller, so game_id cannot be used for path traversal."""
    index = load_index(username)
    record = index["games"].get(game_id)
    if record is None:
        return None
    path = player_dir(username) / GAMES_SUBDIR / record["filename"]
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None
