"""
Tests for the chess.com import feature (chesscom/ package).

No network: the httpx client is driven by httpx.MockTransport.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from chesscom import client as chesscom_client
from chesscom import storage
from chesscom.client import (
    ChessComClient,
    ChessComError,
    PlayerNotFoundError,
    RateLimitedError,
)
from chesscom.importer import GameImporter

BASE = "https://api.chess.com/pub"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_game(
    game_id="game-1",
    white="alice",
    black="bob",
    white_result="win",
    black_result="resigned",
    rules="chess",
    pgn="[Event \"Live Chess\"]\n\n1. e4 e5 *",
    end_time=1705345678,
    time_class="rapid",
):
    game = {
        "uuid": game_id,
        "url": f"https://www.chess.com/game/live/{game_id}",
        "end_time": end_time,
        "time_class": time_class,
        "time_control": "600",
        "rated": True,
        "rules": rules,
        "white": {"username": white, "rating": 1500, "result": white_result},
        "black": {"username": black, "rating": 1450, "result": black_result},
    }
    if pgn is not None:
        game["pgn"] = pgn
    return game


def mock_client(handler):
    return ChessComClient(transport=httpx.MockTransport(handler))


def archives_response(username, keys):
    return {"archives": [f"{BASE}/player/{username}/games/{k}" for k in keys]}


@pytest.fixture
def games_dir(tmp_path, settings):
    settings.IMPORTED_GAMES_DIR = tmp_path / "imported_games"
    return settings.IMPORTED_GAMES_DIR


async def run_import(username, handler):
    importer = GameImporter(username, client=mock_client(handler))
    try:
        archives = await importer.fetch_archives()
        return [event async for event in importer.run(archives)]
    finally:
        await importer.aclose()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

async def test_get_archives_parses_month_keys():
    def handler(request):
        assert "ChessLens" in request.headers["User-Agent"]
        return httpx.Response(200, json=archives_response("alice", ["2023/11", "2023/12"]))

    client = mock_client(handler)
    try:
        assert await client.get_archives("alice") == ["2023/11", "2023/12"]
    finally:
        await client.aclose()


async def test_unknown_player_raises_not_found():
    client = mock_client(lambda request: httpx.Response(404))
    try:
        with pytest.raises(PlayerNotFoundError):
            await client.get_archives("no-such-user")
    finally:
        await client.aclose()


async def test_rate_limit_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(chesscom_client, "RETRY_BACKOFF", (0, 0, 0))
    calls = []

    def handler(request):
        calls.append(request.url)
        if len(calls) < 3:
            return httpx.Response(429)
        return httpx.Response(200, json=archives_response("alice", ["2024/01"]))

    client = mock_client(handler)
    try:
        assert await client.get_archives("alice") == ["2024/01"]
        assert len(calls) == 3
    finally:
        await client.aclose()


async def test_persistent_rate_limit_raises(monkeypatch):
    monkeypatch.setattr(chesscom_client, "RETRY_BACKOFF", (0, 0, 0))
    client = mock_client(lambda request: httpx.Response(429))
    try:
        with pytest.raises(RateLimitedError):
            await client.get_archives("alice")
    finally:
        await client.aclose()


async def test_server_error_raises_chesscom_error():
    client = mock_client(lambda request: httpx.Response(500))
    try:
        with pytest.raises(ChessComError):
            await client.get_archives("alice")
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------

async def test_import_end_to_end(games_dir):
    games_by_archive = {
        "2023/11": [make_game("g1"), make_game("g2", white="bob", black="alice",
                                                 white_result="resigned", black_result="win")],
        "2023/12": [make_game("g3", white_result="agreed", black_result="agreed")],
    }

    def handler(request):
        path = request.url.path
        if path.endswith("/archives"):
            return httpx.Response(200, json=archives_response("alice", list(games_by_archive)))
        archive = "/".join(path.rstrip("/").split("/")[-2:])
        return httpx.Response(200, json={"games": games_by_archive[archive]})

    events = await run_import("alice", handler)

    progress = [e for e in events if "archive" in e]
    assert [e["imported"] for e in progress] == [2, 1]
    assert events[-1] == {
        "done": True, "total_imported": 3, "total_skipped": 0, "total_games": 3,
    }

    games = storage.list_games("alice")
    assert len(games) == 3
    by_id = {g["id"]: g for g in games}
    # alice played white in g1 and won; black in g2 and won; drew g3
    assert by_id["g1"]["player_color"] == "white"
    assert by_id["g1"]["result"] == "win"
    assert by_id["g1"]["opponent"] == "bob"
    assert by_id["g2"]["player_color"] == "black"
    assert by_id["g2"]["result"] == "win"
    assert by_id["g3"]["result"] == "draw"

    for g in games:
        pgn_path = games_dir / "alice" / "games" / g["filename"]
        assert pgn_path.is_file()
        assert storage.read_pgn("alice", g["id"]).startswith("[Event")

    assert storage.list_players() == [
        {"username": "alice", "game_count": 3,
         "updated_at": storage.load_index("alice")["updated_at"]},
    ]


async def test_variants_and_missing_pgn_skipped(games_dir):
    games = [
        make_game("g1"),
        make_game("g2", rules="chess960"),
        make_game("g3", pgn=None),
    ]

    def handler(request):
        if request.url.path.endswith("/archives"):
            return httpx.Response(200, json=archives_response("alice", ["2024/01"]))
        return httpx.Response(200, json={"games": games})

    events = await run_import("alice", handler)
    assert events[-1]["total_imported"] == 1
    assert events[-1]["total_skipped"] == 2
    assert [g["id"] for g in storage.list_games("alice")] == ["g1"]


async def test_second_run_uses_cache_and_dedupes(games_dir):
    now = datetime.now(timezone.utc)
    current_month = f"{now.year:04d}/{now.month:02d}"
    archives = ["2023/05", current_month]
    fetch_counts = {}

    def handler(request):
        path = request.url.path
        if path.endswith("/archives"):
            return httpx.Response(200, json=archives_response("alice", archives))
        archive = "/".join(path.rstrip("/").split("/")[-2:])
        fetch_counts[archive] = fetch_counts.get(archive, 0) + 1
        games = {"2023/05": [make_game("old-1")], current_month: [make_game("new-1")]}
        return httpx.Response(200, json={"games": games[archive]})

    first = await run_import("alice", handler)
    assert first[-1]["total_imported"] == 2

    second = await run_import("alice", handler)
    # Past month served from the ledger; current month re-fetched but deduped.
    cached = [e for e in second if e.get("cached")]
    assert [e["archive"] for e in cached] == ["2023/05"]
    assert fetch_counts["2023/05"] == 1
    assert fetch_counts[current_month] == 2
    assert second[-1]["total_imported"] == 0
    assert second[-1]["total_games"] == 2


async def test_empty_archives_yields_done_only(games_dir):
    def handler(request):
        return httpx.Response(200, json={"archives": []})

    events = await run_import("alice", handler)
    assert events == [
        {"done": True, "total_imported": 0, "total_skipped": 0, "total_games": 0},
    ]


async def test_missing_uuid_falls_back_to_url_hash(games_dir):
    game = make_game("ignored")
    del game["uuid"]

    def handler(request):
        if request.url.path.endswith("/archives"):
            return httpx.Response(200, json=archives_response("alice", ["2024/01"]))
        return httpx.Response(200, json={"games": [game]})

    events = await run_import("alice", handler)
    assert events[-1]["total_games"] == 1
    (record,) = storage.list_games("alice")
    assert len(record["id"]) == 12  # sha1(url)[:12]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def test_sanitize_component():
    assert storage.sanitize_component("../etc/passwd") == ".._etc_passwd"
    assert storage.sanitize_component("user name!") == "user_name_"
    assert len(storage.sanitize_component("x" * 100)) == 40


def test_player_dir_rejects_bad_usernames(games_dir):
    for bad in ("../etc", "a/b", "", "user name", "x" * 51):
        with pytest.raises(ValueError):
            storage.player_dir(bad)


def test_read_pgn_unknown_game_returns_none(games_dir):
    assert storage.read_pgn("alice", "nope") is None


def test_corrupt_index_returns_fresh_skeleton(games_dir):
    directory = games_dir / "alice"
    directory.mkdir(parents=True)
    (directory / "index.json").write_text("{not json", encoding="utf-8")
    index = storage.load_index("alice")
    assert index["games"] == {}
    assert index["archives_completed"] == []


def test_list_games_sorted_most_recent_first(games_dir):
    index = storage.load_index("alice")
    for game_id, end_time in (("a", 100), ("b", 300), ("c", 200)):
        index["games"][game_id] = {"id": game_id, "end_time": end_time, "filename": f"{game_id}.pgn"}
    storage.save_index("alice", index)
    assert [g["id"] for g in storage.list_games("alice")] == ["b", "c", "a"]


@pytest.mark.parametrize("code,expected", [
    ("win", "win"),
    ("checkmated", "loss"),
    ("timeout", "loss"),
    ("resigned", "loss"),
    ("abandoned", "loss"),
    ("agreed", "draw"),
    ("repetition", "draw"),
    ("stalemate", "draw"),
    ("insufficient", "draw"),
    ("50move", "draw"),
    ("timevsinsufficient", "draw"),
    ("bughousepartnerlose", "other"),
])
def test_result_bucket(code, expected):
    assert storage.result_bucket(code) == expected
