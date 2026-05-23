"""
Tests for engine discovery.

These build tiny fake "stockfish" shell scripts that answer the UCI handshake,
so they exercise the real scan + probe path without needing a real binary.
"""

import os
import stat

import pytest

from chess_engine import discovery


def _make_fake_engine(path, version_name):
    """Create an executable fake engine at `path` that prints an `id name` line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        f'echo "id name {version_name}"\n'
        'echo "uciok"\n'
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture(autouse=True)
def _isolate_discovery(tmp_path, monkeypatch):
    """Reset the cache and hide any real engines installed on this machine.

    Points the default search dirs at an empty location and clears PATH so the
    only engines discovered are the fakes a test passes via ``extra_dirs``.
    """
    from django.conf import settings

    discovery.reset_cache()
    empty_base = tmp_path / "_isolated_base"
    (empty_base).mkdir()
    monkeypatch.setattr(settings, "BASE_DIR", empty_base, raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("STOCKFISH_DIR", raising=False)
    yield
    discovery.reset_cache()


def test_discovers_flat_binary(tmp_path):
    engines_dir = tmp_path / "engines"
    _make_fake_engine(engines_dir / "stockfish-16.1", "Stockfish 16.1")

    found = discovery.discover_engines(extra_dirs=[engines_dir])

    names = {e.name for e in found}
    assert "Stockfish 16.1" in names


def test_discovers_binary_in_source_build_tree(tmp_path):
    """A binary nested under src/ (a built source tree) is found via recursion."""
    engines_dir = tmp_path / "engines"
    _make_fake_engine(engines_dir / "stockfish-17" / "src" / "stockfish", "Stockfish 17")

    found = discovery.discover_engines(extra_dirs=[engines_dir])

    assert any(e.name == "Stockfish 17" for e in found)


def test_discovers_multiple_versions(tmp_path):
    engines_dir = tmp_path / "engines"
    _make_fake_engine(engines_dir / "stockfish-16.1" / "src" / "stockfish", "Stockfish 16.1")
    _make_fake_engine(engines_dir / "stockfish-17" / "src" / "stockfish", "Stockfish 17")

    found = discovery.discover_engines(extra_dirs=[engines_dir])

    names = {e.name for e in found}
    assert {"Stockfish 16.1", "Stockfish 17"} <= names
    # Distinct ids even though both binaries are named "stockfish".
    assert len({e.id for e in found}) == len(found)


def test_same_binary_not_listed_twice(tmp_path):
    engines_dir = tmp_path / "engines"
    _make_fake_engine(engines_dir / "stockfish-17" / "src" / "stockfish", "Stockfish 17")

    found = discovery.discover_engines(extra_dirs=[engines_dir, engines_dir])

    assert sum(e.name == "Stockfish 17" for e in found) == 1


def test_non_executable_is_ignored(tmp_path):
    engines_dir = tmp_path / "engines"
    doc = engines_dir / "stockfish-readme"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("not an engine")  # no execute bit

    found = discovery.discover_engines(extra_dirs=[engines_dir])

    assert all(e.path != str(doc.resolve()) for e in found)


def test_choose_default_prefers_16_1():
    engines = [
        discovery.DiscoveredEngine(id="a", name="Stockfish 17", version="Stockfish 17", path="/a"),
        discovery.DiscoveredEngine(id="b", name="Stockfish 16.1", version="Stockfish 16.1", path="/b"),
    ]
    assert discovery.choose_default(engines).id == "b"


def test_choose_default_honors_preference():
    engines = [
        discovery.DiscoveredEngine(id="a", name="Stockfish 17", version="Stockfish 17", path="/a"),
        discovery.DiscoveredEngine(id="b", name="Stockfish 16.1", version="Stockfish 16.1", path="/b"),
    ]
    assert discovery.choose_default(engines, preference="a").id == "a"
    assert discovery.choose_default(engines, preference="17").id == "a"
