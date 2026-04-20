"""
Engine discovery — find Stockfish binaries on disk and probe their versions.

Scans a set of directories for executables whose name starts with "stockfish",
then probes each one with a short UCI handshake to capture the version string.
Results are cached so discovery only pays the subprocess cost once.

Directories scanned (in order):
  1. ./engines/ relative to Django BASE_DIR (recommended drop-in location)
  2. $STOCKFISH_DIR (optional extra directory)
  3. $PATH (resolves the bare `stockfish` command)

Typical layout:
    chesslens-core/
    └── engines/
        ├── stockfish-16.1
        └── stockfish-17
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class DiscoveredEngine:
    id: str          # stable slug, safe for URLs
    name: str        # display name, e.g. "Stockfish 16.1"
    version: str     # raw "id name" line from UCI
    path: str        # absolute path to the binary


_cache: Optional[list[DiscoveredEngine]] = None


def _candidate_paths(extra_dirs: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for d in extra_dirs:
        if not d or not d.is_dir():
            continue
        for entry in sorted(d.iterdir()):
            if not entry.is_file():
                continue
            if not entry.name.lower().startswith("stockfish"):
                continue
            try:
                mode = entry.stat().st_mode
            except OSError:
                continue
            if not (mode & stat.S_IXUSR):
                continue
            resolved = entry.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(resolved)

    which = shutil.which("stockfish")
    if which:
        resolved = Path(which).resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)

    return out


_VERSION_RE = re.compile(r"stockfish[\s_-]*([\d.]+[\w.-]*)", re.IGNORECASE)


def _probe_version(path: Path, timeout: float = 3.0) -> Optional[str]:
    """Run a short UCI handshake and return the `id name` line, or None."""
    try:
        proc = subprocess.run(
            [str(path)],
            input="uci\nquit\n",
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("id name "):
            return line[len("id name "):].strip()
    return None


def _make_id(version: str, path: Path) -> str:
    match = _VERSION_RE.search(version)
    if match:
        slug = "stockfish-" + match.group(1).lower()
    else:
        slug = "stockfish"
    digest = hashlib.sha1(str(path).encode()).hexdigest()[:6]
    return f"{slug}-{digest}"


def discover_engines(extra_dirs: Optional[Iterable[Path]] = None) -> list[DiscoveredEngine]:
    """Return the cached list of discovered engines, scanning on first call."""
    global _cache
    if _cache is not None:
        return _cache

    dirs: list[Path] = []
    try:
        from django.conf import settings
        base = Path(getattr(settings, "BASE_DIR", Path.cwd()))
        dirs.append(base / "engines")
    except Exception:
        dirs.append(Path.cwd() / "engines")

    env_dir = os.environ.get("STOCKFISH_DIR")
    if env_dir:
        dirs.append(Path(env_dir))

    if extra_dirs:
        dirs.extend(Path(d) for d in extra_dirs)

    found: list[DiscoveredEngine] = []
    for path in _candidate_paths(dirs):
        version = _probe_version(path)
        if version is None:
            continue
        found.append(DiscoveredEngine(
            id=_make_id(version, path),
            name=version,
            version=version,
            path=str(path),
        ))

    _cache = found
    return found


def reset_cache() -> None:
    """Clear the discovery cache. Useful for tests and after installing a new binary."""
    global _cache
    _cache = None


def choose_default(engines: list[DiscoveredEngine], preference: Optional[str] = None) -> Optional[DiscoveredEngine]:
    """
    Pick the default engine.

    - If `preference` matches an id exactly, use it.
    - Else if `preference` is a substring of any version, use the first match.
    - Else prefer anything containing "16.1" (chess.com parity).
    - Else fall back to the first engine.
    """
    if not engines:
        return None
    if preference:
        for e in engines:
            if e.id == preference:
                return e
        pref_lower = preference.lower()
        for e in engines:
            if pref_lower in e.version.lower():
                return e
    for e in engines:
        if "16.1" in e.version:
            return e
    return engines[0]


def get_engine_by_id(engine_id: str) -> Optional[DiscoveredEngine]:
    for e in discover_engines():
        if e.id == engine_id:
            return e
    return None
