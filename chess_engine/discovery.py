"""
Engine discovery — find Stockfish binaries on disk and probe their versions.

Scans a set of directories for executables whose name starts with "stockfish",
then probes each one with a short UCI handshake to capture the version string.
Results are cached so discovery only pays the subprocess cost once.

Directories scanned (in order):
  1. ./engines/ relative to Django BASE_DIR (recommended drop-in location)
  2. ../engines/ — sibling of BASE_DIR (handles a chesslens/chesslens-core
     monorepo layout where built engines live next to the app)
  3. $STOCKFISH_DIR — one or more dirs, os.pathsep-separated (`:` on Unix)
  4. $PATH (resolves the bare `stockfish` command and any versioned siblings)

Engine directories (1–3) are scanned recursively up to ``MAX_SCAN_DEPTH``, so
both flat drop-ins and source build trees are found. $PATH entries are scanned
top-level only to keep discovery fast.

Supported layouts (all discovered automatically):
    engines/stockfish-16.1                 # flat binary
    engines/stockfish-17/src/stockfish     # source build tree
    engines/stockfish-17/stockfish         # extracted release dir
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

# How deep to recurse into engine directories. Reaches a source build tree
# such as engines/stockfish-16.1/src/stockfish while bounding the walk.
MAX_SCAN_DEPTH = 4

# Directory names never worth descending into when hunting for binaries.
_PRUNE_DIRS = {".git", "node_modules", "__pycache__", "tests", ".github"}


def _looks_like_engine(entry: Path) -> bool:
    """True if `entry` is an executable file named like a Stockfish binary."""
    if not entry.is_file():
        return False
    if not entry.name.lower().startswith("stockfish"):
        return False
    # Skip docs/assets that happen to share the prefix.
    if entry.suffix.lower() in (".txt", ".md", ".pdf", ".html", ".o", ".a"):
        return False
    try:
        mode = entry.stat().st_mode
    except OSError:
        return False
    return bool(mode & stat.S_IXUSR)


def _scan_dir(root: Path, *, recursive: bool, max_depth: int) -> Iterable[Path]:
    """Yield candidate engine binaries under `root`."""
    if recursive:
        root_str = str(root)
        for dirpath, dirnames, filenames in os.walk(root):
            depth = dirpath[len(root_str):].count(os.sep)
            if depth >= max_depth:
                dirnames[:] = []
            else:
                dirnames[:] = sorted(d for d in dirnames if d not in _PRUNE_DIRS)
            for name in sorted(filenames):
                entry = Path(dirpath) / name
                if _looks_like_engine(entry):
                    yield entry
    else:
        try:
            entries = sorted(root.iterdir())
        except OSError:
            return
        for entry in entries:
            if _looks_like_engine(entry):
                yield entry


def _candidate_paths(engine_dirs: Iterable[Path], path_dirs: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []

    def add(entry: Path) -> None:
        resolved = entry.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        out.append(resolved)

    for d in engine_dirs:
        if d and d.is_dir():
            for entry in _scan_dir(d, recursive=True, max_depth=MAX_SCAN_DEPTH):
                add(entry)

    for d in path_dirs:
        if d and d.is_dir():
            for entry in _scan_dir(d, recursive=False, max_depth=0):
                add(entry)

    which = shutil.which("stockfish")
    if which:
        add(Path(which))

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

    # Directories scanned recursively for engine binaries.
    engine_dirs: list[Path] = []
    try:
        from django.conf import settings
        base = Path(getattr(settings, "BASE_DIR", Path.cwd()))
    except Exception:
        base = Path.cwd()
    engine_dirs.append(base / "engines")
    engine_dirs.append(base.parent / "engines")

    # STOCKFISH_DIR may list several directories, os.pathsep-separated.
    env_dir = os.environ.get("STOCKFISH_DIR")
    if env_dir:
        engine_dirs.extend(Path(p) for p in env_dir.split(os.pathsep) if p)

    if extra_dirs:
        engine_dirs.extend(Path(d) for d in extra_dirs)

    # Each PATH entry is scanned top-level only — catches versioned names like
    # stockfish-16.1 that `shutil.which("stockfish")` misses, without paying to
    # recurse system directories.
    path_dirs: list[Path] = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry:
            path_dirs.append(Path(entry))

    found: list[DiscoveredEngine] = []
    for path in _candidate_paths(engine_dirs, path_dirs):
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
