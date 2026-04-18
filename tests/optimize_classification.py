"""
Parameter optimization for EP classification — grid search over sigmoid
constants, EP thresholds, and engine parameters to maximize accuracy
against chess.com's Classification V2.

Supports multiple modes:
  - Mock mode (default): uses chess.com's eval data (fast, no Stockfish)
  - Stockfish mode (--stockfish): uses cached real Stockfish results
  - Cache generation (--generate-cache): runs Stockfish on all games, saves JSON
  - Engine sweep (--sweep-engine): sweeps depth/multipv to find best engine match
  - Compare mode (--compare): prints detailed move-by-move comparison

Chess.com Game Review reference parameters (best public knowledge):
  - Engine: Stockfish 16.1 NNUE (server-side)
  - Search: node-based (~1M–4M nodes/position), effective depth ~18–22
  - Classification: Expected Points model (sigmoid win probability, Elo-adjusted)
  - Multi-PV: 2–3 candidates per position
  - Advertised strength: ~3430 Elo

Usage:
    # Quick mock comparison with current config
    python tests/optimize_classification.py --compare

    # Mock mode optimization (no Stockfish required)
    python tests/optimize_classification.py

    # Generate Stockfish caches for all games at depth 20, multipv 3
    python tests/optimize_classification.py --generate-cache

    # Generate caches with specific engine params
    python tests/optimize_classification.py --generate-cache --depth 22 --multipv 4

    # Fill in missing caches using settings already present on other games
    python tests/optimize_classification.py --generate-cache --fill-missing

    # Optimize using real Stockfish caches
    python tests/optimize_classification.py --stockfish

    # Sweep engine depth/multipv to find which matches chess.com best
    python tests/optimize_classification.py --sweep-engine

    # Custom Elo
    python tests/optimize_classification.py --elo 1500
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Optional

import chess

# Add project root to path so imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.expected_points import SigmoidWDLProvider, classify_ep_loss
from chess_engine.service import CandidateMove, EngineResult
from config.classification import ClassificationConfig
from tests.test_classification_comparison import (
    CLASS_SEVERITY,
    GAMES_DIR,
    ChessComMove,
    MoveComparison,
    compute_accuracy_stats,
    load_chesscom_csv,
    print_accuracy_report,
    print_comparison_table,
    run_comparison,
)

STOCKFISH_AVAILABLE = shutil.which("stockfish") is not None

# Well-known engine binary locations (relative to project root or absolute)
ENGINE_PATHS = {
    "sf16.1": Path(__file__).resolve().parent.parent.parent / "engines" / "stockfish-16.1" / "src" / "stockfish",
    "sf17": Path(__file__).resolve().parent.parent.parent / "engines" / "stockfish-17" / "src" / "stockfish",
    "sf18": shutil.which("stockfish") or "stockfish",  # system default
}

DEFAULT_EXTERNAL_GAMES_DIR = Path(__file__).resolve().parent.parent.parent / "classification_model" / "games"
DEFAULT_EXTERNAL_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "classification_model"
    / "game_review_scrape"
    / "games_manifest.json"
)
DEFAULT_EXTERNAL_CACHE_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "classification_model"
    / "engine"
    / "cache"
)


@dataclass(frozen=True)
class CacheSpec:
    """One cache configuration discovered from existing JSON cache filenames."""
    engine_tag: str
    multipv: int
    depth: int | None = None
    nodes: int | None = None

    def __post_init__(self):
        if (self.depth is None) == (self.nodes is None):
            raise ValueError("CacheSpec must set exactly one of depth or nodes")

    @property
    def search_tag(self) -> str:
        return f"n{self.nodes}" if self.nodes is not None else f"d{self.depth}"

    @property
    def summary(self) -> str:
        return f"{self.engine_tag} {self.search_tag} pv{self.multipv}"


def _resolve_engine(engine_arg: str | None) -> tuple[str, str]:
    """
    Resolve --engine argument to (binary_path, version_tag).

    Accepts:
      - None / "stockfish" → system default, auto-detect version
      - "sf16.1" / "sf17" / "sf18" → well-known alias
      - "/path/to/binary" → custom path, auto-detect version
    """
    if engine_arg is None or engine_arg == "stockfish":
        path = shutil.which("stockfish") or "stockfish"
    elif engine_arg in ENGINE_PATHS:
        path = str(ENGINE_PATHS[engine_arg])
    else:
        path = engine_arg

    # Auto-detect version from the binary
    tag = _detect_engine_version(path)
    return path, tag


def _detect_engine_version(binary_path: str) -> str:
    """Run the binary with 'uci' to extract its version string."""
    import subprocess
    try:
        result = subprocess.run(
            [binary_path],
            input="uci\nquit\n",
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if line.startswith("id name "):
                name = line[len("id name "):]
                # "Stockfish 16.1" → "sf16.1", "Stockfish 18" → "sf18"
                parts = name.split()
                if len(parts) >= 2 and parts[0] == "Stockfish":
                    return f"sf{parts[1]}"
                return name.replace(" ", "_").lower()
        return "unknown"
    except Exception:
        return "unknown"


def _game_id(csv_path: Path) -> str:
    """Return the numeric/opaque game id without the `game_` prefix."""
    stem = csv_path.stem
    return stem[len("game_"):] if stem.startswith("game_") else stem


def _count_non_book_moves(csv_path: Path) -> int:
    """Count comparable moves (non-book, non-terminal) in a chess.com CSV."""
    return sum(
        1
        for move in load_chesscom_csv(csv_path)
        if not move.is_game_end and move.classification != "book"
    )


def _find_game_csvs(
    games_dir: Path | None = None,
    min_non_book_moves: int = 0,
) -> list[Path]:
    """Find game CSVs, optionally filtering out ultra-short games."""
    base_dir = games_dir or GAMES_DIR
    game_files = sorted(base_dir.glob("game_*.csv"))
    if min_non_book_moves <= 0:
        return game_files
    return [
        csv_path
        for csv_path in game_files
        if _count_non_book_moves(csv_path) >= min_non_book_moves
    ]


def _parse_cache_spec(cache_path: Path) -> CacheSpec | None:
    """
    Parse a cache filename into a CacheSpec.

    Supports both versioned caches like:
      game_01_sf16.1_d20_pv3.json
      game_01_sf16.1_n2000000_pv3.json

    And legacy depth-based caches like:
      game_01_stockfish_d20_pv3.json
    """
    parts = cache_path.stem.split("_")
    if len(parts) < 4:
        return None

    engine_tag = parts[-3]
    search_tag = parts[-2]
    pv_tag = parts[-1]

    if not pv_tag.startswith("pv") or not pv_tag[2:].isdigit():
        return None
    multipv = int(pv_tag[2:])

    if search_tag.startswith("d") and search_tag[1:].isdigit():
        if engine_tag.startswith("sf") or engine_tag == "stockfish":
            return CacheSpec(engine_tag=engine_tag, depth=int(search_tag[1:]), multipv=multipv)
        return None

    if search_tag.startswith("n") and search_tag[1:].isdigit():
        if engine_tag.startswith("sf") or engine_tag == "stockfish":
            return CacheSpec(engine_tag=engine_tag, nodes=int(search_tag[1:]), multipv=multipv)
        return None

    return None


def _discover_cache_specs(cache_files: list[Path]) -> list[CacheSpec]:
    """Discover unique cache configs from a collection of cache filenames."""
    specs = {_parse_cache_spec(cache) for cache in cache_files}
    return sorted(
        (spec for spec in specs if spec is not None),
        key=lambda spec: (
            spec.engine_tag,
            spec.nodes is None,
            spec.depth if spec.depth is not None else -1,
            spec.nodes if spec.nodes is not None else -1,
            spec.multipv,
        ),
    )


def _collect_existing_cache_specs(game_files: list[Path]) -> list[CacheSpec]:
    """Collect all unique cache configs already present for any game."""
    cache_files = []
    for csv_path in game_files:
        cache_files.extend(csv_path.parent.glob(f"{csv_path.stem}_*.json"))
    return _discover_cache_specs(cache_files)


def _missing_games_for_cache_spec(
    game_files: list[Path],
    spec: CacheSpec,
    force: bool = False,
) -> list[Path]:
    """Return the games that still need a cache for the given spec."""
    if force:
        return list(game_files)

    missing = []
    depth = spec.depth if spec.depth is not None else 0
    for csv_path in game_files:
        cache = _cache_path(
            csv_path,
            depth=depth,
            multipv=spec.multipv,
            engine_tag=spec.engine_tag,
            nodes=spec.nodes,
        )
        if not cache.exists():
            missing.append(csv_path)
    return missing


@lru_cache(maxsize=None)
def _load_game_elos(manifest_path: str | None = None) -> dict[str, dict]:
    """
    Load per-game Elo ratings.

    Supported formats:
      - local test fixture: {"game_01": {"white_elo": ..., "black_elo": ...}}
      - external manifest: [{"game_id": "...", "white_elo": ..., "black_elo": ...}, ...]
    """
    if manifest_path is None:
        elo_file = GAMES_DIR / "game_elos.json"
        if elo_file.exists():
            with open(elo_file) as f:
                return json.load(f)
        return {}

    path = Path(manifest_path)
    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        loaded: dict[str, dict] = {}
        for entry in data:
            game_id = str(entry.get("game_id", "")).strip()
            if not game_id:
                continue
            white_elo = entry.get("white_elo")
            black_elo = entry.get("black_elo")
            if white_elo is None or black_elo is None:
                continue
            loaded[game_id] = {
                "white_elo": int(white_elo),
                "black_elo": int(black_elo),
            }
        return loaded

    return {}


def _game_elo(
    csv_path: Path,
    fallback_elo: int = 820,
    manifest_path: Path | None = None,
) -> int:
    """Get the average Elo for a game from metadata, or use fallback."""
    w, b = _game_elos_by_side(csv_path, fallback_elo, manifest_path=manifest_path)
    return (w + b) // 2


def _game_elos_by_side(
    csv_path: Path,
    fallback_elo: int = 820,
    manifest_path: Path | None = None,
) -> tuple[int, int]:
    """Get (white_elo, black_elo) for a game from metadata, or use fallback."""
    elos = _load_game_elos(str(manifest_path) if manifest_path is not None else None)
    stem = csv_path.stem
    if stem in elos:
        return elos[stem]["white_elo"], elos[stem]["black_elo"]
    game_id = _game_id(csv_path)
    if game_id in elos:
        return elos[game_id]["white_elo"], elos[game_id]["black_elo"]
    return fallback_elo, fallback_elo


def _cache_path(
    csv_path: Path, depth: int, multipv: int,
    engine_tag: str = "sf18", nodes: int | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """
    Return the JSON cache file path for a given game + analysis params.

    Includes engine version tag so caches from different Stockfish versions
    are kept separate.

    Node-based:  game_01_sf16.1_n2000000_pv3.json
    Depth-based: game_01_sf16.1_d20_pv3.json

    Falls back to legacy format (no engine tag) for backward compatibility.
    """
    if nodes is not None:
        search_tag = f"n{nodes}"
    else:
        search_tag = f"d{depth}"

    base_dir = cache_dir or csv_path.parent
    flat_external = base_dir / f"{_game_id(csv_path)}.json"
    if cache_dir is not None and (flat_external.exists() or base_dir != csv_path.parent):
        return flat_external

    if engine_tag == "stockfish":
        return base_dir / f"{csv_path.stem}_stockfish_{search_tag}_pv{multipv}.json"

    versioned = base_dir / f"{csv_path.stem}_{engine_tag}_{search_tag}_pv{multipv}.json"
    if versioned.exists():
        return versioned
    # Legacy fallback (depth-based only): game_01_stockfish_d20_pv3.json
    if nodes is None:
        legacy = base_dir / f"{csv_path.stem}_stockfish_d{depth}_pv{multipv}.json"
        if legacy.exists():
            return legacy
    # Return versioned path for new cache creation
    return versioned


# ---------------------------------------------------------------------------
# Stockfish cache generation — runs real engine on all game positions
# ---------------------------------------------------------------------------

async def generate_stockfish_cache(
    csv_path: Path,
    depth: int = 20,
    nodes: int | None = None,
    multipv: int = 3,
    hash_mb: int = 128,
    threads: int = 2,
    engine_path: str = "stockfish",
    engine_tag: str = "sf18",
) -> Path:
    """
    Run real Stockfish analysis for every position in a game and cache results.

    For each position:
      1. Run multi-PV analysis to get top candidates
      2. If played move isn't among them, run constrained analysis for it
      3. Save all engine results to JSON

    If nodes is set, uses node-based search (chess.com style) instead of
    depth-based search. Node-based search typically reaches depth 18-22
    depending on position complexity.

    Returns the cache file path.
    """
    import chess
    from chess_engine.pool import EnginePool
    from chess_engine.service import _parse_candidates

    moves = load_chesscom_csv(csv_path)
    cache_file = _cache_path(csv_path, depth, multipv, engine_tag, nodes=nodes)

    # Build the search limit
    if nodes is not None:
        limit = chess.engine.Limit(nodes=nodes)
        search_desc = f"nodes={nodes:,}"
    else:
        limit = chess.engine.Limit(depth=depth)
        search_desc = f"depth={depth}"

    pool = EnginePool(engine_path=engine_path, size=1, hash_mb=hash_mb, threads=threads)
    await pool.start()

    engine_results = []
    board = chess.Board()
    analyzed = 0
    total_positions = sum(1 for m in moves if not m.is_game_end)

    try:
        for cm in moves:
            try:
                played_move = board.parse_san(cm.move_san)
            except ValueError:
                board.push_san(cm.move_san)
                continue

            if cm.is_game_end:
                board.push(played_move)
                continue

            analyzed += 1
            print(f"    [{analyzed}/{total_positions}] ply {cm.ply} {cm.color} {cm.move_san}...", end="", flush=True)

            async with pool.acquire() as engine:
                await engine.configure({"UCI_ShowWDL": True})
                info_list = await engine.analyse(
                    board,
                    limit,
                    multipv=multipv,
                )
                if not isinstance(info_list, list):
                    info_list = [info_list]

                candidates = _parse_candidates(info_list)

                # Check if played move is in the candidates
                played_in_candidates = any(c.move == played_move for c in candidates)

                if not played_in_candidates:
                    played_info = await engine.analyse(
                        board,
                        limit,
                        root_moves=[played_move],
                    )
                    if not isinstance(played_info, list):
                        played_info = [played_info]
                    played_candidates = _parse_candidates(played_info)
                    candidates.extend(played_candidates)

            # Get the depth actually reached (from the first candidate's info)
            reached_depth = depth
            if info_list and "depth" in info_list[0]:
                reached_depth = info_list[0]["depth"]

            # Serialize
            entry = {
                "fen": board.fen(),
                "depth": reached_depth,
                "candidates": [],
            }
            if nodes is not None:
                entry["nodes"] = nodes
            for c in candidates:
                entry["candidates"].append({
                    "move": c.move.uci(),
                    "score_cp": c.score_cp,
                    "mate_in": c.mate_in,
                    "pv": [m.uci() for m in c.pv],
                    "wdl_win": c.wdl_win,
                    "wdl_draw": c.wdl_draw,
                    "wdl_loss": c.wdl_loss,
                })
            engine_results.append(entry)

            best_cp = candidates[0].score_cp if candidates else None
            best_str = f"cp={best_cp}" if best_cp is not None else f"mate={candidates[0].mate_in}"
            depth_str = f" d={reached_depth}" if nodes is not None else ""
            print(f" {best_str}{depth_str}")

            board.push(played_move)
    finally:
        await pool.stop()

    with open(cache_file, "w") as f:
        json.dump(engine_results, f, indent=2)

    print(f"  Saved {len(engine_results)} positions to {cache_file.name}")
    return cache_file


async def generate_all_caches(
    game_files: list[Path],
    depth: int = 20,
    nodes: int | None = None,
    multipv: int = 3,
    hash_mb: int = 128,
    threads: int = 2,
    force: bool = False,
    engine_path: str = "stockfish",
    engine_tag: str = "sf18",
) -> list[Path]:
    """Generate Stockfish caches for all game files."""
    if nodes is not None:
        search_desc = f"nodes={nodes:,}"
    else:
        search_desc = f"depth={depth}"

    generated = []
    for csv_path in game_files:
        cache = _cache_path(csv_path, depth, multipv, engine_tag, nodes=nodes)
        if cache.exists() and not force:
            print(f"  {cache.name} already exists (use --force to regenerate)")
            generated.append(cache)
            continue

        print(f"\n  Analyzing {csv_path.name} ({engine_tag}, {search_desc}, multipv={multipv})...")
        cache = await generate_stockfish_cache(
            csv_path, depth=depth, nodes=nodes, multipv=multipv,
            hash_mb=hash_mb, threads=threads,
            engine_path=engine_path, engine_tag=engine_tag,
        )
        generated.append(cache)

    return generated


def _resolve_engine_for_cache_spec(
    spec: CacheSpec,
    engine_override: str | None = None,
) -> tuple[str, str] | None:
    """
    Resolve a discovered cache spec to a local Stockfish binary.

    Returns (engine_path, engine_tag_to_use_for_filenames), or None if the
    engine version cannot be resolved on this machine.
    """
    if spec.engine_tag == "stockfish":
        engine_path, _ = _resolve_engine(engine_override)
        return engine_path, "stockfish"

    if spec.engine_tag in ENGINE_PATHS:
        engine_path, resolved_tag = _resolve_engine(spec.engine_tag)
        if resolved_tag not in {spec.engine_tag, "unknown"}:
            print(
                f"  WARNING: alias {spec.engine_tag} resolved to {resolved_tag} "
                f"but caches are tagged {spec.engine_tag}; keeping filename tag."
            )
        return engine_path, spec.engine_tag

    if engine_override:
        engine_path, resolved_tag = _resolve_engine(engine_override)
        if resolved_tag == spec.engine_tag:
            return engine_path, spec.engine_tag

    return None


async def generate_missing_caches_for_discovered_settings(
    game_files: list[Path],
    specs: list[CacheSpec],
    hash_mb: int = 128,
    threads: int = 2,
    force: bool = False,
    engine_override: str | None = None,
) -> list[Path]:
    """Generate missing caches for every already-discovered cache setting."""
    generated = []

    for spec in specs:
        resolved = _resolve_engine_for_cache_spec(spec, engine_override)
        if resolved is None:
            print(f"\nSkipping {spec.summary}: no local engine binary could be resolved")
            print("  Use --engine with a matching binary path if needed.")
            continue

        engine_path, engine_tag = resolved
        if not Path(engine_path).exists() and not shutil.which(engine_path):
            print(f"\nSkipping {spec.summary}: engine binary not found at {engine_path}")
            continue

        target_games = _missing_games_for_cache_spec(game_files, spec, force=force)
        if not target_games:
            print(f"\n{spec.summary}: all games already have caches")
            continue

        print(f"\nGenerating missing caches for {spec.summary}:")
        print(f"  Engine binary: {engine_path}")
        print(f"  Games: {[f.name for f in target_games]}")
        generated.extend(await generate_all_caches(
            target_games,
            depth=spec.depth if spec.depth is not None else 0,
            nodes=spec.nodes,
            multipv=spec.multipv,
            hash_mb=hash_mb,
            threads=threads,
            force=force,
            engine_path=engine_path,
            engine_tag=engine_tag,
        ))

    return generated


# ---------------------------------------------------------------------------
# Stockfish-cached comparison (replays cached engine results with new config)
# ---------------------------------------------------------------------------

def run_comparison_from_cache(
    csv_path: Path,
    cache_path: Path,
    config: ClassificationConfig | None = None,
    player_elo: int | None = None,
    white_elo: int | None = None,
    black_elo: int | None = None,
) -> list[MoveComparison]:
    """
    Run classification comparison using pre-computed Stockfish engine results.
    The engine results are loaded once and only the classification logic varies.

    If white_elo/black_elo are provided, each move uses the Elo of the side
    that played it. Otherwise falls back to player_elo for both sides.
    """
    import chess
    from analysis.context import AssembledContext, ContextAssembler
    from chess_engine.service import CandidateMove, EngineResult

    if white_elo is None:
        white_elo = player_elo
    if black_elo is None:
        black_elo = player_elo

    moves = load_chesscom_csv(csv_path)

    cache_data = _load_cached_engine_results(cache_path)

    assembler = ContextAssembler(classification_config=config)
    comparisons = []
    board = chess.Board()
    prev_context: AssembledContext | None = None
    engine_idx = 0
    external_entries = None
    if isinstance(cache_data, tuple) and cache_data and isinstance(cache_data[0], tuple):
        external_entries = dict(cache_data)
    else:
        engine_results = cache_data

    for i, cm in enumerate(moves):
        try:
            played_move = board.parse_san(cm.move_san)
        except ValueError:
            # Some external review exports contain malformed/desynced SAN.
            # Stop this game rather than aborting the whole corpus run.
            break

        if cm.is_game_end:
            board.push(played_move)
            continue

        if external_entries is not None:
            current_entry = external_entries.get(cm.ply)
            if current_entry is None:
                board.push(played_move)
                prev_context = None
                continue
            next_entry = external_entries.get(cm.ply + 1)
            engine_result = _build_engine_result_from_external_cache(
                current_entry, played_move, next_entry
            )
        else:
            if engine_idx >= len(engine_results):
                board.push(played_move)
                prev_context = None
                continue

            engine_result = engine_results[engine_idx]
            engine_idx += 1

        move_elo = white_elo if board.turn == chess.WHITE else black_elo
        ctx = assembler.assemble(
            board, engine_result, played_move,
            player_elo=move_elo,
            prev_context=prev_context,
        )

        if ctx.ep_loss is None and not any((ctx.is_brilliant, ctx.is_great, ctx.is_miss)):
            prev_context = ctx
            board.push(played_move)
            continue

        our_class = ctx.cp_loss_label
        chesscom_class = cm.classification

        eval_before_pawns = moves[i - 1].eval_pawns if i > 0 else 0.25
        eval_after_pawns = cm.eval_pawns

        comparisons.append(MoveComparison(
            ply=cm.ply,
            color=cm.color,
            move_san=cm.move_san,
            chesscom_class=chesscom_class,
            our_class=our_class,
            match=(our_class == chesscom_class),
            eval_before_wp=ctx.win_pct_before,
            eval_after_wp=ctx.win_pct_after,
            ep_loss=ctx.ep_loss,
            eval_before_pawns=eval_before_pawns,
            eval_after_pawns=eval_after_pawns,
            is_brilliant=ctx.is_brilliant,
            is_great=ctx.is_great,
            is_miss=ctx.is_miss,
        ))

        prev_context = ctx
        board.push(played_move)

    return comparisons


@lru_cache(maxsize=None)
def _load_cached_engine_results(cache_path: Path) -> tuple:
    """Load and decode cached engine results once per cache file."""
    import chess
    from chess_engine.service import CandidateMove, EngineResult

    with open(cache_path) as f:
        cache_data = json.load(f)

    if isinstance(cache_data, dict):
        return tuple(
            (int(ply), entry)
            for ply, entry in sorted(cache_data.items(), key=lambda item: int(item[0]))
        )

    engine_results = []
    for entry in cache_data:
        candidates = []
        for c in entry["candidates"]:
            candidates.append(CandidateMove(
                move=chess.Move.from_uci(c["move"]),
                score_cp=c.get("score_cp"),
                mate_in=c.get("mate_in"),
                pv=[chess.Move.from_uci(m) for m in c.get("pv", [])],
                wdl_win=c.get("wdl_win"),
                wdl_draw=c.get("wdl_draw"),
                wdl_loss=c.get("wdl_loss"),
            ))
        engine_results.append(EngineResult(
            fen=entry["fen"],
            depth=entry["depth"],
            candidates=candidates,
        ))

    return tuple(engine_results)


_MATE_CP_BASE = 10000


def _mate_to_cp(mate_in: int | None) -> int | None:
    if mate_in is None:
        return None
    if mate_in > 0:
        return _MATE_CP_BASE - mate_in * 10
    return -_MATE_CP_BASE - mate_in * 10


def _parse_external_eval(eval_data: dict | None) -> tuple[int | None, int | None]:
    """Convert external cache eval blobs to (score_cp, mate_in)."""
    if not eval_data:
        return None, None
    eval_type = eval_data.get("type")
    value = eval_data.get("value")
    if value is None:
        return None, None
    if eval_type == "cp":
        return int(value), None
    if eval_type == "mate":
        return None, int(value)
    return None, None


def _normalize_external_eval_to_white_pov(
    score_cp: int | None,
    mate_in: int | None,
    side_to_move: chess.Color,
) -> tuple[int | None, int | None]:
    """
    External cache evals are stored from the side-to-move perspective.

    Convert them into the white-perspective convention used everywhere else in
    this repo, matching `python-chess`'s `score.white()` behavior.
    """
    if side_to_move == chess.WHITE:
        return score_cp, mate_in
    return (
        None if score_cp is None else -score_cp,
        None if mate_in is None else -mate_in,
    )


def _normalize_external_wdl_to_white_pov(
    wdl: dict | None,
    side_to_move: chess.Color,
) -> tuple[int | None, int | None, int | None]:
    """Convert side-to-move WDL permille values into white perspective."""
    wdl = wdl or {}
    win = wdl.get("win")
    draw = wdl.get("draw")
    loss = wdl.get("loss")
    if side_to_move == chess.WHITE:
        return win, draw, loss
    return loss, draw, win


def _candidate_from_external_top_move(move_info: dict, side_to_move: chess.Color):
    """Convert a classification_model top-move entry into a CandidateMove."""
    from chess_engine.service import CandidateMove

    score_cp, mate_in = _parse_external_eval(move_info.get("eval"))
    score_cp, mate_in = _normalize_external_eval_to_white_pov(
        score_cp, mate_in, side_to_move
    )
    wdl_win, wdl_draw, wdl_loss = _normalize_external_wdl_to_white_pov(
        move_info.get("wdl"),
        side_to_move,
    )
    return CandidateMove(
        move=chess.Move.from_uci(move_info["move"]),
        score_cp=score_cp,
        mate_in=mate_in,
        pv=[chess.Move.from_uci(m) for m in move_info.get("pv", [])],
        wdl_win=wdl_win,
        wdl_draw=wdl_draw,
        wdl_loss=wdl_loss,
    )


def _played_candidate_from_next_entry(
    played_move,
    next_entry: dict | None,
):
    """Use the next cached position to recover the played move's resulting eval."""
    from chess_engine.service import CandidateMove

    if next_entry is None:
        return None

    score_cp, mate_in = _parse_external_eval(next_entry.get("eval"))
    if score_cp is None and mate_in is None:
        return None
    next_side_to_move = chess.Board(next_entry["fen"]).turn
    score_cp, mate_in = _normalize_external_eval_to_white_pov(
        score_cp, mate_in, next_side_to_move
    )

    next_top_moves = next_entry.get("top_moves") or []
    next_top = next_top_moves[0] if next_top_moves else {}
    next_wdl_win, next_wdl_draw, next_wdl_loss = _normalize_external_wdl_to_white_pov(
        next_top.get("wdl"),
        next_side_to_move,
    )
    continuation = [chess.Move.from_uci(m) for m in next_top.get("pv", [])]
    return CandidateMove(
        move=played_move,
        score_cp=score_cp,
        mate_in=mate_in,
        pv=[played_move, *continuation],
        wdl_win=next_wdl_win,
        wdl_draw=next_wdl_draw,
        wdl_loss=next_wdl_loss,
    )


def _build_engine_result_from_external_cache(
    entry: dict,
    played_move,
    next_entry: dict | None,
):
    """Build an EngineResult from the flat external cache format."""
    from chess_engine.service import EngineResult

    side_to_move = chess.Board(entry["fen"]).turn
    candidates = [
        _candidate_from_external_top_move(move_info, side_to_move)
        for move_info in (entry.get("top_moves") or [])
        if move_info.get("move")
    ]
    if not any(candidate.move == played_move for candidate in candidates):
        played_candidate = _played_candidate_from_next_entry(played_move, next_entry)
        if played_candidate is not None:
            candidates.append(played_candidate)

    if not candidates:
        raise ValueError(f"No candidates found in external cache entry for ply {entry.get('ply')}")

    depth = max(
        (int(move_info.get("depth", 0)) for move_info in (entry.get("top_moves") or [])),
        default=0,
    )
    return EngineResult(
        fen=entry["fen"],
        depth=depth,
        candidates=candidates,
    )


@dataclass(frozen=True)
class MinimalContext:
    """Prior move context needed for miss/great contextual rules."""
    ep_loss: float | None
    label: str


@dataclass(frozen=True)
class ClassificationOutcome:
    label: str
    ep_loss: float | None
    is_brilliant: bool = False
    is_great: bool = False
    is_miss: bool = False

    @property
    def should_record(self) -> bool:
        return self.ep_loss is not None or self.is_brilliant or self.is_great or self.is_miss


@dataclass(frozen=True)
class MoveFeatures:
    """Minimal optimizer feature set for one move."""
    ply: int
    color: str
    move_san: str
    chesscom_class: str
    player_elo: int | None
    side: chess.Color
    best_cp: int | None
    best_mate_in: int | None
    best_wdl_win: int | None
    best_wdl_draw: int | None
    best_wdl_loss: int | None
    played_cp: int | None
    played_mate_in: int | None
    played_wdl_win: int | None
    played_wdl_draw: int | None
    played_wdl_loss: int | None
    second_cp: int | None
    second_mate_in: int | None
    second_wdl_win: int | None
    second_wdl_draw: int | None
    second_wdl_loss: int | None
    is_engine_top: bool
    candidate_gap_cp: int | None
    has_played_eval: bool
    has_concrete_blunder_evidence: bool
    material_delta_after_move: int
    captured_piece_value: int
    move_is_capture: bool
    is_recapture: bool
    in_check_before: bool
    legal_count_if_in_check: int | None
    response_capture_same_square: bool
    played_material_recovery_ply: int | None
    played_checkmate_ply: int | None
    best_material_gain_ply: int | None


@dataclass(frozen=True)
class CorpusGame:
    csv_path: Path
    white_elo: int
    black_elo: int
    features: tuple[MoveFeatures, ...]


@dataclass(frozen=True)
class CorpusSession:
    games: tuple[CorpusGame, ...]
    build_elapsed_s: float
    total_moves: int

    def subset(self, target_games: int) -> "CorpusSession":
        if target_games <= 0 or target_games >= len(self.games):
            return self
        sampled = _stratified_game_sample(self.games, target_games)
        return CorpusSession(
            games=sampled,
            build_elapsed_s=self.build_elapsed_s,
            total_moves=sum(len(game.features) for game in sampled),
        )


class StatsAccumulator:
    """Streaming accuracy stats so optimizer runs avoid huge comparison lists."""

    def __init__(self, include_breakdown: bool = True):
        self.include_breakdown = include_breakdown
        self.total = 0
        self.matches = 0
        self.close_misses = 0
        self.far_misses = 0
        self.by_class: dict[str, dict] = {}

    def add(self, chesscom_class: str, our_class: str) -> None:
        if not chesscom_class or chesscom_class == "book":
            return

        self.total += 1
        match = chesscom_class == our_class
        if match:
            self.matches += 1
        else:
            s1 = CLASS_SEVERITY.get(chesscom_class, 10)
            s2 = CLASS_SEVERITY.get(our_class, 10)
            if abs(s1 - s2) <= 1:
                self.close_misses += 1
            else:
                self.far_misses += 1

        if chesscom_class not in self.by_class:
            self.by_class[chesscom_class] = {"total": 0, "matched": 0}
            if self.include_breakdown:
                self.by_class[chesscom_class]["our_labels"] = {}

        info = self.by_class[chesscom_class]
        info["total"] += 1
        if match:
            info["matched"] += 1
        if self.include_breakdown:
            labels = info["our_labels"]
            labels[our_class] = labels.get(our_class, 0) + 1

    def to_dict(self) -> dict:
        accuracy = self.matches / self.total if self.total else 0.0
        return {
            "total": self.total,
            "matches": self.matches,
            "accuracy": accuracy,
            "close_misses": self.close_misses,
            "far_misses": self.far_misses,
            "by_class": self.by_class,
        }


def _material_balance(board: chess.Board, color: chess.Color) -> int:
    total = 0
    for piece_type, value in {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }.items():
        total += len(board.pieces(piece_type, color)) * value
    return total


def _piece_value(piece) -> int:
    if piece is None:
        return 0
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    return values.get(piece.piece_type, 0)


def _earliest_material_recovery_ply(
    board: chess.Board,
    pv: list[chess.Move],
    side: chess.Color,
    max_depth: int = 8,
) -> int | None:
    initial_balance = _material_balance(board, side)
    temp = board.copy()
    for idx, move in enumerate(pv[:max_depth], start=1):
        try:
            temp.push(move)
        except Exception:
            break
        if _material_balance(temp, side) >= initial_balance:
            return idx
    return None


def _earliest_checkmate_ply(
    board: chess.Board,
    pv: list[chess.Move],
    max_depth: int = 8,
) -> int | None:
    temp = board.copy()
    for idx, move in enumerate(pv[:max_depth], start=1):
        try:
            temp.push(move)
        except Exception:
            break
        if temp.is_checkmate():
            return idx
    return None


def _earliest_best_material_gain_ply(
    board_before: chess.Board,
    best_candidate: CandidateMove | None,
    side: chess.Color,
    max_depth: int = 8,
) -> int | None:
    if best_candidate is None or not best_candidate.pv:
        return None

    initial = _material_balance(board_before, side) - _material_balance(board_before, not side)
    temp = board_before.copy()
    try:
        temp.push(best_candidate.move)
    except Exception:
        return None

    for idx, move in enumerate(best_candidate.pv[1:max_depth + 1], start=1):
        try:
            temp.push(move)
        except Exception:
            break
        current = _material_balance(temp, side) - _material_balance(temp, not side)
        if current > initial:
            return idx
    return None


def _has_hanging_capture_after(board_after: chess.Board, side: chess.Color) -> bool:
    for opp_move in board_after.legal_moves:
        if not board_after.is_capture(opp_move):
            continue
        captured_piece = board_after.piece_at(opp_move.to_square)
        if captured_piece is None:
            if board_after.is_en_passant(opp_move):
                return True
            continue
        captured_value = _piece_value(captured_piece)
        attacker = board_after.piece_at(opp_move.from_square)
        attacker_value = _piece_value(attacker)
        if captured_value > attacker_value:
            return True
        if captured_value > 0 and not board_after.is_attacked_by(side, opp_move.to_square):
            return True
    return False


def _is_recapture(board: chess.Board, move: chess.Move) -> bool:
    if not board.is_capture(move):
        return False
    if len(board.move_stack) == 0:
        return False
    prev_move = board.move_stack[-1]
    if prev_move.to_square != move.to_square:
        return False
    temp = board.copy()
    temp.pop()
    return temp.is_capture(prev_move)


def _build_mock_engine_result(
    board: chess.Board,
    played_move: chess.Move,
    best_cp: int | None,
    best_mate: int | None,
    played_cp: int | None,
    played_mate: int | None,
) -> EngineResult:
    candidates = []
    legal_moves = list(board.legal_moves)
    best_move = (
        played_move
        if (best_cp == played_cp and best_mate == played_mate)
        else legal_moves[0]
    )
    if best_move == played_move and len(legal_moves) > 1:
        if best_cp != played_cp or best_mate != played_mate:
            best_move = legal_moves[1] if legal_moves[0] == played_move else legal_moves[0]

    candidates.append(CandidateMove(
        move=best_move,
        score_cp=best_cp,
        mate_in=best_mate,
        pv=[best_move],
    ))

    if best_move != played_move:
        candidates.append(CandidateMove(
            move=played_move,
            score_cp=played_cp,
            mate_in=played_mate,
            pv=[played_move],
        ))

    return EngineResult(fen=board.fen(), depth=20, candidates=candidates)


def _extract_move_features(
    board: chess.Board,
    cm: ChessComMove,
    engine_result: EngineResult,
    played_move: chess.Move,
    player_elo: int | None,
) -> MoveFeatures:
    best = engine_result.best
    played_candidate = next((candidate for candidate in engine_result.candidates if candidate.move == played_move), None)
    second = engine_result.candidates[1] if len(engine_result.candidates) >= 2 else None

    best_cp_raw = best.score_cp if best.score_cp is not None else _mate_to_cp(best.mate_in)
    second_cp_raw = second.score_cp if second and second.score_cp is not None else _mate_to_cp(second.mate_in if second else None)
    candidate_gap_cp = None
    if best_cp_raw is not None and second_cp_raw is not None:
        candidate_gap_cp = abs(best_cp_raw - second_cp_raw)

    side = board.turn
    board_after = board.copy()
    board_after.push(played_move)

    mat_before = _material_balance(board, side)
    mat_after = _material_balance(board_after, side)
    opp_mat_before = _material_balance(board, not side)
    opp_mat_after = _material_balance(board_after, not side)

    move_is_capture = board.is_capture(played_move)
    captured_piece_value = _piece_value(board.piece_at(played_move.to_square)) if move_is_capture else 0
    has_concrete_blunder_evidence = False
    if played_candidate is not None and played_candidate.mate_in is not None and played_candidate.mate_in < 0:
        has_concrete_blunder_evidence = True
    if (mat_after - opp_mat_after) < (mat_before - opp_mat_before):
        has_concrete_blunder_evidence = True
    if _has_hanging_capture_after(board_after, side):
        has_concrete_blunder_evidence = True

    played_pv_tail = played_candidate.pv[1:] if played_candidate is not None else []
    played_material_recovery_ply = _earliest_material_recovery_ply(
        board_after, played_pv_tail, side
    ) if played_candidate is not None else None
    played_checkmate_ply = _earliest_checkmate_ply(
        board_after, played_pv_tail
    ) if played_candidate is not None else None
    best_material_gain_ply = _earliest_best_material_gain_ply(board, best, side)

    return MoveFeatures(
        ply=cm.ply,
        color=cm.color,
        move_san=cm.move_san,
        chesscom_class=cm.classification,
        player_elo=player_elo,
        side=side,
        best_cp=best.score_cp,
        best_mate_in=best.mate_in,
        best_wdl_win=best.wdl_win,
        best_wdl_draw=best.wdl_draw,
        best_wdl_loss=best.wdl_loss,
        played_cp=played_candidate.score_cp if played_candidate is not None else None,
        played_mate_in=played_candidate.mate_in if played_candidate is not None else None,
        played_wdl_win=played_candidate.wdl_win if played_candidate is not None else None,
        played_wdl_draw=played_candidate.wdl_draw if played_candidate is not None else None,
        played_wdl_loss=played_candidate.wdl_loss if played_candidate is not None else None,
        second_cp=second.score_cp if second is not None else None,
        second_mate_in=second.mate_in if second is not None else None,
        second_wdl_win=second.wdl_win if second is not None else None,
        second_wdl_draw=second.wdl_draw if second is not None else None,
        second_wdl_loss=second.wdl_loss if second is not None else None,
        is_engine_top=(played_move == best.move),
        candidate_gap_cp=candidate_gap_cp,
        has_played_eval=(played_candidate is not None),
        has_concrete_blunder_evidence=has_concrete_blunder_evidence,
        material_delta_after_move=mat_before - mat_after,
        captured_piece_value=captured_piece_value,
        move_is_capture=move_is_capture,
        is_recapture=_is_recapture(board, played_move),
        in_check_before=board.is_check(),
        legal_count_if_in_check=board.legal_moves.count() if board.is_check() else None,
        response_capture_same_square=(
            board.is_capture(played_move)
            and len(board.move_stack) > 0
            and board.move_stack[-1].to_square == played_move.to_square
        ),
        played_material_recovery_ply=played_material_recovery_ply,
        played_checkmate_ply=played_checkmate_ply,
        best_material_gain_ply=best_material_gain_ply,
    )


def _build_corpus_session(
    game_files: list[Path],
    player_elo: int,
    use_stockfish: bool,
    depth: int = 20,
    multipv: int = 3,
    engine_tag: str = "sf18",
    per_game_elo: bool = True,
    manifest_path: Path | None = None,
    cache_dir: Path | None = None,
    nodes: int | None = None,
) -> CorpusSession:
    started_at = time.time()
    games: list[CorpusGame] = []
    total_moves = 0

    for csv_path in game_files:
        if per_game_elo:
            white_elo, black_elo = _game_elos_by_side(
                csv_path,
                fallback_elo=player_elo,
                manifest_path=manifest_path,
            )
        else:
            white_elo = black_elo = player_elo

        moves = load_chesscom_csv(csv_path)
        board = chess.Board()
        game_features: list[MoveFeatures] = []
        engine_idx = 0
        engine_results = None
        external_entries = None

        if use_stockfish:
            cache_path = _cache_path(
                csv_path,
                depth,
                multipv,
                engine_tag,
                nodes=nodes,
                cache_dir=cache_dir,
            )
            if not cache_path.exists():
                continue
            cache_data = _load_cached_engine_results(cache_path)
            if isinstance(cache_data, tuple) and cache_data and isinstance(cache_data[0], tuple):
                external_entries = dict(cache_data)
            else:
                engine_results = cache_data

        for i, cm in enumerate(moves):
            try:
                played_move = board.parse_san(cm.move_san)
            except ValueError:
                break

            if cm.is_game_end:
                board.push(played_move)
                continue

            move_elo = white_elo if board.turn == chess.WHITE else black_elo

            if use_stockfish:
                if external_entries is not None:
                    current_entry = external_entries.get(cm.ply)
                    if current_entry is None:
                        board.push(played_move)
                        continue
                    next_entry = external_entries.get(cm.ply + 1)
                    engine_result = _build_engine_result_from_external_cache(
                        current_entry, played_move, next_entry
                    )
                else:
                    if engine_results is None or engine_idx >= len(engine_results):
                        board.push(played_move)
                        continue
                    engine_result = engine_results[engine_idx]
                    engine_idx += 1
            else:
                if i == 0:
                    eval_before_pawns = 0.25
                    eval_before_mate = None
                else:
                    prev = moves[i - 1]
                    eval_before_pawns = prev.eval_pawns
                    eval_before_mate = prev.mate_in
                eval_after_pawns = cm.eval_pawns
                eval_after_mate = cm.mate_in
                best_cp = round(eval_before_pawns * 100) if eval_before_pawns is not None else None
                played_cp = round(eval_after_pawns * 100) if eval_after_pawns is not None else None
                engine_result = _build_mock_engine_result(
                    board,
                    played_move,
                    best_cp,
                    eval_before_mate,
                    played_cp,
                    eval_after_mate,
                )

            game_features.append(_extract_move_features(
                board,
                cm,
                engine_result,
                played_move,
                move_elo,
            ))
            board.push(played_move)

        games.append(CorpusGame(
            csv_path=csv_path,
            white_elo=white_elo,
            black_elo=black_elo,
            features=tuple(game_features),
        ))
        total_moves += len(game_features)

    return CorpusSession(
        games=tuple(games),
        build_elapsed_s=time.time() - started_at,
        total_moves=total_moves,
    )


def _stratified_game_sample(games: tuple[CorpusGame, ...], target_games: int) -> tuple[CorpusGame, ...]:
    buckets: dict[str, list[CorpusGame]] = defaultdict(list)
    for game in games:
        avg_elo = (game.white_elo + game.black_elo) // 2
        if avg_elo < 600:
            bucket = "elo_0_599"
        elif avg_elo < 900:
            bucket = "elo_600_899"
        elif avg_elo < 1200:
            bucket = "elo_900_1199"
        else:
            bucket = "elo_1200_plus"
        buckets[bucket].append(game)

    ordered_buckets = []
    for key in sorted(buckets):
        ordered = sorted(
            buckets[key],
            key=lambda game: (len(game.features), game.csv_path.name),
        )
        ordered_buckets.append(ordered)

    selected: list[CorpusGame] = []
    seen = set()
    while len(selected) < target_games:
        progressed = False
        for bucket in ordered_buckets:
            if not bucket:
                continue
            index = len(selected) % len(bucket)
            candidate = bucket[index]
            if candidate.csv_path not in seen:
                selected.append(candidate)
                seen.add(candidate.csv_path)
                progressed = True
                if len(selected) >= target_games:
                    break
            else:
                for fallback in bucket:
                    if fallback.csv_path not in seen:
                        selected.append(fallback)
                        seen.add(fallback.csv_path)
                        progressed = True
                        break
                if len(selected) >= target_games:
                    break
        if not progressed:
            break

    if len(selected) < target_games:
        for game in sorted(games, key=lambda game: (len(game.features), game.csv_path.name)):
            if game.csv_path in seen:
                continue
            selected.append(game)
            if len(selected) >= target_games:
                break

    return tuple(selected)


def _white_pov_to_side_win_pct(win_pct: float, side: chess.Color) -> float:
    return 1.0 - win_pct if side == chess.BLACK else win_pct


def _is_heavy_piece_sacrifice(feature: MoveFeatures, config: ClassificationConfig) -> bool:
    if feature.player_elo is not None and feature.player_elo < config.brilliant_low_elo_threshold:
        min_sacrifice = config.brilliant_low_elo_material_sacrifice
    else:
        min_sacrifice = config.brilliant_min_material_sacrifice
    if feature.material_delta_after_move < min_sacrifice:
        return False
    if feature.move_is_capture and feature.captured_piece_value >= feature.material_delta_after_move:
        return False
    return True


def _detect_brilliant_precomputed(
    feature: MoveFeatures,
    config: ClassificationConfig,
    ep_loss: float,
    best_win_pct: float,
    played_win_pct: float,
) -> bool:
    mover_win_pct_before = _white_pov_to_side_win_pct(best_win_pct, feature.side)
    mover_win_pct_after = _white_pov_to_side_win_pct(played_win_pct, feature.side)
    if feature.player_elo is not None and feature.player_elo < config.brilliant_low_elo_max_win_pct_threshold:
        max_win_before = config.brilliant_low_elo_max_win_pct_before
    else:
        max_win_before = config.brilliant_max_win_pct_before
    if ep_loss > config.brilliant_ep_tolerance:
        return False
    if not _is_heavy_piece_sacrifice(feature, config):
        return False
    if mover_win_pct_after < config.brilliant_min_win_pct_after:
        return False
    if mover_win_pct_before > max_win_before:
        return False
    if (feature.played_material_recovery_ply is not None
            and feature.played_material_recovery_ply <= config.brilliant_pv_depth_check):
        return False
    has_mate = feature.played_mate_in is not None and feature.played_mate_in > 0
    if not has_mate and feature.played_checkmate_ply is not None:
        has_mate = feature.played_checkmate_ply <= config.brilliant_pv_depth_check
    if not has_mate and mover_win_pct_after < config.brilliant_decisive_win_pct:
        return False
    return True


def _detect_great_precomputed(
    feature: MoveFeatures,
    config: ClassificationConfig,
    prev_context: MinimalContext | None,
    ep_loss: float,
    best_win_pct: float,
    played_win_pct: float,
) -> bool:
    mover_win_pct_before = _white_pov_to_side_win_pct(best_win_pct, feature.side)
    if not feature.is_engine_top or ep_loss > config.great_ep_tolerance:
        return False
    if feature.is_recapture:
        return False
    if feature.in_check_before and feature.legal_count_if_in_check is not None:
        if feature.legal_count_if_in_check <= config.great_max_forced_check_moves:
            return False
    if (feature.candidate_gap_cp is not None
            and feature.candidate_gap_cp >= config.great_min_candidate_gap_cp
            and feature.best_mate_in is not None
            and feature.best_mate_in > 0
            and feature.second_mate_in is None):
        return True
    if (feature.candidate_gap_cp is not None
            and feature.candidate_gap_cp >= config.great_min_candidate_gap_cp
            and config.great_min_win_pct_before <= mover_win_pct_before <= config.great_max_win_pct_before):
        return True

    capitalization_gap_cp = max(
        1,
        int(round(config.great_min_candidate_gap_cp * config.great_capitalization_gap_scale)),
    )
    if (prev_context is not None
            and feature.candidate_gap_cp is not None
            and config.great_min_win_pct_before <= mover_win_pct_before <= config.great_max_win_pct_before):
        prev_ep_loss = prev_context.ep_loss
        effective_cap_gap = capitalization_gap_cp
        if feature.response_capture_same_square:
            effective_cap_gap = max(
                effective_cap_gap,
                int(round(capitalization_gap_cp * 1.5)),
            )
        if (prev_ep_loss is not None
                and prev_ep_loss >= config.great_capitalization_min_ep_loss
                and prev_ep_loss < config.great_capitalization_max_ep_loss
                and feature.candidate_gap_cp >= effective_cap_gap):
            return True

    if prev_context is not None and feature.candidate_gap_cp is not None:
        prev_ep_loss = prev_context.ep_loss
        effective_gap = config.great_post_blunder_min_gap_cp
        if feature.response_capture_same_square:
            effective_gap = max(effective_gap, int(round(effective_gap * 1.5)))
        if (
            prev_ep_loss is not None
            and prev_ep_loss >= config.great_post_blunder_min_prev_ep_loss
            and prev_context.label in {"mistake", "blunder", "miss"}
            and feature.candidate_gap_cp >= effective_gap
            and mover_win_pct_before <= config.great_post_blunder_max_win_pct_before
        ):
            return True

    return False


def _detect_miss_precomputed(
    feature: MoveFeatures,
    config: ClassificationConfig,
    prev_context: MinimalContext,
    best_win_pct: float,
    ep_loss: float,
) -> bool:
    mover_best_win_pct = _white_pov_to_side_win_pct(best_win_pct, feature.side)
    prev_ep_loss = prev_context.ep_loss
    if (
        (prev_ep_loss is None or prev_ep_loss < config.miss_opponent_ep_loss_threshold)
        and prev_context.label not in {"mistake", "blunder", "miss"}
    ):
        return False
    if ep_loss < config.miss_min_ep_loss:
        return False

    has_concrete_opportunity = False
    if feature.best_mate_in is not None and feature.best_mate_in > 0:
        has_concrete_opportunity = True
    if not has_concrete_opportunity and feature.best_material_gain_ply is not None:
        has_concrete_opportunity = feature.best_material_gain_ply <= config.miss_best_wins_material_depth
    if not has_concrete_opportunity and mover_best_win_pct >= config.miss_best_win_pct_threshold:
        has_concrete_opportunity = True
    return has_concrete_opportunity


def classify_precomputed_move(
    feature: MoveFeatures,
    config: ClassificationConfig,
    provider: SigmoidWDLProvider,
    prev_context: MinimalContext | None,
) -> ClassificationOutcome:
    best_win_pct = provider.get_win_pct(
        cp=feature.best_cp,
        mate_in=feature.best_mate_in,
        elo=feature.player_elo,
        wdl_win=feature.best_wdl_win,
        wdl_draw=feature.best_wdl_draw,
        wdl_loss=feature.best_wdl_loss,
    )
    played_win_pct = None
    if feature.has_played_eval:
        played_win_pct = provider.get_win_pct(
            cp=feature.played_cp,
            mate_in=feature.played_mate_in,
            elo=feature.player_elo,
            wdl_win=feature.played_wdl_win,
            wdl_draw=feature.played_wdl_draw,
            wdl_loss=feature.played_wdl_loss,
        )

    ep_loss = None
    if best_win_pct is not None and played_win_pct is not None:
        if feature.side == chess.WHITE:
            ep_loss = max(0.0, best_win_pct - played_win_pct)
        else:
            ep_loss = max(0.0, played_win_pct - best_win_pct)

    effective_config = config
    if feature.player_elo is not None and feature.player_elo < config.excellent_elo_threshold:
        effective_config = replace(config, ep_excellent=config.ep_excellent_low_elo)

    label = classify_ep_loss(ep_loss, effective_config)

    if feature.is_engine_top and label in ("best", "excellent"):
        if feature.candidate_gap_cp is not None and feature.candidate_gap_cp >= config.best_promotion_min_gap_cp:
            label = "best"
        elif feature.candidate_gap_cp is None:
            label = "best"
    elif label == "best" and not feature.is_engine_top and feature.candidate_gap_cp is not None:
        in_gap_band = (
            config.best_non_top_excellent_min_gap_cp
            <= feature.candidate_gap_cp
            < config.best_non_top_excellent_max_gap_cp
        )
        if feature.best_cp is not None and feature.played_cp is not None:
            cp_loss = abs(feature.best_cp - feature.played_cp)
            cp_loss_meaningful = cp_loss >= config.best_non_top_excellent_min_cp_loss
        else:
            cp_loss_meaningful = True
        if in_gap_band and cp_loss_meaningful:
            label = "excellent"

    if label == "blunder" and feature.has_played_eval:
        if not (feature.has_concrete_blunder_evidence or (ep_loss is not None and ep_loss >= 0.35)):
            label = "mistake"

    is_brilliant = False
    is_great = False
    is_miss = False
    if feature.has_played_eval and ep_loss is not None and best_win_pct is not None and played_win_pct is not None:
        is_brilliant = _detect_brilliant_precomputed(
            feature, config, ep_loss, best_win_pct, played_win_pct
        )
        is_great = _detect_great_precomputed(
            feature, config, prev_context, ep_loss, best_win_pct, played_win_pct
        )
        if prev_context is not None:
            is_miss = _detect_miss_precomputed(
                feature, config, prev_context, best_win_pct, ep_loss
            )

        if is_brilliant:
            label = "brilliant"
        elif is_great and label in ("best", "excellent", "good"):
            label = "great"
        elif is_miss:
            if label == "blunder":
                if (prev_context is not None
                        and (
                            (prev_context.ep_loss is not None
                             and prev_context.ep_loss >= config.miss_blunder_override_min_prev_ep)
                            or prev_context.label in {"mistake", "blunder", "miss"}
                        )):
                    label = "miss"
            else:
                label = "miss"

    return ClassificationOutcome(
        label=label,
        ep_loss=ep_loss,
        is_brilliant=is_brilliant,
        is_great=is_great,
        is_miss=is_miss,
    )


def _evaluate_corpus_session(
    session: CorpusSession,
    config: ClassificationConfig,
    include_breakdown: bool = True,
) -> dict:
    provider = SigmoidWDLProvider(config)
    stats = StatsAccumulator(include_breakdown=include_breakdown)

    for game in session.games:
        prev_context: MinimalContext | None = None
        for feature in game.features:
            outcome = classify_precomputed_move(feature, config, provider, prev_context)
            prev_context = MinimalContext(outcome.ep_loss, outcome.label)
            if not outcome.should_record:
                continue
            stats.add(feature.chesscom_class, outcome.label)

    return stats.to_dict()


# ---------------------------------------------------------------------------
# Grid search helpers
# ---------------------------------------------------------------------------

OptimizationResult = tuple[tuple, dict, ClassificationConfig]


@dataclass(frozen=True)
class ScoreProfile:
    key: str
    label: str
    class_weights: dict[str, float] | None = None


STANDARD_SCORE = ScoreProfile(key="standard", label="Standard")
WEIGHTED_SCORE = ScoreProfile(
    key="weighted",
    label="Weighted",
    class_weights={
        # Keep core labels balanced, but give contextual/special labels a bit
        # more influence without letting the single "brilliant" sample dominate.
        "best": 1.0,
        "excellent": 1.0,
        "good": 1.0,
        "inaccuracy": 1.0,
        "mistake": 1.0,
        "blunder": 1.0,
        "great": 2.0,
        "miss": 2.0,
        "brilliant": 1.5,
    },
)
SCORE_PROFILES = {
    STANDARD_SCORE.key: STANDARD_SCORE,
    WEIGHTED_SCORE.key: WEIGHTED_SCORE,
}


def _config_key(config: ClassificationConfig) -> tuple:
    """Stable cache key for a ClassificationConfig."""
    return tuple(asdict(config).items())


def _weighted_class_recall(stats: dict, class_weights: dict[str, float]) -> float:
    """Weighted mean per-class recall, to avoid majority classes dominating."""
    by_class = stats.get("by_class", {})
    weighted_total = 0.0
    weighted_hits = 0.0
    for cls, info in by_class.items():
        total = info.get("total", 0)
        if total <= 0:
            continue
        weight = class_weights.get(cls, 1.0)
        weighted_total += weight
        weighted_hits += weight * (info.get("matched", 0) / total)
    return weighted_hits / weighted_total if weighted_total else 0.0


def _score_stats(stats: dict, profile: ScoreProfile = STANDARD_SCORE) -> tuple:
    """Return a sortable score tuple for the requested optimization objective."""
    if profile.class_weights is None:
        return (stats["accuracy"], -stats.get("far_misses", 0))

    weighted_recall = _weighted_class_recall(stats, profile.class_weights)
    return (
        weighted_recall,
        stats["accuracy"],
        -stats.get("far_misses", 0),
    )


def _format_score(score: tuple | None, profile: ScoreProfile = STANDARD_SCORE) -> str:
    if score is None:
        return "n/a"
    if profile.class_weights is None:
        return f"acc={score[0]:.1%}, far={-score[1]}"
    return f"wrecall={score[0]:.1%}, acc={score[1]:.1%}, far={-score[2]}"


def _describe_stats(stats: dict, profile: ScoreProfile = STANDARD_SCORE) -> str:
    if profile.class_weights is None:
        return f"acc={stats['accuracy']:.1%}, far={stats.get('far_misses', 0)}"
    return (
        f"wrecall={_weighted_class_recall(stats, profile.class_weights):.1%}, "
        f"acc={stats['accuracy']:.1%}, far={stats.get('far_misses', 0)}"
    )


def _print_sweep_progress(
    label: str,
    done: int,
    total: int,
    started_at: float,
    best_score: tuple | None = None,
    profile: ScoreProfile = STANDARD_SCORE,
) -> None:
    """Print a one-line progress update with ETA and current best score."""
    elapsed = max(0.001, time.time() - started_at)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else 0.0
    best_str = f", best={_format_score(best_score, profile)}" if best_score is not None else ""
    print(
        f"    {label}: {done}/{total} ({done / total:.0%}), "
        f"elapsed={elapsed:.1f}s, eta={eta:.1f}s{best_str}",
        end="\r" if done < total else "\n",
        flush=True,
    )


def _coerce_param_value(template: int | float, value: int | float) -> int | float:
    if isinstance(template, int):
        return int(round(value))
    return round(float(value), 6)


def _is_valid_phase3_config(config: ClassificationConfig) -> bool:
    # EP thresholds must stay pinned to chess.com published values
    if not (config.ep_excellent == 0.02 and config.ep_good == 0.05
            and config.ep_inaccuracy == 0.10 and config.ep_mistake == 0.20):
        return False
    return (
        config.elo_scale_floor >= 0.1
        and config.elo_scale_range >= 0.1
        and config.elo_scale_floor + config.elo_scale_range <= 1.05
        and config.great_min_candidate_gap_cp >= 50
        and config.best_promotion_min_gap_cp >= 5
        and config.best_non_top_excellent_min_gap_cp >= 0
        and config.best_non_top_excellent_min_gap_cp < config.best_non_top_excellent_max_gap_cp
        and config.great_capitalization_min_ep_loss >= 0.02
        and config.great_capitalization_min_ep_loss < config.great_capitalization_max_ep_loss
        and 0.2 <= config.great_capitalization_gap_scale <= 1.0
        and 0.0 < config.great_min_win_pct_before < config.great_max_win_pct_before < 1.0
        and config.miss_opponent_ep_loss_threshold >= 0.02
        and config.miss_min_ep_loss >= 0.02
        and config.ep_excellent_low_elo > 0
        and 0.6 <= config.brilliant_low_elo_max_win_pct_before <= 0.98
        and 600 <= config.elo_scale_midpoint <= 2200
        and 0.3 <= config.miss_best_win_pct_threshold <= 0.95
    )


def _unique_configs(configs: list[ClassificationConfig]) -> list[ClassificationConfig]:
    seen = set()
    unique = []
    for config in configs:
        key = _config_key(config)
        if key in seen:
            continue
        seen.add(key)
        unique.append(config)
    return unique


def _top_unique_configs(results: list[OptimizationResult], limit: int) -> list[ClassificationConfig]:
    seen = set()
    top = []
    for _, _, config in sorted(results, key=lambda x: x[0], reverse=True):
        key = _config_key(config)
        if key in seen:
            continue
        seen.add(key)
        top.append(config)
        if len(top) >= limit:
            break
    return top


def _unique_optimization_results(results: list[OptimizationResult]) -> list[OptimizationResult]:
    unique: dict[tuple, OptimizationResult] = {}
    for score, stats, config in sorted(results, key=lambda x: x[0], reverse=True):
        key = _config_key(config)
        if key not in unique:
            unique[key] = (score, stats, config)
    return list(unique.values())


class OptimizerRunner:
    """Memoized wrapper around `run_single` for repeated optimizer evaluations."""

    def __init__(
        self,
        game_files: list[Path],
        player_elo: int,
        use_stockfish: bool,
        depth: int = 20,
        multipv: int = 3,
        engine_tag: str = "sf18",
        nodes: int | None = None,
        per_game_elo: bool = True,
        manifest_path: Path | None = None,
        cache_dir: Path | None = None,
        coarse_sample_games: int = 128,
        coarse_top_k: int = 24,
    ):
        self.game_files = game_files
        self.player_elo = player_elo
        self.use_stockfish = use_stockfish
        self.depth = depth
        self.multipv = multipv
        self.engine_tag = engine_tag
        self.nodes = nodes
        self.per_game_elo = per_game_elo
        self.manifest_path = manifest_path
        self.cache_dir = cache_dir
        self.coarse_sample_games = coarse_sample_games
        self.coarse_top_k = coarse_top_k
        self.session = _build_corpus_session(
            game_files,
            player_elo,
            use_stockfish,
            depth=depth,
            multipv=multipv,
            engine_tag=engine_tag,
            per_game_elo=per_game_elo,
            manifest_path=manifest_path,
            cache_dir=cache_dir,
            nodes=nodes,
        )
        self.sample_session = self.session.subset(coarse_sample_games)
        self._summary_cache: dict[tuple, dict] = {}
        self._detailed_cache: dict[tuple, dict] = {}
        self._sample_summary_cache: dict[tuple, dict] = {}
        self._sample_detailed_cache: dict[tuple, dict] = {}
        self._timings = {
            "full_evals": 0,
            "full_eval_s": 0.0,
            "sample_evals": 0,
            "sample_eval_s": 0.0,
        }

    def evaluate(
        self,
        config: ClassificationConfig,
        include_breakdown: bool = False,
        sample: bool = False,
    ) -> dict:
        key = _config_key(config)
        session = self.sample_session if sample else self.session
        summary_cache = self._sample_summary_cache if sample else self._summary_cache
        detailed_cache = self._sample_detailed_cache if sample else self._detailed_cache
        timing_prefix = "sample" if sample else "full"

        if include_breakdown:
            if key not in detailed_cache:
                started_at = time.time()
                detailed_cache[key] = _evaluate_corpus_session(
                    session,
                    config,
                    include_breakdown=True,
                )
                self._timings[f"{timing_prefix}_evals"] += 1
                self._timings[f"{timing_prefix}_eval_s"] += time.time() - started_at
            return detailed_cache[key]

        if key in detailed_cache:
            return detailed_cache[key]

        if key not in summary_cache:
            started_at = time.time()
            summary_cache[key] = _evaluate_corpus_session(
                session,
                config,
                include_breakdown=False,
            )
            self._timings[f"{timing_prefix}_evals"] += 1
            self._timings[f"{timing_prefix}_eval_s"] += time.time() - started_at
        return summary_cache[key]

    @property
    def unique_evaluations(self) -> int:
        return len(set(self._summary_cache) | set(self._detailed_cache))

    @property
    def sample_games(self) -> int:
        return len(self.sample_session.games)

    def timing_summary(self) -> dict[str, float | int]:
        summary = dict(self._timings)
        summary["build_elapsed_s"] = self.session.build_elapsed_s
        summary["full_games"] = len(self.session.games)
        summary["full_moves"] = self.session.total_moves
        summary["sample_games"] = len(self.sample_session.games)
        summary["sample_moves"] = self.sample_session.total_moves
        return summary


def run_single(
    config: ClassificationConfig,
    game_files: list[Path],
    player_elo: int,
    use_stockfish: bool,
    depth: int = 20,
    multipv: int = 3,
    engine_tag: str = "sf18",
    per_game_elo: bool = True,
    manifest_path: Path | None = None,
    cache_dir: Path | None = None,
    nodes: int | None = None,
    include_breakdown: bool = True,
    session: CorpusSession | None = None,
) -> dict:
    """Run comparison across all games with a given config, return aggregate stats.

    When per_game_elo=True (default), uses Elo from game_elos.json for each game,
    falling back to player_elo if the game isn't listed.
    """
    corpus = session or _build_corpus_session(
        game_files,
        player_elo,
        use_stockfish,
        depth=depth,
        multipv=multipv,
        engine_tag=engine_tag,
        per_game_elo=per_game_elo,
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        nodes=nodes,
    )
    return _evaluate_corpus_session(corpus, config, include_breakdown=include_breakdown)


def _evaluate_config_candidates(
    configs: list[ClassificationConfig],
    evaluator: OptimizerRunner,
    profile: ScoreProfile,
    label: str,
    progress_every: int = 25,
    full_top_k: int | None = None,
) -> list[OptimizationResult]:
    """Evaluate many configs using a stratified sample first, then full corpus."""
    configs = _unique_configs(configs)
    if not configs:
        return []

    use_sample = evaluator.sample_games < len(evaluator.game_files)
    if not use_sample:
        results: list[OptimizationResult] = []
        best_score = None
        started_at = time.time()
        for done, config in enumerate(configs, start=1):
            stats = evaluator.evaluate(config)
            if stats["total"] == 0:
                continue
            score = _score_stats(stats, profile)
            results.append((score, stats, config))
            if best_score is None or score > best_score:
                best_score = score
            if done % progress_every == 0 or done == len(configs):
                _print_sweep_progress(label, done, len(configs), started_at, best_score, profile)
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    started_at = time.time()
    sample_best = None
    sample_results: list[OptimizationResult] = []
    for done, config in enumerate(configs, start=1):
        stats = evaluator.evaluate(config, sample=True)
        if stats["total"] == 0:
            continue
        score = _score_stats(stats, profile)
        sample_results.append((score, stats, config))
        if sample_best is None or score > sample_best:
            sample_best = score
        if done % progress_every == 0 or done == len(configs):
            _print_sweep_progress(
                f"{label} sample", done, len(configs), started_at, sample_best, profile
            )

    validate_count = min(full_top_k or evaluator.coarse_top_k, len(sample_results))
    validated_configs = _top_unique_configs(sample_results, validate_count)
    print(
        f"    {label}: validating top {len(validated_configs)} "
        f"of {len(sample_results)} sample-ranked configs on full corpus"
    )

    results: list[OptimizationResult] = []
    best_score = None
    full_started_at = time.time()
    for done, config in enumerate(validated_configs, start=1):
        stats = evaluator.evaluate(config)
        if stats["total"] == 0:
            continue
        score = _score_stats(stats, profile)
        results.append((score, stats, config))
        if best_score is None or score > best_score:
            best_score = score
        if done % max(1, min(progress_every, 10)) == 0 or done == len(validated_configs):
            _print_sweep_progress(
                f"{label} full", done, len(validated_configs), full_started_at, best_score, profile
            )

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def phase1_sigmoid_sweep(
    evaluator: OptimizerRunner,
    profile: ScoreProfile = STANDARD_SCORE,
) -> list[OptimizationResult]:
    """Sweep sigmoid parameters with default thresholds."""
    floors = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ranges = [0.2, 0.3, 0.4, 0.5, 0.6]
    steepnesses = [0.003, 0.005, 0.008]
    midpoints = [1000.0, 1250.0, 1500.0, 1750.0]

    base = ClassificationConfig()
    configs = []
    for floor, range_, steep, midpoint in product(floors, ranges, steepnesses, midpoints):
        if floor + range_ > 1.05:
            continue
        configs.append(replace(
            base,
            elo_scale_floor=floor,
            elo_scale_range=range_,
            elo_scale_steepness=steep,
            elo_scale_midpoint=midpoint,
        ))
    return _evaluate_config_candidates(
        configs,
        evaluator,
        profile,
        label=f"phase 1 ({profile.key})",
        progress_every=10,
    )


def phase2_threshold_sweep(
    best_sigmoid: ClassificationConfig,
    evaluator: OptimizerRunner,
    profile: ScoreProfile = STANDARD_SCORE,
) -> list[OptimizationResult]:
    """Sweep contextual parameters with EP thresholds fixed at chess.com published values.

    EP thresholds are pinned to chess.com's published table and should NOT be tuned.
    Instead, sweep candidate gap and great move transition parameters.
    """
    candidate_gaps = [100, 150, 200, 250]

    configs = [
        replace(
            best_sigmoid,
            ep_excellent=0.02,
            ep_good=0.05,
            ep_inaccuracy=0.10,
            ep_mistake=0.20,
            great_min_candidate_gap_cp=cg,
        )
        for cg in candidate_gaps
    ]
    return _evaluate_config_candidates(
        configs,
        evaluator,
        profile,
        label=f"phase 2 ({profile.key})",
        progress_every=50,
    )


def phase2b_special_sweep(
    best_config: ClassificationConfig,
    evaluator: OptimizerRunner,
    profile: ScoreProfile = STANDARD_SCORE,
) -> list[OptimizationResult]:
    """Coordinate-descent sweep for special move parameters."""
    miss_opp_thresholds = [0.04, 0.06, 0.08, 0.10, 0.15]
    miss_min_eps = [0.04, 0.06, 0.08, 0.10, 0.15]
    miss_best_win_pcts = [0.60, 0.65, 0.70, 0.75, 0.80]
    best_gaps = [15, 20, 30, 50, 75]
    best_non_top_min_gaps = [0, 5, 10, 15]
    best_non_top_max_gaps = [20, 30, 40, 50]
    excellent_low_elos = [0.004, 0.006, 0.008, 0.010, 0.012]
    excellent_elo_thresholds = [800, 1000, 1200]

    rounds = [
        ("miss_opponent_ep_loss_threshold", miss_opp_thresholds),
        ("miss_min_ep_loss", miss_min_eps),
        ("miss_best_win_pct_threshold", miss_best_win_pcts),
        ("best_promotion_min_gap_cp", best_gaps),
        ("best_non_top_excellent_min_gap_cp", best_non_top_min_gaps),
        ("best_non_top_excellent_max_gap_cp", best_non_top_max_gaps),
        ("ep_excellent_low_elo", excellent_low_elos),
        ("excellent_elo_threshold", excellent_elo_thresholds),
    ]

    results: list[OptimizationResult] = []
    current = best_config
    for round_idx in range(2):
        print(f"    phase 2b ({profile.key}) round {round_idx + 1}/2")
        for field_name, values in rounds:
            configs = [replace(current, **{field_name: value}) for value in values]
            sweep_results = _evaluate_config_candidates(
                configs,
                evaluator,
                profile,
                label=f"phase 2b {field_name} ({profile.key})",
                progress_every=max(1, len(configs)),
                full_top_k=min(len(configs), max(3, evaluator.coarse_top_k // 4)),
            )
            if not sweep_results:
                continue
            results.extend(sweep_results)
            current = sweep_results[0][2]

    results = sorted(_unique_optimization_results(results), key=lambda x: x[0], reverse=True)
    return results


def phase3_fine_tune(
    best_config: ClassificationConfig,
    evaluator: OptimizerRunner,
    profile: ScoreProfile = STANDARD_SCORE,
    seed_configs: list[ClassificationConfig] | None = None,
    beam_width: int = 4,
    rounds: int = 3,
) -> list[OptimizationResult]:
    """
    Fine-tune around several strong configs using beam/local search.

    This replaces the old 10-D Cartesian product, which could easily explode
    into tens of thousands of runs with no intermediate visibility.
    """
    if seed_configs is None:
        seed_configs = [best_config]

    seed_configs = _unique_configs([best_config, *seed_configs])
    # EP thresholds are pinned to chess.com published values and excluded from tuning.
    # Only contextual parameters (great triggers, miss detection, sigmoid) are perturbed.
    step_schedule = [
        {
            "elo_scale_floor": (-0.05, 0.05),
            "elo_scale_range": (-0.05, 0.05),
            "elo_scale_midpoint": (-200, 200),
            "great_min_candidate_gap_cp": (-25, 25),
            "great_capitalization_min_ep_loss": (-0.02, 0.02),
            "great_capitalization_max_ep_loss": (-0.03, 0.03),
            "great_capitalization_gap_scale": (-0.15, 0.15),
            "great_min_win_pct_before": (-0.05, 0.05),
            "great_max_win_pct_before": (-0.05, 0.05),
            "best_promotion_min_gap_cp": (-10, 10),
            "best_non_top_excellent_min_gap_cp": (-5, 5),
            "best_non_top_excellent_max_gap_cp": (-10, 10),
            "miss_opponent_ep_loss_threshold": (-0.02, 0.02),
            "miss_min_ep_loss": (-0.03, 0.03),
            "miss_best_win_pct_threshold": (-0.05, 0.05),
            "brilliant_low_elo_max_win_pct_before": (-0.05, 0.05),
            "ep_excellent_low_elo": (-0.002, 0.002),
            "excellent_elo_threshold": (-200, 200),
        },
        {
            "elo_scale_floor": (-0.02, 0.02),
            "elo_scale_range": (-0.02, 0.02),
            "elo_scale_midpoint": (-100, 100),
            "great_min_candidate_gap_cp": (-10, 10),
            "great_capitalization_min_ep_loss": (-0.01, 0.01),
            "great_capitalization_max_ep_loss": (-0.015, 0.015),
            "great_capitalization_gap_scale": (-0.08, 0.08),
            "great_min_win_pct_before": (-0.03, 0.03),
            "great_max_win_pct_before": (-0.03, 0.03),
            "best_promotion_min_gap_cp": (-5, 5),
            "best_non_top_excellent_min_gap_cp": (-3, 3),
            "best_non_top_excellent_max_gap_cp": (-5, 5),
            "miss_opponent_ep_loss_threshold": (-0.01, 0.01),
            "miss_min_ep_loss": (-0.015, 0.015),
            "miss_best_win_pct_threshold": (-0.03, 0.03),
            "brilliant_low_elo_max_win_pct_before": (-0.03, 0.03),
            "ep_excellent_low_elo": (-0.001, 0.001),
            "excellent_elo_threshold": (-100, 100),
        },
        {
            "elo_scale_floor": (-0.01, 0.01),
            "elo_scale_range": (-0.01, 0.01),
            "elo_scale_midpoint": (-50, 50),
            "great_min_candidate_gap_cp": (-5, 5),
            "great_capitalization_min_ep_loss": (-0.005, 0.005),
            "great_capitalization_max_ep_loss": (-0.01, 0.01),
            "great_capitalization_gap_scale": (-0.04, 0.04),
            "great_min_win_pct_before": (-0.015, 0.015),
            "great_max_win_pct_before": (-0.015, 0.015),
            "best_promotion_min_gap_cp": (-2, 2),
            "best_non_top_excellent_min_gap_cp": (-2, 2),
            "best_non_top_excellent_max_gap_cp": (-3, 3),
            "miss_opponent_ep_loss_threshold": (-0.005, 0.005),
            "miss_min_ep_loss": (-0.01, 0.01),
            "miss_best_win_pct_threshold": (-0.015, 0.015),
            "brilliant_low_elo_max_win_pct_before": (-0.015, 0.015),
            "ep_excellent_low_elo": (-0.0005, 0.0005),
            "excellent_elo_threshold": (-50, 50),
        },
    ][:rounds]

    results: list[OptimizationResult] = []
    baseline_start = time.time()
    baseline_best = None
    for idx, config in enumerate(seed_configs, start=1):
        stats = evaluator.evaluate(config)
        if stats["total"] == 0:
            continue
        score = _score_stats(stats, profile)
        results.append((score, stats, config))
        if baseline_best is None or score > baseline_best:
            baseline_best = score
        _print_sweep_progress(
            f"phase 3 seeds ({profile.key})",
            idx,
            len(seed_configs),
            baseline_start,
            baseline_best,
            profile,
        )

    if not results:
        return []

    beam = _top_unique_configs(results, beam_width)
    best_score = baseline_best

    for round_idx, step_map in enumerate(step_schedule, start=1):
        frontier = []
        seen_frontier = {_config_key(config) for _, _, config in results}

        for config in beam:
            for field_name, deltas in step_map.items():
                current_value = getattr(config, field_name)
                for delta in deltas:
                    candidate = replace(
                        config,
                        **{
                            field_name: _coerce_param_value(
                                current_value,
                                current_value + delta,
                            )
                        },
                    )
                    candidate_key = _config_key(candidate)
                    if candidate_key in seen_frontier or not _is_valid_phase3_config(candidate):
                        continue
                    seen_frontier.add(candidate_key)
                    frontier.append(candidate)

        if not frontier:
            print(f"    phase 3 round {round_idx}: no new neighbors to evaluate")
            break

        started_at = time.time()
        round_best = best_score
        total = len(frontier)
        for done, candidate in enumerate(frontier, start=1):
            stats = evaluator.evaluate(candidate)
            if stats["total"] == 0:
                continue
            score = _score_stats(stats, profile)
            results.append((score, stats, candidate))
            if score > round_best:
                round_best = score
            if done % 25 == 0 or done == total:
                _print_sweep_progress(
                    f"phase 3 round {round_idx} ({profile.key})",
                    done,
                    total,
                    started_at,
                    round_best,
                    profile,
                )

        results.sort(key=lambda x: x[0], reverse=True)
        beam = _top_unique_configs(results, beam_width)
        current_best = results[0]
        improved = "improved" if current_best[0] > best_score else "held"
        best_score = current_best[0]
        print(
            f"    phase 3 round {round_idx} ({profile.key}) {improved}: "
            f"{_describe_stats(current_best[1], profile)} "
            f"(beam={len(beam)}, candidates={len(frontier)})"
        )

    results.sort(key=lambda x: x[0], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Engine parameter sweep — find which depth/multipv best matches chess.com
# ---------------------------------------------------------------------------

def sweep_engine_params(
    game_files: list[Path],
    player_elo: int,
    depths: list[int] | None = None,
    multipvs: list[int] | None = None,
    engine_tags: list[str] | None = None,
    node_counts: list[int] | None = None,
    config: ClassificationConfig | None = None,
) -> list[dict]:
    """
    Sweep depth/nodes/multipv/engine-version combinations using pre-generated caches.
    Each combination requires a cache file (generate with --generate-cache).

    Returns sorted list of {engine, depth/nodes, multipv, accuracy, far_misses, ...} dicts.
    """
    if depths is None:
        depths = [14, 16, 18, 20, 22, 24, 26]
    if multipvs is None:
        multipvs = [2, 3, 4, 5]
    if engine_tags is None:
        # Auto-discover available engine tags from cache filenames
        engine_tags = set()
        for csv_path in game_files:
            for cache in csv_path.parent.glob(f"{csv_path.stem}_sf*_*_pv*.json"):
                # Extract engine tag: game_01_sf16.1_d20_pv3.json → sf16.1
                parts = cache.stem.split("_")
                for j, p in enumerate(parts):
                    if p.startswith("sf") and j + 1 < len(parts) and (parts[j + 1].startswith("d") or parts[j + 1].startswith("n")):
                        engine_tags.add(p)
                        break
        engine_tags = sorted(engine_tags) if engine_tags else ["sf18"]
    if config is None:
        config = ClassificationConfig()

    # Auto-discover node-based caches if no explicit node_counts
    if node_counts is None:
        node_counts_set = set()
        for csv_path in game_files:
            for cache in csv_path.parent.glob(f"{csv_path.stem}_sf*_n*_pv*.json"):
                parts = cache.stem.split("_")
                for p in parts:
                    if p.startswith("n") and p[1:].isdigit():
                        node_counts_set.add(int(p[1:]))
        node_counts = sorted(node_counts_set) if node_counts_set else []

    results = []

    # Sweep depth-based configs
    for etag, depth, mpv in product(engine_tags, depths, multipvs):
        available_games = []
        for csv_path in game_files:
            cache = _cache_path(csv_path, depth, mpv, etag)
            if cache.exists():
                available_games.append(csv_path)

        if not available_games:
            continue

        stats = run_single(config, available_games, player_elo, True, depth, mpv, etag)
        if stats["total"] == 0:
            continue

        results.append({
            "engine": etag,
            "search": f"d{depth}",
            "depth": depth,
            "nodes": None,
            "multipv": mpv,
            "games": len(available_games),
            "total_moves": stats["total"],
            "accuracy": stats["accuracy"],
            "matches": stats["matches"],
            "close_misses": stats.get("close_misses", 0),
            "far_misses": stats.get("far_misses", 0),
            "by_class": stats.get("by_class", {}),
        })

    # Sweep node-based configs
    for etag, nc, mpv in product(engine_tags, node_counts, multipvs):
        available_games = []
        for csv_path in game_files:
            cache = _cache_path(csv_path, 0, mpv, etag, nodes=nc)
            if cache.exists():
                available_games.append(csv_path)

        if not available_games:
            continue

        stats = run_single(config, available_games, player_elo, True, 0, mpv, etag, nodes=nc)
        if stats["total"] == 0:
            continue

        results.append({
            "engine": etag,
            "search": f"n{nc // 1000}k" if nc >= 1000 else f"n{nc}",
            "depth": None,
            "nodes": nc,
            "multipv": mpv,
            "games": len(available_games),
            "total_moves": stats["total"],
            "accuracy": stats["accuracy"],
            "matches": stats["matches"],
            "close_misses": stats.get("close_misses", 0),
            "far_misses": stats.get("far_misses", 0),
            "by_class": stats.get("by_class", {}),
        })

    results.sort(key=lambda r: (r["accuracy"], -r["far_misses"]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Printing / reporting
# ---------------------------------------------------------------------------

def print_results(
    title: str,
    results: list[OptimizationResult],
    top_n: int = 10,
    evaluator: OptimizerRunner | None = None,
    profile: ScoreProfile = STANDARD_SCORE,
):
    print(f"\n{'='*80}")
    print(f" {title} -- Top {min(top_n, len(results))} of {len(results)} runs")
    print(f"{'='*80}")
    for i, (score, stats, config) in enumerate(results[:top_n]):
        if evaluator is not None and (
            "by_class" not in stats
            or any("our_labels" not in info for info in stats.get("by_class", {}).values())
        ):
            stats = evaluator.evaluate(config, include_breakdown=True)
        print(
            f"\n  #{i+1}  Score: {_format_score(score, profile)}  "
            f"(close_misses={stats.get('close_misses', '?')})"
        )
        print(
            f"       sigmoid: floor={config.elo_scale_floor:.2f}, "
            f"range={config.elo_scale_range:.2f}, "
            f"steepness={config.elo_scale_steepness:.4f}, "
            f"midpoint={config.elo_scale_midpoint:.0f}"
        )
        print(
            f"       thresholds: excellent<={config.ep_excellent:.3f}, "
            f"good<={config.ep_good:.3f}, "
            f"inaccuracy<{config.ep_inaccuracy:.3f}, "
            f"mistake<{config.ep_mistake:.3f}"
        )
        print(
            f"       specials: best_gap>={config.best_promotion_min_gap_cp}, "
            f"best_non_top_excellent=[{config.best_non_top_excellent_min_gap_cp}, "
            f"{config.best_non_top_excellent_max_gap_cp}), "
            f"great_gap>={config.great_min_candidate_gap_cp}, "
            f"great_cap=[{config.great_capitalization_min_ep_loss:.3f}, "
            f"{config.great_capitalization_max_ep_loss:.3f}), "
            f"great_cap_gap_scale={config.great_capitalization_gap_scale:.2f}"
        )
        print(
            f"       filters: great_wp_before=[{config.great_min_win_pct_before:.2f}, "
            f"{config.great_max_win_pct_before:.2f}], "
            f"brilliant_low_elo_max_wp={config.brilliant_low_elo_max_win_pct_before:.2f}"
        )
        by_class = stats.get("by_class", {})
        for cls in sorted(by_class, key=lambda x: by_class[x]["total"], reverse=True):
            info = by_class[cls]
            rate = info["matched"] / info["total"] if info["total"] else 0
            labels = info.get("our_labels")
            if labels:
                label_text = ", ".join(f"{k}={v}" for k, v in sorted(labels.items()))
                print(f"         {cls:<12} {info['matched']}/{info['total']} ({rate:.0%})  -> {label_text}")
            else:
                print(f"         {cls:<12} {info['matched']}/{info['total']} ({rate:.0%})")


def print_engine_sweep_results(results: list[dict]):
    """Print engine parameter sweep results in a readable table."""
    print(f"\n{'='*80}")
    print(f" ENGINE PARAMETER SWEEP -- {len(results)} configurations tested")
    print(f"{'='*80}")
    print(f"\n  {'Engine':<8} {'Search':>10} {'MPV':>4} {'Games':>5} {'Moves':>6} {'Accuracy':>8} {'Matches':>7} {'Close':>5} {'Far':>4}")
    print(f"  {'-'*63}")

    for r in results:
        print(
            f"  {r['engine']:<8} {r['search']:>10} {r['multipv']:>4} {r['games']:>5} "
            f"{r['total_moves']:>6} {r['accuracy']:>7.1%} "
            f"{r['matches']:>7} {r['close_misses']:>5} {r['far_misses']:>4}"
        )

    if results:
        best = results[0]
        print(f"\n  Best match: {best['engine']} {best['search']}, multipv={best['multipv']} "
              f"({best['accuracy']:.1%} accuracy, {best['far_misses']} far misses)")

        # Per-class breakdown for best config
        print(f"\n  Per-class breakdown ({best['engine']}, {best['search']}, mpv={best['multipv']}):")
        by_class = best.get("by_class", {})
        for cls in sorted(by_class, key=lambda x: CLASS_SEVERITY.get(x, 99)):
            info = by_class[cls]
            rate = info["matched"] / info["total"] if info["total"] else 0
            labels = ", ".join(f"{k}={v}" for k, v in sorted(info["our_labels"].items()))
            print(f"    {cls:<12} {info['matched']}/{info['total']} ({rate:.0%})  -> {labels}")


def print_mismatch_details(comparisons: list[MoveComparison]):
    """Print detailed info for every mismatched move."""
    mismatches = [c for c in comparisons if not c.match and c.chesscom_class != "book"]
    if not mismatches:
        print("\n  No mismatches!")
        return

    print(f"\n  MISMATCHES ({len(mismatches)} moves):")
    print(f"  {'Ply':>3} {'Color':<5} {'Move':<8} {'chess.com':<12} {'Ours':<12} "
          f"{'EP Loss':>8} {'Distance':>8}")
    print(f"  {'-'*65}")

    for c in mismatches:
        s1 = CLASS_SEVERITY.get(c.chesscom_class, 10)
        s2 = CLASS_SEVERITY.get(c.our_class, 10)
        dist = abs(s1 - s2)
        dist_label = f"{'FAR' if dist >= 2 else 'close'} ({dist})"
        ep_str = f"{c.ep_loss:.4f}" if c.ep_loss is not None else "N/A"

        special = ""
        if c.is_brilliant:
            special = " [!!]"
        elif c.is_great:
            special = " [!]"
        elif c.is_miss:
            special = " [?]"

        print(
            f"  {c.ply:>3} {c.color:<5} {c.move_san:<8} {c.chesscom_class:<12} "
            f"{c.our_class + special:<12} {ep_str:>8} {dist_label:>8}"
        )


def print_config_as_python(config: ClassificationConfig, label: str = "Best config"):
    """Print a config as copy-pasteable Python for config/classification.py."""
    print(f"\n  {label} (paste into config/classification.py):")
    print(f"    elo_scale_floor     = {config.elo_scale_floor}")
    print(f"    elo_scale_range     = {config.elo_scale_range}")
    print(f"    elo_scale_steepness = {config.elo_scale_steepness}")
    print(f"    elo_scale_midpoint  = {config.elo_scale_midpoint}")
    print(f"    ep_excellent        = {config.ep_excellent}")
    print(f"    ep_good             = {config.ep_good}")
    print(f"    ep_inaccuracy       = {config.ep_inaccuracy}")
    print(f"    ep_mistake          = {config.ep_mistake}")
    print(f"    best_non_top_excellent_min_gap_cp = {config.best_non_top_excellent_min_gap_cp}")
    print(f"    best_non_top_excellent_max_gap_cp = {config.best_non_top_excellent_max_gap_cp}")
    print(f"    great_min_candidate_gap_cp = {config.great_min_candidate_gap_cp}")
    print(f"    great_capitalization_min_ep_loss = {config.great_capitalization_min_ep_loss}")
    print(f"    great_capitalization_max_ep_loss = {config.great_capitalization_max_ep_loss}")
    print(f"    great_capitalization_gap_scale = {config.great_capitalization_gap_scale}")
    print(f"    great_min_win_pct_before = {config.great_min_win_pct_before}")
    print(f"    great_max_win_pct_before = {config.great_max_win_pct_before}")
    print(f"    best_promotion_min_gap_cp = {config.best_promotion_min_gap_cp}")
    print(f"    excellent_elo_threshold   = {config.excellent_elo_threshold}")
    print(f"    ep_excellent_low_elo      = {config.ep_excellent_low_elo}")
    print(f"    miss_opponent_ep_loss_threshold = {config.miss_opponent_ep_loss_threshold}")
    print(f"    miss_min_ep_loss = {config.miss_min_ep_loss}")
    print(f"    brilliant_low_elo_max_win_pct_before = {config.brilliant_low_elo_max_win_pct_before}")
    print(f"    brilliant_low_elo_max_win_pct_threshold = {config.brilliant_low_elo_max_win_pct_threshold}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_compare(args):
    """Print detailed move-by-move comparison for all games."""
    game_files = _find_game_csvs(args.games_dir, min_non_book_moves=args.min_non_book_moves)
    config = ClassificationConfig()
    engine_path, engine_tag = _resolve_engine(args.engine)
    nodes = args.nodes

    print(f"\nCompare mode: {'Stockfish (cached)' if args.stockfish else 'Mock (chess.com evals)'}")
    if args.stockfish:
        print(f"Engine: {engine_tag} ({engine_path})")
        if nodes:
            print(f"Search: nodes={nodes:,}")
        else:
            print(f"Search: depth={args.depth}")
    print(f"Fallback Elo: {args.elo} (per-game Elos from {args.manifest_path or 'game_elos.json'})")
    if args.min_non_book_moves:
        print(f"Skipping games with < {args.min_non_book_moves} non-book moves")
    print(f"Games: {[f.name for f in game_files]}")

    all_comparisons = []
    for csv_path in game_files:
        w_elo, b_elo = _game_elos_by_side(
            csv_path,
            fallback_elo=args.elo,
            manifest_path=args.manifest_path,
        )
        print(f"\n{'='*60}")
        print(f" {csv_path.name}  (White: {w_elo}, Black: {b_elo})")
        print(f"{'='*60}")

        if args.stockfish:
            cache = _cache_path(
                csv_path,
                args.depth,
                args.multipv,
                engine_tag,
                nodes=nodes,
                cache_dir=args.cache_dir,
            )
            if not cache.exists():
                print(f"  No cache at {cache.name} -- run --generate-cache first")
                continue
            comparisons = run_comparison_from_cache(
                csv_path, cache, config=config,
                white_elo=w_elo, black_elo=b_elo,
            )
        else:
            comparisons = run_comparison(csv_path, config=config, player_elo=(w_elo + b_elo) // 2)

        print_comparison_table(comparisons)
        stats = compute_accuracy_stats(comparisons)
        print_accuracy_report(stats)
        print_mismatch_details(comparisons)
        all_comparisons.extend(comparisons)

    if len(game_files) > 1:
        print(f"\n{'='*60}")
        print(f" AGGREGATE ({len(game_files)} games, {len(all_comparisons)} moves)")
        print(f"{'='*60}")
        agg_stats = compute_accuracy_stats(all_comparisons)
        print_accuracy_report(agg_stats)


def cmd_generate_cache(args):
    """Generate Stockfish caches for all game files."""
    nodes = args.nodes
    game_files = _find_game_csvs(args.games_dir, min_non_book_moves=args.min_non_book_moves)

    if args.fill_missing:
        specs = _collect_existing_cache_specs(game_files)
        if not specs:
            print("\nNo existing caches found to infer settings from.")
            print("Run --generate-cache with an explicit engine/search setting first.")
            return

        print(f"\nGenerating missing caches from discovered settings:")
        print(f"  Settings: {[spec.summary for spec in specs]}")
        print(f"  Games: {[f.name for f in game_files]}")
        if args.engine:
            print(f"  Engine override: {args.engine}")
        print(f"  Hash: {args.hash_mb}MB, threads={args.threads}")

        start = time.time()
        generated = asyncio.run(generate_missing_caches_for_discovered_settings(
            game_files,
            specs,
            hash_mb=args.hash_mb,
            threads=args.threads,
            force=args.force,
            engine_override=args.engine,
        ))
        elapsed = time.time() - start
        print(f"\n  Generated {len(generated)} cache files in {elapsed:.1f}s")
        return

    engine_path, engine_tag = _resolve_engine(args.engine)

    if not Path(engine_path).exists() and not shutil.which(engine_path):
        print(f"ERROR: Engine binary not found: {engine_path}")
        print("Install Stockfish or use --engine to specify a path/alias.")
        print(f"  Available aliases: {', '.join(ENGINE_PATHS.keys())}")
        sys.exit(1)

    print(f"\nGenerating Stockfish caches:")
    print(f"  Engine: {engine_tag} ({engine_path})")
    if nodes:
        print(f"  Search: nodes={nodes:,}, multipv={args.multipv}, "
              f"hash={args.hash_mb}MB, threads={args.threads}")
    else:
        print(f"  Search: depth={args.depth}, multipv={args.multipv}, "
              f"hash={args.hash_mb}MB, threads={args.threads}")
    print(f"  Games: {[f.name for f in game_files]}")

    start = time.time()
    asyncio.run(generate_all_caches(
        game_files,
        depth=args.depth,
        nodes=nodes,
        multipv=args.multipv,
        hash_mb=args.hash_mb,
        threads=args.threads,
        force=args.force,
        engine_path=engine_path,
        engine_tag=engine_tag,
    ))
    elapsed = time.time() - start
    print(f"\n  Done in {elapsed:.1f}s")


def cmd_sweep_engine(args):
    """Sweep depth/nodes/multipv/engine-version to find which best match chess.com."""
    game_files = _find_game_csvs(args.games_dir, min_non_book_moves=args.min_non_book_moves)
    config = ClassificationConfig()

    depths = [int(d) for d in args.depths.split(",")] if args.depths else None
    multipvs = [int(m) for m in args.multipvs.split(",")] if args.multipvs else None
    engine_tags = [t.strip() for t in args.engine_tags.split(",")] if args.engine_tags else None
    node_counts = [int(n) for n in args.node_counts.split(",")] if args.node_counts else None

    print(f"\nEngine parameter sweep:")
    print(f"  Depths: {depths or [14, 16, 18, 20, 22, 24, 26]}")
    print(f"  Node counts: {node_counts or '(auto-discover from caches)'}")
    print(f"  MultiPVs: {multipvs or [2, 3, 4, 5]}")
    print(f"  Engine versions: {engine_tags or '(auto-discover from caches)'}")
    print(f"  Elo: {args.elo}")
    if args.min_non_book_moves:
        print(f"  Skipping games with < {args.min_non_book_moves} non-book moves")
    print(f"  Games: {[f.name for f in game_files]}")

    # Check available caches (both depth-based and node-based)
    available_any = False
    for csv_path in game_files:
        caches = list(csv_path.parent.glob(f"{csv_path.stem}_sf*_*_pv*.json"))
        if caches:
            available_any = True
            print(f"  {csv_path.name}: {', '.join(c.name for c in sorted(caches))}")
        else:
            print(f"  {csv_path.name}: no caches")

    if not available_any:
        print("\n  No Stockfish caches found. Run --generate-cache first. Example:")
        print("    python tests/optimize_classification.py --generate-cache --engine sf16.1 --depth 20")
        print("    python tests/optimize_classification.py --generate-cache --engine sf16.1 --nodes 2000000")
        print("    python tests/optimize_classification.py --generate-cache --depth 20  # uses system SF")
        return

    results = sweep_engine_params(
        game_files, args.elo, depths=depths, multipvs=multipvs,
        engine_tags=engine_tags, node_counts=node_counts, config=config,
    )
    print_engine_sweep_results(results)


def cmd_optimize(args):
    """Run 3-phase parameter optimization (sigmoid + thresholds + fine-tune)."""
    game_files = _find_game_csvs(args.games_dir, min_non_book_moves=args.min_non_book_moves)
    engine_path, engine_tag = _resolve_engine(args.engine)
    nodes = args.nodes
    evaluator = OptimizerRunner(
        game_files,
        args.elo,
        args.stockfish,
        depth=args.depth,
        multipv=args.multipv,
        engine_tag=engine_tag,
        nodes=nodes,
        manifest_path=args.manifest_path,
        cache_dir=args.cache_dir,
        coarse_sample_games=args.coarse_sample_games,
        coarse_top_k=args.coarse_top_k,
    )
    if args.score_mode == "both":
        score_profiles = [STANDARD_SCORE, WEIGHTED_SCORE]
    else:
        score_profiles = [SCORE_PROFILES[args.score_mode]]
    mode = "Stockfish (cached)" if args.stockfish else "Mock (chess.com evals)"
    print(f"\nMode: {mode}")
    if args.stockfish:
        print(f"Engine: {engine_tag} ({engine_path})")
    print(f"Fallback Elo: {args.elo} (per-game Elos from {args.manifest_path or 'game_elos.json'})")
    if args.min_non_book_moves:
        print(f"Skipping games with < {args.min_non_book_moves} non-book moves")
    if args.stockfish:
        if nodes:
            print(f"Engine params: nodes={nodes:,}, multipv={args.multipv}")
        else:
            print(f"Engine params: depth={args.depth}, multipv={args.multipv}")
    timing = evaluator.timing_summary()
    print(
        f"Corpus precompute: {timing['build_elapsed_s']:.1f}s "
        f"for {timing['full_games']} games / {timing['full_moves']} moves"
    )
    if timing["sample_games"] < timing["full_games"]:
        print(
            f"Coarse sample: {timing['sample_games']} games / {timing['sample_moves']} moves "
            f"(top {args.coarse_top_k} full-corpus validations per sweep)"
        )
    else:
        print("Coarse sample: disabled (sample covers full corpus)")
    print(f"Games:")
    for gf in game_files:
        w, b = _game_elos_by_side(
            gf,
            fallback_elo=args.elo,
            manifest_path=args.manifest_path,
        )
        print(f"  {gf.name} (W: {w}, B: {b})")

    if args.stockfish:
        missing = []
        for gf in game_files:
            cache = _cache_path(
                gf,
                args.depth,
                args.multipv,
                engine_tag,
                nodes=nodes,
                cache_dir=args.cache_dir,
            )
            if cache.exists():
                print(f"  + {cache.name}")
            else:
                print(f"  - {cache.name} MISSING")
                missing.append(gf.name)
        if missing:
            print(f"\n  WARNING: {len(missing)} games have no cache.")
            if nodes:
                print(f"  Run: python tests/optimize_classification.py --generate-cache "
                      f"--engine {engine_tag} --nodes {nodes} --multipv {args.multipv}")
            else:
                print(f"  Run: python tests/optimize_classification.py --generate-cache "
                      f"--engine {engine_tag} --depth {args.depth} --multipv {args.multipv}")

    start = time.time()
    profile_summaries = []
    baseline_started_at = time.time()
    baseline_stats = evaluator.evaluate(ClassificationConfig())
    baseline_elapsed = time.time() - baseline_started_at
    print(
        f"\nTiming checkpoint: baseline full-corpus eval "
        f"{baseline_elapsed:.2f}s ({baseline_stats['total']} comparable moves)"
    )
    if timing["sample_games"] < timing["full_games"]:
        sample_started_at = time.time()
        evaluator.evaluate(ClassificationConfig(), sample=True)
        sample_elapsed = time.time() - sample_started_at
        print(f"Timing checkpoint: coarse-sample eval {sample_elapsed:.2f}s")

    for profile in score_profiles:
        print(f"\n{'='*80}")
        print(f" OPTIMIZING FOR {profile.label.upper()} SCORE")
        print(f"{'='*80}")

        # --- Phase 1: Sigmoid sweep ---
        print(f"\n[Phase 1/{profile.key}] Sweeping sigmoid parameters...")
        p1 = phase1_sigmoid_sweep(evaluator, profile)
        if not p1:
            print("No results -- check that game files and caches exist.")
            return
        print_results(
            f"Phase 1: Sigmoid Sweep ({profile.label})",
            p1,
            top_n=5,
            evaluator=evaluator,
            profile=profile,
        )
        best_sigmoid = p1[0][2]

        # --- Phase 2: Threshold sweep ---
        print(
            f"\n[Phase 2/{profile.key}] Sweeping EP thresholds "
            f"(best sigmoid midpoint={best_sigmoid.elo_scale_midpoint:.0f})..."
        )
        p2 = phase2_threshold_sweep(best_sigmoid, evaluator, profile)
        if not p2:
            print("No threshold results.")
            return
        print_results(
            f"Phase 2: Threshold Sweep ({profile.label})",
            p2,
            top_n=5,
            evaluator=evaluator,
            profile=profile,
        )
        best_combined = p2[0][2]

        # --- Phase 2b: Special parameter sweep ---
        print(f"\n[Phase 2b/{profile.key}] Sweeping special classification parameters...")
        p2b = phase2b_special_sweep(best_combined, evaluator, profile)
        if p2b:
            print_results(
                f"Phase 2b: Special Sweep ({profile.label})",
                p2b,
                top_n=5,
                evaluator=evaluator,
                profile=profile,
            )
            best_combined = p2b[0][2]

        # --- Phase 3: Fine-tune ---
        phase3_seeds = [config for _, _, config in p2[:3]]
        if p2b:
            phase3_seeds.extend(config for _, _, config in p2b[:3])
        print(
            f"\n[Phase 3/{profile.key}] Local fine-tuning from "
            f"{len(_unique_configs(phase3_seeds))} strong seeds..."
        )
        p3 = phase3_fine_tune(
            best_combined,
            evaluator,
            profile,
            seed_configs=phase3_seeds,
        )
        if not p3:
            print("No fine-tune results.")
            return
        print_results(
            f"Phase 3: Fine-Tuning ({profile.label})",
            p3,
            top_n=5,
            evaluator=evaluator,
            profile=profile,
        )
        best_final = p3[0][2]

        final_stats = evaluator.evaluate(best_final, include_breakdown=True)
        default_stats = evaluator.evaluate(ClassificationConfig(), include_breakdown=True)
        delta = final_stats["accuracy"] - default_stats["accuracy"] if default_stats["total"] > 0 else 0.0

        print(f"\n  {profile.label} best config:")
        print_config_as_python(best_final, label=f"{profile.label} best config")
        print(f"\n  Final metrics: {_describe_stats(final_stats, profile)}")
        print(f"  Exact matches: {final_stats['matches']}/{final_stats['total']}")
        print(f"  Close misses: {final_stats.get('close_misses', 0)}")
        print(f"  Far misses: {final_stats.get('far_misses', 0)}")
        if default_stats["total"] > 0:
            print(
                f"  vs default exact accuracy: "
                f"{default_stats['accuracy']:.1%} -> {final_stats['accuracy']:.1%} ({delta:+.1%})"
            )

        profile_summaries.append({
            "profile": profile,
            "config": best_final,
            "stats": final_stats,
            "delta": delta,
        })

    elapsed = time.time() - start
    print(f"\n{'='*80}")
    print(f" FINAL RESULT -- {evaluator.unique_evaluations} unique configs in {elapsed:.1f}s")
    print(f"{'='*80}")
    for summary in profile_summaries:
        profile = summary["profile"]
        stats = summary["stats"]
        print(
            f"  {profile.label:<8} {_describe_stats(stats, profile)}  "
            f"delta_exact={summary['delta']:+.1%}"
        )
    timing = evaluator.timing_summary()
    if timing["full_evals"]:
        print(
            f"  Avg full eval: {timing['full_eval_s'] / timing['full_evals']:.2f}s "
            f"across {timing['full_evals']} runs"
        )
    if timing["sample_evals"]:
        print(
            f"  Avg sample eval: {timing['sample_eval_s'] / timing['sample_evals']:.2f}s "
            f"across {timing['sample_evals']} runs"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optimize EP classification parameters against chess.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick comparison with current config
  %(prog)s --compare

  # Generate caches with different Stockfish versions (depth-based)
  %(prog)s --generate-cache --engine sf16.1 --depth 20

  # Generate caches with node-based search (chess.com style)
  %(prog)s --generate-cache --engine sf16.1 --nodes 2000000
  %(prog)s --generate-cache --engine sf16.1 --nodes 4000000

  # Fill in missing caches for settings already present on other games
  %(prog)s --generate-cache --fill-missing

  # Sweep engine versions + depth/nodes to find best match to chess.com
  %(prog)s --sweep-engine

  # Run optimization using real Stockfish 16.1 node-based caches
  %(prog)s --stockfish --engine sf16.1 --nodes 2000000

  # Run optimization using depth-based caches
  %(prog)s --stockfish --engine sf16.1 --depth 20

  # Run 3-phase optimization using mock evals (no Stockfish)
  %(prog)s --elo 820

Engine aliases: sf16.1, sf17, sf18 (or provide a full path)
Chess.com reference: Stockfish 16.1 NNUE, node-based search (~1-4M nodes),
multi-PV 2-3, Expected Points classification with Elo-adjusted sigmoid.
""",
    )

    # Mode flags
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--compare", action="store_true",
        help="Print detailed move-by-move comparison (no optimization)",
    )
    mode_group.add_argument(
        "--generate-cache", action="store_true",
        help="Run Stockfish on all games and save analysis caches",
    )
    mode_group.add_argument(
        "--sweep-engine", action="store_true",
        help="Sweep depth/multipv combos to find best engine match",
    )

    # Engine parameters
    parser.add_argument(
        "--engine", type=str, default=None,
        help="Stockfish binary path or alias: sf16.1, sf17, sf18 (default: system stockfish)",
    )
    parser.add_argument(
        "--stockfish", action="store_true",
        help="Use cached Stockfish results instead of mock evals",
    )
    parser.add_argument(
        "--depth", type=int, default=20,
        help="Stockfish analysis depth (default: 20, ignored if --nodes is set)",
    )
    parser.add_argument(
        "--nodes", type=int, default=None,
        help="Node-based search limit (chess.com uses ~1-4M). Overrides --depth.",
    )
    parser.add_argument(
        "--multipv", type=int, default=3,
        help="Number of principal variations (default: 3)",
    )
    parser.add_argument(
        "--hash-mb", type=int, default=128,
        help="Stockfish hash table size in MB (default: 128)",
    )
    parser.add_argument(
        "--threads", type=int, default=2,
        help="Stockfish threads per engine (default: 2)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force regeneration of existing caches",
    )
    parser.add_argument(
        "--fill-missing", action="store_true",
        help="Infer existing cache settings from other games and generate only missing game/setting caches",
    )

    # Classification parameters
    parser.add_argument(
        "--elo", type=int, default=820,
        help="Player Elo for classification (default: 820)",
    )
    parser.add_argument(
        "--games-dir",
        type=Path,
        default=GAMES_DIR,
        help="Directory containing chess.com review CSVs (default: tests/test_games)",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional manifest/elo JSON for per-game white_elo/black_elo lookup",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory; supports flat <game_id>.json caches",
    )
    parser.add_argument(
        "--min-non-book-moves",
        type=int,
        default=0,
        help="Skip games with fewer than this many comparable non-book moves",
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        choices=["standard", "weighted", "both"],
        default="both",
        help="Optimization objective: standard accuracy, weighted per-class recall, or both (default: both)",
    )
    parser.add_argument(
        "--coarse-sample-games",
        type=int,
        default=128,
        help="Number of games in the stratified coarse-search sample (default: 128)",
    )
    parser.add_argument(
        "--coarse-top-k",
        type=int,
        default=24,
        help="How many sample-ranked configs to validate on the full corpus (default: 24)",
    )

    # Engine sweep options
    parser.add_argument(
        "--depths", type=str, default=None,
        help="Comma-separated depths to sweep (default: 14,16,18,20,22,24,26)",
    )
    parser.add_argument(
        "--multipvs", type=str, default=None,
        help="Comma-separated multipv values to sweep (default: 2,3,4,5)",
    )
    parser.add_argument(
        "--engine-tags", type=str, default=None,
        help="Comma-separated engine version tags to sweep (default: auto-discover from caches)",
    )
    parser.add_argument(
        "--node-counts", type=str, default=None,
        help="Comma-separated node counts to sweep (default: auto-discover from caches)",
    )

    args = parser.parse_args()

    # Dispatch to command
    if args.compare:
        cmd_compare(args)
    elif args.generate_cache:
        cmd_generate_cache(args)
    elif args.sweep_engine:
        cmd_sweep_engine(args)
    else:
        cmd_optimize(args)


if __name__ == "__main__":
    main()
