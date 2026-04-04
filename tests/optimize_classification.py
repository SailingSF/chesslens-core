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
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Optional

# Add project root to path so imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def _find_game_csvs() -> list[Path]:
    """Find all game CSV files in the test_games directory."""
    return sorted(GAMES_DIR.glob("game_*.csv"))


@lru_cache(maxsize=1)
def _load_game_elos() -> dict[str, dict]:
    """
    Load per-game Elo ratings from game_elos.json.

    Returns dict mapping game stem (e.g. "game_01") to
    {"white_elo": int, "black_elo": int}.
    """
    elo_file = GAMES_DIR / "game_elos.json"
    if elo_file.exists():
        with open(elo_file) as f:
            return json.load(f)
    return {}


def _game_elo(csv_path: Path, fallback_elo: int = 820) -> int:
    """Get the average Elo for a game from metadata, or use fallback."""
    w, b = _game_elos_by_side(csv_path, fallback_elo)
    return (w + b) // 2


def _game_elos_by_side(csv_path: Path, fallback_elo: int = 820) -> tuple[int, int]:
    """Get (white_elo, black_elo) for a game from metadata, or use fallback."""
    elos = _load_game_elos()
    stem = csv_path.stem
    if stem in elos:
        return elos[stem]["white_elo"], elos[stem]["black_elo"]
    return fallback_elo, fallback_elo


def _cache_path(
    csv_path: Path, depth: int, multipv: int,
    engine_tag: str = "sf18", nodes: int | None = None,
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

    versioned = csv_path.parent / f"{csv_path.stem}_{engine_tag}_{search_tag}_pv{multipv}.json"
    if versioned.exists():
        return versioned
    # Legacy fallback (depth-based only): game_01_stockfish_d20_pv3.json
    if nodes is None:
        legacy = csv_path.parent / f"{csv_path.stem}_stockfish_d{depth}_pv{multipv}.json"
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

    engine_results = _load_cached_engine_results(cache_path)

    assembler = ContextAssembler(classification_config=config)
    comparisons = []
    board = chess.Board()
    prev_context: AssembledContext | None = None
    engine_idx = 0

    for i, cm in enumerate(moves):
        try:
            played_move = board.parse_san(cm.move_san)
        except ValueError:
            board.push_san(cm.move_san)
            continue

        if cm.is_game_end:
            board.push(played_move)
            continue

        if engine_idx >= len(engine_results):
            board.push(played_move)
            continue

        engine_result = engine_results[engine_idx]
        engine_idx += 1

        move_elo = white_elo if board.turn == chess.WHITE else black_elo
        ctx = assembler.assemble(
            board, engine_result, played_move,
            player_elo=move_elo,
            prev_context=prev_context,
        )

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
        and config.great_capitalization_min_ep_loss >= 0.02
        and config.great_capitalization_min_ep_loss < config.great_capitalization_max_ep_loss
        and 0.2 <= config.great_capitalization_gap_scale <= 1.0
        and 0.0 < config.great_min_win_pct_before < config.great_max_win_pct_before < 1.0
        and config.miss_opponent_ep_loss_threshold >= 0.02
        and config.miss_min_ep_loss >= 0.02
        and config.ep_excellent_low_elo > 0
        and 0.6 <= config.brilliant_low_elo_max_win_pct_before <= 0.98
        and 600 <= config.elo_scale_midpoint <= 2200
        and config.great_transition_min_gap_cp >= 10
        and 0.1 <= config.great_defensive_losing_threshold <= 0.5
        and 0.5 <= config.great_seizing_winning_threshold <= 0.9
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
    ):
        self.game_files = game_files
        self.player_elo = player_elo
        self.use_stockfish = use_stockfish
        self.depth = depth
        self.multipv = multipv
        self.engine_tag = engine_tag
        self.nodes = nodes
        self.per_game_elo = per_game_elo
        self._summary_cache: dict[tuple, dict] = {}
        self._detailed_cache: dict[tuple, dict] = {}

    def evaluate(
        self,
        config: ClassificationConfig,
        include_breakdown: bool = False,
    ) -> dict:
        key = _config_key(config)

        if include_breakdown:
            if key not in self._detailed_cache:
                self._detailed_cache[key] = run_single(
                    config,
                    self.game_files,
                    self.player_elo,
                    self.use_stockfish,
                    self.depth,
                    self.multipv,
                    self.engine_tag,
                    self.per_game_elo,
                    nodes=self.nodes,
                    include_breakdown=True,
                )
            return self._detailed_cache[key]

        if key in self._detailed_cache:
            return self._detailed_cache[key]

        if key not in self._summary_cache:
            self._summary_cache[key] = run_single(
                config,
                self.game_files,
                self.player_elo,
                self.use_stockfish,
                self.depth,
                self.multipv,
                self.engine_tag,
                self.per_game_elo,
                nodes=self.nodes,
                include_breakdown=False,
            )
        return self._summary_cache[key]

    @property
    def unique_evaluations(self) -> int:
        return len(set(self._summary_cache) | set(self._detailed_cache))


def run_single(
    config: ClassificationConfig,
    game_files: list[Path],
    player_elo: int,
    use_stockfish: bool,
    depth: int = 20,
    multipv: int = 3,
    engine_tag: str = "sf18",
    per_game_elo: bool = True,
    nodes: int | None = None,
    include_breakdown: bool = True,
) -> dict:
    """Run comparison across all games with a given config, return aggregate stats.

    When per_game_elo=True (default), uses Elo from game_elos.json for each game,
    falling back to player_elo if the game isn't listed.
    """
    all_comparisons = []
    for csv_path in game_files:
        if per_game_elo:
            w_elo, b_elo = _game_elos_by_side(csv_path, fallback_elo=player_elo)
        else:
            w_elo, b_elo = player_elo, player_elo
        if use_stockfish:
            cache_path = _cache_path(csv_path, depth, multipv, engine_tag, nodes=nodes)
            if not cache_path.exists():
                continue
            comparisons = run_comparison_from_cache(
                csv_path, cache_path, config=config,
                white_elo=w_elo, black_elo=b_elo,
            )
        else:
            # Mock mode uses average Elo (no side distinction in mock data)
            comparisons = run_comparison(csv_path, config=config, player_elo=(w_elo + b_elo) // 2)
        all_comparisons.extend(comparisons)

    return compute_accuracy_stats(all_comparisons, include_breakdown=include_breakdown)


def phase1_sigmoid_sweep(
    evaluator: OptimizerRunner,
    profile: ScoreProfile = STANDARD_SCORE,
) -> list[OptimizationResult]:
    """Sweep sigmoid parameters with default thresholds."""
    floors = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ranges = [0.2, 0.3, 0.4, 0.5, 0.6]
    steepnesses = [0.003, 0.005, 0.008]
    midpoints = [1000.0, 1250.0, 1500.0, 1750.0]

    results = []
    base = ClassificationConfig()
    total = sum(
        1
        for f, r, s, m in product(floors, ranges, steepnesses, midpoints)
        if f + r <= 1.05
    )
    done = 0
    started_at = time.time()
    best_score = None

    for floor, range_, steep, midpoint in product(floors, ranges, steepnesses, midpoints):
        if floor + range_ > 1.05:
            continue
        done += 1
        config = replace(
            base,
            elo_scale_floor=floor,
            elo_scale_range=range_,
            elo_scale_steepness=steep,
            elo_scale_midpoint=midpoint,
        )
        stats = evaluator.evaluate(config)
        if stats["total"] == 0:
            continue
        score = _score_stats(stats, profile)
        results.append((score, stats, config))
        if best_score is None or score > best_score:
            best_score = score

        if done % 10 == 0 or done == total:
            _print_sweep_progress(
                f"phase 1 ({profile.key})", done, total, started_at, best_score, profile
            )

    results.sort(key=lambda x: x[0], reverse=True)
    return results


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
    great_transition_gaps = [30, 50, 75, 100]
    great_defensive_losing = [0.30, 0.35, 0.40]
    great_seizing_winning = [0.65, 0.70, 0.75]

    combos = list(product(
        candidate_gaps, great_transition_gaps,
        great_defensive_losing, great_seizing_winning,
    ))
    total = len(combos)
    done = 0
    started_at = time.time()
    best_score = None

    results = []
    for cg, tg, dl, sw in combos:
        done += 1
        config = replace(
            best_sigmoid,
            # EP thresholds pinned to chess.com published values
            ep_excellent=0.02,
            ep_good=0.05,
            ep_inaccuracy=0.10,
            ep_mistake=0.20,
            great_min_candidate_gap_cp=cg,
            great_transition_min_gap_cp=tg,
            great_defensive_losing_threshold=dl,
            great_seizing_winning_threshold=sw,
        )
        stats = evaluator.evaluate(config)
        if stats["total"] == 0:
            continue
        score = _score_stats(stats, profile)
        results.append((score, stats, config))
        if best_score is None or score > best_score:
            best_score = score

        if done % 50 == 0 or done == total:
            _print_sweep_progress(
                f"phase 2 ({profile.key})", done, total, started_at, best_score, profile
            )

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def phase2b_special_sweep(
    best_config: ClassificationConfig,
    evaluator: OptimizerRunner,
    profile: ScoreProfile = STANDARD_SCORE,
) -> list[OptimizationResult]:
    """Sweep miss, best-promotion, and Elo-dependent excellent parameters."""
    miss_opp_thresholds = [0.04, 0.06, 0.08, 0.10, 0.15]
    miss_min_eps = [0.04, 0.06, 0.08, 0.10, 0.15]
    miss_best_win_pcts = [0.60, 0.65, 0.70, 0.75, 0.80]
    best_gaps = [15, 20, 30, 50, 75]
    excellent_low_elos = [0.004, 0.006, 0.008, 0.010, 0.012]
    excellent_elo_thresholds = [800, 1000, 1200]

    combos = list(product(
        miss_opp_thresholds, miss_min_eps, miss_best_win_pcts,
        best_gaps, excellent_low_elos, excellent_elo_thresholds,
    ))
    total = len(combos)
    done = 0
    started_at = time.time()
    best_score = None

    results = []
    for miss_opp, miss_ep, miss_wp, bg, ele, elet in combos:
        done += 1
        config = replace(
            best_config,
            miss_opponent_ep_loss_threshold=miss_opp,
            miss_min_ep_loss=miss_ep,
            miss_best_win_pct_threshold=miss_wp,
            best_promotion_min_gap_cp=bg,
            ep_excellent_low_elo=ele,
            excellent_elo_threshold=elet,
        )
        stats = evaluator.evaluate(config)
        if stats["total"] == 0:
            continue
        score = _score_stats(stats, profile)
        results.append((score, stats, config))
        if best_score is None or score > best_score:
            best_score = score

        if done % 100 == 0 or done == total:
            _print_sweep_progress(
                f"phase 2b ({profile.key})", done, total, started_at, best_score, profile
            )

    results.sort(key=lambda x: x[0], reverse=True)
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
            "great_transition_min_gap_cp": (-15, 15),
            "great_defensive_losing_threshold": (-0.05, 0.05),
            "great_seizing_winning_threshold": (-0.05, 0.05),
            "best_promotion_min_gap_cp": (-10, 10),
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
            "great_transition_min_gap_cp": (-8, 8),
            "great_defensive_losing_threshold": (-0.03, 0.03),
            "great_seizing_winning_threshold": (-0.03, 0.03),
            "best_promotion_min_gap_cp": (-5, 5),
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
            "great_transition_min_gap_cp": (-4, 4),
            "great_defensive_losing_threshold": (-0.015, 0.015),
            "great_seizing_winning_threshold": (-0.015, 0.015),
            "best_promotion_min_gap_cp": (-2, 2),
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
    game_files = _find_game_csvs()
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
    print(f"Fallback Elo: {args.elo} (per-game Elos from game_elos.json)")
    print(f"Games: {[f.name for f in game_files]}")

    all_comparisons = []
    for csv_path in game_files:
        w_elo, b_elo = _game_elos_by_side(csv_path, fallback_elo=args.elo)
        print(f"\n{'='*60}")
        print(f" {csv_path.name}  (White: {w_elo}, Black: {b_elo})")
        print(f"{'='*60}")

        if args.stockfish:
            cache = _cache_path(csv_path, args.depth, args.multipv, engine_tag, nodes=nodes)
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
    engine_path, engine_tag = _resolve_engine(args.engine)
    nodes = args.nodes

    if not Path(engine_path).exists() and not shutil.which(engine_path):
        print(f"ERROR: Engine binary not found: {engine_path}")
        print("Install Stockfish or use --engine to specify a path/alias.")
        print(f"  Available aliases: {', '.join(ENGINE_PATHS.keys())}")
        sys.exit(1)

    game_files = _find_game_csvs()
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
    game_files = _find_game_csvs()
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
    game_files = _find_game_csvs()
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
    )
    if args.score_mode == "both":
        score_profiles = [STANDARD_SCORE, WEIGHTED_SCORE]
    else:
        score_profiles = [SCORE_PROFILES[args.score_mode]]
    mode = "Stockfish (cached)" if args.stockfish else "Mock (chess.com evals)"
    print(f"\nMode: {mode}")
    if args.stockfish:
        print(f"Engine: {engine_tag} ({engine_path})")
    print(f"Fallback Elo: {args.elo} (per-game Elos from game_elos.json)")
    if args.stockfish:
        if nodes:
            print(f"Engine params: nodes={nodes:,}, multipv={args.multipv}")
        else:
            print(f"Engine params: depth={args.depth}, multipv={args.multipv}")
    print(f"Games:")
    for gf in game_files:
        w, b = _game_elos_by_side(gf, fallback_elo=args.elo)
        print(f"  {gf.name} (W: {w}, B: {b})")

    if args.stockfish:
        missing = []
        for gf in game_files:
            cache = _cache_path(gf, args.depth, args.multipv, engine_tag, nodes=nodes)
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

    # Classification parameters
    parser.add_argument(
        "--elo", type=int, default=820,
        help="Player Elo for classification (default: 820)",
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        choices=["standard", "weighted", "both"],
        default="both",
        help="Optimization objective: standard accuracy, weighted per-class recall, or both (default: both)",
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
