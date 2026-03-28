"""
Parameter optimization for EP classification — grid search over sigmoid
constants and EP thresholds to maximize accuracy against chess.com.

Supports two modes:
  - Mock mode (default): uses chess.com's eval data (fast, no Stockfish)
  - Stockfish mode (--stockfish): uses cached real Stockfish results

Usage:
    python tests/optimize_classification.py                 # mock mode
    python tests/optimize_classification.py --stockfish     # Stockfish mode
    python tests/optimize_classification.py --elo 1500      # custom Elo
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

# Add project root to path so imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.classification import ClassificationConfig
from tests.test_classification_comparison import (
    GAMES_DIR,
    MoveComparison,
    compute_accuracy_stats,
    load_chesscom_csv,
    run_comparison,
)


def _find_game_csvs() -> list[Path]:
    """Find all game CSV files in the test_games directory."""
    return sorted(GAMES_DIR.glob("game_*.csv"))


# ---------------------------------------------------------------------------
# Stockfish-cached comparison (replays cached engine results with new config)
# ---------------------------------------------------------------------------

def run_comparison_from_cache(
    csv_path: Path,
    cache_path: Path,
    config: ClassificationConfig | None = None,
    player_elo: int | None = None,
) -> list[MoveComparison]:
    """
    Run classification comparison using pre-computed Stockfish engine results.

    This avoids re-running Stockfish for each parameter config — the engine
    results are loaded once and only the classification logic varies.
    """
    import json

    import chess

    from analysis.context import AssembledContext, ContextAssembler
    from chess_engine.service import CandidateMove, EngineResult

    moves = load_chesscom_csv(csv_path)

    # Load cached engine results
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

    # Replay with given config
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

        ctx = assembler.assemble(
            board, engine_result, played_move,
            player_elo=player_elo,
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


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_single(
    config: ClassificationConfig,
    game_files: list[Path],
    player_elo: int,
    use_stockfish: bool,
) -> dict:
    """Run comparison across all games with a given config, return aggregate stats."""
    all_comparisons = []
    for csv_path in game_files:
        if use_stockfish:
            # Look for cached Stockfish results
            cache_path = csv_path.parent / f"{csv_path.stem}_stockfish_d20_pv3.json"
            if not cache_path.exists():
                print(f"  WARNING: No Stockfish cache for {csv_path.name}, skipping")
                continue
            comparisons = run_comparison_from_cache(
                csv_path, cache_path, config=config, player_elo=player_elo,
            )
        else:
            comparisons = run_comparison(csv_path, config=config, player_elo=player_elo)
        all_comparisons.extend(comparisons)

    return compute_accuracy_stats(all_comparisons)


def phase1_sigmoid_sweep(
    game_files: list[Path], player_elo: int, use_stockfish: bool,
) -> list[tuple[float, dict, ClassificationConfig]]:
    """Sweep sigmoid parameters with default thresholds."""
    floors = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ranges = [0.2, 0.3, 0.4, 0.5, 0.6]
    steepnesses = [0.003, 0.005, 0.008]

    results = []
    base = ClassificationConfig()

    for floor, range_, steep in product(floors, ranges, steepnesses):
        if floor + range_ > 1.05:
            continue
        config = replace(
            base,
            elo_scale_floor=floor,
            elo_scale_range=range_,
            elo_scale_steepness=steep,
        )
        stats = run_single(config, game_files, player_elo, use_stockfish)
        if stats["total"] == 0:
            continue
        score = (stats["accuracy"], -stats.get("far_misses", 0))
        results.append((score, stats, config))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def phase2_threshold_sweep(
    best_sigmoid: ClassificationConfig,
    game_files: list[Path], player_elo: int, use_stockfish: bool,
) -> list[tuple[float, dict, ClassificationConfig]]:
    """Sweep EP thresholds with fixed sigmoid parameters."""
    excellents = [0.005, 0.01, 0.015, 0.02, 0.025]
    goods = [0.02, 0.03, 0.04, 0.05, 0.06]
    inaccuracies = [0.06, 0.07, 0.08, 0.10, 0.12]
    mistakes = [0.12, 0.15, 0.18, 0.20, 0.22, 0.25]

    results = []
    for exc, good, inacc, mist in product(excellents, goods, inaccuracies, mistakes):
        if not (exc < good < inacc < mist):
            continue
        config = replace(
            best_sigmoid,
            ep_excellent=exc,
            ep_good=good,
            ep_inaccuracy=inacc,
            ep_mistake=mist,
        )
        stats = run_single(config, game_files, player_elo, use_stockfish)
        if stats["total"] == 0:
            continue
        score = (stats["accuracy"], -stats.get("far_misses", 0))
        results.append((score, stats, config))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def phase3_fine_tune(
    best_config: ClassificationConfig,
    game_files: list[Path], player_elo: int, use_stockfish: bool,
) -> list[tuple[float, dict, ClassificationConfig]]:
    """Fine-tune around the best config from phases 1+2."""
    bc = best_config
    floors = [bc.elo_scale_floor + d for d in [-0.05, 0.0, 0.05]]
    ranges = [bc.elo_scale_range + d for d in [-0.05, 0.0, 0.05]]
    excellents = [bc.ep_excellent + d for d in [-0.005, 0.0, 0.005]]
    goods = [bc.ep_good + d for d in [-0.005, 0.0, 0.005]]
    inaccuracies = [bc.ep_inaccuracy + d for d in [-0.01, 0.0, 0.01]]
    mistakes = [bc.ep_mistake + d for d in [-0.02, 0.0, 0.02]]

    results = []
    for floor, range_, exc, good, inacc, mist in product(
        floors, ranges, excellents, goods, inaccuracies, mistakes
    ):
        if floor + range_ > 1.05 or floor < 0.1 or range_ < 0.1:
            continue
        if not (0 < exc < good < inacc < mist):
            continue
        config = replace(
            bc,
            elo_scale_floor=floor,
            elo_scale_range=range_,
            ep_excellent=exc,
            ep_good=good,
            ep_inaccuracy=inacc,
            ep_mistake=mist,
        )
        stats = run_single(config, game_files, player_elo, use_stockfish)
        if stats["total"] == 0:
            continue
        score = (stats["accuracy"], -stats.get("far_misses", 0))
        results.append((score, stats, config))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def print_results(title: str, results: list, top_n: int = 10):
    print(f"\n{'='*80}")
    print(f" {title} — Top {min(top_n, len(results))} of {len(results)} runs")
    print(f"{'='*80}")
    for i, (score, stats, config) in enumerate(results[:top_n]):
        acc = stats["accuracy"]
        fm = stats.get("far_misses", "?")
        cm = stats.get("close_misses", "?")
        print(
            f"\n  #{i+1}  Accuracy: {acc:.1%}  "
            f"(far_misses={fm}, close_misses={cm})"
        )
        print(
            f"       sigmoid: floor={config.elo_scale_floor:.2f}, "
            f"range={config.elo_scale_range:.2f}, "
            f"steepness={config.elo_scale_steepness:.4f}"
        )
        print(
            f"       thresholds: excellent≤{config.ep_excellent:.3f}, "
            f"good≤{config.ep_good:.3f}, "
            f"inaccuracy<{config.ep_inaccuracy:.3f}, "
            f"mistake<{config.ep_mistake:.3f}"
        )
        by_class = stats.get("by_class", {})
        for cls in sorted(by_class, key=lambda x: by_class[x]["total"], reverse=True):
            info = by_class[cls]
            rate = info["matched"] / info["total"] if info["total"] else 0
            labels = ", ".join(f"{k}={v}" for k, v in sorted(info["our_labels"].items()))
            print(f"         {cls:<12} {info['matched']}/{info['total']} ({rate:.0%})  → {labels}")


def main():
    parser = argparse.ArgumentParser(description="Optimize EP classification parameters")
    parser.add_argument("--stockfish", action="store_true",
                        help="Use cached Stockfish results instead of mock evals")
    parser.add_argument("--elo", type=int, default=820,
                        help="Player Elo for classification (default: 820)")
    args = parser.parse_args()

    game_files = _find_game_csvs()
    mode = "Stockfish (cached)" if args.stockfish else "Mock (chess.com evals)"
    print(f"\nMode: {mode}")
    print(f"Elo: {args.elo}")
    print(f"Games: {[f.name for f in game_files]}")

    if args.stockfish:
        # Check for caches
        for gf in game_files:
            cache = gf.parent / f"{gf.stem}_stockfish_d20_pv3.json"
            if cache.exists():
                print(f"  ✓ {cache.name}")
            else:
                print(f"  ✗ {cache.name} — run pytest Stockfish test first to generate")

    start = time.time()

    # --- Phase 1: Sigmoid sweep ---
    print("\n[Phase 1] Sweeping sigmoid parameters (default thresholds)...")
    p1 = phase1_sigmoid_sweep(game_files, args.elo, args.stockfish)
    if not p1:
        print("No results — check that game files and caches exist.")
        return
    print_results("Phase 1: Sigmoid Sweep", p1, top_n=10)
    best_sigmoid = p1[0][2]

    # --- Phase 2: Threshold sweep ---
    print(f"\n[Phase 2] Sweeping EP thresholds (best sigmoid: "
          f"floor={best_sigmoid.elo_scale_floor}, range={best_sigmoid.elo_scale_range}, "
          f"steep={best_sigmoid.elo_scale_steepness})...")
    p2 = phase2_threshold_sweep(best_sigmoid, game_files, args.elo, args.stockfish)
    print_results("Phase 2: Threshold Sweep", p2, top_n=10)
    best_combined = p2[0][2]

    # --- Phase 3: Fine-tune ---
    print(f"\n[Phase 3] Fine-tuning around best combined config...")
    p3 = phase3_fine_tune(best_combined, game_files, args.elo, args.stockfish)
    print_results("Phase 3: Fine-Tuning", p3, top_n=10)
    best_final = p3[0][2]

    # --- Summary ---
    elapsed = time.time() - start
    total_runs = len(p1) + len(p2) + len(p3)
    print(f"\n{'='*80}")
    print(f" FINAL RESULT — {total_runs} total runs in {elapsed:.1f}s")
    print(f"{'='*80}")
    print(f"\n  Best config to apply to config/classification.py:")
    print(f"    elo_scale_floor:     {best_final.elo_scale_floor}")
    print(f"    elo_scale_range:     {best_final.elo_scale_range}")
    print(f"    elo_scale_steepness: {best_final.elo_scale_steepness}")
    print(f"    ep_excellent:        {best_final.ep_excellent}")
    print(f"    ep_good:             {best_final.ep_good}")
    print(f"    ep_inaccuracy:       {best_final.ep_inaccuracy}")
    print(f"    ep_mistake:          {best_final.ep_mistake}")
    print()


if __name__ == "__main__":
    main()
