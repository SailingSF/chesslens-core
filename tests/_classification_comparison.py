"""
Shared comparison harness for chess.com classification benchmarks.

This module intentionally keeps the reusable data loading, comparison, caching,
and reporting logic out of the pytest file so the tests stay focused.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import chess

from analysis.context import AssembledContext, ContextAssembler
from chess_engine.service import CandidateMove, EngineResult
from config.classification import ClassificationConfig

GAMES_DIR = Path(__file__).parent / "test_games"


@dataclass
class ChessComMove:
    """A single move's data as recorded from chess.com's game review."""

    ply: int
    color: str
    move_san: str
    classification: str
    eval_pawns: Optional[float]
    mate_in: Optional[int]
    points_diff: Optional[float]
    is_game_end: bool


def _parse_eval(raw: str) -> tuple[Optional[float], Optional[int], bool]:
    """Parse eval_points field -> (eval_pawns, mate_in, is_game_end)."""
    raw = raw.strip()
    if raw in ("1-0", "0-1", "1/2-1/2"):
        return None, None, True
    if raw.startswith("M"):
        return None, int(raw[1:]), False
    if raw.startswith("-M"):
        return None, -int(raw[2:]), False
    return float(raw), None, False


@lru_cache(maxsize=None)
def load_chesscom_csv(path: Path) -> list[ChessComMove]:
    """Load a chess.com game review CSV into structured records."""
    moves = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            eval_pawns, mate_in, is_end = _parse_eval(row["eval_points"])
            diff_raw = row["points_difference"].strip()
            points_diff = float(diff_raw) if diff_raw else None

            moves.append(
                ChessComMove(
                    ply=int(row["ply"]),
                    color=row["color"].strip(),
                    move_san=row["move"].strip(),
                    classification=row["classification"].strip().lower(),
                    eval_pawns=eval_pawns,
                    mate_in=mate_in,
                    points_diff=points_diff,
                    is_game_end=is_end,
                )
            )
    return moves


def _build_engine_result(
    board: chess.Board,
    played_move: chess.Move,
    best_cp: Optional[int],
    best_mate: Optional[int],
    played_cp: Optional[int],
    played_mate: Optional[int],
) -> EngineResult:
    """
    Build a mock EngineResult from CSV eval data.

    If the played move is not the best move, the first legal move is used as a
    placeholder because the CSV only exposes the evaluation, not chess.com's
    exact preferred continuation.
    """

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

    candidates.append(
        CandidateMove(
            move=best_move,
            score_cp=best_cp,
            mate_in=best_mate,
            pv=[best_move],
        )
    )

    if best_move != played_move:
        candidates.append(
            CandidateMove(
                move=played_move,
                score_cp=played_cp,
                mate_in=played_mate,
                pv=[played_move],
            )
        )

    return EngineResult(fen=board.fen(), depth=20, candidates=candidates)


@dataclass
class MoveComparison:
    """Result of comparing our classification vs chess.com's for a single move."""

    ply: int
    color: str
    move_san: str
    chesscom_class: str
    our_class: str
    match: bool
    eval_before_wp: Optional[float]
    eval_after_wp: Optional[float]
    ep_loss: Optional[float]
    eval_before_pawns: Optional[float]
    eval_after_pawns: Optional[float]
    is_brilliant: bool
    is_great: bool
    is_miss: bool


CLASS_SEVERITY = {
    "book": 0,
    "brilliant": 1,
    "great": 2,
    "best": 3,
    "excellent": 4,
    "good": 5,
    "miss": 6,
    "inaccuracy": 7,
    "mistake": 8,
    "blunder": 9,
    "unknown": 10,
}


def run_comparison(
    csv_path: Path,
    config: Optional[ClassificationConfig] = None,
    player_elo: Optional[int] = None,
) -> list[MoveComparison]:
    """Run the mock comparison pipeline for a single game."""

    moves = load_chesscom_csv(csv_path)
    assembler = ContextAssembler(classification_config=config)
    comparisons = []
    board = chess.Board()
    prev_context: Optional[AssembledContext] = None

    for i, cm in enumerate(moves):
        try:
            played_move = board.parse_san(cm.move_san)
        except ValueError:
            board.push_san(cm.move_san)
            continue

        if cm.is_game_end:
            board.push(played_move)
            continue

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
        best_mate = eval_before_mate
        played_cp = round(eval_after_pawns * 100) if eval_after_pawns is not None else None
        played_mate = eval_after_mate

        engine_result = _build_engine_result(
            board,
            played_move,
            best_cp,
            best_mate,
            played_cp,
            played_mate,
        )

        ctx = assembler.assemble(
            board,
            engine_result,
            played_move,
            player_elo=player_elo,
            prev_context=prev_context,
        )

        comparisons.append(
            MoveComparison(
                ply=cm.ply,
                color=cm.color,
                move_san=cm.move_san,
                chesscom_class=cm.classification,
                our_class=ctx.cp_loss_label,
                match=(ctx.cp_loss_label == cm.classification),
                eval_before_wp=ctx.win_pct_before,
                eval_after_wp=ctx.win_pct_after,
                ep_loss=ctx.ep_loss,
                eval_before_pawns=eval_before_pawns,
                eval_after_pawns=eval_after_pawns,
                is_brilliant=ctx.is_brilliant,
                is_great=ctx.is_great,
                is_miss=ctx.is_miss,
            )
        )

        prev_context = ctx
        board.push(played_move)

    return comparisons


def _cache_path(csv_path: Path, depth: int, multipv: int) -> Path:
    """Return the JSON cache file path for a given game and analysis params."""
    stem = csv_path.stem
    return csv_path.parent / f"{stem}_stockfish_d{depth}_pv{multipv}.json"


def _save_engine_cache(cache_file: Path, results: list[EngineResult]) -> None:
    """Serialize engine results to JSON for fast re-use."""
    data = []
    for er in results:
        candidates = []
        for c in er.candidates:
            candidates.append(
                {
                    "move": c.move.uci(),
                    "score_cp": c.score_cp,
                    "mate_in": c.mate_in,
                    "pv": [m.uci() for m in c.pv],
                    "wdl_win": c.wdl_win,
                    "wdl_draw": c.wdl_draw,
                    "wdl_loss": c.wdl_loss,
                }
            )
        data.append({"fen": er.fen, "depth": er.depth, "candidates": candidates})
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def _load_engine_cache(cache_file: Path) -> list[EngineResult]:
    """Deserialize engine results from JSON cache."""
    with open(cache_file) as f:
        data = json.load(f)

    results = []
    for entry in data:
        candidates = []
        for c in entry["candidates"]:
            candidates.append(
                CandidateMove(
                    move=chess.Move.from_uci(c["move"]),
                    score_cp=c.get("score_cp"),
                    mate_in=c.get("mate_in"),
                    pv=[chess.Move.from_uci(m) for m in c.get("pv", [])],
                    wdl_win=c.get("wdl_win"),
                    wdl_draw=c.get("wdl_draw"),
                    wdl_loss=c.get("wdl_loss"),
                )
            )
        results.append(
            EngineResult(
                fen=entry["fen"],
                depth=entry["depth"],
                candidates=candidates,
            )
        )
    return results


async def _analyze_game_positions(
    moves: list[ChessComMove],
    depth: int = 20,
    multipv: int = 3,
) -> list[EngineResult]:
    """Run real Stockfish analysis at each position in the game."""

    from chess_engine.pool import EnginePool
    from chess_engine.service import _parse_candidates

    pool = EnginePool(size=1, hash_mb=128, threads=2)
    await pool.start()

    results = []
    board = chess.Board()

    try:
        for cm in moves:
            try:
                played_move = board.parse_san(cm.move_san)
            except ValueError:
                board.push_san(cm.move_san)
                results.append(None)
                continue

            if cm.is_game_end:
                board.push(played_move)
                results.append(None)
                continue

            async with pool.acquire() as engine:
                await engine.configure({"UCI_ShowWDL": True})
                info_list = await engine.analyse(
                    board,
                    chess.engine.Limit(depth=depth),
                    multipv=multipv,
                )
                if not isinstance(info_list, list):
                    info_list = [info_list]

                candidates = _parse_candidates(info_list)
                played_in_candidates = any(c.move == played_move for c in candidates)

                if not played_in_candidates:
                    played_info = await engine.analyse(
                        board,
                        chess.engine.Limit(depth=depth),
                        root_moves=[played_move],
                    )
                    if not isinstance(played_info, list):
                        played_info = [played_info]
                    candidates.extend(_parse_candidates(played_info))

            results.append(EngineResult(fen=board.fen(), depth=depth, candidates=candidates))
            board.push(played_move)
    finally:
        await pool.stop()

    return results


async def run_comparison_stockfish(
    csv_path: Path,
    config: Optional[ClassificationConfig] = None,
    player_elo: Optional[int] = None,
    depth: int = 20,
    multipv: int = 3,
) -> list[MoveComparison]:
    """Run comparison using real Stockfish analysis with JSON caching."""

    moves = load_chesscom_csv(csv_path)
    cache_file = _cache_path(csv_path, depth, multipv)

    if cache_file.exists():
        engine_results = _load_engine_cache(cache_file)
    else:
        raw_results = await _analyze_game_positions(moves, depth=depth, multipv=multipv)
        engine_results = [er for er in raw_results if er is not None]
        _save_engine_cache(cache_file, engine_results)

    assembler = ContextAssembler(classification_config=config)
    comparisons = []
    board = chess.Board()
    prev_context: Optional[AssembledContext] = None
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
            board,
            engine_result,
            played_move,
            player_elo=player_elo,
            prev_context=prev_context,
        )

        if ctx.win_pct_before is None:
            eval_before_pawns = None
        elif i == 0:
            eval_before_pawns = 0.25
        else:
            eval_before_pawns = moves[i - 1].eval_pawns

        comparisons.append(
            MoveComparison(
                ply=cm.ply,
                color=cm.color,
                move_san=cm.move_san,
                chesscom_class=cm.classification,
                our_class=ctx.cp_loss_label,
                match=(ctx.cp_loss_label == cm.classification),
                eval_before_wp=ctx.win_pct_before,
                eval_after_wp=ctx.win_pct_after,
                ep_loss=ctx.ep_loss,
                eval_before_pawns=eval_before_pawns,
                eval_after_pawns=cm.eval_pawns,
                is_brilliant=ctx.is_brilliant,
                is_great=ctx.is_great,
                is_miss=ctx.is_miss,
            )
        )

        prev_context = ctx
        board.push(played_move)

    return comparisons


def print_comparison_table(comparisons: list[MoveComparison]) -> None:
    """Print a formatted comparison table for debugging."""
    header = (
        f"{'Ply':>3} {'Color':<5} {'Move':<8} {'chess.com':<12} {'Ours':<12} "
        f"{'Match':<5} {'EP Loss':>8} {'WP Before':>9} {'WP After':>9} "
        f"{'Eval Bef':>8} {'Eval Aft':>8}"
    )
    print()
    print(header)
    print("-" * len(header))

    for c in comparisons:
        match_str = "  Y" if c.match else "  N ***"
        ep_str = f"{c.ep_loss:.4f}" if c.ep_loss is not None else "   N/A"
        wp_before = f"{c.eval_before_wp:.4f}" if c.eval_before_wp is not None else "  N/A"
        wp_after = f"{c.eval_after_wp:.4f}" if c.eval_after_wp is not None else "  N/A"
        eb = f"{c.eval_before_pawns:+.2f}" if c.eval_before_pawns is not None else "  N/A"
        ea = f"{c.eval_after_pawns:+.2f}" if c.eval_after_pawns is not None else " mate"

        special = ""
        if c.is_brilliant:
            special = " [!!]"
        elif c.is_great:
            special = " [!]"
        elif c.is_miss:
            special = " [?]"

        print(
            f"{c.ply:>3} {c.color:<5} {c.move_san:<8} {c.chesscom_class:<12} "
            f"{c.our_class + special:<12} {match_str:<7} {ep_str:>8} {wp_before:>9} "
            f"{wp_after:>9} {eb:>8} {ea:>8}"
        )


def compute_accuracy_stats(
    comparisons: list[MoveComparison],
    include_breakdown: bool = True,
) -> dict:
    """Compute summary accuracy statistics."""

    comparable = [c for c in comparisons if c.chesscom_class and c.chesscom_class != "book"]
    if not comparable:
        return {"total": 0, "matches": 0, "accuracy": 0.0}

    matches = sum(1 for c in comparable if c.match)
    total = len(comparable)

    close_misses = 0
    far_misses = 0
    for c in comparable:
        if not c.match:
            s1 = CLASS_SEVERITY.get(c.chesscom_class, 10)
            s2 = CLASS_SEVERITY.get(c.our_class, 10)
            if abs(s1 - s2) <= 1:
                close_misses += 1
            else:
                far_misses += 1

    by_class: dict[str, dict] = {}
    for c in comparable:
        cc = c.chesscom_class
        if cc not in by_class:
            by_class[cc] = {"total": 0, "matched": 0}
            if include_breakdown:
                by_class[cc]["our_labels"] = {}
        by_class[cc]["total"] += 1
        if c.match:
            by_class[cc]["matched"] += 1
        if include_breakdown:
            labels = by_class[cc]["our_labels"]
            labels[c.our_class] = labels.get(c.our_class, 0) + 1

    return {
        "total": total,
        "matches": matches,
        "accuracy": matches / total,
        "close_misses": close_misses,
        "far_misses": far_misses,
        "by_class": by_class,
    }


def print_accuracy_report(stats: dict) -> None:
    """Print a human-readable accuracy report."""
    print(f"\n{'=' * 60}")
    print("ACCURACY REPORT")
    print(f"{'=' * 60}")
    print(f"Total non-book moves: {stats['total']}")
    print(f"Exact matches:        {stats['matches']} ({stats['accuracy']:.1%})")
    print(f"Close misses (±1):    {stats['close_misses']}")
    print(f"Far misses (±2+):     {stats['far_misses']}")

    print("\nBreakdown by chess.com classification:")
    print(f"  {'Class':<12} {'Total':>5} {'Match':>5} {'Rate':>6}  Our labels")
    print(f"  {'-' * 60}")
    for cls in sorted(stats["by_class"], key=lambda x: CLASS_SEVERITY.get(x, 99)):
        info = stats["by_class"][cls]
        rate = info["matched"] / info["total"] if info["total"] > 0 else 0
        labels = ", ".join(f"{k}={v}" for k, v in sorted(info["our_labels"].items()))
        print(f"  {cls:<12} {info['total']:>5} {info['matched']:>5} {rate:>5.0%}   {labels}")


STOCKFISH_AVAILABLE = shutil.which("stockfish") is not None
