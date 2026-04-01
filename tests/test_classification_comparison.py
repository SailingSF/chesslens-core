"""
Classification comparison test — compares our EP-based move classifications
against chess.com's Classification V2 output for the same game.

Two modes:
  1. Mock mode (run_comparison) — uses chess.com's eval data to build fake engine
     results. Fast, no Stockfish required. Used by the optimizer and CI.
  2. Stockfish mode (run_comparison_stockfish) — runs real Stockfish analysis at
     each position. Matches production behavior exactly (multi-PV, native WDL).
     Results are cached to JSON for instant re-runs.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chess
import pytest

from analysis.context import AssembledContext, ContextAssembler
from analysis.expected_points import SigmoidWDLProvider, classify_ep_loss
from chess_engine.service import CandidateMove, EngineResult
from config.classification import ClassificationConfig

GAMES_DIR = Path(__file__).parent / "test_games"


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

@dataclass
class ChessComMove:
    """A single move's data as recorded from chess.com's game review."""
    ply: int
    color: str
    move_san: str
    classification: str  # book, best, excellent, good, inaccuracy, mistake, blunder, great, miss, brilliant
    eval_pawns: Optional[float]   # engine eval in pawns (from white's POV), None for mate
    mate_in: Optional[int]        # mate-in-N (positive = white mates), None if not mate
    points_diff: Optional[float]  # eval change from previous position
    is_game_end: bool             # "1-0", "0-1", "1/2-1/2"


def _parse_eval(raw: str) -> tuple[Optional[float], Optional[int], bool]:
    """Parse eval_points field → (eval_pawns, mate_in, is_game_end)."""
    raw = raw.strip()
    if raw in ("1-0", "0-1", "1/2-1/2"):
        return None, None, True
    if raw.startswith("M"):
        # Mate-in-N — always from the moving side's perspective in chess.com CSV
        return None, int(raw[1:]), False
    if raw.startswith("-M"):
        return None, -int(raw[2:]), False
    return float(raw), None, False


def load_chesscom_csv(path: Path) -> list[ChessComMove]:
    """Load a chess.com game review CSV into structured records."""
    moves = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            eval_pawns, mate_in, is_end = _parse_eval(row["eval_points"])

            # eval_points is always from white's perspective: "M5" means
            # white mates in 5, regardless of which side just moved.
            # No negation needed.

            diff_raw = row["points_difference"].strip()
            points_diff = float(diff_raw) if diff_raw else None

            moves.append(ChessComMove(
                ply=int(row["ply"]),
                color=row["color"].strip(),
                move_san=row["move"].strip(),
                classification=row["classification"].strip().lower(),
                eval_pawns=eval_pawns,
                mate_in=mate_in,
                points_diff=points_diff,
                is_game_end=is_end,
            ))
    return moves


# ---------------------------------------------------------------------------
# Engine result construction from CSV eval data
# ---------------------------------------------------------------------------

def _build_engine_result(
    board: chess.Board,
    played_move: chess.Move,
    best_cp: Optional[int],
    best_mate: Optional[int],
    played_cp: Optional[int],
    played_mate: Optional[int],
) -> EngineResult:
    """
    Build a mock EngineResult from the CSV eval data.

    The "best" candidate gets the eval of the position before the move
    (what the engine thinks is the best continuation). If the played move
    IS the best move, both candidates have the same eval.

    We use the board's first legal move as the "best" move placeholder
    when it differs from the played move, since we don't know chess.com's
    actual best move — only their eval.
    """
    candidates = []

    # Best move candidate — use the actual best move if it's the played move,
    # otherwise use the first legal move as a placeholder
    legal_moves = list(board.legal_moves)
    best_move = played_move if (best_cp == played_cp and best_mate == played_mate) else legal_moves[0]
    if best_move == played_move and len(legal_moves) > 1:
        # If they happen to collide but evals differ, use a different move
        if best_cp != played_cp or best_mate != played_mate:
            best_move = legal_moves[1] if legal_moves[0] == played_move else legal_moves[0]

    candidates.append(CandidateMove(
        move=best_move, score_cp=best_cp, mate_in=best_mate,
        pv=[best_move],
    ))

    # Played move candidate (if different from best)
    if best_move != played_move:
        candidates.append(CandidateMove(
            move=played_move, score_cp=played_cp, mate_in=played_mate,
            pv=[played_move],
        ))

    return EngineResult(fen=board.fen(), depth=20, candidates=candidates)


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

@dataclass
class MoveComparison:
    """Result of comparing our classification vs chess.com's for a single move."""
    ply: int
    color: str
    move_san: str
    chesscom_class: str
    our_class: str
    match: bool
    # Debug data
    eval_before_wp: Optional[float]  # our win_pct before move (from white's POV)
    eval_after_wp: Optional[float]   # our win_pct after move
    ep_loss: Optional[float]
    eval_before_pawns: Optional[float]
    eval_after_pawns: Optional[float]
    is_brilliant: bool
    is_great: bool
    is_miss: bool


# Classification severity for ordering — used to measure "close" vs "far" mismatches
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
    """
    Run the full comparison pipeline for a single game.

    Returns a list of MoveComparison records, one per non-book/non-end ply.
    """
    moves = load_chesscom_csv(csv_path)
    assembler = ContextAssembler(classification_config=config)
    comparisons = []

    board = chess.Board()
    prev_context: Optional[AssembledContext] = None

    for i, cm in enumerate(moves):
        # Parse the SAN move
        try:
            played_move = board.parse_san(cm.move_san)
        except ValueError:
            board.push_san(cm.move_san)
            continue

        # Skip game-end markers
        if cm.is_game_end:
            board.push(played_move)
            continue

        # Compute eval_before from the previous move's eval
        # eval_before = the engine's eval of the position before this move
        # (i.e., what the best move would achieve)
        if i == 0:
            # Starting position — approximate as 0.25 for white
            eval_before_pawns = 0.25
            eval_before_mate = None
        else:
            prev = moves[i - 1]
            eval_before_pawns = prev.eval_pawns
            eval_before_mate = prev.mate_in

        eval_after_pawns = cm.eval_pawns
        eval_after_mate = cm.mate_in

        # Convert to centipawns from white's POV (our engine convention)
        best_cp = round(eval_before_pawns * 100) if eval_before_pawns is not None else None
        best_mate = eval_before_mate
        played_cp = round(eval_after_pawns * 100) if eval_after_pawns is not None else None
        played_mate = eval_after_mate

        # Build mock engine result and run our classifier
        engine_result = _build_engine_result(
            board, played_move, best_cp, best_mate, played_cp, played_mate,
        )

        ctx = assembler.assemble(
            board, engine_result, played_move,
            player_elo=player_elo,
            prev_context=prev_context,
        )

        our_class = ctx.cp_loss_label
        chesscom_class = cm.classification

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
# Stockfish engine result caching
# ---------------------------------------------------------------------------

def _cache_path(csv_path: Path, depth: int, multipv: int) -> Path:
    """Return the JSON cache file path for a given game + analysis params."""
    stem = csv_path.stem  # e.g. "game_01"
    return csv_path.parent / f"{stem}_stockfish_d{depth}_pv{multipv}.json"


def _save_engine_cache(
    cache_file: Path,
    results: list[EngineResult],
) -> None:
    """Serialize engine results to JSON for fast re-use."""
    data = []
    for er in results:
        candidates = []
        for c in er.candidates:
            candidates.append({
                "move": c.move.uci(),
                "score_cp": c.score_cp,
                "mate_in": c.mate_in,
                "pv": [m.uci() for m in c.pv],
                "wdl_win": c.wdl_win,
                "wdl_draw": c.wdl_draw,
                "wdl_loss": c.wdl_loss,
            })
        data.append({
            "fen": er.fen,
            "depth": er.depth,
            "candidates": candidates,
        })
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
            candidates.append(CandidateMove(
                move=chess.Move.from_uci(c["move"]),
                score_cp=c.get("score_cp"),
                mate_in=c.get("mate_in"),
                pv=[chess.Move.from_uci(m) for m in c.get("pv", [])],
                wdl_win=c.get("wdl_win"),
                wdl_draw=c.get("wdl_draw"),
                wdl_loss=c.get("wdl_loss"),
            ))
        results.append(EngineResult(
            fen=entry["fen"],
            depth=entry["depth"],
            candidates=candidates,
        ))
    return results


# ---------------------------------------------------------------------------
# Stockfish-based comparison (production-equivalent)
# ---------------------------------------------------------------------------

async def _analyze_game_positions(
    moves: list[ChessComMove],
    depth: int = 20,
    multipv: int = 3,
) -> list[EngineResult]:
    """
    Run real Stockfish analysis at each position in the game.

    For each position, runs multi-PV analysis to get the top candidates.
    If the played move isn't among them, runs a second constrained analysis
    to get the played move's eval (so EP loss is always computable).
    """
    from chess_engine.pool import EnginePool

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

            # Primary multi-PV analysis
            async with pool.acquire() as engine:
                await engine.configure({"UCI_ShowWDL": True})
                info_list = await engine.analyse(
                    board,
                    chess.engine.Limit(depth=depth),
                    multipv=multipv,
                )
                if not isinstance(info_list, list):
                    info_list = [info_list]

                from chess_engine.service import _parse_candidates
                candidates = _parse_candidates(info_list)

                # Check if played move is in the candidates
                played_in_candidates = any(c.move == played_move for c in candidates)

                if not played_in_candidates:
                    # Run constrained analysis for just the played move
                    played_info = await engine.analyse(
                        board,
                        chess.engine.Limit(depth=depth),
                        root_moves=[played_move],
                    )
                    if not isinstance(played_info, list):
                        played_info = [played_info]
                    played_candidates = _parse_candidates(played_info)
                    candidates.extend(played_candidates)

            engine_result = EngineResult(
                fen=board.fen(), depth=depth, candidates=candidates,
            )
            results.append(engine_result)
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
    """
    Run comparison using real Stockfish analysis (production-equivalent).

    Uses cached engine results if available, otherwise runs Stockfish and
    saves the cache for future runs.
    """
    moves = load_chesscom_csv(csv_path)
    cache_file = _cache_path(csv_path, depth, multipv)

    # Load or compute engine results
    if cache_file.exists():
        engine_results = _load_engine_cache(cache_file)
    else:
        raw_results = await _analyze_game_positions(moves, depth=depth, multipv=multipv)
        # Filter out None entries and save only real results
        # We need to track which plies have results
        engine_results_with_nones = raw_results
        # Save only non-None results with their index
        saveable = []
        for er in engine_results_with_nones:
            if er is not None:
                saveable.append(er)
        _save_engine_cache(cache_file, saveable)
        engine_results = saveable

    # Now replay the game and run our classifier with cached engine results
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
            board, engine_result, played_move,
            player_elo=player_elo,
            prev_context=prev_context,
        )

        our_class = ctx.cp_loss_label
        chesscom_class = cm.classification

        eval_before_pawns = None
        eval_after_pawns = None
        if ctx.win_pct_before is not None:
            # Approximate pawns from win_pct for display only
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


def compute_accuracy_stats(comparisons: list[MoveComparison]) -> dict:
    """Compute summary accuracy statistics."""
    # Exclude book moves from accuracy calculation (we don't have a book classifier)
    non_book = [c for c in comparisons if c.chesscom_class != "book"]
    if not non_book:
        return {"total": 0, "matches": 0, "accuracy": 0.0}

    matches = sum(1 for c in non_book if c.match)
    total = len(non_book)

    # Count by severity distance
    close_misses = 0  # off by 1 severity level
    far_misses = 0    # off by 2+ severity levels
    for c in non_book:
        if not c.match:
            s1 = CLASS_SEVERITY.get(c.chesscom_class, 10)
            s2 = CLASS_SEVERITY.get(c.our_class, 10)
            if abs(s1 - s2) <= 1:
                close_misses += 1
            else:
                far_misses += 1

    # Breakdown by chess.com classification
    by_class: dict[str, dict] = {}
    for c in non_book:
        cc = c.chesscom_class
        if cc not in by_class:
            by_class[cc] = {"total": 0, "matched": 0, "our_labels": {}}
        by_class[cc]["total"] += 1
        if c.match:
            by_class[cc]["matched"] += 1
        our = c.our_class
        by_class[cc]["our_labels"][our] = by_class[cc]["our_labels"].get(our, 0) + 1

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
    print(f"\n{'='*60}")
    print(f"ACCURACY REPORT")
    print(f"{'='*60}")
    print(f"Total non-book moves: {stats['total']}")
    print(f"Exact matches:        {stats['matches']} ({stats['accuracy']:.1%})")
    print(f"Close misses (±1):    {stats['close_misses']}")
    print(f"Far misses (±2+):     {stats['far_misses']}")

    print(f"\nBreakdown by chess.com classification:")
    print(f"  {'Class':<12} {'Total':>5} {'Match':>5} {'Rate':>6}  Our labels")
    print(f"  {'-'*60}")
    for cls in sorted(stats["by_class"], key=lambda x: CLASS_SEVERITY.get(x, 99)):
        info = stats["by_class"][cls]
        rate = info["matched"] / info["total"] if info["total"] > 0 else 0
        labels = ", ".join(f"{k}={v}" for k, v in sorted(info["our_labels"].items()))
        print(f"  {cls:<12} {info['total']:>5} {info['matched']:>5} {rate:>5.0%}   {labels}")


# ===========================================================================
# Tests
# ===========================================================================

class TestGame01Comparison:
    """Compare our classification against chess.com for game_01."""

    @pytest.fixture
    def comparisons(self):
        csv_path = GAMES_DIR / "game_01.csv"
        return run_comparison(csv_path)

    def test_run_comparison_completes(self, comparisons):
        """Sanity check: comparison produces results for all moves."""
        assert len(comparisons) > 0
        # game_01 has 59 plies (minus the final checkmate marker = 59 moves)
        # but game-end (1-0) is skipped, so we get 58 non-end moves
        # Also ply 59 is Qf7# which has eval "1-0" so it's skipped
        assert len(comparisons) >= 55

    def test_print_comparison_table(self, comparisons, capsys):
        """Print the full comparison table for manual review."""
        print_comparison_table(comparisons)
        stats = compute_accuracy_stats(comparisons)
        print_accuracy_report(stats)

        captured = capsys.readouterr()
        assert "ACCURACY REPORT" in captured.out

    def test_best_moves_classified_correctly(self, comparisons):
        """Moves chess.com classified as 'best' should ideally be 'best' for us too."""
        best_moves = [c for c in comparisons if c.chesscom_class == "best"]
        assert len(best_moves) > 0
        # Check that most 'best' moves have ep_loss of 0 or very close
        for c in best_moves:
            if c.ep_loss is not None:
                assert c.ep_loss < 0.03, (
                    f"Ply {c.ply} {c.move_san}: chess.com='best' but our ep_loss={c.ep_loss:.4f}"
                )

    def test_blunders_and_mistakes_detected(self, comparisons):
        """Moves chess.com classified as 'mistake' should not be classified as 'best'/'excellent' by us."""
        mistakes = [c for c in comparisons if c.chesscom_class in ("mistake", "blunder")]
        for c in mistakes:
            assert c.our_class not in ("best", "excellent"), (
                f"Ply {c.ply} {c.move_san}: chess.com='{c.chesscom_class}' "
                f"but we said '{c.our_class}' (ep_loss={c.ep_loss})"
            )

    def test_report_accuracy(self, comparisons):
        """Report overall accuracy — this test always passes but prints diagnostics."""
        stats = compute_accuracy_stats(comparisons)
        print_accuracy_report(stats)
        # Log the accuracy for visibility
        print(f"\n>>> Overall accuracy: {stats['accuracy']:.1%}")
        print(f">>> Far misses: {stats['far_misses']}")


class TestGame01ComparisonWithElo:
    """Same comparison but with player Elo to test Elo-adjusted curves."""

    def test_elo_adjusted_comparison(self):
        """Run comparison with Elo 1500 and print results."""
        csv_path = GAMES_DIR / "game_01.csv"
        comparisons = run_comparison(csv_path, player_elo=1500)
        stats = compute_accuracy_stats(comparisons)

        print(f"\n>>> With Elo 1500 — accuracy: {stats['accuracy']:.1%}")
        print_comparison_table(comparisons)
        print_accuracy_report(stats)


# ===========================================================================
# Stockfish-based comparison tests (production-equivalent)
# ===========================================================================

STOCKFISH_AVAILABLE = shutil.which("stockfish") is not None


@pytest.mark.skipif(not STOCKFISH_AVAILABLE, reason="Stockfish not installed")
class TestGame01StockfishComparison:
    """Compare using real Stockfish analysis — matches production behavior."""

    @pytest.fixture
    def comparisons(self):
        csv_path = GAMES_DIR / "game_01.csv"
        return asyncio.run(
            run_comparison_stockfish(csv_path, player_elo=820)
        )

    def test_stockfish_comparison_completes(self, comparisons):
        """Stockfish comparison produces results for all non-end moves."""
        assert len(comparisons) >= 55

    def test_stockfish_print_comparison(self, comparisons, capsys):
        """Print Stockfish comparison table for review."""
        print("\n[STOCKFISH MODE — real engine analysis]")
        print_comparison_table(comparisons)
        stats = compute_accuracy_stats(comparisons)
        print_accuracy_report(stats)
        print(f"\n>>> Stockfish accuracy at Elo 820: {stats['accuracy']:.1%}")

        captured = capsys.readouterr()
        assert "ACCURACY REPORT" in captured.out

    def test_stockfish_uses_native_wdl(self, comparisons):
        """At least some moves should have non-None WDL-derived win_pct."""
        with_wp = [c for c in comparisons if c.eval_before_wp is not None]
        assert len(with_wp) > 0

    def test_stockfish_mistakes_detected(self, comparisons):
        """chess.com mistakes/blunders should not be classified as best/excellent."""
        mistakes = [c for c in comparisons if c.chesscom_class in ("mistake", "blunder")]
        for c in mistakes:
            assert c.our_class not in ("best", "excellent"), (
                f"Ply {c.ply} {c.move_san}: chess.com='{c.chesscom_class}' "
                f"but Stockfish-based says '{c.our_class}' (ep_loss={c.ep_loss})"
            )

    def test_stockfish_report_accuracy(self, comparisons):
        """Report accuracy with real Stockfish — always passes, prints diagnostics."""
        stats = compute_accuracy_stats(comparisons)
        print_accuracy_report(stats)
        print(f"\n>>> Stockfish accuracy: {stats['accuracy']:.1%}")
        print(f">>> Far misses: {stats['far_misses']}")
