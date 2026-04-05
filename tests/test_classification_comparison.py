"""
Classification comparison tests.

The reusable harness lives in `tests/_classification_comparison.py` so the test
module can stay focused, while this file still re-exports the public helpers
used by `tests/optimize_classification.py`.
"""

import asyncio

import pytest

from tests._classification_comparison import (
    CLASS_SEVERITY,
    GAMES_DIR,
    ChessComMove,
    MoveComparison,
    STOCKFISH_AVAILABLE,
    compute_accuracy_stats,
    load_chesscom_csv,
    print_accuracy_report,
    print_comparison_table,
    run_comparison,
    run_comparison_stockfish,
)

__all__ = [
    "CLASS_SEVERITY",
    "GAMES_DIR",
    "ChessComMove",
    "MoveComparison",
    "compute_accuracy_stats",
    "load_chesscom_csv",
    "print_accuracy_report",
    "print_comparison_table",
    "run_comparison",
]


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
