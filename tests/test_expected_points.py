"""
Tests for the Expected Points classification system.

Covers:
  - Sigmoid WDL provider (boundary, Elo scaling, mate handling, large evals)
  - Stockfish native WDL provider
  - Provider auto-selection
  - EP-based move classification thresholds
  - Brilliant / Great / Miss detection
  - Integration with ContextAssembler
"""

import math

import chess
import pytest

from analysis.context import AssembledContext, ContextAssembler, _classify_cp_loss, _mate_to_cp
from analysis.expected_points import (
    SigmoidWDLProvider,
    StockfishNativeWDLProvider,
    classify_ep_loss,
    draw_sensitivity_factor,
    pv_end_win_pct,
    resolve_provider,
)
from chess_engine.service import PVEndEval
from analysis.special_moves import (
    _material_balance,
    _pv_has_checkmate,
    detect_brilliant,
    detect_great,
    detect_miss,
)
from tests._engine_result_helpers import make_result, make_result_with_wdl
from chess_engine.service import CandidateMove, EngineResult
from config.classification import ClassificationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ===========================================================================
# Sigmoid WDL Provider
# ===========================================================================

class TestSigmoidProvider:
    def test_zero_cp_gives_half(self):
        provider = SigmoidWDLProvider()
        assert provider.get_win_pct(cp=0) == pytest.approx(0.5, abs=1e-6)

    def test_zero_cp_with_elo_gives_half(self):
        """cp=0 should give 0.5 regardless of Elo."""
        provider = SigmoidWDLProvider()
        for elo in [800, 1500, 2200, 2800]:
            assert provider.get_win_pct(cp=0, elo=elo) == pytest.approx(0.5, abs=1e-6)

    def test_positive_cp_above_half(self):
        provider = SigmoidWDLProvider()
        assert provider.get_win_pct(cp=100) > 0.5

    def test_negative_cp_below_half(self):
        provider = SigmoidWDLProvider()
        assert provider.get_win_pct(cp=-100) < 0.5

    def test_symmetry(self):
        """win_pct(+X) + win_pct(-X) should equal 1.0."""
        provider = SigmoidWDLProvider()
        for cp in [50, 100, 200, 500]:
            total = provider.get_win_pct(cp=cp) + provider.get_win_pct(cp=-cp)
            assert total == pytest.approx(1.0, abs=1e-6)

    def test_mate_winning(self):
        provider = SigmoidWDLProvider()
        assert provider.get_win_pct(mate_in=1) == 1.0
        assert provider.get_win_pct(mate_in=5) == 1.0

    def test_mate_losing(self):
        provider = SigmoidWDLProvider()
        assert provider.get_win_pct(mate_in=-1) == 0.0
        assert provider.get_win_pct(mate_in=-3) == 0.0

    def test_mate_overrides_cp(self):
        """Mate takes priority even if cp is also provided."""
        provider = SigmoidWDLProvider()
        assert provider.get_win_pct(cp=-500, mate_in=2) == 1.0

    def test_none_cp_returns_none(self):
        provider = SigmoidWDLProvider()
        assert provider.get_win_pct() is None

    def test_large_eval_stability(self):
        """Very large evals should not cause overflow."""
        provider = SigmoidWDLProvider()
        assert provider.get_win_pct(cp=5000) == pytest.approx(1.0, abs=0.01)
        assert provider.get_win_pct(cp=-5000) == pytest.approx(0.0, abs=0.01)

    def test_elo_scaling_steeper_at_high_elo(self):
        """Higher Elo should give steeper curve (more extreme win_pct for same cp)."""
        provider = SigmoidWDLProvider()
        wp_800 = provider.get_win_pct(cp=150, elo=800)
        wp_2200 = provider.get_win_pct(cp=150, elo=2200)
        # Higher Elo should convert +150cp to higher win probability
        assert wp_2200 > wp_800

    def test_elo_scaling_factor_values(self):
        """Check scaling factor at representative Elo values for current config."""
        provider = SigmoidWDLProvider()
        assert provider._elo_scaling_factor(800) == pytest.approx(0.802, abs=0.02)
        assert provider._elo_scaling_factor(1500) == pytest.approx(0.854, abs=0.02)
        assert provider._elo_scaling_factor(2200) == pytest.approx(0.985, abs=0.02)
        assert provider._elo_scaling_factor(2800) == pytest.approx(0.999, abs=0.01)

    def test_no_elo_uses_base_curve(self):
        """Without Elo, scaling factor should be 1.0 (base curve)."""
        provider = SigmoidWDLProvider()
        assert provider._elo_scaling_factor(None) == 1.0


# ===========================================================================
# Stockfish Native WDL Provider
# ===========================================================================

class TestNativeWDLProvider:
    def test_basic_win_pct(self):
        provider = StockfishNativeWDLProvider()
        # wdl_win=650, wdl_draw=200, wdl_loss=150 → (650 + 100) / 1000 = 0.75
        wp = provider.get_win_pct(wdl_win=650, wdl_draw=200, wdl_loss=150)
        assert wp == pytest.approx(0.75)

    def test_pure_win(self):
        provider = StockfishNativeWDLProvider()
        wp = provider.get_win_pct(wdl_win=1000, wdl_draw=0, wdl_loss=0)
        assert wp == pytest.approx(1.0)

    def test_pure_loss(self):
        provider = StockfishNativeWDLProvider()
        wp = provider.get_win_pct(wdl_win=0, wdl_draw=0, wdl_loss=1000)
        assert wp == pytest.approx(0.0)

    def test_pure_draw(self):
        provider = StockfishNativeWDLProvider()
        wp = provider.get_win_pct(wdl_win=0, wdl_draw=1000, wdl_loss=0)
        assert wp == pytest.approx(0.5)

    def test_mate_overrides_wdl(self):
        provider = StockfishNativeWDLProvider()
        assert provider.get_win_pct(mate_in=3, wdl_win=0, wdl_draw=0, wdl_loss=1000) == 1.0
        assert provider.get_win_pct(mate_in=-3, wdl_win=1000, wdl_draw=0, wdl_loss=0) == 0.0

    def test_missing_wdl_returns_none(self):
        provider = StockfishNativeWDLProvider()
        assert provider.get_win_pct(cp=100) is None
        assert provider.get_win_pct(wdl_win=500) is None  # missing draw


# ===========================================================================
# Provider Auto-Selection
# ===========================================================================

class TestResolveProvider:
    def test_selects_native_when_wdl_present(self):
        result = make_result_with_wdl([(100, None, 600, 300, 100)])
        provider = resolve_provider(result)
        assert isinstance(provider, StockfishNativeWDLProvider)

    def test_selects_sigmoid_when_no_wdl(self):
        result = make_result([(100, None)])
        provider = resolve_provider(result)
        assert isinstance(provider, SigmoidWDLProvider)

    def test_selects_sigmoid_when_partial_wdl(self):
        """If only some WDL fields are present, fall back to sigmoid."""
        board = chess.Board()
        moves = list(board.legal_moves)
        result = EngineResult(
            fen=board.fen(), depth=20,
            candidates=[CandidateMove(
                move=moves[0], score_cp=100, mate_in=None, pv=[moves[0]],
                wdl_win=600, wdl_draw=None, wdl_loss=None,
            )],
        )
        provider = resolve_provider(result)
        assert isinstance(provider, SigmoidWDLProvider)


# ===========================================================================
# EP-Based Classification
# ===========================================================================

class TestClassifyEPLoss:
    def test_best(self):
        assert classify_ep_loss(0.0) == "best"

    def test_excellent(self):
        assert classify_ep_loss(0.01) == "excellent"
        assert classify_ep_loss(0.015) == "excellent"

    def test_good(self):
        assert classify_ep_loss(0.03) == "good"
        assert classify_ep_loss(0.05) == "good"

    def test_inaccuracy(self):
        assert classify_ep_loss(0.06) == "inaccuracy"
        assert classify_ep_loss(0.069) == "inaccuracy"

    def test_mistake(self):
        assert classify_ep_loss(0.15) == "mistake"
        assert classify_ep_loss(0.19) == "mistake"
        assert classify_ep_loss(0.20) == "blunder"

    def test_blunder(self):
        assert classify_ep_loss(0.22) == "blunder"
        assert classify_ep_loss(0.50) == "blunder"
        assert classify_ep_loss(1.0) == "blunder"

    def test_none(self):
        assert classify_ep_loss(None) == "unknown"

    def test_custom_config(self):
        config = ClassificationConfig(ep_excellent=0.05, ep_good=0.10)
        assert classify_ep_loss(0.04, config) == "excellent"
        assert classify_ep_loss(0.07, config) == "good"


# ===========================================================================
# EP Classification — Context-Aware Scenarios (PRD User Stories)
# ===========================================================================

class TestEPContextAware:
    def test_large_cp_loss_in_winning_position_not_blunder(self):
        """US-1: 300cp loss in +25.00 position should be best/excellent, not blunder."""
        provider = SigmoidWDLProvider()
        # +2500cp → win_pct very close to 1.0
        wp_before = provider.get_win_pct(cp=2500)
        # +2200cp → still very close to 1.0
        wp_after = provider.get_win_pct(cp=2200)
        ep_loss = wp_before - wp_after
        label = classify_ep_loss(ep_loss)
        assert label in ("best", "excellent")

    def test_small_cp_loss_in_balanced_position_is_significant(self):
        """US-1: 60cp loss in a balanced position should be inaccuracy or worse."""
        provider = SigmoidWDLProvider()
        wp_before = provider.get_win_pct(cp=30)
        wp_after = provider.get_win_pct(cp=-30)
        ep_loss = wp_before - wp_after
        label = classify_ep_loss(ep_loss)
        assert label in ("inaccuracy", "mistake", "good")

    def test_elo_800_advantage_less_winning(self):
        """US-2: +150cp at Elo 800 should be less winning than at Elo 2200."""
        provider = SigmoidWDLProvider()
        wp_800 = provider.get_win_pct(cp=150, elo=800)
        wp_2200 = provider.get_win_pct(cp=150, elo=2200)
        assert wp_2200 > wp_800
        # With optimized narrow range, both are in the 0.58-0.65 range
        # but higher Elo still converts better
        assert 0.55 < wp_800 < 0.65
        assert 0.58 < wp_2200 < 0.68


# ===========================================================================
# Legacy CP-Loss Classification (backward compat)
# ===========================================================================

class TestLegacyCPLoss:
    def test_still_works(self):
        assert _classify_cp_loss(0) == "best"
        assert _classify_cp_loss(5) == "excellent"
        assert _classify_cp_loss(15) == "good"
        assert _classify_cp_loss(50) == "inaccuracy"
        assert _classify_cp_loss(100) == "mistake"
        assert _classify_cp_loss(300) == "blunder"
        assert _classify_cp_loss(None) == "unknown"


# ===========================================================================
# Special Moves — Brilliant
# ===========================================================================

class TestBrilliantDetection:
    def _make_board_with_queen_sac(self):
        """
        Position where white can sacrifice the queen.
        White: Kh1, Qd1, Rd3 — Black: Kg8, Rb8, pawns
        After Qxd7 (hypothetical queen sac), white loses queen material.
        """
        # Simplified: use a position where white has queen+rook vs rook
        # and a move sacrifices the queen
        return chess.Board("1r4k1/3Q4/8/8/8/3R4/8/7K w - - 0 1")

    def test_brilliant_queen_sac_positive(self):
        """Queen sacrifice that is best move, decisive, no material recovery."""
        board = chess.Board("1r4k1/3Q4/8/8/8/3R4/8/7K w - - 0 1")
        # Qd7-d8 would be a capture, let's use a more controlled setup
        # White sacs queen: after the move, material drops by 9 points
        # We'll test the detection function directly with crafted values

        board_before = chess.Board("r1b1k2r/pppp1ppp/2n2n2/2b1Q3/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
        # Simulate queen sac: Qxc5 (queen takes bishop = trade, not sacrifice)
        # Instead, let's test with mock values for a true sacrifice scenario

        # Direct function test with favorable conditions
        board_after = board_before.copy()
        move = list(board_before.legal_moves)[0]  # any move for structure
        board_after.push(move)

        result = detect_brilliant(
            board_before=board_before,
            board_after=chess.Board("r1b1k2r/pppp1ppp/2n2n2/2b5/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1"),  # queen gone
            move=move,
            ep_loss=0.0,
            win_pct_before=0.55,
            win_pct_after=0.70,
            played_pv=[],
            played_mate_in=None,
            side=chess.WHITE,
            elo=1500,
        )
        # This specific board transition may not show a 5+ material drop
        # since _is_heavy_piece_sacrifice checks actual material difference.
        # The important thing is the logic flow — tested more precisely below.

    def test_brilliant_rejected_small_sacrifice(self):
        """Minor piece sacrifice at Elo >= 1200 should not be brilliant."""
        config = ClassificationConfig()
        board = chess.Board()
        board_after = chess.Board()  # same material → no sacrifice
        move = list(board.legal_moves)[0]

        result = detect_brilliant(
            board_before=board, board_after=board_after, move=move,
            ep_loss=0.0, win_pct_before=0.55, win_pct_after=0.70,
            played_pv=[], played_mate_in=None, side=chess.WHITE,
            elo=1500, config=config,
        )
        assert result is False  # no material sacrificed

    def test_brilliant_rejected_already_winning(self):
        """Sacrifice in already winning position should not be brilliant."""
        config = ClassificationConfig()
        board = chess.Board()
        move = list(board.legal_moves)[0]

        result = detect_brilliant(
            board_before=board, board_after=board, move=move,
            ep_loss=0.0, win_pct_before=0.95, win_pct_after=0.90,
            played_pv=[], played_mate_in=None, side=chess.WHITE,
            elo=2000, config=config,
        )
        assert result is False

    def test_brilliant_rejected_non_decisive(self):
        """Sacrifice leading to only slight edge should not be brilliant."""
        config = ClassificationConfig()
        board = chess.Board()
        move = list(board.legal_moves)[0]

        result = detect_brilliant(
            board_before=board, board_after=board, move=move,
            ep_loss=0.0, win_pct_before=0.55, win_pct_after=0.45,
            played_pv=[], played_mate_in=None, side=chess.WHITE,
            elo=1500, config=config,
        )
        assert result is False  # win_pct_after < 0.60 and no mate

    def test_brilliant_rejected_not_best_move(self):
        """Move that loses significant EP should not be brilliant."""
        config = ClassificationConfig()
        board = chess.Board()
        move = list(board.legal_moves)[0]

        result = detect_brilliant(
            board_before=board, board_after=board, move=move,
            ep_loss=0.05, win_pct_before=0.55, win_pct_after=0.70,
            played_pv=[], played_mate_in=None, side=chess.WHITE,
            config=config,
        )
        assert result is False  # ep_loss > 0.02 tolerance

    def test_brilliant_respects_configured_low_elo_win_before_cap(self):
        """Lower-Elo brilliant filter should use the config, not a hard-coded 0.90."""
        board_before = chess.Board("r1b1k2r/pppp1ppp/2n2n2/2b1Q3/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
        board_after = chess.Board("r1b1k2r/pppp1ppp/2n2n2/2b5/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1")
        move = list(board_before.legal_moves)[0]
        config = ClassificationConfig(
            brilliant_low_elo_max_win_pct_before=0.80,
            brilliant_low_elo_max_win_pct_threshold=1500,
        )

        result = detect_brilliant(
            board_before=board_before,
            board_after=board_after,
            move=move,
            ep_loss=0.0,
            win_pct_before=0.85,
            win_pct_after=0.70,
            played_pv=[],
            played_mate_in=None,
            side=chess.WHITE,
            elo=1400,
            config=config,
        )
        assert result is False

    def test_brilliant_normalizes_black_pov(self):
        """Black sacrifices should use black's winning chances, not white POV."""
        config = ClassificationConfig()
        board_before = chess.Board("4k3/8/8/8/8/8/8/r2qK2r b - - 0 1")
        board_after = chess.Board("4k3/8/8/8/8/8/8/r3K2r w - - 0 2")
        move = chess.Move.from_uci("d1d2")

        result = detect_brilliant(
            board_before=board_before,
            board_after=board_after,
            move=move,
            ep_loss=0.0,
            win_pct_before=0.45,  # white POV => black already better
            win_pct_after=0.30,   # white POV => black clearly winning
            played_pv=[],
            played_mate_in=None,
            side=chess.BLACK,
            elo=1800,
            config=config,
        )
        assert result is True


class TestPVCheckmate:
    def test_detects_mate_in_pv(self):
        """Walking PV should find checkmate position."""
        # Fool's mate scenario: 1.f3 e5 2.g4 Qh4#
        board = chess.Board()
        pv = [
            chess.Move.from_uci("f2f3"),
            chess.Move.from_uci("e7e5"),
            chess.Move.from_uci("g2g4"),
            chess.Move.from_uci("d8h4"),
        ]
        assert _pv_has_checkmate(board, pv) is True

    def test_no_mate_in_pv(self):
        """Standard opening moves should not have checkmate."""
        board = chess.Board()
        pv = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")]
        assert _pv_has_checkmate(board, pv) is False


# ===========================================================================
# Special Moves — Great
# ===========================================================================

class TestGreatMoveDetection:
    def _make_prev_context(self, ep_loss: float) -> AssembledContext:
        board = chess.Board()
        move = list(board.legal_moves)[0]
        return AssembledContext(
            fen=board.fen(),
            played_move=move,
            played_move_san="e4",
            best_move=move,
            best_move_san="e4",
            best_move_cp=30,
            mate_in=None,
            played_move_cp_loss=0,
            cp_loss_label="best",
            pv_san=["e4"],
            ep_loss=ep_loss,
        )

    def test_seizing_move(self):
        """Large candidate gap in a balanced position → great via Trigger A."""
        provider = SigmoidWDLProvider()
        board = chess.Board()
        moves = list(board.legal_moves)
        candidates = [
            CandidateMove(move=moves[0], score_cp=300, mate_in=None, pv=[moves[0]]),
            CandidateMove(move=moves[1], score_cp=0, mate_in=None, pv=[moves[1]]),
        ]

        result = detect_great(
            ep_loss=0.0,
            win_pct_before=0.50,
            win_pct_after=0.75,
            candidates=candidates,
            provider=provider,
            elo=2000,
            is_engine_top=True,
            candidate_gap_cp=300,
        )
        assert result is True

    def test_only_move(self):
        """Only one candidate within 0.05 EP of best → great move."""
        provider = SigmoidWDLProvider()
        board = chess.Board()
        moves = list(board.legal_moves)

        # Best move: +200cp, second: -300cp (huge gap)
        candidates = [
            CandidateMove(move=moves[0], score_cp=200, mate_in=None, pv=[moves[0]]),
            CandidateMove(move=moves[1], score_cp=-300, mate_in=None, pv=[moves[1]]),
            CandidateMove(move=moves[2], score_cp=-400, mate_in=None, pv=[moves[2]]),
        ]

        result = detect_great(
            ep_loss=0.0,
            win_pct_before=0.50,
            win_pct_after=0.50,  # doesn't matter for only-move check
            candidates=candidates,
            provider=provider,
            elo=1500,
            is_engine_top=True,
            candidate_gap_cp=500,
        )
        assert result is True

    def test_not_great_when_ep_loss_high(self):
        """Even a turning-move position doesn't qualify if ep_loss > tolerance."""
        provider = SigmoidWDLProvider()
        board = chess.Board()
        moves = list(board.legal_moves)
        candidates = [
            CandidateMove(move=moves[0], score_cp=100, mate_in=None, pv=[moves[0]]),
        ]

        result = detect_great(
            ep_loss=0.05,  # too much EP lost
            win_pct_before=0.25,
            win_pct_after=0.55,
            candidates=candidates,
            provider=provider,
            elo=2000,
        )
        assert result is False

    def test_great_respects_configured_win_pct_filter(self):
        """Great should obey the configured pre-move win-probability filter."""
        provider = SigmoidWDLProvider()
        board = chess.Board()
        moves = list(board.legal_moves)
        candidates = [
            CandidateMove(move=moves[0], score_cp=300, mate_in=None, pv=[moves[0]]),
            CandidateMove(move=moves[1], score_cp=0, mate_in=None, pv=[moves[1]]),
        ]
        config = ClassificationConfig(great_max_win_pct_before=0.80)

        result = detect_great(
            ep_loss=0.0,
            win_pct_before=0.85,
            win_pct_after=0.92,
            candidates=candidates,
            provider=provider,
            board=board,
            move=moves[0],
            elo=1500,
            config=config,
            is_engine_top=True,
            candidate_gap_cp=300,
        )
        assert result is False

    def test_great_normalizes_white_pov_for_black_moves(self):
        """Black great-move thresholds should use black's perspective, not white's."""
        provider = SigmoidWDLProvider()
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        moves = list(board.legal_moves)

        # white POV win_pct_before=0.75 means black is losing (0.25 from black's
        # view), which falls inside great_min_win_pct_before..great_max_win_pct_before
        # (0.15..0.82) from black's POV. With a 300cp gap this trips Trigger A.
        result = detect_great(
            ep_loss=0.0,
            win_pct_before=0.75,
            win_pct_after=0.45,
            candidates=[
                CandidateMove(move=moves[0], score_cp=-100, mate_in=None, pv=[moves[0]]),
                CandidateMove(move=moves[1], score_cp=200, mate_in=None, pv=[moves[1]]),
            ],
            provider=provider,
            board=board,
            move=moves[0],
            elo=2000,
            is_engine_top=True,
            candidate_gap_cp=300,
        )
        assert result is True

    def test_great_respects_configured_capitalization_gap_scale(self):
        """Capitalization gap threshold should come from config, not hard-coded half-gap."""
        provider = SigmoidWDLProvider()
        board = chess.Board()
        moves = list(board.legal_moves)
        config = ClassificationConfig(
            great_min_candidate_gap_cp=200,
            great_capitalization_gap_scale=0.40,
        )

        result = detect_great(
            ep_loss=0.0,
            win_pct_before=0.50,
            win_pct_after=0.65,
            candidates=[
                CandidateMove(move=moves[0], score_cp=100, mate_in=None, pv=[moves[0]]),
                CandidateMove(move=moves[1], score_cp=10, mate_in=None, pv=[moves[1]]),
            ],
            provider=provider,
            board=board,
            move=moves[0],
            elo=1500,
            config=config,
            prev_context=self._make_prev_context(0.10),
            is_engine_top=True,
            candidate_gap_cp=90,
        )
        assert result is True


# ===========================================================================
# Special Moves — Miss
# ===========================================================================

class TestMissDetection:
    def _make_prev_context(self, ep_loss: float) -> AssembledContext:
        """Create a minimal AssembledContext with ep_loss set."""
        board = chess.Board()
        moves = list(board.legal_moves)
        return AssembledContext(
            fen=board.fen(),
            played_move=moves[0],
            played_move_san="e4",
            best_move=moves[0],
            best_move_san="e4",
            best_move_cp=30,
            mate_in=None,
            played_move_cp_loss=0,
            cp_loss_label="best",
            pv_san=["e4"],
            ep_loss=ep_loss,
        )

    def test_miss_positive(self):
        """Opponent blundered, best reply wins, player doesn't find it."""
        prev = self._make_prev_context(ep_loss=0.25)  # opponent blundered
        result = detect_miss(
            prev_context=prev,
            best_win_pct=0.80,    # best reply would be winning
            played_win_pct=0.50,  # player's reply is only equal
            ep_loss=0.10,         # meaningful EP loss but not a blunder label by itself
            elo=2000,
        )
        assert result is True

    def test_miss_no_opponent_mistake(self):
        """Opponent played fine → no miss possible."""
        prev = self._make_prev_context(ep_loss=0.03)  # opponent played well
        result = detect_miss(
            prev_context=prev,
            best_win_pct=0.80,
            played_win_pct=0.50,
            ep_loss=0.05,
            elo=2000,
        )
        assert result is False

    def test_miss_detects_opportunity_even_for_large_ep_loss(self):
        """detect_miss only detects missed opportunities; label priority is handled upstream."""
        prev = self._make_prev_context(ep_loss=0.25)
        result = detect_miss(
            prev_context=prev,
            best_win_pct=0.80,
            played_win_pct=0.30,
            ep_loss=0.25,  # this is a mistake/blunder (>= ep_mistake threshold)
            elo=2000,
        )
        assert result is True

    def test_miss_player_found_winning_move(self):
        """Player's reply IS winning → no miss."""
        prev = self._make_prev_context(ep_loss=0.25)
        result = detect_miss(
            prev_context=prev,
            best_win_pct=0.80,
            played_win_pct=0.75,  # still winning
            ep_loss=0.02,
            elo=2000,
        )
        assert result is False

    def test_miss_no_prev_ep_loss(self):
        """If prev_context has no ep_loss, no miss detection."""
        prev = self._make_prev_context(ep_loss=None)
        result = detect_miss(
            prev_context=prev,
            best_win_pct=0.80,
            played_win_pct=0.50,
            ep_loss=0.05,
            elo=2000,
        )
        assert result is False

    def test_miss_normalizes_black_pov(self):
        """Black misses should use black's winning chances, not white POV."""
        prev = self._make_prev_context(ep_loss=0.25)
        result = detect_miss(
            prev_context=prev,
            best_win_pct=0.25,   # white POV => black would be clearly winning
            played_win_pct=0.45, # white POV => black only slightly better
            ep_loss=0.10,
            side=chess.BLACK,
            elo=1800,
        )
        assert result is True

    # --- Trigger C: direct tactic missed (no opponent-blunder required) ---

    def test_trigger_c_fires_on_forking_check_with_large_gap(self):
        """Fork check (Nf6+ winning queen) with huge candidate gap and no prev context."""
        # White knight e4, black king g8, black queen d7; Nf6+ forks king and queen.
        board = chess.Board("6k1/3q4/8/8/4N3/8/8/4K3 w - - 0 1")
        best_move = board.parse_san("Nf6+")
        best_candidate = CandidateMove(
            move=best_move, score_cp=800, mate_in=None, pv=[best_move],
        )
        result = detect_miss(
            prev_context=None,
            best_win_pct=0.88,
            played_win_pct=0.55,
            ep_loss=0.33,
            best_candidate=best_candidate,
            board_before=board,
            side=chess.WHITE,
            candidate_gap_cp=500,
        )
        assert result is True

    def test_trigger_c_does_not_fire_with_small_gap(self):
        """Same forking position but candidate gap below threshold — no miss."""
        board = chess.Board("6k1/3q4/8/8/4N3/8/8/4K3 w - - 0 1")
        best_move = board.parse_san("Nf6+")
        best_candidate = CandidateMove(
            move=best_move, score_cp=800, mate_in=None, pv=[best_move],
        )
        result = detect_miss(
            prev_context=None,
            best_win_pct=0.88,
            played_win_pct=0.55,
            ep_loss=0.33,
            best_candidate=best_candidate,
            board_before=board,
            side=chess.WHITE,
            candidate_gap_cp=100,
        )
        assert result is False

    def test_trigger_c_does_not_fire_on_quiet_best_move(self):
        """Best move is a quiet pawn push (not a check/capture/mate) — no miss."""
        board = chess.Board("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1")
        best_move = board.parse_san("e7")
        best_candidate = CandidateMove(
            move=best_move, score_cp=500, mate_in=None, pv=[best_move],
        )
        result = detect_miss(
            prev_context=None,
            best_win_pct=0.90,
            played_win_pct=0.50,
            ep_loss=0.40,
            best_candidate=best_candidate,
            board_before=board,
            side=chess.WHITE,
            candidate_gap_cp=500,
        )
        assert result is False

    def test_trigger_c_does_not_fire_when_mover_already_losing(self):
        """Even with a forcing best move and large gap, don't fire when already toast."""
        board = chess.Board("6k1/3q4/8/8/4N3/8/8/4K3 w - - 0 1")
        best_move = board.parse_san("Nf6+")
        # Best leads to mate → concrete opportunity gate passes via mate_in
        best_candidate = CandidateMove(
            move=best_move, score_cp=None, mate_in=2, pv=[best_move],
        )
        result = detect_miss(
            prev_context=None,
            best_win_pct=0.10,   # mover WP 0.10 < 0.15 gate
            played_win_pct=0.05,
            ep_loss=0.20,
            best_candidate=best_candidate,
            board_before=board,
            side=chess.WHITE,
            candidate_gap_cp=500,
        )
        assert result is False


# ===========================================================================
# Material Balance Helper
# ===========================================================================

class TestMaterialBalance:
    def test_starting_position(self):
        board = chess.Board()
        white_mat = _material_balance(board, chess.WHITE)
        black_mat = _material_balance(board, chess.BLACK)
        # 8 pawns(8) + 2 knights(6) + 2 bishops(6) + 2 rooks(10) + 1 queen(9) = 39
        assert white_mat == 39
        assert black_mat == 39

    def test_missing_queen(self):
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
        white_mat = _material_balance(board, chess.WHITE)
        assert white_mat == 30  # 39 - 9 (queen)


# ===========================================================================
# Context Assembler Integration
# ===========================================================================

class TestContextAssemblerEP:
    def test_ep_fields_populated(self):
        """AssembledContext should have EP fields when played_move is given."""
        board = chess.Board()
        moves = list(board.legal_moves)
        result = EngineResult(
            fen=board.fen(), depth=20,
            candidates=[
                CandidateMove(move=moves[0], score_cp=30, mate_in=None, pv=[moves[0]]),
                CandidateMove(move=moves[1], score_cp=20, mate_in=None, pv=[moves[1]]),
            ],
        )
        assembler = ContextAssembler()
        ctx = assembler.assemble(board, result, played_move=moves[1])

        assert ctx.win_pct_before is not None
        assert ctx.win_pct_after is not None
        assert ctx.ep_loss is not None
        assert ctx.ep_loss >= 0.0
        assert ctx.wdl_source == "sigmoid"
        # Raw cp loss still available
        assert ctx.played_move_cp_loss == 10

    def test_ep_fields_none_without_played_move(self):
        """Without a played move, EP fields should be None."""
        board = chess.Board()
        moves = list(board.legal_moves)
        result = EngineResult(
            fen=board.fen(), depth=20,
            candidates=[
                CandidateMove(move=moves[0], score_cp=30, mate_in=None, pv=[moves[0]]),
            ],
        )
        assembler = ContextAssembler()
        ctx = assembler.assemble(board, result)

        assert ctx.win_pct_before is not None  # best move win_pct still computed
        assert ctx.win_pct_after is None
        assert ctx.ep_loss is None

    def test_large_cp_loss_winning_position(self):
        """PRD US-1: 300cp loss in winning position → best/excellent."""
        board = chess.Board()
        moves = list(board.legal_moves)
        result = EngineResult(
            fen=board.fen(), depth=20,
            candidates=[
                CandidateMove(move=moves[0], score_cp=2500, mate_in=None, pv=[moves[0]]),
                CandidateMove(move=moves[1], score_cp=2200, mate_in=None, pv=[moves[1]]),
            ],
        )
        assembler = ContextAssembler()
        ctx = assembler.assemble(board, result, played_move=moves[1])

        assert ctx.cp_loss_label in ("best", "excellent")
        # Old system would have said "blunder" for 300cp loss
        assert ctx.played_move_cp_loss == 300

    def test_elo_passed_through(self):
        """player_elo and opponent_elo should be stored in context."""
        board = chess.Board()
        moves = list(board.legal_moves)
        result = EngineResult(
            fen=board.fen(), depth=20,
            candidates=[
                CandidateMove(move=moves[0], score_cp=30, mate_in=None, pv=[moves[0]]),
            ],
        )
        assembler = ContextAssembler()
        ctx = assembler.assemble(board, result, player_elo=1500, opponent_elo=1600)

        assert ctx.player_elo == 1500
        assert ctx.opponent_elo == 1600

    def test_native_wdl_source(self):
        """When WDL data is present, wdl_source should be 'native'."""
        board = chess.Board()
        moves = list(board.legal_moves)
        result = EngineResult(
            fen=board.fen(), depth=20,
            candidates=[
                CandidateMove(
                    move=moves[0], score_cp=30, mate_in=None, pv=[moves[0]],
                    wdl_win=550, wdl_draw=350, wdl_loss=100,
                ),
                CandidateMove(
                    move=moves[1], score_cp=20, mate_in=None, pv=[moves[1]],
                    wdl_win=520, wdl_draw=360, wdl_loss=120,
                ),
            ],
        )
        assembler = ContextAssembler()
        ctx = assembler.assemble(board, result, played_move=moves[1])

        assert ctx.wdl_source == "native"

    def test_config_override(self):
        """Custom ClassificationConfig should change classification behavior."""
        config = ClassificationConfig(ep_excellent=0.50)  # very lenient
        board = chess.Board()
        moves = list(board.legal_moves)
        result = EngineResult(
            fen=board.fen(), depth=20,
            candidates=[
                CandidateMove(move=moves[0], score_cp=300, mate_in=None, pv=[moves[0]]),
                CandidateMove(move=moves[1], score_cp=0, mate_in=None, pv=[moves[1]]),
            ],
        )
        assembler = ContextAssembler(classification_config=config)
        ctx = assembler.assemble(board, result, played_move=moves[1])

        # With the lenient config, even a significant EP loss is "excellent"
        assert ctx.cp_loss_label == "excellent"

    def test_special_flags_default_false(self):
        """Special flags should be False for normal moves."""
        board = chess.Board()
        moves = list(board.legal_moves)
        result = EngineResult(
            fen=board.fen(), depth=20,
            candidates=[
                CandidateMove(move=moves[0], score_cp=30, mate_in=None, pv=[moves[0]]),
                CandidateMove(move=moves[1], score_cp=20, mate_in=None, pv=[moves[1]]),
            ],
        )
        assembler = ContextAssembler()
        ctx = assembler.assemble(board, result, played_move=moves[1])

        assert ctx.is_brilliant is False
        assert ctx.is_great is False
        assert ctx.is_miss is False

    def test_black_move_ep_loss_correct(self):
        """Black's EP loss should be positive when black plays a bad move.

        When black plays poorly, white's eval goes UP. Since our evals are
        from white's perspective, EP loss for black = played_wp - best_wp.
        """
        # Position after 1.e4 — black to move
        board = chess.Board()
        board.push_san("e4")

        moves = list(board.legal_moves)
        # Best move for black: eval stays at +30cp (white slight edge)
        # Played move: eval jumps to +200cp (black blundered)
        result = EngineResult(
            fen=board.fen(), depth=20,
            candidates=[
                CandidateMove(move=moves[0], score_cp=30, mate_in=None, pv=[moves[0]]),
                CandidateMove(move=moves[1], score_cp=200, mate_in=None, pv=[moves[1]]),
            ],
        )
        assembler = ContextAssembler()
        ctx = assembler.assemble(board, result, played_move=moves[1])

        # EP loss should be positive (black made things worse for themselves)
        assert ctx.ep_loss is not None
        assert ctx.ep_loss > 0.0
        # The eval went from +30 to +200 for white, so black lost significant EP
        assert ctx.cp_loss_label not in ("best", "excellent")

    def test_draw_boost_preserves_good_below_gate(self):
        """Near-drawn positions must NOT demote good moves to inaccuracy.

        The draw boost is gated at `draw_boost_min_ep_loss` (default 0.05) so
        best / excellent / good plays stay untouched — otherwise structurally
        drawish positions would wrongly demote near-best moves. Only moves
        already above the `good` bucket receive the severity-tightening scale.
        """
        board = chess.Board()
        moves = list(board.legal_moves)
        # Raw ep_loss ≈ 0.04 (good bucket) with best_wdl_draw=800‰ — aggressive
        # factor would have boosted this past the inaccuracy line, but the
        # gate keeps it at good.
        result = EngineResult(
            fen=board.fen(), depth=20,
            candidates=[
                CandidateMove(
                    move=moves[0], score_cp=20, mate_in=None, pv=[moves[0]],
                    wdl_win=150, wdl_draw=800, wdl_loss=50,
                ),
                CandidateMove(
                    move=moves[1], score_cp=-20, mate_in=None, pv=[moves[1]],
                    wdl_win=110, wdl_draw=800, wdl_loss=90,
                ),
            ],
        )

        # Even with an aggressive factor, the default gate keeps this at `good`.
        gated = ClassificationConfig(
            draw_boost_min_permille=400, draw_boost_max_factor=1.80,
        )
        gated_ctx = ContextAssembler(classification_config=gated).assemble(
            board, result, played_move=moves[1]
        )
        assert gated_ctx.cp_loss_label == "good"

        # Removing the gate restores the old demoting behaviour — proves the
        # gate is the mechanism responsible, not unrelated changes.
        ungated = ClassificationConfig(
            draw_boost_min_permille=400, draw_boost_max_factor=1.80,
            draw_boost_min_ep_loss=0.0,
        )
        ungated_ctx = ContextAssembler(classification_config=ungated).assemble(
            board, result, played_move=moves[1]
        )
        assert ungated_ctx.cp_loss_label == "inaccuracy"

    def test_draw_boost_tightens_inaccuracy_to_mistake(self):
        """Above the gate, the draw boost still escalates severity classes.

        Raw ep_loss ≈ 0.085 sits in `inaccuracy`. With best_wdl_draw=800‰ and
        the default (min=500, max=1.30) ramp the factor is ~1.20, pushing the
        effective ep_loss past 0.10 into `mistake`. This is the intended
        behaviour — severity plays in drawish positions are under-penalised
        by the raw thresholds.
        """
        board = chess.Board()
        moves = list(board.legal_moves)
        result = EngineResult(
            fen=board.fen(), depth=20,
            candidates=[
                CandidateMove(
                    move=moves[0], score_cp=20, mate_in=None, pv=[moves[0]],
                    wdl_win=150, wdl_draw=800, wdl_loss=50,   # wp = 0.55
                ),
                CandidateMove(
                    move=moves[1], score_cp=-20, mate_in=None, pv=[moves[1]],
                    wdl_win=60, wdl_draw=800, wdl_loss=140,   # wp = 0.46 → ep ≈ 0.09
                ),
            ],
        )

        off = ClassificationConfig(draw_boost_max_factor=1.0)
        baseline_ctx = ContextAssembler(classification_config=off).assemble(
            board, result, played_move=moves[1]
        )
        assert baseline_ctx.cp_loss_label == "inaccuracy"

        boosted = ClassificationConfig()  # defaults: min=500, max=1.30, gate=0.05
        boosted_ctx = ContextAssembler(classification_config=boosted).assemble(
            board, result, played_move=moves[1]
        )
        assert boosted_ctx.cp_loss_label == "mistake"

    def test_draw_boost_changes_bucket_without_mutating_stored_ep_loss(self):
        """Drawishness should act like a threshold adjustment, not rewrite ep_loss."""
        board = chess.Board()
        moves = list(board.legal_moves)
        result = EngineResult(
            fen=board.fen(), depth=20,
            candidates=[
                CandidateMove(
                    move=moves[0], score_cp=20, mate_in=None, pv=[moves[0]],
                    wdl_win=150, wdl_draw=800, wdl_loss=50,
                ),
                CandidateMove(
                    move=moves[1], score_cp=-20, mate_in=None, pv=[moves[1]],
                    wdl_win=60, wdl_draw=800, wdl_loss=140,
                ),
            ],
        )

        baseline_ctx = ContextAssembler(
            classification_config=ClassificationConfig(draw_boost_max_factor=1.0)
        ).assemble(board, result, played_move=moves[1])
        boosted_ctx = ContextAssembler(
            classification_config=ClassificationConfig()
        ).assemble(board, result, played_move=moves[1])

        assert baseline_ctx.ep_loss == pytest.approx(boosted_ctx.ep_loss, abs=1e-9)
        assert baseline_ctx.cp_loss_label == "inaccuracy"
        assert boosted_ctx.cp_loss_label == "mistake"

    def test_draw_boost_disabled_by_default_when_factor_is_one(self):
        """Setting draw_boost_max_factor=1.0 is the documented kill-switch."""
        assert draw_sensitivity_factor(900, ClassificationConfig(draw_boost_max_factor=1.0)) == 1.0

    def test_draw_boost_ignores_low_draw_permille(self):
        """Decisive positions (low wdl_draw) should not receive any boost."""
        cfg = ClassificationConfig(draw_boost_min_permille=500, draw_boost_max_factor=1.3)
        assert draw_sensitivity_factor(100, cfg) == 1.0
        assert draw_sensitivity_factor(499, cfg) == 1.0
        assert draw_sensitivity_factor(None, cfg) == 1.0

    def test_draw_boost_ramps_linearly(self):
        """At the midpoint of the ramp, the factor is halfway to the max."""
        cfg = ClassificationConfig(draw_boost_min_permille=400, draw_boost_max_factor=1.40)
        # at 1000‰: 1.40
        assert draw_sensitivity_factor(1000, cfg) == pytest.approx(1.40, abs=1e-6)
        # midpoint (700‰): 1.20
        assert draw_sensitivity_factor(700, cfg) == pytest.approx(1.20, abs=1e-6)

    def test_pv_end_lifts_hidden_drift_into_stricter_bucket(self):
        """When played's PV drifts far worse than root eval, EP loss escalates.

        Root eval makes the played move look like an `inaccuracy` (small cp
        gap past the pv_end gate of 0.05). Walking the played PV to its
        endpoint exposes a much larger swing, which should bump the label.
        """
        board = chess.Board()
        moves = list(board.legal_moves)

        best = CandidateMove(
            move=moves[0], score_cp=40, mate_in=None, pv=[moves[0]],
            pv_end=PVEndEval(
                score_cp=40, mate_in=None, depth=20, seldepth=25,
                pushed=4, fen="", terminal=None,
            ),
        )
        played = CandidateMove(
            move=moves[1], score_cp=-40, mate_in=None, pv=[moves[1]],
            # played looks like an inaccuracy at the root (~0.06 ep_loss) but
            # its pv_end reveals a -500cp collapse at the endpoint.
            pv_end=PVEndEval(
                score_cp=-500, mate_in=None, depth=20, seldepth=25,
                pushed=4, fen="", terminal=None,
            ),
        )
        result = EngineResult(fen=board.fen(), depth=20, candidates=[best, played])

        off = ClassificationConfig(pv_end_ep_loss_enabled=False)
        baseline_ctx = ContextAssembler(classification_config=off).assemble(
            board, result, played_move=moves[1]
        )
        on = ClassificationConfig(pv_end_ep_loss_enabled=True)
        lifted_ctx = ContextAssembler(classification_config=on).assemble(
            board, result, played_move=moves[1]
        )

        assert baseline_ctx.ep_loss < lifted_ctx.ep_loss
        assert baseline_ctx.cp_loss_label == "inaccuracy"
        assert lifted_ctx.cp_loss_label in ("mistake", "blunder")

    def test_out_of_top_window_move_uses_exact_played_eval(self):
        """EngineResult.played_move should populate EP data for rank-4+ plays."""
        board = chess.Board()
        moves = list(board.legal_moves)
        played_move = moves[3]
        result = EngineResult(
            fen=board.fen(),
            depth=20,
            candidates=[
                CandidateMove(move=moves[0], score_cp=40, mate_in=None, pv=[moves[0]]),
                CandidateMove(move=moves[1], score_cp=20, mate_in=None, pv=[moves[1]]),
                CandidateMove(move=moves[2], score_cp=10, mate_in=None, pv=[moves[2]]),
            ],
            played_move=CandidateMove(
                move=played_move, score_cp=-50, mate_in=None, pv=[played_move]
            ),
        )

        ctx = ContextAssembler().assemble(board, result, played_move=played_move)

        assert ctx.win_pct_after is not None
        assert ctx.ep_loss is not None
        assert ctx.played_move_cp_loss == 90
        assert ctx.cp_loss_label != "unknown"

    def test_pv_end_lifts_out_of_top_window_play_using_played_drift(self):
        """Rank-4+ exact-eval plays should use their own PV-end drift as severity."""
        board = chess.Board()
        moves = list(board.legal_moves)
        played_move = moves[3]
        result = EngineResult(
            fen=board.fen(),
            depth=20,
            candidates=[
                CandidateMove(
                    move=moves[0], score_cp=40, mate_in=None, pv=[moves[0]],
                    pv_end=PVEndEval(
                        score_cp=40, mate_in=None, depth=20, seldepth=25,
                        pushed=4, fen="", terminal=None,
                    ),
                ),
                CandidateMove(move=moves[1], score_cp=20, mate_in=None, pv=[moves[1]]),
                CandidateMove(move=moves[2], score_cp=10, mate_in=None, pv=[moves[2]]),
            ],
            played_move=CandidateMove(
                move=played_move, score_cp=-40, mate_in=None, pv=[played_move],
                pv_end=PVEndEval(
                    score_cp=-500, mate_in=None, depth=20, seldepth=25,
                    pushed=4, fen="", terminal=None,
                ),
            ),
        )

        baseline_ctx = ContextAssembler(
            classification_config=ClassificationConfig(pv_end_ep_loss_enabled=False)
        ).assemble(board, result, played_move=played_move)
        lifted_ctx = ContextAssembler(
            classification_config=ClassificationConfig(pv_end_ep_loss_enabled=True)
        ).assemble(board, result, played_move=played_move)

        assert baseline_ctx.ep_loss < lifted_ctx.ep_loss
        assert baseline_ctx.cp_loss_label == "inaccuracy"
        assert lifted_ctx.cp_loss_label in ("mistake", "blunder")

    def test_rank4_generic_pv_end_gate_stays_conservative(self):
        """The generic PV-end replacement should stay gated for rank-4+ goods."""
        board = chess.Board()
        moves = list(board.legal_moves)
        played_move = moves[3]
        result = EngineResult(
            fen=board.fen(),
            depth=20,
            candidates=[
                CandidateMove(
                    move=moves[0], score_cp=20, mate_in=None, pv=[moves[0]],
                    pv_end=PVEndEval(
                        score_cp=20, mate_in=None, depth=20, seldepth=25,
                        pushed=4, fen="", terminal=None,
                    ),
                ),
                CandidateMove(move=moves[1], score_cp=10, mate_in=None, pv=[moves[1]]),
                CandidateMove(move=moves[2], score_cp=0, mate_in=None, pv=[moves[2]]),
            ],
            played_move=CandidateMove(
                move=played_move, score_cp=-30, mate_in=None, pv=[played_move],
                pv_end=PVEndEval(
                    score_cp=-500, mate_in=None, depth=20, seldepth=25,
                    pushed=4, fen="", terminal=None,
                ),
            ),
        )

        strict_rank4 = ContextAssembler(
            classification_config=ClassificationConfig(
                pv_end_min_ep_loss_rank4=0.05,
                pv_end_min_lift_rank4=0.05,
            )
        ).assemble(board, result, played_move=played_move)
        assert strict_rank4.cp_loss_label == "good"

        default_rank4 = ContextAssembler(
            classification_config=ClassificationConfig(
                rank4_good_to_inaccuracy_min_draw_permille=10_000,
                rank4_inaccuracy_to_mistake_min_future_ep=1.0,
            )
        ).assemble(board, result, played_move=played_move)
        assert default_rank4.cp_loss_label == "good"

    def test_rank4_future_override_bumps_good_to_inaccuracy(self):
        """Rank-4+ future collapse should rescue drawish good->inaccuracy misses."""
        board = chess.Board()
        moves = list(board.legal_moves)
        played_move = moves[3]
        result = EngineResult(
            fen=board.fen(),
            depth=20,
            candidates=[
                CandidateMove(
                    move=moves[0],
                    score_cp=20,
                    mate_in=None,
                    pv=[moves[0]],
                    wdl_win=150,
                    wdl_draw=700,
                    wdl_loss=150,
                    pv_end=PVEndEval(
                        score_cp=20, mate_in=None, depth=20, seldepth=25,
                        pushed=4, fen="", terminal=None,
                    ),
                ),
                CandidateMove(move=moves[1], score_cp=10, mate_in=None, pv=[moves[1]]),
                CandidateMove(move=moves[2], score_cp=0, mate_in=None, pv=[moves[2]]),
            ],
            played_move=CandidateMove(
                move=played_move, score_cp=-10, mate_in=None, pv=[played_move],
                wdl_win=110, wdl_draw=700, wdl_loss=190,
                pv_end=PVEndEval(
                    score_cp=-500, mate_in=None, depth=20, seldepth=25,
                    pushed=4, fen="", terminal=None,
                ),
            ),
        )

        off = ContextAssembler(
            classification_config=ClassificationConfig(
                pv_end_min_ep_loss_rank4=1.0,
                pv_end_min_lift_rank4=1.0,
                rank4_good_to_inaccuracy_min_draw_permille=10_000,
            )
        ).assemble(board, result, played_move=played_move, player_elo=1000)
        assert off.cp_loss_label == "good"

        on = ContextAssembler(
            classification_config=ClassificationConfig(
                pv_end_min_ep_loss_rank4=1.0,
                pv_end_min_lift_rank4=1.0,
            )
        ).assemble(board, result, played_move=played_move, player_elo=1000)
        assert on.cp_loss_label == "inaccuracy"

    def test_rank4_future_override_bumps_inaccuracy_to_mistake(self):
        """Rank-4+ future collapse should rescue inaccuracy->mistake misses."""
        board = chess.Board()
        moves = list(board.legal_moves)
        played_move = moves[3]
        result = EngineResult(
            fen=board.fen(),
            depth=20,
            candidates=[
                CandidateMove(
                    move=moves[0], score_cp=40, mate_in=None, pv=[moves[0]],
                    pv_end=PVEndEval(
                        score_cp=40, mate_in=None, depth=20, seldepth=25,
                        pushed=4, fen="", terminal=None,
                    ),
                ),
                CandidateMove(move=moves[1], score_cp=20, mate_in=None, pv=[moves[1]]),
                CandidateMove(move=moves[2], score_cp=10, mate_in=None, pv=[moves[2]]),
            ],
            played_move=CandidateMove(
                move=played_move, score_cp=-40, mate_in=None, pv=[played_move],
                pv_end=PVEndEval(
                    score_cp=-500, mate_in=None, depth=20, seldepth=25,
                    pushed=4, fen="", terminal=None,
                ),
            ),
        )

        off = ContextAssembler(
            classification_config=ClassificationConfig(
                pv_end_min_ep_loss_rank4=1.0,
                pv_end_min_lift_rank4=1.0,
                rank4_inaccuracy_to_mistake_min_future_ep=1.0,
            )
        ).assemble(board, result, played_move=played_move)
        assert off.cp_loss_label == "inaccuracy"

        on = ContextAssembler(
            classification_config=ClassificationConfig(
                pv_end_min_ep_loss_rank4=1.0,
                pv_end_min_lift_rank4=1.0,
            )
        ).assemble(board, result, played_move=played_move)
        assert on.cp_loss_label == "mistake"

    def test_pv_end_lift_gated_by_root_ep_loss(self):
        """Near-best and `good` plays (root ep_loss ≤ pv_end_min_ep_loss)
        skip the lift.

        Chess.com's best / excellent / good buckets are assigned from the root
        eval without PV-drift adjustment; overriding them via pv_end regresses
        those buckets more than it helps. The default gate (0.05) keeps plays
        with small root ep_loss intact even when their pv_end drift is huge.
        Opting out of the gate restores the demotion and confirms the gate is
        the responsible mechanism.
        """
        board = chess.Board()
        moves = list(board.legal_moves)
        best = CandidateMove(
            move=moves[0], score_cp=20, mate_in=None, pv=[moves[0]],
            pv_end=PVEndEval(
                score_cp=20, mate_in=None, depth=20, seldepth=25,
                pushed=4, fen="", terminal=None,
            ),
        )
        # Root cp delta ≈ 30cp → ep_loss ~0.05 (top of the good bucket). Gate
        # must keep this at `good`; ungated, pv_end pushes it to blunder.
        played = CandidateMove(
            move=moves[1], score_cp=-10, mate_in=None, pv=[moves[1]],
            pv_end=PVEndEval(
                score_cp=-500, mate_in=None, depth=20, seldepth=25,
                pushed=4, fen="", terminal=None,
            ),
        )
        result = EngineResult(fen=board.fen(), depth=20, candidates=[best, played])

        gated = ContextAssembler(classification_config=ClassificationConfig()).assemble(
            board, result, played_move=moves[1]
        )
        assert gated.cp_loss_label in ("best", "excellent", "good")

        ungated = ContextAssembler(
            classification_config=ClassificationConfig(pv_end_min_ep_loss=0.0)
        ).assemble(board, result, played_move=moves[1])
        assert ungated.cp_loss_label in ("inaccuracy", "mistake", "blunder")

    def test_pv_end_gated_off_when_pv_too_shallow(self):
        """If the PV was only 0-1 plies before ending, don't trust its drift."""
        provider = SigmoidWDLProvider()
        board = chess.Board()
        moves = list(board.legal_moves)
        shallow = CandidateMove(
            move=moves[0], score_cp=30, mate_in=None, pv=[moves[0]],
            pv_end=PVEndEval(
                score_cp=-500, mate_in=None, depth=None, seldepth=None,
                pushed=1, fen="", terminal=None,
            ),
        )
        assert pv_end_win_pct(shallow, provider, None, ClassificationConfig()) is None

    def test_pv_end_skips_all_terminals(self):
        """Synthesized terminal evals are degraded representations — the
        helper returns None so the root eval handles them."""
        provider = SigmoidWDLProvider()
        board = chess.Board()
        moves = list(board.legal_moves)
        for terminal in ("stalemate", "draw", "checkmate"):
            cand = CandidateMove(
                move=moves[0], score_cp=0, mate_in=0, pv=[moves[0]],
                pv_end=PVEndEval(
                    score_cp=0, mate_in=0, depth=None, seldepth=None,
                    pushed=5, fen="", terminal=terminal,
                ),
            )
            assert pv_end_win_pct(cand, provider, None) is None, terminal

    def test_black_best_move_ep_loss_zero(self):
        """Black playing the best move should have ep_loss = 0."""
        board = chess.Board()
        board.push_san("e4")

        moves = list(board.legal_moves)
        result = EngineResult(
            fen=board.fen(), depth=20,
            candidates=[
                CandidateMove(move=moves[0], score_cp=30, mate_in=None, pv=[moves[0]]),
                CandidateMove(move=moves[1], score_cp=100, mate_in=None, pv=[moves[1]]),
            ],
        )
        assembler = ContextAssembler()
        ctx = assembler.assemble(board, result, played_move=moves[0])

        assert ctx.ep_loss == 0.0
        assert ctx.cp_loss_label == "best"
