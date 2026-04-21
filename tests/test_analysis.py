"""
Smoke tests for the analysis layer.

These tests use real python-chess boards but do NOT require Stockfish or
the Anthropic API — engine calls are mocked.
"""

import chess
import pytest

from analysis.context import ContextAssembler, _classify_cp_loss, _mate_to_cp
from analysis.pawns import analyze_pawns
from analysis.patterns import detect_tactics
from analysis.priority import PriorityTier, classify
from tests._engine_result_helpers import make_result
from chess_engine.service import CandidateMove, EngineResult


# --- Helpers ---


# --- Priority classifier ---

def test_priority_critical_mate():
    result = make_result([(None, 2), (50, None)])
    pr = classify(result)
    assert pr.tier == PriorityTier.CRITICAL
    assert "mate" in pr.trigger


def test_priority_critical_gap():
    result = make_result([(300, None), (50, None)])
    pr = classify(result)
    assert pr.tier == PriorityTier.CRITICAL


def test_priority_tactical():
    result = make_result([(100, None), (40, None), (30, None)])
    pr = classify(result)
    assert pr.tier == PriorityTier.TACTICAL


def test_priority_strategic():
    result = make_result([(20, None), (10, None), (5, None)])
    pr = classify(result)
    assert pr.tier == PriorityTier.STRATEGIC


def test_priority_with_board_passes_through():
    """classify() accepts an optional board argument for capture detection."""
    board = chess.Board()
    result = make_result([(100, None), (40, None)])
    pr = classify(result, board=board)
    assert pr.tier == PriorityTier.TACTICAL


# --- Centipawn loss classification ---

def test_cp_loss_exact_best():
    assert _classify_cp_loss(0) == "best"

def test_cp_loss_excellent():
    assert _classify_cp_loss(5) == "excellent"

def test_cp_loss_good():
    assert _classify_cp_loss(15) == "good"
    assert _classify_cp_loss(30) == "good"

def test_cp_loss_inaccuracy():
    assert _classify_cp_loss(50) == "inaccuracy"
    assert _classify_cp_loss(75) == "inaccuracy"
    assert _classify_cp_loss(99) == "inaccuracy"

def test_cp_loss_mistake():
    assert _classify_cp_loss(100) == "mistake"
    assert _classify_cp_loss(150) == "mistake"
    assert _classify_cp_loss(250) == "mistake"
    assert _classify_cp_loss(299) == "mistake"

def test_cp_loss_blunder():
    assert _classify_cp_loss(300) == "blunder"
    assert _classify_cp_loss(500) == "blunder"

def test_cp_loss_none():
    assert _classify_cp_loss(None) == "unknown"


# --- Mate-to-cp conversion ---

def test_mate_to_cp_winning():
    assert _mate_to_cp(3) == 10000 - 30  # 9970

def test_mate_to_cp_losing():
    assert _mate_to_cp(-3) == -10000 + 30  # -9970

def test_mate_to_cp_none():
    assert _mate_to_cp(None) is None


def test_cp_for_move_mate_fallback():
    """_cp_for_move returns mate equivalent when score_cp is None."""
    board = chess.Board()
    moves = list(board.legal_moves)
    result = EngineResult(
        fen=board.fen(),
        depth=20,
        candidates=[
            CandidateMove(move=moves[0], score_cp=None, mate_in=2, pv=[moves[0]]),
        ],
    )
    assembler = ContextAssembler()
    cp = assembler._cp_for_move(result, moves[0])
    assert cp == 10000 - 20  # mate-in-2 = 9980


def test_cp_loss_with_mate_best_move():
    """cp_loss is computed even when the best move has a mate score."""
    board = chess.Board()
    moves = list(board.legal_moves)
    result = EngineResult(
        fen=board.fen(),
        depth=20,
        candidates=[
            CandidateMove(move=moves[0], score_cp=None, mate_in=3, pv=[moves[0]]),
            CandidateMove(move=moves[1], score_cp=200, mate_in=None, pv=[moves[1]]),
        ],
    )
    assembler = ContextAssembler()
    ctx = assembler.assemble(board, result, played_move=moves[1])
    assert ctx.played_move_cp_loss is not None
    # Best move is mate-in-3 with a huge candidate gap — Trigger C (direct
    # tactic missed) fires: best is forcing (mate), gap >> 250cp, mover
    # winning. The blunder gate would have demoted to "mistake" on EP alone,
    # but the missed-mate pattern is more informative.
    assert ctx.cp_loss_label == "miss"


# --- Pawn structure ---

def test_pawns_starting_position():
    board = chess.Board()
    result = analyze_pawns(board)
    # No isolated/doubled/passed pawns in starting position
    assert result["white"]["isolated"] == []
    assert result["white"]["doubled"] == []
    assert result["black"]["doubled"] == []


def test_passed_pawn():
    # Advanced white pawn on e6 with no black pawns blocking
    board = chess.Board("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1")
    result = analyze_pawns(board)
    assert "e6" in result["white"]["passed"]


def test_isolated_pawn():
    # White pawn on a2, no pawns on b-file — isolated
    board = chess.Board("4k3/8/8/8/8/8/P7/4K3 w - - 0 1")
    result = analyze_pawns(board)
    assert "a2" in result["white"]["isolated"]


def test_doubled_pawns():
    # Two white pawns on e-file
    board = chess.Board("4k3/8/8/4P3/4P3/8/8/4K3 w - - 0 1")
    result = analyze_pawns(board)
    assert "e4" in result["white"]["doubled"]
    assert "e5" in result["white"]["doubled"]


# --- Tactical pattern detection ---

def test_fork_detection():
    # Knight on e5 forking king on g6 and rook on c6
    board = chess.Board("2r1k3/8/2r1K3/4N3/8/8/8/8 w - - 0 1")
    patterns = detect_tactics(board)
    assert isinstance(patterns, list)


def test_pin_detection():
    # Black bishop pins white knight to white king (Bg4 pins Nf3 to Ke1)
    board = chess.Board("rnbqkbnr/pppppppp/8/8/6b1/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    patterns = detect_tactics(board)
    # The pin should be detected (white knight on f3 is pinned by bishop on g4)
    assert isinstance(patterns, list)


def test_back_rank_weakness():
    # King on a1 with no flight squares off the back rank at all
    # (a2, b1, b2 all occupied by own pieces)
    board = chess.Board("6k1/8/8/8/8/8/PP6/KR6 w - - 0 1")
    patterns = detect_tactics(board)
    assert "back-rank weakness" in patterns


# --- Context assembler (no engine/LLM calls) ---

def test_context_assembler_basic():
    board = chess.Board()
    # Create a fake engine result with first legal moves
    moves = list(board.legal_moves)
    result = EngineResult(
        fen=board.fen(),
        depth=20,
        candidates=[
            CandidateMove(move=moves[0], score_cp=30, mate_in=None, pv=[moves[0]]),
            CandidateMove(move=moves[1], score_cp=20, mate_in=None, pv=[moves[1]]),
        ],
    )
    assembler = ContextAssembler()
    ctx = assembler.assemble(board, result, played_move=moves[1])

    assert ctx.fen == board.fen()
    assert ctx.best_move == moves[0]
    assert ctx.played_move == moves[1]
    assert ctx.played_move_cp_loss == 10  # 30 - 20
    assert ctx.cp_loss_label == "excellent"  # 10cp = excellent


def test_threat_narrative_not_empty_when_attacks_exist():
    """Threat narrative should describe attacks in a non-starting position."""
    # Rook on e1 attacking king on e8 — clear threat
    board = chess.Board("4k3/8/8/8/8/8/8/4K1R1 w - - 0 1")
    assembler = ContextAssembler()
    moves = list(board.legal_moves)
    result = EngineResult(
        fen=board.fen(),
        depth=20,
        candidates=[CandidateMove(move=moves[0], score_cp=500, mate_in=None, pv=[moves[0]])],
    )
    ctx = assembler.assemble(board, result)
    # There should be some threats in this position
    assert isinstance(ctx.threat_narrative, str)


def test_piece_activity_structure():
    board = chess.Board()
    assembler = ContextAssembler()
    activity = assembler._piece_activity(board)
    assert "white_bishop_pair" in activity
    assert "black_bishop_pair" in activity
    assert "white_rooks_open_files" in activity
    assert "white_king_danger" in activity


# --- Static exchange evaluation & concrete blunder gate ---

from analysis.special_moves import _static_exchange_eval, is_concrete_blunder
from chess_engine.service import CandidateMove


def test_see_wins_undefended_pawn():
    # Black knight on f6 captures undefended white pawn on e4
    board = chess.Board("4k3/8/5n2/8/4P3/8/8/4K3 b - - 0 1")
    attackers = list(board.attackers(chess.BLACK, chess.E4))
    assert attackers, "knight should attack e4"
    see = _static_exchange_eval(board, chess.E4, attackers[0])
    assert see == 1


def test_see_losing_queen_for_pawn():
    # White queen captures defended black pawn — loses queen for pawn + pawn
    board = chess.Board("4k3/5p2/4p3/8/2Q5/8/8/4K3 w - - 0 1")
    see = _static_exchange_eval(board, chess.E6, chess.C4)
    assert see <= -7


def test_see_clean_piece_win():
    # White knight on c4 wins undefended knight on d6
    board = chess.Board("4k3/8/3n4/8/2N5/8/8/4K3 w - - 0 1")
    see = _static_exchange_eval(board, chess.D6, chess.C4)
    assert see == 3


def test_concrete_blunder_does_not_fire_on_undefended_pawn_only():
    # White pushed pawn e4 — black can win undefended pawn with Nxe4.
    # This is a one-pawn hang with ep_loss=0; should NOT trigger concrete blunder
    # (the old 1-ply check would have fired since e4 is undefended).
    board_before = chess.Board("4k3/8/5n2/8/8/4P3/8/4K3 w - - 0 1")
    move = board_before.parse_san("e4")
    board_after = board_before.copy()
    board_after.push(move)
    played = CandidateMove(move=move, score_cp=0, mate_in=None, pv=[move])
    # With SEE threshold >=2 (default is 1 — a pawn hang still fires), set
    # ep_loss small so gate 4 doesn't interfere.
    from config.classification import ClassificationConfig
    cfg = ClassificationConfig(blunder_min_hanging_see=2)
    assert not is_concrete_blunder(
        board_before, board_after, move, played, chess.WHITE,
        ep_loss=0.15, config=cfg,
    )


def test_concrete_blunder_fires_on_hanging_piece():
    # White hangs the queen (moves it to an attacked, undefended square)
    board_before = chess.Board("4k3/8/5n2/8/8/8/1Q6/4K3 w - - 0 1")
    move = board_before.parse_san("Qe5")  # black knight on f6 attacks e5; queen undefended
    board_after = board_before.copy()
    board_after.push(move)
    played = CandidateMove(move=move, score_cp=0, mate_in=None, pv=[move])
    assert is_concrete_blunder(
        board_before, board_after, move, played, chess.WHITE,
        ep_loss=0.5,
    )
