"""
Integration tests: Stockfish engine container.

Verifies that the Docker-hosted Stockfish returns correct, structured results
for a variety of well-known positions. No mocking — all calls hit the real
engine container.
"""

import chess
import pytest

from chess_engine.service import EngineResult, RemoteEngineService

from .conftest import (
    BACK_RANK_FEN,
    COMPLEX_MIDDLEGAME_FEN,
    CRUSHING_ADVANTAGE_FEN,
    ITALIAN_GAME_FEN,
    MATE_IN_1_FEN,
    QUEENS_GAMBIT_FEN,
    SICILIAN_FEN,
    STARTING_FEN,
)


class TestEngineHealth:
    """Basic connectivity and response structure."""

    @pytest.mark.asyncio
    async def test_engine_returns_result(self, engine_service: RemoteEngineService):
        result = await engine_service.analyze(STARTING_FEN, depth=10, multipv=3)
        assert isinstance(result, EngineResult)
        assert result.fen == STARTING_FEN
        assert result.depth == 10
        assert len(result.candidates) >= 1

    @pytest.mark.asyncio
    async def test_candidates_have_valid_moves(self, engine_service):
        result = await engine_service.analyze(STARTING_FEN, depth=10, multipv=3)
        board = chess.Board(STARTING_FEN)
        for candidate in result.candidates:
            assert candidate.move in board.legal_moves, (
                f"{candidate.move.uci()} is not legal in the starting position"
            )

    @pytest.mark.asyncio
    async def test_multipv_returns_requested_count(self, engine_service):
        result = await engine_service.analyze(STARTING_FEN, depth=10, multipv=5)
        assert len(result.candidates) >= 3

    @pytest.mark.asyncio
    async def test_pv_lines_are_populated(self, engine_service):
        result = await engine_service.analyze(STARTING_FEN, depth=12, multipv=3)
        for candidate in result.candidates:
            assert len(candidate.pv) >= 1, "PV should contain at least the candidate move"


class TestMateDetection:
    """Verify the engine correctly identifies mate threats."""

    @pytest.mark.asyncio
    async def test_mate_in_1_detected(self, engine_service):
        """Re8# is mate in 1. Engine must find it."""
        result = await engine_service.analyze(MATE_IN_1_FEN, depth=10, multipv=1)
        best = result.best
        assert best.mate_in is not None, "Engine should detect a forced mate"
        assert best.mate_in == 1
        expected_move = chess.Move.from_uci("e1e8")
        assert best.move == expected_move, f"Expected Re8#, got {best.move.uci()}"

    @pytest.mark.asyncio
    async def test_mate_score_has_no_cp(self, engine_service):
        """When mate is found, score_cp should be None."""
        result = await engine_service.analyze(MATE_IN_1_FEN, depth=10, multipv=1)
        assert result.best.score_cp is None


class TestKnownPositions:
    """Verify engine produces reasonable evaluations for well-studied positions."""

    @pytest.mark.asyncio
    async def test_starting_position_roughly_equal(self, engine_service):
        result = await engine_service.analyze(STARTING_FEN, depth=14, multipv=1)
        best = result.best
        assert best.score_cp is not None
        assert abs(best.score_cp) < 80, (
            f"Starting position should be roughly equal, got {best.score_cp}cp"
        )

    @pytest.mark.asyncio
    async def test_crushing_advantage_high_eval(self, engine_service):
        """White has a queen vs nothing — eval should be very high."""
        result = await engine_service.analyze(CRUSHING_ADVANTAGE_FEN, depth=14, multipv=1)
        best = result.best
        if best.mate_in is not None:
            assert best.mate_in > 0
        else:
            assert best.score_cp is not None
            assert best.score_cp > 500, (
                f"White has a queen up, expected >500cp, got {best.score_cp}"
            )

    @pytest.mark.asyncio
    async def test_italian_game_playable(self, engine_service):
        result = await engine_service.analyze(ITALIAN_GAME_FEN, depth=14, multipv=3)
        assert len(result.candidates) >= 2
        best = result.best
        assert best.score_cp is not None

    @pytest.mark.asyncio
    async def test_complex_middlegame_multiple_candidates(self, engine_service):
        result = await engine_service.analyze(COMPLEX_MIDDLEGAME_FEN, depth=14, multipv=3)
        assert len(result.candidates) >= 3
        scores = [c.score_cp for c in result.candidates if c.score_cp is not None]
        assert len(scores) >= 2

    @pytest.mark.asyncio
    async def test_queens_gambit_reasonable(self, engine_service):
        result = await engine_service.analyze(QUEENS_GAMBIT_FEN, depth=14, multipv=3)
        best = result.best
        assert best.score_cp is not None
        assert abs(best.score_cp) < 150, "Queen's Gambit should be within normal range"

    @pytest.mark.asyncio
    async def test_sicilian_reasonable(self, engine_service):
        result = await engine_service.analyze(SICILIAN_FEN, depth=14, multipv=3)
        best = result.best
        assert best.score_cp is not None
        assert abs(best.score_cp) < 150


class TestUCIOptions:
    """Verify that UCI options are forwarded to the engine."""

    @pytest.mark.asyncio
    async def test_custom_hash_still_returns_result(self, engine_service):
        """Setting a universally-supported UCI option (Hash) still works."""
        result = await engine_service.analyze(
            STARTING_FEN,
            depth=10,
            multipv=1,
            uci_options={"Hash": 64},
        )
        assert len(result.candidates) >= 1
        assert result.best.move in chess.Board(STARTING_FEN).legal_moves
