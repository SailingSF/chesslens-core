"""
Integration tests: full pipeline — Stockfish → Context Assembly → Anthropic API.

Every test hits the real Stockfish container and the real Claude API.
No mocking anywhere.
"""

import chess
import pytest

from analysis.context import AssembledContext, ContextAssembler
from analysis.priority import PriorityResult, PriorityTier, classify
from chess_engine.service import RemoteEngineService
from explanation.coach import CoachNudgeGenerator
from explanation.generator import ExplanationGenerator

from .conftest import (
    COMPLEX_MIDDLEGAME_FEN,
    CRUSHING_ADVANTAGE_FEN,
    ITALIAN_GAME_FEN,
    MATE_IN_1_FEN,
    STARTING_FEN,
)


class TestContextAssemblyWithRealEngine:
    """Context assembly using real Stockfish output (not hand-crafted fakes)."""

    @pytest.mark.asyncio
    async def test_assemble_starting_position(
        self, engine_service: RemoteEngineService, context_assembler: ContextAssembler
    ):
        board = chess.Board(STARTING_FEN)
        result = await engine_service.analyze(STARTING_FEN, depth=14, multipv=3)

        played_move = list(board.legal_moves)[5]
        ctx = context_assembler.assemble(board, result, played_move=played_move)

        assert isinstance(ctx, AssembledContext)
        assert ctx.fen == STARTING_FEN
        assert ctx.best_move in board.legal_moves
        assert ctx.best_move_san is not None
        assert ctx.played_move == played_move
        assert ctx.cp_loss_label in ("best", "inaccuracy", "mistake", "blunder", "unknown")

    @pytest.mark.asyncio
    async def test_assemble_mate_position(self, engine_service, context_assembler):
        board = chess.Board(MATE_IN_1_FEN)
        result = await engine_service.analyze(MATE_IN_1_FEN, depth=10, multipv=1)

        ctx = context_assembler.assemble(board, result)
        assert ctx.mate_in is not None
        assert ctx.mate_in == 1
        assert ctx.best_move_san == "Re8#"

    @pytest.mark.asyncio
    async def test_priority_classification_with_real_engine(
        self, engine_service, context_assembler
    ):
        result = await engine_service.analyze(MATE_IN_1_FEN, depth=10, multipv=2)
        board = chess.Board(MATE_IN_1_FEN)

        priority = classify(result, board=board)
        assert isinstance(priority, PriorityResult)
        assert priority.tier == PriorityTier.CRITICAL
        assert "mate" in priority.trigger.lower()

    @pytest.mark.asyncio
    async def test_complex_middlegame_context(self, engine_service, context_assembler):
        board = chess.Board(COMPLEX_MIDDLEGAME_FEN)
        result = await engine_service.analyze(COMPLEX_MIDDLEGAME_FEN, depth=14, multipv=3)

        ctx = context_assembler.assemble(board, result)
        assert ctx.pawn_structure is not None
        assert ctx.piece_activity is not None
        assert ctx.threat_narrative is not None
        assert len(ctx.pv_san) >= 1


class TestExplanationGeneration:
    """Full pipeline: engine → context → priority → Claude explanation."""

    @pytest.mark.asyncio
    async def test_generate_explanation_for_blunder(
        self, engine_service, context_assembler, explainer: ExplanationGenerator
    ):
        """Simulate a blunder in the starting position and get a real explanation."""
        board = chess.Board(STARTING_FEN)
        result = await engine_service.analyze(STARTING_FEN, depth=14, multipv=3)

        worst_candidate = result.candidates[-1]
        ctx = context_assembler.assemble(board, result, played_move=worst_candidate.move)
        priority = classify(result, board=board)

        explanation = await explainer.generate(ctx, priority, skill_level="intermediate")

        assert isinstance(explanation, str)
        assert len(explanation) > 20, "Explanation should be substantive"
        assert len(explanation) < 2000, "Explanation should be concise"

    @pytest.mark.asyncio
    async def test_generate_explanation_for_mate_threat(
        self, engine_service, context_assembler, explainer
    ):
        board = chess.Board(MATE_IN_1_FEN)
        result = await engine_service.analyze(MATE_IN_1_FEN, depth=10, multipv=1)

        ctx = context_assembler.assemble(board, result)
        priority = classify(result, board=board)

        explanation = await explainer.generate(ctx, priority, skill_level="beginner")

        assert isinstance(explanation, str)
        assert len(explanation) > 20

    @pytest.mark.asyncio
    async def test_generate_explanation_for_middlegame(
        self, engine_service, context_assembler, explainer
    ):
        board = chess.Board(COMPLEX_MIDDLEGAME_FEN)
        result = await engine_service.analyze(COMPLEX_MIDDLEGAME_FEN, depth=14, multipv=3)

        played = list(board.legal_moves)[0]
        ctx = context_assembler.assemble(board, result, played_move=played)
        priority = classify(result, board=board)

        explanation = await explainer.generate(ctx, priority, skill_level="advanced")

        assert isinstance(explanation, str)
        assert len(explanation) > 20

    @pytest.mark.asyncio
    async def test_explanation_across_skill_levels(
        self, engine_service, context_assembler, explainer
    ):
        """Verify explanations are generated for all three skill levels."""
        board = chess.Board(ITALIAN_GAME_FEN)
        result = await engine_service.analyze(ITALIAN_GAME_FEN, depth=12, multipv=3)

        played = list(board.legal_moves)[0]
        ctx = context_assembler.assemble(board, result, played_move=played)
        priority = classify(result, board=board)

        for level in ("beginner", "intermediate", "advanced"):
            explanation = await explainer.generate(ctx, priority, skill_level=level)
            assert isinstance(explanation, str)
            assert len(explanation) > 20, f"Empty explanation for {level}"


class TestCoachNudgeGeneration:
    """Full pipeline: engine → context → priority → Claude coach nudge."""

    @pytest.mark.asyncio
    async def test_coach_nudge_returns_question(
        self, engine_service, context_assembler, coach: CoachNudgeGenerator
    ):
        board = chess.Board(COMPLEX_MIDDLEGAME_FEN)
        result = await engine_service.analyze(COMPLEX_MIDDLEGAME_FEN, depth=12, multipv=3)

        played = list(board.legal_moves)[0]
        ctx = context_assembler.assemble(board, result, played_move=played)
        priority = classify(result, board=board)

        nudge = await coach.generate(ctx, priority, skill_level="intermediate")

        assert isinstance(nudge, str)
        assert len(nudge) > 10
        assert "?" in nudge, "Coach nudge should end with a question"

    @pytest.mark.asyncio
    async def test_coach_nudge_no_move_notation(
        self, engine_service, context_assembler, coach
    ):
        """The coach nudge validator rejects responses containing move notation.
        If this test passes, Claude respected the prompt constraint AND the
        validator didn't raise."""
        board = chess.Board(MATE_IN_1_FEN)
        result = await engine_service.analyze(MATE_IN_1_FEN, depth=10, multipv=1)

        ctx = context_assembler.assemble(board, result)
        priority = classify(result, board=board)

        nudge = await coach.generate(ctx, priority, skill_level="beginner")
        assert isinstance(nudge, str)
        assert len(nudge) > 10
