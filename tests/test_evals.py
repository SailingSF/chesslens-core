"""
Ad-hoc evaluation tests — quality checks on real LLM output.

These tests go beyond "does it return a string" and check whether the
explanations are *correct* and *useful* for specific chess scenarios.
All calls hit the real Stockfish container and the real Anthropic API.

Run with: pytest tests/test_evals.py -v -s
The -s flag lets all print output stream to the terminal.
"""

import chess
import pytest

from analysis.context import AssembledContext, ContextAssembler
from analysis.priority import PriorityResult, PriorityTier, classify
from chess_engine.service import EngineResult, RemoteEngineService
from explanation.coach import CoachNudgeGenerator
from explanation.generator import ExplanationGenerator
from explanation.templates.game_review import build_game_review_prompt
from explanation.templates.coach_nudge import build_coach_nudge_prompt

from .conftest import (
    COMPLEX_MIDDLEGAME_FEN,
    CRUSHING_ADVANTAGE_FEN,
    ITALIAN_GAME_FEN,
    MATE_IN_1_FEN,
    STARTING_FEN,
)


SEPARATOR = "─" * 72


def _log_engine_result(fen: str, result: EngineResult) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  ENGINE ANALYSIS")
    print(f"{SEPARATOR}")
    print(f"  FEN:   {fen}")
    print(f"  Depth: {result.depth}")
    print(f"  Candidates ({len(result.candidates)}):")
    board = chess.Board(fen)
    for i, c in enumerate(result.candidates):
        san = board.san(c.move)
        if c.mate_in is not None:
            score_str = f"mate in {c.mate_in}"
        else:
            score_str = f"{c.score_cp}cp"
        pv_san = []
        temp = board.copy()
        for m in c.pv[:6]:
            try:
                pv_san.append(temp.san(m))
                temp.push(m)
            except Exception:
                break
        pv_str = " ".join(pv_san) if pv_san else c.move.uci()
        rank = "★" if i == 0 else " "
        print(f"    {rank} {i+1}. {san:<8} {score_str:<16} PV: {pv_str}")


def _log_context(ctx: AssembledContext, priority: PriorityResult) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  ASSEMBLED CONTEXT")
    print(f"{SEPARATOR}")
    print(f"  Best move:      {ctx.best_move_san} ({ctx.best_move_cp}cp / mate={ctx.mate_in})")
    print(f"  Played move:    {ctx.played_move_san or 'N/A'}")
    print(f"  CP loss:        {ctx.played_move_cp_loss} ({ctx.cp_loss_label})")
    print(f"  Opening:        {ctx.opening_name or 'unknown'} ({ctx.eco_code or '—'})")
    print(f"  Theory dev:     {ctx.theory_deviation}")
    print(f"  Tactics:        {ctx.tactical_patterns or 'none'}")
    print(f"  Threats:        {ctx.threat_narrative}")
    print(f"  Priority:       {priority.tier.value} — {priority.trigger}")
    if priority.score_delta is not None:
        print(f"  Score delta:    {priority.score_delta}cp")

    pw = ctx.pawn_structure.get("white", {})
    pb = ctx.pawn_structure.get("black", {})
    pawn_parts = []
    for label in ("isolated", "doubled", "passed"):
        ws = pw.get(label, [])
        bs = pb.get(label, [])
        if ws:
            pawn_parts.append(f"W {label}: {','.join(ws)}")
        if bs:
            pawn_parts.append(f"B {label}: {','.join(bs)}")
    print(f"  Pawn structure:  {'; '.join(pawn_parts) if pawn_parts else 'normal'}")

    pa = ctx.piece_activity
    print(f"  Piece activity:  W bishop pair={pa.get('white_bishop_pair')}, "
          f"B bishop pair={pa.get('black_bishop_pair')}, "
          f"W king danger={pa.get('white_king_danger')}, "
          f"B king danger={pa.get('black_king_danger')}")


def _log_prompt(prompt: str, label: str = "PROMPT → CLAUDE") -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {label}")
    print(f"{SEPARATOR}")
    for line in prompt.splitlines():
        print(f"  │ {line}")


def _log_response(response: str, label: str = "CLAUDE → RESPONSE") -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {label}")
    print(f"{SEPARATOR}")
    for line in response.splitlines():
        print(f"  │ {line}")
    print(f"{SEPARATOR}\n")


async def _run_pipeline(engine_service, context_assembler, fen, played_move=None, depth=14):
    """Run the full engine → context → priority pipeline, logging everything."""
    board = chess.Board(fen)
    result = await engine_service.analyze(fen, depth=depth, multipv=3)
    _log_engine_result(fen, result)

    if played_move is None:
        played_move = result.candidates[-1].move if len(result.candidates) > 1 else None

    ctx = context_assembler.assemble(board, result, played_move=played_move)
    priority = classify(result, board=board)
    _log_context(ctx, priority)

    return ctx, priority


async def _generate_and_log_explanation(explainer, ctx, priority, skill_level):
    """Generate an explanation, log the prompt and response, return the text."""
    prompt = build_game_review_prompt(ctx, priority, skill_level)
    _log_prompt(prompt, label=f"PROMPT → CLAUDE (skill={skill_level})")
    explanation = await explainer.generate(ctx, priority, skill_level=skill_level)
    _log_response(explanation, label=f"CLAUDE → EXPLANATION (skill={skill_level})")
    return explanation


async def _generate_and_log_nudge(coach, ctx, priority, skill_level):
    """Generate a coach nudge, log the prompt and response, return the text."""
    prompt = build_coach_nudge_prompt(ctx, priority, skill_level)
    _log_prompt(prompt, label=f"PROMPT → CLAUDE COACH (skill={skill_level})")
    nudge = await coach.generate(ctx, priority, skill_level=skill_level)
    _log_response(nudge, label=f"CLAUDE → COACH NUDGE (skill={skill_level})")
    return nudge


# ---------------------------------------------------------------------------
# Eval: Mate threats
# ---------------------------------------------------------------------------

class TestEvalMateThreats:

    @pytest.mark.asyncio
    async def test_mate_explanation_mentions_checkmate(
        self, engine_service, context_assembler, explainer: ExplanationGenerator
    ):
        """When there's a forced mate, the explanation should mention it."""
        print("\n\n══════════════════════════════════════════════════════════════════════════")
        print("  EVAL: Mate explanation mentions checkmate")
        print("══════════════════════════════════════════════════════════════════════════")

        ctx, priority = await _run_pipeline(engine_service, context_assembler, MATE_IN_1_FEN)
        assert priority.tier == PriorityTier.CRITICAL

        explanation = await _generate_and_log_explanation(explainer, ctx, priority, "beginner")
        explanation_lower = explanation.lower()
        assert any(
            term in explanation_lower
            for term in ("mate", "checkmate", "mating", "deliver mate")
        ), f"Explanation for a mate-in-1 should mention checkmate: {explanation!r}"

    @pytest.mark.asyncio
    async def test_mate_explanation_mentions_rook(
        self, engine_service, context_assembler, explainer
    ):
        """In the mate-in-1 position, Re8# — explanation should reference the rook."""
        print("\n\n══════════════════════════════════════════════════════════════════════════")
        print("  EVAL: Mate explanation mentions rook / back rank")
        print("══════════════════════════════════════════════════════════════════════════")

        ctx, priority = await _run_pipeline(engine_service, context_assembler, MATE_IN_1_FEN)

        explanation = await _generate_and_log_explanation(explainer, ctx, priority, "intermediate")
        explanation_lower = explanation.lower()
        assert any(
            term in explanation_lower
            for term in ("rook", "re8", "back rank", "back-rank", "eighth rank")
        ), f"Explanation should reference the rook or back rank: {explanation!r}"


# ---------------------------------------------------------------------------
# Eval: Material advantage
# ---------------------------------------------------------------------------

class TestEvalMaterialAdvantage:

    @pytest.mark.asyncio
    async def test_crushing_advantage_acknowledged(
        self, engine_service, context_assembler, explainer
    ):
        """White has a queen vs nothing — explanation should reflect dominance."""
        print("\n\n══════════════════════════════════════════════════════════════════════════")
        print("  EVAL: Crushing advantage acknowledged")
        print("══════════════════════════════════════════════════════════════════════════")

        ctx, priority = await _run_pipeline(
            engine_service, context_assembler, CRUSHING_ADVANTAGE_FEN
        )

        explanation = await _generate_and_log_explanation(explainer, ctx, priority, "beginner")
        explanation_lower = explanation.lower()
        assert any(
            term in explanation_lower
            for term in ("queen", "winning", "decisive", "advantage", "dominant", "material")
        ), f"Explanation should acknowledge the crushing advantage: {explanation!r}"


# ---------------------------------------------------------------------------
# Eval: Opening recognition
# ---------------------------------------------------------------------------

class TestEvalOpeningRecognition:

    @pytest.mark.asyncio
    async def test_italian_game_context_has_opening(
        self, engine_service, context_assembler
    ):
        """The Italian Game position should be recognized by ECO lookup."""
        print("\n\n══════════════════════════════════════════════════════════════════════════")
        print("  EVAL: Italian Game opening recognition")
        print("══════════════════════════════════════════════════════════════════════════")

        ctx, _ = await _run_pipeline(engine_service, context_assembler, ITALIAN_GAME_FEN)

        print(f"\n  Result: opening_name={ctx.opening_name!r}, eco_code={ctx.eco_code!r}")

        assert ctx.opening_name is not None or ctx.eco_code is not None, (
            "Italian Game position should be recognized in the ECO database"
        )


# ---------------------------------------------------------------------------
# Eval: Skill level differentiation
# ---------------------------------------------------------------------------

class TestEvalSkillDifferentiation:

    @pytest.mark.asyncio
    async def test_beginner_explanation_avoids_jargon(
        self, engine_service, context_assembler, explainer
    ):
        print("\n\n══════════════════════════════════════════════════════════════════════════")
        print("  EVAL: Beginner vs Advanced skill differentiation")
        print("══════════════════════════════════════════════════════════════════════════")

        ctx, priority = await _run_pipeline(
            engine_service, context_assembler, COMPLEX_MIDDLEGAME_FEN
        )

        beginner = await _generate_and_log_explanation(explainer, ctx, priority, "beginner")
        advanced = await _generate_and_log_explanation(explainer, ctx, priority, "advanced")

        assert beginner != advanced, "Beginner and advanced explanations should differ"

        jargon_terms = ["outpost", "prophylaxis", "initiative", "tempo", "fianchetto"]
        beginner_jargon_count = sum(1 for t in jargon_terms if t in beginner.lower())
        advanced_jargon_count = sum(1 for t in jargon_terms if t in advanced.lower())

        print(f"\n  Jargon check: beginner={beginner_jargon_count}, advanced={advanced_jargon_count}")
        print(f"  (terms checked: {', '.join(jargon_terms)})")

        assert beginner_jargon_count <= advanced_jargon_count + 1, (
            f"Beginner explanation should not be more jargon-heavy than advanced.\n"
            f"Beginner ({beginner_jargon_count} jargon terms): {beginner!r}\n"
            f"Advanced ({advanced_jargon_count} jargon terms): {advanced!r}"
        )


# ---------------------------------------------------------------------------
# Eval: Coach nudge quality
# ---------------------------------------------------------------------------

class TestEvalCoachNudge:

    @pytest.mark.asyncio
    async def test_nudge_is_a_single_sentence(
        self, engine_service, context_assembler, coach: CoachNudgeGenerator
    ):
        print("\n\n══════════════════════════════════════════════════════════════════════════")
        print("  EVAL: Coach nudge is a single sentence")
        print("══════════════════════════════════════════════════════════════════════════")

        ctx, priority = await _run_pipeline(
            engine_service, context_assembler, COMPLEX_MIDDLEGAME_FEN
        )
        nudge = await _generate_and_log_nudge(coach, ctx, priority, "intermediate")

        sentence_count = nudge.count(".") + nudge.count("?") + nudge.count("!")
        print(f"  Sentence terminators: {sentence_count}")

        assert sentence_count <= 3, (
            f"Coach nudge should be ~1 sentence, got {sentence_count} terminators: {nudge!r}"
        )

    @pytest.mark.asyncio
    async def test_nudge_asks_a_question(
        self, engine_service, context_assembler, coach
    ):
        print("\n\n══════════════════════════════════════════════════════════════════════════")
        print("  EVAL: Coach nudge asks a question")
        print("══════════════════════════════════════════════════════════════════════════")

        ctx, priority = await _run_pipeline(
            engine_service, context_assembler, ITALIAN_GAME_FEN
        )
        nudge = await _generate_and_log_nudge(coach, ctx, priority, "beginner")
        assert "?" in nudge, f"Coach nudge should ask a question: {nudge!r}"

    @pytest.mark.asyncio
    async def test_nudge_references_board_features(
        self, engine_service, context_assembler, coach
    ):
        """The nudge should reference something concrete on the board."""
        print("\n\n══════════════════════════════════════════════════════════════════════════")
        print("  EVAL: Coach nudge references board features")
        print("══════════════════════════════════════════════════════════════════════════")

        ctx, priority = await _run_pipeline(
            engine_service, context_assembler, COMPLEX_MIDDLEGAME_FEN
        )
        nudge = await _generate_and_log_nudge(coach, ctx, priority, "intermediate")

        board_terms = [
            "pawn", "knight", "bishop", "rook", "queen", "king",
            "center", "castle", "attack", "defend", "threat",
            "control", "pressure", "piece", "square", "file", "rank",
            "diagonal", "pin", "fork", "check", "develop",
        ]
        nudge_lower = nudge.lower()
        matched = [t for t in board_terms if t in nudge_lower]
        print(f"  Board terms found: {matched}")

        assert matched, (
            f"Coach nudge should reference concrete board features: {nudge!r}"
        )


# ---------------------------------------------------------------------------
# Eval: Priority tier correctness
# ---------------------------------------------------------------------------

class TestEvalPriorityTiers:

    @pytest.mark.asyncio
    async def test_mate_position_is_critical(self, engine_service, context_assembler):
        print("\n\n══════════════════════════════════════════════════════════════════════════")
        print("  EVAL: Mate position → CRITICAL tier")
        print("══════════════════════════════════════════════════════════════════════════")

        _, priority = await _run_pipeline(engine_service, context_assembler, MATE_IN_1_FEN)
        assert priority.tier == PriorityTier.CRITICAL

    @pytest.mark.asyncio
    async def test_starting_position_is_strategic(self, engine_service, context_assembler):
        print("\n\n══════════════════════════════════════════════════════════════════════════")
        print("  EVAL: Starting position → STRATEGIC tier")
        print("══════════════════════════════════════════════════════════════════════════")

        _, priority = await _run_pipeline(engine_service, context_assembler, STARTING_FEN)
        assert priority.tier == PriorityTier.STRATEGIC, (
            f"Starting position should be STRATEGIC, got {priority.tier}: {priority.trigger}"
        )

    @pytest.mark.asyncio
    async def test_crushing_advantage_is_critical_or_tactical(
        self, engine_service, context_assembler
    ):
        print("\n\n══════════════════════════════════════════════════════════════════════════")
        print("  EVAL: Crushing advantage → CRITICAL or TACTICAL tier")
        print("══════════════════════════════════════════════════════════════════════════")

        _, priority = await _run_pipeline(
            engine_service, context_assembler, CRUSHING_ADVANTAGE_FEN
        )
        assert priority.tier in (PriorityTier.CRITICAL, PriorityTier.TACTICAL), (
            f"Queen vs nothing should be CRITICAL or TACTICAL, got {priority.tier}"
        )


# ---------------------------------------------------------------------------
# Eval: Explanation-context coherence
# ---------------------------------------------------------------------------

class TestEvalCoherence:

    @pytest.mark.asyncio
    async def test_explanation_references_played_move_or_best_move(
        self, engine_service, context_assembler, explainer
    ):
        """The explanation should mention at least one of the moves involved."""
        print("\n\n══════════════════════════════════════════════════════════════════════════")
        print("  EVAL: Explanation references played/best move")
        print("══════════════════════════════════════════════════════════════════════════")

        board = chess.Board(STARTING_FEN)
        result = await engine_service.analyze(STARTING_FEN, depth=14, multipv=3)
        _log_engine_result(STARTING_FEN, result)

        played = result.candidates[-1].move
        played_san = board.san(played)
        best_san = board.san(result.best.move)

        ctx = context_assembler.assemble(board, result, played_move=played)
        priority = classify(result, board=board)
        _log_context(ctx, priority)

        explanation = await _generate_and_log_explanation(
            explainer, ctx, priority, "intermediate"
        )

        assert played_san in explanation or best_san in explanation or any(
            term in explanation.lower()
            for term in ("best", "played", "move", "pawn", "knight", "bishop")
        ), (
            f"Explanation should reference the moves or pieces involved.\n"
            f"Played: {played_san}, Best: {best_san}\n"
            f"Explanation: {explanation!r}"
        )
