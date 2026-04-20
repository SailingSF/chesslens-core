"""
REST API views (native async Django views).

Endpoints:
  POST /api/game-review/       — submit a PGN for analysis (SSE streaming)
  POST /api/position-explorer/ — analyze a single FEN position
  POST /api/bot-match/start/   — create a new bot match session
  POST /api/opening-lab/start/ — create a new opening lab session
"""

from __future__ import annotations

import json
import uuid

import chess
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from rest_framework.exceptions import ValidationError

from analysis.context import ContextAssembler
from analysis.priority import classify
from api.serializers import (
    BotMatchStartSerializer,
    GameReviewSerializer,
    OpeningLabStartSerializer,
    PositionExplorerSerializer,
)
from chess_engine.service import get_engine_service
from explanation.generator import get_explainer
from game.analyzer import GameAnalyzer
from game.session import GameSession, SessionManager


def _engine_defaults():
    """Read per-request engine kwargs from Django settings. None values are
    stripped so callers can overlay them with their own overrides."""
    from django.conf import settings
    raw = {
        "nodes": getattr(settings, "STOCKFISH_DEFAULT_NODES", None),
        "multipv": getattr(settings, "STOCKFISH_DEFAULT_MULTIPV", None),
        "pv_length": getattr(settings, "STOCKFISH_PV_LENGTH", None),
        "pv_end_nodes": getattr(settings, "STOCKFISH_PV_END_NODES", None),
        "played_move_nodes": getattr(settings, "STOCKFISH_PLAYED_MOVE_NODES", None),
    }
    return {k: v for k, v in raw.items() if v is not None}

# Shared assembler instance (stateless, safe to reuse)
_assembler = ContextAssembler()


def _parse_json_body(request):
    """Parse JSON body from a Django HttpRequest."""
    try:
        return json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return None


def _validate(serializer_class, data):
    """Validate data with a DRF serializer, return validated_data or raise."""
    serializer = serializer_class(data=data)
    serializer.is_valid(raise_exception=True)
    return serializer.validated_data


def _error_response(message, status=400):
    return JsonResponse({"error": message}, status=status)


async def engines_list(request):
    """
    List Stockfish engines this server can route to, plus the default id.

    Response: {"engines": [{"id", "name", "version"}, ...], "default_id": "..."}
    """
    engine = get_engine_service()
    engines = [
        {"id": e.id, "name": e.name, "version": e.version}
        for e in engine.list_engines()
    ]
    return JsonResponse({
        "engines": engines,
        "default_id": engine.default_engine_id(),
    })


@csrf_exempt
@require_POST
async def game_review(request):
    """
    Submit a PGN and stream analysis results via SSE.

    Request body: {"pgn": "...", "skill_level": "intermediate"}
    Response: text/event-stream with MoveAnalysis JSON objects
    """
    body = _parse_json_body(request)
    if body is None:
        return _error_response("Invalid JSON body")

    try:
        data = _validate(GameReviewSerializer, body)
    except ValidationError as e:
        return _error_response(str(e.detail))

    pgn_text = data["pgn"]
    skill_level = data.get("skill_level", "intermediate")
    player_color = data.get("player_color", "white")
    engine_id = data.get("engine_id") or None

    api_key = getattr(request, "anthropic_api_key", None)
    engine = get_engine_service()
    explainer = get_explainer()
    analyzer = GameAnalyzer(
        engine, assembler=_assembler, explainer=explainer,
        engine_defaults={**_engine_defaults(), **({"engine_id": engine_id} if engine_id else {})},
    )

    async def event_stream():
        async for move_analysis in analyzer.analyze_pgn(
            pgn_text,
            skill_level=skill_level,
            player_color=player_color,
            api_key=api_key,
        ):
            data = {
                "move_number": move_analysis.move_number,
                "color": move_analysis.color,
                "move_san": move_analysis.context.played_move_san,
                "fen": move_analysis.context.fen,
                "eval_cp": move_analysis.context.best_move_cp,
                "mate_in": move_analysis.context.mate_in,
                "cp_loss": move_analysis.context.played_move_cp_loss,
                "cp_loss_label": move_analysis.context.cp_loss_label,
                "priority_tier": move_analysis.priority.tier.value,
                "explanation": move_analysis.explanation,
            }
            yield f"data: {json.dumps(data)}\n\n"

        # Generate and send the final game summary
        summary = await analyzer.generate_summary(api_key=api_key)
        yield f'data: {json.dumps({"done": True, "summary": summary})}\n\n'

    return StreamingHttpResponse(event_stream(), content_type="text/event-stream")


@csrf_exempt
@require_POST
async def position_explorer(request):
    """
    Analyze a single FEN position.

    Request body: {"fen": "...", "skill_level": "intermediate", "question": "optional"}
    Response: {"explanation": "...", "priority_tier": "...", "best_move": "..."}
    """
    body = _parse_json_body(request)
    if body is None:
        return _error_response("Invalid JSON body")

    try:
        data = _validate(PositionExplorerSerializer, body)
    except ValidationError as e:
        return _error_response(str(e.detail))

    fen = data["fen"]
    skill_level = data.get("skill_level", "intermediate")
    engine_id = data.get("engine_id") or None

    engine = get_engine_service()
    analyze_kwargs = {"depth": 20, **_engine_defaults()}
    if engine_id:
        analyze_kwargs["engine_id"] = engine_id
    result = await engine.analyze(fen, **analyze_kwargs)

    board = chess.Board(fen)
    context = _assembler.assemble(board, result)
    priority = classify(result, board=board)

    # Build candidate list with SAN notation
    candidates = []
    for candidate in result.candidates:
        # Convert move to SAN
        move_san = board.san(candidate.move)
        # Convert PV to SAN
        pv_san = []
        temp = board.copy()
        for pv_move in candidate.pv[:6]:
            try:
                pv_san.append(temp.san(pv_move))
                temp.push(pv_move)
            except Exception:
                break
        candidates.append({
            "move": move_san,
            "cp": candidate.score_cp,
            "mate_in": candidate.mate_in,
            "pv": pv_san,
        })

    api_key = getattr(request, "anthropic_api_key", None)
    explainer = get_explainer()
    explanation = await explainer.generate(context, priority, skill_level, api_key=api_key)

    return JsonResponse({
        "explanation": explanation,
        "priority_tier": priority.tier.value,
        "best_move": context.best_move_san,
        "pv": context.pv_san,
        "cp": context.best_move_cp,
        "candidates": candidates,
        "opening": context.opening_name,
        "eco": context.eco_code,
    })


@csrf_exempt
@require_POST
async def bot_match_start(request):
    """
    Create a new bot match session. Returns a session_id for the WebSocket.

    Request body: {"elo": 1200, "player_color": "white", "coaching_mode": "nudge",
                   "skill_level": "intermediate", "opening": ""}
    Response: {"session_id": "...", "fen": "...", "player_color": "white"}
    """
    body = _parse_json_body(request)
    if body is None:
        return _error_response("Invalid JSON body")

    try:
        data = _validate(BotMatchStartSerializer, body)
    except ValidationError as e:
        return _error_response(str(e.detail))

    session_id = uuid.uuid4().hex[:12]
    starting_fen = chess.STARTING_FEN

    session = GameSession(
        session_id=session_id,
        fen=starting_fen,
        pgn_moves=[],
        bot_elo=data["elo"],
        bot_opening=data.get("opening") or None,
        player_color=data["player_color"],
        coaching_mode=data["coaching_mode"],
        skill_level=data["skill_level"],
        move_count=0,
        mode="bot_match",
    )

    from django.conf import settings
    sessions = SessionManager(settings.REDIS_URL)
    await sessions.create(session)

    response_data = {
        "session_id": session_id,
        "fen": starting_fen,
        "player_color": session.player_color,
    }

    # If the player chose black, the bot moves first
    if session.player_color == "black":
        from game.bot import BotConfig, BotService
        engine = get_engine_service()
        bot = BotService(engine)
        board = chess.Board(starting_fen)
        config = BotConfig(elo=session.bot_elo, opening_book_path=session.bot_opening)
        bot_move = await bot.get_move(board, config)
        bot_move_san = board.san(bot_move)
        board.push(bot_move)
        await sessions.update_after_move(session_id, board.fen(), bot_move_san)
        response_data["first_move"] = bot_move.uci()
        response_data["fen"] = board.fen()

    return JsonResponse(response_data)


@csrf_exempt
@require_POST
async def opening_lab_start(request):
    """
    Create a new opening lab session. Returns a session_id for the WebSocket.

    Request body: {"eco_code": "C60", "player_color": "white", "skill_level": "intermediate"}
    Response: {"session_id": "...", "fen": "...", "opening_eco": "C60"}
    """
    body = _parse_json_body(request)
    if body is None:
        return _error_response("Invalid JSON body")

    try:
        data = _validate(OpeningLabStartSerializer, body)
    except ValidationError as e:
        return _error_response(str(e.detail))

    session_id = uuid.uuid4().hex[:12]
    starting_fen = chess.STARTING_FEN

    session = GameSession(
        session_id=session_id,
        fen=starting_fen,
        pgn_moves=[],
        bot_elo=2000,  # Opening lab bot plays at full strength
        bot_opening=None,  # Will use ECO book lookup instead
        player_color=data["player_color"],
        coaching_mode="nudge",
        skill_level=data["skill_level"],
        move_count=0,
        mode="opening_lab",
        opening_eco=data["eco_code"],
    )

    from django.conf import settings
    sessions = SessionManager(settings.REDIS_URL)
    await sessions.create(session)

    return JsonResponse({
        "session_id": session_id,
        "fen": starting_fen,
        "opening_eco": session.opening_eco,
        "player_color": session.player_color,
    })
