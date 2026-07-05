"""
REST API views (native async Django views).

Endpoints:
  POST /api/game-review/       — submit a PGN for analysis (SSE streaming)
  POST /api/position-explorer/ — analyze a single FEN position
"""

from __future__ import annotations

import asyncio
import json

import chess
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from rest_framework.exceptions import ValidationError

from analysis.context import ContextAssembler, format_candidates
from analysis.priority import classify
from api.serializers import (
    ChessComImportSerializer,
    GameReviewSerializer,
    PositionChatSerializer,
    PositionExplorerSerializer,
)
from chess_engine.service import get_engine_service
from chesscom import storage as chesscom_storage
from chesscom.client import ChessComError, PlayerNotFoundError, RateLimitedError
from chesscom.importer import GameImporter
from explanation.chat import get_chat_agent
from explanation.generator import get_explainer
from explanation.prompts import build_chat_system_prompt, format_position_context
from explanation.providers import make_llm_config
from game.analyzer import GameAnalyzer


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


def _build_llm_config(request, data: dict):
    """Resolve LLMConfig from request headers (middleware) + body fields.

    Body fields take precedence over headers so that per-request overrides
    from programmatic API callers work without modifying headers.
    """
    provider = data.get("llm_provider") or getattr(request, "llm_provider", None)
    model = data.get("llm_model") or getattr(request, "llm_model", None)
    reasoning = data.get("llm_reasoning_effort") or getattr(request, "llm_reasoning_effort", None)

    # Pick the correct API key based on the resolved provider
    if provider == "openai" or (model and not model.startswith("claude-")):
        api_key = getattr(request, "openai_api_key", None)
    else:
        api_key = getattr(request, "anthropic_api_key", None)

    return make_llm_config(
        provider=provider,
        model=model or None,
        api_key=api_key,
        reasoning_effort=reasoning or None,
    )


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

    llm_config = _build_llm_config(request, data)
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
            llm_config=llm_config,
        ):
            payload = {
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
            yield f"data: {json.dumps(payload)}\n\n"

        # Generate and send the final game summary
        summary = await analyzer.generate_summary(llm_config=llm_config)
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

    candidates = format_candidates(board, result)

    llm_config = _build_llm_config(request, data)
    explainer = get_explainer()
    explanation = await explainer.generate(context, priority, skill_level, llm_config=llm_config)

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
async def position_chat(request):
    """
    Chat about a position. Stateless: the client sends the FEN and the full
    visible thread; the server rebuilds the engine analysis as context and runs
    the agentic loop (the LLM may call Stockfish via the analyze_position tool).

    Request body: {"fen": "...", "skill_level": "...", "messages": [{"role", "content"}, ...]}
    Response: {"reply": "...", "tool_calls": [{"tool", "fen"}, ...]}
    """
    body = _parse_json_body(request)
    if body is None:
        return _error_response("Invalid JSON body")

    try:
        data = _validate(PositionChatSerializer, body)
    except ValidationError as e:
        return _error_response(str(e.detail))

    fen = data["fen"]
    skill_level = data.get("skill_level", "intermediate")
    engine_id = data.get("engine_id") or None
    messages = [dict(m) for m in data["messages"]]

    engine_defaults = {**_engine_defaults()}
    if engine_id:
        engine_defaults["engine_id"] = engine_id

    engine = get_engine_service()
    result = await engine.analyze(fen, depth=20, **engine_defaults)

    board = chess.Board(fen)
    context = _assembler.assemble(board, result)
    candidates = format_candidates(board, result)
    position_context = format_position_context(context, candidates, board)
    system = build_chat_system_prompt(skill_level, position_context)

    llm_config = _build_llm_config(request, data)
    chat_result = await get_chat_agent().run(
        messages,
        system=system,
        llm_config=llm_config,
        engine_defaults=engine_defaults,
    )

    return JsonResponse({
        "reply": chat_result.reply,
        "tool_calls": chat_result.tool_calls,
    })


@csrf_exempt
@require_POST
async def chesscom_import(request):
    """
    Import a player's game history from chess.com into the local games folder.

    Request body: {"username": "hikaru"}
    Response: text/event-stream with per-archive progress events, then
    {"done": true, "total_imported": N, "total_games": M}
    """
    body = _parse_json_body(request)
    if body is None:
        return _error_response("Invalid JSON body")

    try:
        data = _validate(ChessComImportSerializer, body)
    except ValidationError as e:
        return _error_response(str(e.detail))

    username = data["username"]
    importer = GameImporter(username)

    # Pre-flight the archive list so bad usernames / rate limits become real
    # HTTP errors instead of failing after the SSE stream has started.
    try:
        archives = await importer.fetch_archives()
    except PlayerNotFoundError:
        await importer.aclose()
        return _error_response(f"chess.com user '{username}' not found", status=404)
    except RateLimitedError as e:
        await importer.aclose()
        return _error_response(str(e), status=503)
    except ChessComError as e:
        await importer.aclose()
        return _error_response(str(e), status=502)

    async def event_stream():
        try:
            async for event in importer.run(archives):
                yield f"data: {json.dumps(event)}\n\n"
        except ChessComError as e:
            # Archives saved before the failure stay on disk; re-run resumes.
            yield f'data: {json.dumps({"error": str(e), "resumable": True})}\n\n'
        finally:
            await importer.aclose()

    return StreamingHttpResponse(event_stream(), content_type="text/event-stream")


async def imported_games_list(request):
    """
    List imported chess.com games.

    Without params: {"players": [{"username", "game_count", "updated_at"}, ...]}
    With ?username=x: {"username": "x", "games": [ ...metadata records... ]}
    """
    username = request.GET.get("username")
    if username is None:
        players = await asyncio.to_thread(chesscom_storage.list_players)
        return JsonResponse({"players": players})

    if not chesscom_storage.USERNAME_RE.match(username):
        return _error_response("Invalid username")
    games = await asyncio.to_thread(chesscom_storage.list_games, username)
    return JsonResponse({"username": username.lower(), "games": games})


async def imported_game_pgn(request, username, game_id):
    """
    Fetch one imported game's PGN plus its metadata record.

    Response: {"pgn": "...", "player_color": "white", ...}
    """
    if not chesscom_storage.USERNAME_RE.match(username):
        return _error_response("Invalid username")

    def _load():
        index = chesscom_storage.load_index(username)
        record = index["games"].get(game_id)
        if record is None:
            return None
        return record, chesscom_storage.read_pgn(username, game_id)

    result = await asyncio.to_thread(_load)
    if result is None or result[1] is None:
        return _error_response("game not found", status=404)
    record, pgn = result
    return JsonResponse({**record, "pgn": pgn})
