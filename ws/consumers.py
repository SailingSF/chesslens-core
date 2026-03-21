"""
Django Channels WebSocket consumers.

BotMatchConsumer   — handles live bot match sessions
OpeningLabConsumer — handles opening lab sessions

Message protocol (JSON):

Client → Server:
  {"type": "player_move", "move": "e2e4"}
  {"type": "resign"}
  {"type": "request_hint"}

Server → Client:
  {"type": "bot_move", "move": "e7e5", "fen": "...", "move_number": 2}
  {"type": "coach_nudge", "text": "..."}
  {"type": "game_over", "result": "1-0", "reason": "checkmate"}
  {"type": "theory_deviation", "book_move": "Nf3", "opening": "Ruy Lopez", "explanation": "..."}
  {"type": "error", "message": "..."}
"""

from __future__ import annotations

import json

import chess
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings

from analysis.context import ContextAssembler
from analysis.priority import classify
from chess_engine.service import get_engine_service
from explanation.coach import get_coach
from explanation.generator import get_explainer
from explanation.templates.opening_lab import build_opening_lab_prompt
from game.bot import BotConfig, BotService, OpeningLabBotService
from game.session import SessionManager


# Shared stateless services (created once, reused across all consumers)
_assembler = ContextAssembler()


def _get_session_manager() -> SessionManager:
    return SessionManager(settings.REDIS_URL)


class BotMatchConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session_id = self.scope["url_route"]["kwargs"]["session_id"]
        self._engine = get_engine_service()
        self._bot = BotService(self._engine)
        self._sessions = _get_session_manager()
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data: str):
        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            await self._send_error("invalid JSON")
            return

        msg_type = data.get("type")

        if msg_type == "player_move":
            await self._handle_player_move(data)
        elif msg_type == "resign":
            await self._send({"type": "game_over", "result": "resign", "reason": "resignation"})
            await self._sessions.delete(self.session_id)
        elif msg_type == "request_hint":
            await self._handle_hint()
        else:
            await self._send_error(f"unknown message type: {msg_type!r}")

    async def _handle_player_move(self, data: dict):
        session = await self._sessions.get(self.session_id)
        if session is None:
            await self._send_error("session not found")
            return

        move_uci = data.get("move", "")
        board = chess.Board(session.fen)

        try:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                raise ValueError("illegal move")
        except (ValueError, chess.InvalidMoveError) as e:
            await self._send_error(f"invalid move: {e}")
            return

        # Compute SAN BEFORE pushing (san() requires the move to be legal in current position)
        player_move_san = board.san(move)
        board.push(move)
        await self._sessions.update_after_move(self.session_id, board.fen(), player_move_san)

        if board.is_game_over():
            result = board.result()
            await self._send({"type": "game_over", "result": result, "reason": "game_over"})
            return

        # Bot response
        config = BotConfig(elo=session.bot_elo, opening_book_path=session.bot_opening)
        bot_move = await self._bot.get_move(board, config)
        bot_move_san = board.san(bot_move)  # SAN before push
        board.push(bot_move)
        await self._sessions.update_after_move(self.session_id, board.fen(), bot_move_san)

        await self._send({
            "type": "bot_move",
            "move": bot_move.uci(),
            "fen": board.fen(),
            "move_number": board.fullmove_number,
        })

        if board.is_game_over():
            await self._send({"type": "game_over", "result": board.result(), "reason": "game_over"})
            return

        # Coach nudge (if coaching mode is not silent)
        if session.coaching_mode != "silent":
            engine_result = await self._engine.analyze(board.fen(), depth=14, multipv=3)
            context = _assembler.assemble(board, engine_result)
            priority = classify(engine_result, board=board)

            should_nudge = (
                session.coaching_mode == "verbose"
                or (context.played_move_cp_loss or 0) >= 30
            )
            if should_nudge:
                coach = get_coach()
                nudge = await coach.generate(context, priority, session.skill_level)
                await self._send({"type": "coach_nudge", "text": nudge})

    async def _handle_hint(self):
        session = await self._sessions.get(self.session_id)
        if session is None:
            await self._send_error("session not found")
            return
        board = chess.Board(session.fen)
        engine_result = await self._engine.analyze(board.fen(), depth=14, multipv=3)
        context = _assembler.assemble(board, engine_result)
        priority = classify(engine_result, board=board)
        coach = get_coach()
        nudge = await coach.generate(context, priority, session.skill_level)
        await self._send({"type": "coach_nudge", "text": nudge})

    async def _send(self, data: dict):
        await self.send(text_data=json.dumps(data))

    async def _send_error(self, message: str):
        await self._send({"type": "error", "message": message})


class OpeningLabConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session_id = self.scope["url_route"]["kwargs"]["session_id"]
        self._engine = get_engine_service()
        self._bot = OpeningLabBotService(self._engine)
        self._sessions = _get_session_manager()
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data: str):
        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            await self._send_error("invalid JSON")
            return

        msg_type = data.get("type")
        if msg_type == "player_move":
            await self._handle_player_move(data)
        else:
            await self._send_error(f"unknown message type: {msg_type!r}")

    async def _handle_player_move(self, data: dict):
        session = await self._sessions.get(self.session_id)
        if session is None:
            await self._send_error("session not found")
            return

        board = chess.Board(session.fen)
        move_uci = data.get("move", "")

        try:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                raise ValueError("illegal move")
        except (ValueError, chess.InvalidMoveError) as e:
            await self._send_error(f"invalid move: {e}")
            return

        # Check theory deviation BEFORE pushing
        config = BotConfig(elo=2000, opening_book_path=session.bot_opening)
        book_move = self._bot._strict_book_move(board, config)

        # Compute SAN before push
        player_move_san = board.san(move)
        board.push(move)
        await self._sessions.update_after_move(self.session_id, board.fen(), player_move_san)

        if book_move and move != book_move:
            # Theory deviation — generate opening lab explanation
            # We need SAN of the book move in the PRE-push position, but we already
            # pushed. Use a temp board at the pre-push state.
            pre_push = board.copy()
            pre_push.pop()
            book_move_san = pre_push.san(book_move)

            engine_result = await self._engine.analyze(board.fen(), depth=14, multipv=1)
            context = _assembler.assemble(board, engine_result)
            priority = classify(engine_result, board=board)

            explainer = get_explainer()
            prompt = build_opening_lab_prompt(
                context,
                priority,
                session.skill_level,
                book_move_san=book_move_san,
                opening_name=session.opening_eco or "this opening",
                eco_code=session.opening_eco or "",
            )
            explanation = await explainer.generate(context, priority, session.skill_level)
            await self._send({
                "type": "theory_deviation",
                "book_move": book_move_san,
                "opening": session.opening_eco or "",
                "explanation": explanation,
            })

        # Bot response
        bot_move = await self._bot.get_move(board, config)
        bot_move_san = board.san(bot_move)  # SAN before push
        board.push(bot_move)
        await self._sessions.update_after_move(self.session_id, board.fen(), bot_move_san)

        await self._send({
            "type": "bot_move",
            "move": bot_move.uci(),
            "fen": board.fen(),
            "move_number": board.fullmove_number,
        })

    async def _send(self, data: dict):
        await self.send(text_data=json.dumps(data))

    async def _send_error(self, message: str):
        await self._send({"type": "error", "message": message})
