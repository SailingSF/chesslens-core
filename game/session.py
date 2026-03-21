"""
SessionManager — Redis-backed state for active bot match / opening lab sessions.

Session TTL: 4 hours, extended on every move.
Key format: game_session:{session_id}

Fields stored:
  fen, pgn_moves, bot_config, player_color, coaching_mode,
  skill_level, move_count, mode, opening_eco (opening lab only)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal, Optional

import redis.asyncio as aioredis

SESSION_TTL = 4 * 60 * 60  # 4 hours in seconds

CoachingMode = Literal["silent", "nudge", "verbose"]
GameMode = Literal["bot_match", "opening_lab"]


@dataclass
class GameSession:
    session_id: str
    fen: str
    pgn_moves: list[str]
    bot_elo: int
    bot_opening: Optional[str]
    player_color: str                    # "white" or "black"
    coaching_mode: CoachingMode
    skill_level: str
    move_count: int
    mode: GameMode
    opening_eco: Optional[str] = None   # opening lab only


class SessionManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self._redis = aioredis.from_url(redis_url, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"game_session:{session_id}"

    async def create(self, session: GameSession) -> None:
        key = self._key(session.session_id)
        data = {
            "fen": session.fen,
            "pgn_moves": json.dumps(session.pgn_moves),
            "bot_elo": session.bot_elo,
            "bot_opening": session.bot_opening or "",
            "player_color": session.player_color,
            "coaching_mode": session.coaching_mode,
            "skill_level": session.skill_level,
            "move_count": session.move_count,
            "mode": session.mode,
            "opening_eco": session.opening_eco or "",
        }
        await self._redis.hset(key, mapping=data)
        await self._redis.expire(key, SESSION_TTL)

    async def get(self, session_id: str) -> Optional[GameSession]:
        key = self._key(session_id)
        data = await self._redis.hgetall(key)
        if not data:
            return None
        return GameSession(
            session_id=session_id,
            fen=data["fen"],
            pgn_moves=json.loads(data["pgn_moves"]),
            bot_elo=int(data["bot_elo"]),
            bot_opening=data.get("bot_opening") or None,
            player_color=data["player_color"],
            coaching_mode=data["coaching_mode"],  # type: ignore[arg-type]
            skill_level=data["skill_level"],
            move_count=int(data["move_count"]),
            mode=data["mode"],  # type: ignore[arg-type]
            opening_eco=data.get("opening_eco") or None,
        )

    async def update_after_move(
        self, session_id: str, new_fen: str, move_san: str
    ) -> None:
        key = self._key(session_id)
        session = await self.get(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} not found")
        session.pgn_moves.append(move_san)
        session.move_count += 1
        session.fen = new_fen
        await self._redis.hset(
            key,
            mapping={
                "fen": new_fen,
                "pgn_moves": json.dumps(session.pgn_moves),
                "move_count": session.move_count,
            },
        )
        await self._redis.expire(key, SESSION_TTL)

    async def delete(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))
