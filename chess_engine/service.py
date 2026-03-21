"""
EngineService — interface and implementations.

Implementations:
  - PooledEngineService — reuses Stockfish processes via EnginePool (default)
  - RemoteEngineService — HTTP dispatch to engine container (Docker mode)

Factory:
  get_engine_service() returns the correct implementation based on settings.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import chess
import chess.engine

from chess_engine.pool import EnginePool


@dataclass
class CandidateMove:
    """A single engine candidate with its evaluation."""
    move: chess.Move
    score_cp: Optional[int]        # centipawns (None if mate)
    mate_in: Optional[int]         # moves to mate (None if not mate)
    pv: list[chess.Move] = field(default_factory=list)


@dataclass
class EngineResult:
    """Structured output from a single-position analysis."""
    fen: str
    depth: int
    candidates: list[CandidateMove]   # multi-PV results, best move first

    @property
    def best(self) -> CandidateMove:
        return self.candidates[0]


class EngineService:
    """
    Abstract interface for the chess engine layer.

    All implementations must provide analyze() with the same signature.
    """

    async def analyze(
        self,
        fen: str,
        *,
        depth: int = 20,
        multipv: int = 3,
        uci_options: Optional[dict] = None,
    ) -> EngineResult:
        raise NotImplementedError

    async def shutdown(self) -> None:
        """Clean up resources. Called on application shutdown."""
        pass


class PooledEngineService(EngineService):
    """
    Reuses Stockfish processes via an EnginePool.

    The pool keeps N Stockfish subprocesses alive across requests,
    avoiding the ~200ms cold-start cost per analysis call.
    """

    def __init__(self, pool: EnginePool):
        self._pool = pool
        self._started = False

    async def _ensure_started(self) -> None:
        if not self._started:
            await self._pool.start()
            self._started = True

    async def analyze(
        self,
        fen: str,
        *,
        depth: int = 20,
        multipv: int = 3,
        uci_options: Optional[dict] = None,
    ) -> EngineResult:
        await self._ensure_started()

        board = chess.Board(fen)

        async with self._pool.acquire() as engine:
            # Apply per-request UCI options (e.g. UCI_LimitStrength, UCI_Elo)
            if uci_options:
                await engine.configure(uci_options)

            info_list = await engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
                multipv=multipv,
            )

            # Reset Elo constraints after request so next caller gets clean state
            if uci_options and "UCI_LimitStrength" in uci_options:
                await engine.configure({"UCI_LimitStrength": False})

        # python-chess engine.analyse with multipv returns a list of InfoDicts
        if not isinstance(info_list, list):
            info_list = [info_list]

        candidates = _parse_candidates(info_list)
        return EngineResult(fen=fen, depth=depth, candidates=candidates)

    async def shutdown(self) -> None:
        if self._started:
            await self._pool.stop()
            self._started = False


class RemoteEngineService(EngineService):
    """
    Dispatches analysis to the engine service container via HTTP.

    Used when ENGINE_URL is set (Docker mode). The engine container runs
    engine_service/server.py and exposes POST /analyze.
    """

    def __init__(self, engine_url: str):
        self._engine_url = engine_url.rstrip("/")

    async def analyze(
        self,
        fen: str,
        *,
        depth: int = 20,
        multipv: int = 3,
        uci_options: Optional[dict] = None,
    ) -> EngineResult:
        import httpx

        payload = {
            "fen": fen,
            "depth": depth,
            "multipv": multipv,
            "uci_options": uci_options or {},
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{self._engine_url}/analyze", json=payload)
            resp.raise_for_status()
            data = resp.json()

        board = chess.Board(fen)
        candidates = []
        for c in data.get("candidates", []):
            move = chess.Move.from_uci(c["move"])
            pv = [chess.Move.from_uci(m) for m in c.get("pv", [])]
            candidates.append(CandidateMove(
                move=move,
                score_cp=c.get("score_cp"),
                mate_in=c.get("mate_in"),
                pv=pv,
            ))

        return EngineResult(fen=fen, depth=depth, candidates=candidates)


def _parse_candidates(info_list: list) -> list[CandidateMove]:
    """Convert python-chess InfoDict list to CandidateMove list."""
    candidates = []
    for info in info_list:
        pv = list(info.get("pv", []))
        move = pv[0] if pv else None
        if move is None:
            continue
        score = info.get("score")
        score_cp = None
        mate_in = None
        if score is not None:
            pov = score.white()
            if pov.is_mate():
                mate_in = pov.mate()
            else:
                score_cp = pov.score()
        candidates.append(CandidateMove(
            move=move, score_cp=score_cp, mate_in=mate_in, pv=pv
        ))
    return candidates


# ---------------------------------------------------------------------------
# Singleton factory — returns the right implementation based on Django settings
# ---------------------------------------------------------------------------

_engine_service: Optional[EngineService] = None


def get_engine_service() -> EngineService:
    """
    Return a shared EngineService instance.

    - If ENGINE_URL is set → RemoteEngineService (Docker / cloud)
    - Otherwise → PooledEngineService with a local Stockfish pool
    """
    global _engine_service
    if _engine_service is not None:
        return _engine_service

    from django.conf import settings

    engine_url = getattr(settings, "ENGINE_URL", None)
    if engine_url:
        _engine_service = RemoteEngineService(engine_url)
    else:
        import os
        cpu_count = os.cpu_count() or 2
        # Use 2 engine processes with multiple threads each.
        # This is faster than many single-threaded processes because
        # Stockfish scales well with threads for a single search.
        pool_size = min(cpu_count, 2)
        threads_per_engine = max(1, cpu_count // pool_size)
        pool = EnginePool(size=pool_size, threads=threads_per_engine)
        _engine_service = PooledEngineService(pool)

    return _engine_service
