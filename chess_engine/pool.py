"""
Engine process pool — keeps Stockfish instances alive across requests.

Stockfish is expensive to start (loads neural network weights). The pool
keeps N processes alive and hands them out for analysis, returning them
when done. Pool size = number of available CPU cores.

Usage:
    pool = EnginePool(size=2)
    await pool.start()

    async with pool.acquire() as engine:
        result = await engine.analyse(board, chess.engine.Limit(depth=20))

    await pool.stop()
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import chess.engine


class EnginePool:
    def __init__(
        self,
        engine_path: str = "stockfish",
        size: int = 2,
        hash_mb: int = 128,
        threads: int = 1,
    ):
        self._engine_path = engine_path
        self._size = size
        self._hash_mb = hash_mb
        self._threads = threads
        self._queue: asyncio.Queue[chess.engine.UciProtocol] = asyncio.Queue()
        self._engines: list[chess.engine.UciProtocol] = []

    @property
    def size(self) -> int:
        return self._size

    async def start(self) -> None:
        """Initialize the pool by spawning Stockfish processes."""
        for _ in range(self._size):
            _, engine = await chess.engine.popen_uci(self._engine_path)
            # Configure each engine once at startup
            await engine.configure({
                "Hash": self._hash_mb,
                "Threads": self._threads,
            })
            self._engines.append(engine)
            await self._queue.put(engine)

    async def stop(self) -> None:
        """Shut down all Stockfish processes."""
        for engine in self._engines:
            try:
                await engine.quit()
            except Exception:
                pass
        self._engines.clear()

    @asynccontextmanager
    async def acquire(self):
        """Check out an engine from the pool, return it when done."""
        engine = await self._queue.get()
        try:
            yield engine
        finally:
            await self._queue.put(engine)
