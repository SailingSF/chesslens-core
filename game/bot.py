"""
BotService — Elo-calibrated bot move generation.

Wraps EngineService with:
  - UCI_LimitStrength + UCI_Elo for target rating
  - Reduced depth at lower Elo levels
  - Opening book integration via python-chess polyglot reader

Deviation rates per Elo band (probability of ignoring book move):
  800  → 20%
  1200 → 10%
  1600 →  5%
  2000 →  0%
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import chess
import chess.polyglot

from chess_engine.service import EngineService


@dataclass
class BotConfig:
    elo: int
    opening_book_path: Optional[str] = None
    deviation_rate: Optional[float] = None   # None = derive from elo

    def effective_deviation_rate(self) -> float:
        if self.deviation_rate is not None:
            return self.deviation_rate
        if self.elo >= 2000:
            return 0.0
        if self.elo >= 1600:
            return 0.05
        if self.elo >= 1200:
            return 0.10
        return 0.20

    def depth(self) -> int:
        if self.elo >= 2000:
            return 18
        if self.elo >= 1600:
            return 14
        if self.elo >= 1200:
            return 12
        return 8


class BotService:
    def __init__(self, engine: EngineService):
        self._engine = engine

    async def get_move(self, board: chess.Board, config: BotConfig) -> chess.Move:
        """Return the bot's chosen move for the current position."""
        # Try opening book first
        book_move = self._book_move(board, config)
        if book_move is not None:
            return book_move

        # Fall back to Elo-constrained engine
        uci_options = {
            "UCI_LimitStrength": True,
            "UCI_Elo": config.elo,
        }
        result = await self._engine.analyze(
            board.fen(),
            depth=config.depth(),
            multipv=1,
            uci_options=uci_options,
        )
        return result.best.move

    def _book_move(self, board: chess.Board, config: BotConfig) -> Optional[chess.Move]:
        if not config.opening_book_path or not os.path.exists(config.opening_book_path):
            return None
        if random.random() < config.effective_deviation_rate():
            return None  # Deliberate deviation from book
        try:
            with chess.polyglot.open_reader(config.opening_book_path) as reader:
                entry = reader.weighted_choice(board)
                return entry.move
        except (IndexError, KeyError):
            return None  # Position not in book


class OpeningLabBotService(BotService):
    """
    Opening lab variant — always plays the book continuation with zero deviation.
    Falls back to full-strength Stockfish when beyond book coverage.
    """

    async def get_move(self, board: chess.Board, config: BotConfig) -> chess.Move:
        book_move = self._strict_book_move(board, config)
        if book_move is not None:
            return book_move
        # Beyond book: full-strength engine
        result = await self._engine.analyze(board.fen(), depth=20, multipv=1)
        return result.best.move

    def _strict_book_move(self, board: chess.Board, config: BotConfig) -> Optional[chess.Move]:
        if not config.opening_book_path or not os.path.exists(config.opening_book_path):
            return None
        try:
            with chess.polyglot.open_reader(config.opening_book_path) as reader:
                entry = reader.find(board)
                return entry.move
        except (IndexError, KeyError):
            return None
