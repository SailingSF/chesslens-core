"""
ECO opening lookup.

Maps the current board position to a named opening and ECO code by matching
the FEN of the position against a lookup table of ~500 ECO codes / 2,000 variations.

The lookup table (eco_data.json) ships with the repo. Format:
  [{"eco": "B90", "name": "Sicilian, Najdorf", "fen": "...stripped FEN..."}]

Stripped FEN: position string only (no move clocks), for reliable matching.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional

import chess

_ECO_DATA_PATH = os.path.join(os.path.dirname(__file__), "eco_data.json")


@lru_cache(maxsize=1)
def _load_eco_data() -> dict[str, tuple[str, str]]:
    """Returns {stripped_fen: (eco_code, opening_name)}."""
    if not os.path.exists(_ECO_DATA_PATH):
        return {}
    with open(_ECO_DATA_PATH) as f:
        entries = json.load(f)
    return {entry["fen"]: (entry["eco"], entry["name"]) for entry in entries}


def _strip_fen(fen: str) -> str:
    """Remove move clocks from FEN for position-only matching."""
    parts = fen.split()
    return " ".join(parts[:4])  # piece placement, active color, castling, en-passant


class ECOLookup:
    def __init__(self):
        self._data = _load_eco_data()

    def lookup(self, board: chess.Board) -> tuple[Optional[str], Optional[str]]:
        """Return (eco_code, opening_name) or (None, None) if not found."""
        key = _strip_fen(board.fen())
        result = self._data.get(key)
        if result:
            return result
        return None, None
