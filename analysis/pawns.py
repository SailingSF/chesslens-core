"""
Pawn structure analysis.

Detects: isolated, doubled, passed, and backward pawns for each side.
Returns a dict keyed by color with lists of pawn square names.
"""

from __future__ import annotations

import chess


def analyze_pawns(board: chess.Board) -> dict:
    return {
        "white": _analyze_side(board, chess.WHITE),
        "black": _analyze_side(board, chess.BLACK),
    }


def _analyze_side(board: chess.Board, color: chess.Color) -> dict:
    pawns = list(board.pieces(chess.PAWN, color))
    files = [chess.square_file(sq) for sq in pawns]

    return {
        "isolated": [chess.square_name(sq) for sq in pawns if _is_isolated(sq, files)],
        "doubled": [chess.square_name(sq) for sq in pawns if _is_doubled(sq, files)],
        "passed": [chess.square_name(sq) for sq in pawns if _is_passed(sq, color, board)],
    }


def _is_isolated(square: chess.Square, all_files: list[int]) -> bool:
    f = chess.square_file(square)
    return (f - 1) not in all_files and (f + 1) not in all_files


def _is_doubled(square: chess.Square, all_files: list[int]) -> bool:
    return all_files.count(chess.square_file(square)) > 1


def _is_passed(square: chess.Square, color: chess.Color, board: chess.Board) -> bool:
    """A pawn is passed if no enemy pawn can block or capture it on its path."""
    f = chess.square_file(square)
    r = chess.square_rank(square)
    enemy = not color

    if color == chess.WHITE:
        ranks_ahead = range(r + 1, 8)
    else:
        ranks_ahead = range(r - 1, -1, -1)

    adjacent_files = [af for af in (f - 1, f, f + 1) if 0 <= af <= 7]

    for rank in ranks_ahead:
        for af in adjacent_files:
            sq = chess.square(af, rank)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == enemy:
                return False
    return True
