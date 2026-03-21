"""
Tactical pattern detection using python-chess attack maps.

Detects: forks, pins, skewers, discovered attacks, back-rank weaknesses.
Returns a list of human-readable pattern names present in the position.
"""

from __future__ import annotations

import chess


def detect_tactics(board: chess.Board) -> list[str]:
    patterns = []

    if _has_fork(board):
        patterns.append("fork")
    if _has_pin(board):
        patterns.append("pin")
    if _has_skewer(board):
        patterns.append("skewer")
    if _has_discovered_attack(board):
        patterns.append("discovered attack")
    if _has_back_rank_weakness(board):
        patterns.append("back-rank weakness")

    return patterns


def _has_fork(board: chess.Board) -> bool:
    """Detect if the side to move has a piece attacking two or more valuable targets."""
    side = board.turn
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.color != side:
            continue
        attacked = board.attacks(square)
        valuable_targets = [
            sq for sq in attacked
            if board.piece_at(sq) is not None
            and board.piece_at(sq).color != side
            and board.piece_at(sq).piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.KING)
        ]
        if len(valuable_targets) >= 2:
            return True
    return False


def _has_pin(board: chess.Board) -> bool:
    """Detect if any piece is pinned (moving it would expose the king)."""
    side = board.turn
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.color != side or piece.piece_type == chess.KING:
            continue
        if board.is_pinned(side, square):
            return True
    return False


def _has_skewer(board: chess.Board) -> bool:
    """
    Detect skewers: a sliding piece attacks a valuable piece which, when moved,
    exposes a less valuable piece behind it on the same ray.

    Algorithm:
      1. For each enemy sliding piece (bishop, rook, queen):
      2. Cast rays in each direction the piece can move
      3. Find the first two pieces on that ray
      4. If both are friendly to the side to move, and the first piece is
         more valuable than the second → skewer detected
    """
    side = board.turn
    enemy = not side

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.color != enemy:
            continue
        if piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue

        # Get the ray directions this piece type can use
        if piece.piece_type == chess.BISHOP:
            ray_dirs = _DIAGONAL_DIRS
        elif piece.piece_type == chess.ROOK:
            ray_dirs = _ORTHOGONAL_DIRS
        else:  # queen
            ray_dirs = _DIAGONAL_DIRS + _ORTHOGONAL_DIRS

        for d_file, d_rank in ray_dirs:
            pieces_on_ray = _pieces_along_ray(board, square, d_file, d_rank)
            if len(pieces_on_ray) < 2:
                continue
            first_sq, first_piece = pieces_on_ray[0]
            second_sq, second_piece = pieces_on_ray[1]

            # Both must be side-to-move's pieces, first more valuable than second
            if first_piece.color == side and second_piece.color == side:
                if _piece_value(first_piece.piece_type) > _piece_value(second_piece.piece_type):
                    return True

    return False


def _has_discovered_attack(board: chess.Board) -> bool:
    """
    Detect potential discovered attacks: moving a piece reveals an attack
    from a sliding piece behind it.

    Algorithm:
      1. For each friendly sliding piece (bishop, rook, queen):
      2. Check if there is exactly one friendly piece between it and an
         enemy valuable piece on the same ray
      3. If that blocking piece has a legal move that clears the ray → discovered attack

    Pseudocode for full implementation:
      for each friendly slider:
        for each ray direction:
          find pieces on ray
          if first piece is friendly (blocker) and second piece is enemy (target):
            if blocker has any legal move that leaves the ray:
              return True  # discovered attack available
    """
    side = board.turn

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.color != side:
            continue
        if piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue

        if piece.piece_type == chess.BISHOP:
            ray_dirs = _DIAGONAL_DIRS
        elif piece.piece_type == chess.ROOK:
            ray_dirs = _ORTHOGONAL_DIRS
        else:
            ray_dirs = _DIAGONAL_DIRS + _ORTHOGONAL_DIRS

        for d_file, d_rank in ray_dirs:
            pieces_on_ray = _pieces_along_ray(board, square, d_file, d_rank)
            if len(pieces_on_ray) < 2:
                continue
            blocker_sq, blocker = pieces_on_ray[0]
            target_sq, target = pieces_on_ray[1]

            # Blocker is friendly, target is enemy and valuable
            if (
                blocker.color == side
                and target.color != side
                and target.piece_type in (chess.QUEEN, chess.ROOK, chess.KING)
            ):
                return True

    return False


def _has_back_rank_weakness(board: chess.Board) -> bool:
    """Detect if the king is on the back rank with no escape squares (potential back-rank mate)."""
    for color in (chess.WHITE, chess.BLACK):
        king_sq = board.king(color)
        if king_sq is None:
            continue
        back_rank = chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8
        if not (chess.BB_SQUARES[king_sq] & back_rank):
            continue
        # Check if king has any flight squares off the back rank that
        # are not blocked by own pieces
        own_pieces = board.occupied_co[color]
        king_attacks = board.attacks(king_sq)
        # Escape squares = attacked by king, not on back rank, not occupied by own pieces
        escape_squares = king_attacks & ~back_rank & ~own_pieces
        if not escape_squares:
            return True
    return False


# --- Ray helpers ---

_ORTHOGONAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
_DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def _piece_value(piece_type: chess.PieceType) -> int:
    return _PIECE_VALUES.get(piece_type, 0)


def _pieces_along_ray(
    board: chess.Board,
    from_square: chess.Square,
    d_file: int,
    d_rank: int,
) -> list[tuple[chess.Square, chess.Piece]]:
    """Walk along a ray from from_square and return pieces encountered in order."""
    f = chess.square_file(from_square)
    r = chess.square_rank(from_square)
    found = []

    for step in range(1, 8):
        nf = f + d_file * step
        nr = r + d_rank * step
        if not (0 <= nf <= 7 and 0 <= nr <= 7):
            break
        sq = chess.square(nf, nr)
        piece = board.piece_at(sq)
        if piece is not None:
            found.append((sq, piece))
            if len(found) == 2:
                break

    return found
