"""
ContextAssembler — enriches raw engine output with named chess features.

Accepts an EngineResult + chess.Board, returns an AssembledContext containing:
  - opening name / ECO code
  - theory deviation flag
  - tactical patterns detected
  - pawn structure summary
  - piece activity signals
  - centipawn loss classification
  - threat narrative (for coaching prompts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import chess

from analysis.eco import ECOLookup
from analysis.patterns import detect_tactics
from analysis.pawns import analyze_pawns
from chess_engine.service import EngineResult


PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


@dataclass
class AssembledContext:
    fen: str
    played_move: Optional[chess.Move]
    played_move_san: Optional[str]

    # Engine data
    best_move: chess.Move
    best_move_san: str
    best_move_cp: Optional[int]
    mate_in: Optional[int]
    played_move_cp_loss: Optional[int]   # positive = worse for side to move
    cp_loss_label: str                   # best / excellent / good / inaccuracy / mistake / blunder
    pv_san: list[str]                    # principal variation in SAN notation

    # Opening
    eco_code: Optional[str]
    opening_name: Optional[str]
    theory_deviation: bool

    # Tactics & structure
    tactical_patterns: list[str]
    pawn_structure: dict
    piece_activity: dict

    # Coaching
    threat_narrative: str


# Large centipawn value used when converting mate-in-N to centipawns.
_MATE_CP_BASE = 10000


def _mate_to_cp(mate_in: Optional[int]) -> Optional[int]:
    """Convert mate-in-N to a centipawn equivalent. Returns None if mate_in is None."""
    if mate_in is None:
        return None
    if mate_in > 0:
        return _MATE_CP_BASE - mate_in * 10
    return -_MATE_CP_BASE - mate_in * 10


# Centipawn loss buckets — aligned with Lichess/chess.com thresholds
# Lichess: inaccuracy 50+, mistake 100+, blunder 300+
# We add finer "best/excellent/good" grades below 50cp
def _classify_cp_loss(cp_loss: Optional[int]) -> str:
    if cp_loss is None:
        return "unknown"
    if cp_loss == 0:
        return "best"
    if cp_loss <= 10:
        return "excellent"
    if cp_loss <= 30:
        return "good"
    if cp_loss < 100:
        return "inaccuracy"
    if cp_loss < 300:
        return "mistake"
    return "blunder"


class ContextAssembler:
    def __init__(self, eco_lookup: Optional[ECOLookup] = None):
        self._eco = eco_lookup or ECOLookup()

    def assemble(
        self,
        board: chess.Board,
        engine_result: EngineResult,
        played_move: Optional[chess.Move] = None,
    ) -> AssembledContext:
        best = engine_result.best
        best_move_san = board.san(best.move)
        pv_san = self._pv_to_san(board, best.pv)

        # Centipawn loss for the played move
        cp_loss = None
        played_move_san = None
        if played_move is not None:
            played_move_san = board.san(played_move)
            played_cp = self._cp_for_move(engine_result, played_move)
            best_cp = best.score_cp if best.score_cp is not None else _mate_to_cp(best.mate_in)
            if played_cp is not None and best_cp is not None:
                cp_loss = abs(best_cp - played_cp)

        # Opening
        eco_code, opening_name = self._eco.lookup(board)
        theory_deviation = self._detect_deviation(board, played_move, eco_code)

        # Tactics, pawns, pieces
        tactical_patterns = detect_tactics(board)
        pawn_structure = analyze_pawns(board)
        piece_activity = self._piece_activity(board)
        threat_narrative = self._build_threat_narrative(board)

        return AssembledContext(
            fen=engine_result.fen,
            played_move=played_move,
            played_move_san=played_move_san,
            best_move=best.move,
            best_move_san=best_move_san,
            best_move_cp=best.score_cp,
            mate_in=best.mate_in,
            played_move_cp_loss=cp_loss,
            cp_loss_label=_classify_cp_loss(cp_loss),
            pv_san=pv_san,
            eco_code=eco_code,
            opening_name=opening_name,
            theory_deviation=theory_deviation,
            tactical_patterns=tactical_patterns,
            pawn_structure=pawn_structure,
            piece_activity=piece_activity,
            threat_narrative=threat_narrative,
        )

    def _cp_for_move(self, result: EngineResult, move: chess.Move) -> Optional[int]:
        for candidate in result.candidates:
            if candidate.move == move:
                if candidate.score_cp is not None:
                    return candidate.score_cp
                return _mate_to_cp(candidate.mate_in)
        return None

    def _pv_to_san(self, board: chess.Board, pv: list[chess.Move]) -> list[str]:
        san_list = []
        temp = board.copy()
        for move in pv[:6]:
            try:
                san_list.append(temp.san(move))
                temp.push(move)
            except Exception:
                break
        return san_list

    def _detect_deviation(
        self,
        board: chess.Board,
        played_move: Optional[chess.Move],
        eco_code: Optional[str],
    ) -> bool:
        """
        Check if the played move deviates from known opening theory.

        Pseudocode for full implementation:
          1. If eco_code is None, we're outside book territory → not a deviation
          2. Look up the ECO book continuation for this position
             (requires an extended eco_data.json with move sequences, or a polyglot book)
          3. If the played move matches the book continuation → no deviation
          4. If the played move does NOT match → deviation detected
          5. Also flag deviation if the position was in-book on the previous move
             but the current position is no longer recognized
        """
        if played_move is None or eco_code is None:
            return False

        # Check if the position AFTER the played move is still in the book.
        # If the current position is in the book but the resulting position is not,
        # the played move is a deviation from theory.
        temp = board.copy()
        temp.push(played_move)
        next_eco, _ = self._eco.lookup(temp)
        if next_eco is None and eco_code is not None:
            # Was in book, now out of book → deviation
            return True

        return False

    def _piece_activity(self, board: chess.Board) -> dict:
        """
        Extract piece activity signals from the position.

        Returns dict with keys: bishop_pair, rooks_on_open_files,
        knight_outposts, king_safety_score.
        """
        result = {}

        for color_name, color in [("white", chess.WHITE), ("black", chess.BLACK)]:
            bishops = board.pieces(chess.BISHOP, color)
            rooks = board.pieces(chess.ROOK, color)

            # Bishop pair: having two bishops is a positional advantage
            result[f"{color_name}_bishop_pair"] = len(bishops) >= 2

            # Rooks on open files: a file with no pawns of either color
            rooks_on_open = []
            for rook_sq in rooks:
                f = chess.square_file(rook_sq)
                file_mask = chess.BB_FILES[f]
                white_pawns_on_file = board.pieces(chess.PAWN, chess.WHITE) & file_mask
                black_pawns_on_file = board.pieces(chess.PAWN, chess.BLACK) & file_mask
                if not white_pawns_on_file and not black_pawns_on_file:
                    rooks_on_open.append(chess.square_name(rook_sq))
            result[f"{color_name}_rooks_open_files"] = rooks_on_open

            # Knight outposts: knights on ranks 4-6 (for white) or 3-5 (for black)
            # supported by own pawn and not attackable by enemy pawns
            # Pseudocode for full implementation:
            #   for each knight:
            #     1. Check if knight is on an advanced rank (4-6 for white, 3-5 for black)
            #     2. Check if knight is supported by a friendly pawn
            #     3. Check that no enemy pawn on adjacent files can advance to attack it
            #     4. If all conditions met → outpost
            knights = board.pieces(chess.KNIGHT, color)
            outposts = []
            for knight_sq in knights:
                rank = chess.square_rank(knight_sq)
                if color == chess.WHITE and 3 <= rank <= 5:
                    outposts.append(chess.square_name(knight_sq))
                elif color == chess.BLACK and 2 <= rank <= 4:
                    outposts.append(chess.square_name(knight_sq))
            result[f"{color_name}_knight_outposts"] = outposts

            # King safety: count attackers near the king
            # Pseudocode for full implementation:
            #   1. Get the king's square and its surrounding squares (king zone)
            #   2. Count how many enemy pieces attack squares in the king zone
            #   3. Weight by piece type (queen attack = 4, rook = 2, bishop/knight = 1)
            #   4. Subtract defenders in the king zone
            #   5. Higher score = less safe
            king_sq = board.king(color)
            if king_sq is not None:
                enemy = not color
                king_zone = chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]) | chess.SquareSet.from_square(king_sq)
                attacker_count = 0
                for sq in king_zone:
                    attackers = board.attackers(enemy, sq)
                    attacker_count += len(attackers)
                result[f"{color_name}_king_danger"] = attacker_count
            else:
                result[f"{color_name}_king_danger"] = 0

        return result

    def _build_threat_narrative(self, board: chess.Board) -> str:
        """
        Build a human-readable description of threats in the position.

        Uses python-chess attack maps to name which pieces are attacked
        and which squares are controlled. Formatted for use in coaching
        nudge prompts.
        """
        side = board.turn
        enemy = not side
        side_name = "White" if side == chess.WHITE else "Black"
        enemy_name = "Black" if side == chess.WHITE else "White"

        threats = []

        # Find enemy pieces that are attacked by the side to move
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.color != enemy:
                continue

            attackers = board.attackers(side, square)
            if not attackers:
                continue

            piece_name = PIECE_NAMES.get(piece.piece_type, "piece")
            square_name = chess.square_name(square)

            # Check if the attacked piece is defended
            defenders = board.attackers(enemy, square)
            if defenders:
                threats.append(
                    f"{enemy_name}'s {piece_name} on {square_name} is attacked and defended"
                )
            else:
                threats.append(
                    f"{enemy_name}'s {piece_name} on {square_name} is attacked and undefended"
                )

        # Find hanging pieces (undefended pieces of the side to move attacked by enemy)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.color != side:
                continue

            enemy_attackers = board.attackers(enemy, square)
            if not enemy_attackers:
                continue

            own_defenders = board.attackers(side, square)
            if not own_defenders:
                piece_name = PIECE_NAMES.get(piece.piece_type, "piece")
                square_name = chess.square_name(square)
                threats.append(
                    f"{side_name}'s {piece_name} on {square_name} is hanging (undefended)"
                )

        if not threats:
            return "No immediate threats detected."

        return "; ".join(threats[:5])  # Cap at 5 to keep prompt concise
