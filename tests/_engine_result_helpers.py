import chess

from chess_engine.service import CandidateMove, EngineResult


def make_result(candidates: list[tuple], fen: str | None = None) -> EngineResult:
    """Build EngineResult from (score_cp, mate_in) tuples."""
    board = chess.Board(fen) if fen else chess.Board()
    moves = list(board.legal_moves)
    built_candidates = []
    for i, (cp, mate) in enumerate(candidates):
        built_candidates.append(
            CandidateMove(
                move=moves[i],
                score_cp=cp,
                mate_in=mate,
                pv=[moves[i]],
            )
        )
    return EngineResult(fen=board.fen(), depth=20, candidates=built_candidates)


def make_result_with_wdl(
    candidates: list[tuple], fen: str | None = None
) -> EngineResult:
    """Build EngineResult from (score_cp, mate_in, wdl_win, wdl_draw, wdl_loss) tuples."""
    board = chess.Board(fen) if fen else chess.Board()
    moves = list(board.legal_moves)
    built_candidates = []
    for i, (cp, mate, ww, wd, wl) in enumerate(candidates):
        built_candidates.append(
            CandidateMove(
                move=moves[i],
                score_cp=cp,
                mate_in=mate,
                pv=[moves[i]],
                wdl_win=ww,
                wdl_draw=wd,
                wdl_loss=wl,
            )
        )
    return EngineResult(fen=board.fen(), depth=20, candidates=built_candidates)
