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
class PVEndEval:
    """Re-evaluation of a candidate's PV endpoint.

    Produced when analyze() is called with pv_end_nodes set. The PV is pushed
    on a board copy, then a fresh search is run from the endpoint. Exposes
    quiet-leverage: moves whose root eval is small but whose PV resolves far
    from it. Eval is white-POV (matching CandidateMove.score_cp).
    """
    score_cp: Optional[int]
    mate_in: Optional[int]
    depth: Optional[int]
    seldepth: Optional[int]
    pushed: int               # number of PV plies actually played
    fen: str                  # FEN at the endpoint
    terminal: Optional[str]   # "checkmate" | "stalemate" | "draw" | None


@dataclass
class CandidateMove:
    """A single engine candidate with its evaluation."""
    move: chess.Move
    score_cp: Optional[int]        # centipawns (None if mate)
    mate_in: Optional[int]         # moves to mate (None if not mate)
    pv: list[chess.Move] = field(default_factory=list)
    # Native WDL from Stockfish 16+ (optional, permille 0–1000)
    wdl_win: Optional[int] = None
    wdl_draw: Optional[int] = None
    wdl_loss: Optional[int] = None
    seldepth: Optional[int] = None
    # Populated only when analyze() is called with pv_end_nodes set.
    pv_end: Optional[PVEndEval] = None


@dataclass
class EngineResult:
    """Structured output from a single-position analysis."""
    fen: str
    depth: int
    candidates: list[CandidateMove]   # multi-PV results, best move first
    # Populated only when analyze() is called with played_move_uci set and the
    # played move fell outside the multipv window (searchmoves exact eval).
    played_move: Optional[CandidateMove] = None

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
        nodes: Optional[int] = None,
        multipv: int = 3,
        uci_options: Optional[dict] = None,
        pv_length: Optional[int] = None,
        pv_end_nodes: Optional[int] = None,
        played_move_uci: Optional[str] = None,
        played_move_nodes: Optional[int] = None,
    ) -> EngineResult:
        """
        Analyze a position.

        If nodes is set, uses node-based search (chess.com style) instead of
        depth-based search. When nodes is set, depth is ignored.

        Optional enrichments (all opt-in, produce the same output shape as
        classification_model/engine/run_stockfish.py when set):
          - pv_length: truncate each candidate's PV to this many plies.
          - pv_end_nodes: if set, push each candidate's PV on a board copy and
            re-search the endpoint with this node budget. Populates
            CandidateMove.pv_end.
          - played_move_uci: if set and the move is not in the multipv window,
            run a `searchmoves`-restricted search for an exact eval of that
            move. Populates EngineResult.played_move.
          - played_move_nodes: node budget for the played-move search
            (defaults to the main search's nodes, or None for depth-based).
        """
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
        nodes: Optional[int] = None,
        multipv: int = 3,
        uci_options: Optional[dict] = None,
        pv_length: Optional[int] = None,
        pv_end_nodes: Optional[int] = None,
        played_move_uci: Optional[str] = None,
        played_move_nodes: Optional[int] = None,
    ) -> EngineResult:
        await self._ensure_started()

        board = chess.Board(fen)

        # Node-based search (chess.com style) takes priority over depth
        if nodes is not None:
            limit = chess.engine.Limit(nodes=nodes)
        else:
            limit = chess.engine.Limit(depth=depth)

        # Resolve the played move up front so we can branch on top-N membership.
        played_move: Optional[chess.Move] = None
        if played_move_uci:
            try:
                mv = chess.Move.from_uci(played_move_uci)
                if mv in board.legal_moves:
                    played_move = mv
            except ValueError:
                played_move = None

        pv_end_results: list[Optional[PVEndEval]] = []
        played_move_candidate: Optional[CandidateMove] = None

        async with self._pool.acquire() as engine:
            # Apply per-request UCI options (e.g. UCI_LimitStrength, UCI_Elo)
            if uci_options:
                await engine.configure(uci_options)

            info_list = await engine.analyse(board, limit, multipv=multipv)

            # python-chess engine.analyse with multipv returns a list of InfoDicts
            if not isinstance(info_list, list):
                info_list = [info_list]

            candidates = _parse_candidates(info_list)
            if pv_length is not None:
                for c in candidates:
                    c.pv = c.pv[:pv_length]

            # PV-end re-eval: push each candidate's PV on a board copy and
            # re-search the endpoint with a shallower node budget. Keeps the
            # same engine checked out so transposition tables stay warm.
            if pv_end_nodes is not None:
                pv_end_limit = chess.engine.Limit(nodes=pv_end_nodes)
                for c in candidates:
                    pv_end_results.append(
                        await _compute_pv_end(engine, board, c.pv, pv_end_limit)
                    )

            # Played-move exact eval via searchmoves when it's outside top_moves.
            if played_move is not None:
                in_top = any(c.move == played_move for c in candidates)
                if not in_top:
                    pm_nodes = played_move_nodes if played_move_nodes is not None else nodes
                    pm_limit = (
                        chess.engine.Limit(nodes=pm_nodes)
                        if pm_nodes is not None
                        else chess.engine.Limit(depth=depth)
                    )
                    pm_info = await engine.analyse(
                        board, pm_limit, multipv=1, root_moves=[played_move]
                    )
                    if not isinstance(pm_info, list):
                        pm_info = [pm_info]
                    pm_candidates = _parse_candidates(pm_info)
                    if pm_candidates:
                        played_move_candidate = pm_candidates[0]
                        if pv_length is not None:
                            played_move_candidate.pv = played_move_candidate.pv[:pv_length]

            # Reset Elo constraints after request so next caller gets clean state
            if uci_options and "UCI_LimitStrength" in uci_options:
                await engine.configure({"UCI_LimitStrength": False})

        # Attach pv_end results after releasing the engine.
        if pv_end_nodes is not None:
            for c, pv_end in zip(candidates, pv_end_results):
                c.pv_end = pv_end

        reached_depth = depth
        if info_list and "depth" in info_list[0]:
            reached_depth = info_list[0]["depth"]
        return EngineResult(
            fen=fen,
            depth=reached_depth,
            candidates=candidates,
            played_move=played_move_candidate,
        )

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
        nodes: Optional[int] = None,
        multipv: int = 3,
        uci_options: Optional[dict] = None,
        pv_length: Optional[int] = None,
        pv_end_nodes: Optional[int] = None,
        played_move_uci: Optional[str] = None,
        played_move_nodes: Optional[int] = None,
    ) -> EngineResult:
        import httpx

        payload = {
            "fen": fen,
            "depth": depth,
            "multipv": multipv,
            "uci_options": uci_options or {},
        }
        if nodes is not None:
            payload["nodes"] = nodes
        if pv_length is not None:
            payload["pv_length"] = pv_length
        if pv_end_nodes is not None:
            payload["pv_end_nodes"] = pv_end_nodes
        if played_move_uci is not None:
            payload["played_move_uci"] = played_move_uci
        if played_move_nodes is not None:
            payload["played_move_nodes"] = played_move_nodes

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{self._engine_url}/analyze", json=payload)
            resp.raise_for_status()
            data = resp.json()

        candidates = [_candidate_from_dict(c) for c in data.get("candidates", [])]
        played_move_dict = data.get("played_move")
        played_move = _candidate_from_dict(played_move_dict) if played_move_dict else None

        return EngineResult(
            fen=fen,
            depth=data.get("depth", depth),
            candidates=candidates,
            played_move=played_move,
        )


def _candidate_from_dict(c: dict) -> CandidateMove:
    move = chess.Move.from_uci(c["move"])
    pv = [chess.Move.from_uci(m) for m in c.get("pv", [])]
    pv_end = None
    pv_end_data = c.get("pv_end")
    if pv_end_data:
        pv_end = PVEndEval(
            score_cp=pv_end_data.get("score_cp"),
            mate_in=pv_end_data.get("mate_in"),
            depth=pv_end_data.get("depth"),
            seldepth=pv_end_data.get("seldepth"),
            pushed=pv_end_data.get("pushed", 0),
            fen=pv_end_data.get("fen", ""),
            terminal=pv_end_data.get("terminal"),
        )
    return CandidateMove(
        move=move,
        score_cp=c.get("score_cp"),
        mate_in=c.get("mate_in"),
        pv=pv,
        wdl_win=c.get("wdl_win"),
        wdl_draw=c.get("wdl_draw"),
        wdl_loss=c.get("wdl_loss"),
        seldepth=c.get("seldepth"),
        pv_end=pv_end,
    )


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

        # Parse native WDL from Stockfish 16+ (UCI_ShowWDL)
        # python-chess exposes this as a PovWdl in info["wdl"]
        wdl_win = None
        wdl_draw = None
        wdl_loss = None
        wdl = info.get("wdl")
        if wdl is not None:
            pov_wdl = wdl.white()
            wdl_win = pov_wdl.wins
            wdl_draw = pov_wdl.draws
            wdl_loss = pov_wdl.losses

        seldepth = info.get("seldepth")

        candidates.append(CandidateMove(
            move=move, score_cp=score_cp, mate_in=mate_in, pv=pv,
            wdl_win=wdl_win, wdl_draw=wdl_draw, wdl_loss=wdl_loss,
            seldepth=seldepth,
        ))
    return candidates


async def _compute_pv_end(
    engine: "chess.engine.UciProtocol",
    root_board: chess.Board,
    pv: list[chess.Move],
    limit: "chess.engine.Limit",
) -> Optional[PVEndEval]:
    """Push `pv` on a copy of `root_board` and re-search the endpoint.

    Mirrors run_stockfish.py's evaluate_pv_end. Returns None if the PV is
    empty. If the PV ends the game, synthesizes a terminal eval instead of
    calling the engine.
    """
    if not pv:
        return None
    board = root_board.copy(stack=False)
    pushed = 0
    for mv in pv:
        if mv not in board.legal_moves:
            break
        board.push(mv)
        pushed += 1
    if pushed == 0:
        return None

    end_fen = board.fen()

    if board.is_checkmate():
        return PVEndEval(
            score_cp=None, mate_in=0, depth=None, seldepth=None,
            pushed=pushed, fen=end_fen, terminal="checkmate",
        )
    if board.is_stalemate():
        return PVEndEval(
            score_cp=0, mate_in=None, depth=None, seldepth=None,
            pushed=pushed, fen=end_fen, terminal="stalemate",
        )
    if (
        board.is_insufficient_material()
        or board.is_fivefold_repetition()
        or board.is_seventyfive_moves()
    ):
        return PVEndEval(
            score_cp=0, mate_in=None, depth=None, seldepth=None,
            pushed=pushed, fen=end_fen, terminal="draw",
        )

    info = await engine.analyse(board, limit, multipv=1)
    if isinstance(info, list):
        info = info[0] if info else None
    if info is None:
        return None
    score = info.get("score")
    score_cp = None
    mate_in = None
    if score is not None:
        pov = score.white()
        if pov.is_mate():
            mate_in = pov.mate()
        else:
            score_cp = pov.score()
    return PVEndEval(
        score_cp=score_cp,
        mate_in=mate_in,
        depth=info.get("depth"),
        seldepth=info.get("seldepth"),
        pushed=pushed,
        fen=end_fen,
        terminal=None,
    )


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
