"""
Thin HTTP server for the engine service container.

Exposes a single endpoint:
  POST /analyze
  Body: {"fen": "...", "depth": 20, "multipv": 3, "uci_options": {}}
  Response: {"fen": "...", "depth": 20, "candidates": [...]}

The web container calls this via ENGINE_URL=http://engine:8001/analyze
when running in Docker. In local dev (no Docker), the web container uses
a local Stockfish subprocess directly.
"""

import asyncio
import json
import shutil
from http.server import BaseHTTPRequestHandler, HTTPServer

import chess
import chess.engine

STOCKFISH_PATH = shutil.which("stockfish") or "/usr/games/stockfish"


class AnalyzeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/analyze":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        result = asyncio.run(self._analyze(body))
        response = json.dumps(result).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response))
        self.end_headers()
        self.wfile.write(response)

    async def _analyze(self, body: dict) -> dict:
        fen = body["fen"]
        depth = body.get("depth", 20)
        multipv = body.get("multipv", 3)
        uci_options = body.get("uci_options", {})

        transport, engine = await chess.engine.popen_uci(STOCKFISH_PATH)
        try:
            # Apply per-request UCI options (e.g. UCI_LimitStrength, UCI_Elo)
            if uci_options:
                await engine.configure(uci_options)

            board = chess.Board(fen)
            info_list = await engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
                multipv=multipv,
            )
            candidates = []
            for info in (info_list if isinstance(info_list, list) else [info_list]):
                pv = [m.uci() for m in info.get("pv", [])]
                score = info.get("score")
                score_cp = None
                mate_in = None
                if score:
                    pov = score.white()
                    if pov.is_mate():
                        mate_in = pov.mate()
                    else:
                        score_cp = pov.score()
                if pv:
                    candidates.append({
                        "move": pv[0],
                        "score_cp": score_cp,
                        "mate_in": mate_in,
                        "pv": pv,
                    })
            return {"fen": fen, "depth": depth, "candidates": candidates}
        finally:
            await engine.quit()

    def log_message(self, format, *args):
        pass  # Silence request logs


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8001), AnalyzeHandler)
    print("Engine service running on :8001")
    server.serve_forever()
