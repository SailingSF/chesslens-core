"""
Shared fixtures for integration tests.

Manages a real Stockfish Docker container (engine_service/) and provides
real RemoteEngineService + Anthropic-backed generators. NOTHING is mocked.
"""

import os
import subprocess
import time

import chess
import httpx
import pytest

from analysis.context import ContextAssembler
from analysis.priority import classify
from chess_engine.service import RemoteEngineService

ENGINE_CONTAINER_NAME = "chesslens-test-engine"
ENGINE_HOST_PORT = 18001


# ---------------------------------------------------------------------------
# Docker engine container (session-scoped — one container for all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def engine_url():
    """Build and start the Stockfish engine container for the entire test session."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    engine_service_dir = os.path.join(project_root, "engine_service")

    subprocess.run(
        ["docker", "build", "-t", "chesslens-engine-test", "."],
        cwd=engine_service_dir,
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["docker", "rm", "-f", ENGINE_CONTAINER_NAME],
        capture_output=True,
    )

    subprocess.run(
        [
            "docker", "run", "-d",
            "--name", ENGINE_CONTAINER_NAME,
            "-p", f"{ENGINE_HOST_PORT}:8001",
            "chesslens-engine-test",
        ],
        check=True,
        capture_output=True,
    )

    url = f"http://localhost:{ENGINE_HOST_PORT}"
    _wait_for_engine(url, timeout=90)

    yield url

    subprocess.run(
        ["docker", "rm", "-f", ENGINE_CONTAINER_NAME],
        capture_output=True,
    )


def _wait_for_engine(url: str, timeout: int = 90) -> None:
    """Poll the engine service until it responds to a real analysis request."""
    start = time.time()
    payload = {
        "fen": chess.STARTING_FEN,
        "depth": 1,
        "multipv": 1,
        "uci_options": {},
    }
    while time.time() - start < timeout:
        try:
            resp = httpx.post(f"{url}/analyze", json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("candidates"):
                    return
        except (httpx.HTTPError, ConnectionError, OSError):
            pass
        time.sleep(3)
    raise RuntimeError(f"Engine service at {url} failed to become healthy within {timeout}s")


# ---------------------------------------------------------------------------
# Service fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def engine_service(engine_url):
    """Real RemoteEngineService connected to the Docker Stockfish container."""
    return RemoteEngineService(engine_url)


@pytest.fixture(scope="session")
def context_assembler():
    return ContextAssembler()


@pytest.fixture(scope="session")
def anthropic_client():
    """Real AsyncAnthropic client using the API key from .env / Django settings."""
    anthropic = pytest.importorskip("anthropic")
    from django.conf import settings
    api_key = settings.ANTHROPIC_API_KEY
    assert api_key and api_key != "sk-ant-...", (
        "ANTHROPIC_API_KEY must be set in .env to run integration tests"
    )
    return anthropic.AsyncAnthropic(api_key=api_key)


@pytest.fixture(scope="session")
def explainer(anthropic_client):
    """Real ExplanationGenerator backed by the Anthropic API."""
    from explanation.generator import ExplanationGenerator
    return ExplanationGenerator(client=anthropic_client)


@pytest.fixture(scope="session")
def coach(anthropic_client):
    """Real CoachNudgeGenerator backed by the Anthropic API."""
    from explanation.coach import CoachNudgeGenerator
    return CoachNudgeGenerator(client=anthropic_client)


# ---------------------------------------------------------------------------
# Convenience: well-known test positions
# ---------------------------------------------------------------------------

STARTING_FEN = chess.STARTING_FEN

ITALIAN_GAME_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"

MATE_IN_1_FEN = "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1"

BACK_RANK_FEN = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"

QUEENS_GAMBIT_FEN = "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2"

SICILIAN_FEN = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"

CRUSHING_ADVANTAGE_FEN = "6k1/5ppp/8/8/8/8/5PPP/1Q4K1 w - - 0 1"

COMPLEX_MIDDLEGAME_FEN = "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9"
