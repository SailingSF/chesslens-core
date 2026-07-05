# ChessLens Core — Developer Guide

## Project Overview

`chesslens-core` is the open-source chess analysis library and Django app that lets players run chess.com-style game reviews on their own machine. It combines Stockfish engine analysis with LLM narration (Anthropic Claude or OpenAI) to classify every move (best, excellent, good, inaccuracy, mistake, blunder, brilliant, great, miss) and explain positions in plain language — the same categories chess.com uses, without requiring a subscription.

**Key design principle: the LLM never calculates — it only narrates.** All ground truth (best moves, scores, tactics) comes from Stockfish and the context assembly layer. The model receives structured, verified data and is asked only to explain it.

This repo is the source of truth for all chess intelligence. The private `chesslens-cloud` repo imports this as a dependency and wraps it with production infrastructure.

Beyond game review, the repo includes:
- **Classification optimizer** (`tests/optimize_classification.py`) — grid search tool for tuning EP model parameters to match chess.com's Classification V2 system
- **Chess.com review bookmarklet** (`chesscom-review-download/`) — browser bookmarklet that extracts move-by-move review data from chess.com into CSVs for optimizer ground truth
- **Classification config** (`config/classification.py`) — all tunable thresholds (sigmoid constants, EP boundaries, brilliant/great/miss detection) in one dataclass

## Architecture

Four layers, each with a distinct responsibility:

| Layer | Module | Responsibility |
|---|---|---|
| Chess Engine | `chess_engine/` | Run Stockfish via UCI, manage process pool |
| Context Assembly | `analysis/` | Detect tactics, classify openings, pawn structure, threat narrative |
| Priority Classifier | `analysis/priority.py` | Derive CRITICAL/TACTICAL/STRATEGIC tier from engine output + board |
| Explanation Layer | `explanation/` | Call LLM (Anthropic or OpenAI, async) with structured context |

## Directory Structure

```
chesslens-core/
├── config/              # Django project config (settings, urls, asgi)
│   └── classification.py # ClassificationConfig — all tunable EP/sigmoid thresholds
├── chess_engine/        # EngineService interface, PooledEngineService, EnginePool
├── analysis/            # ContextAssembler, PriorityClassifier, ECO lookup, tactical patterns, pawns
│   └── eco_data.json    # ECO opening database (~80 entries, expandable)
├── explanation/         # ExplanationGenerator (async), provider abstraction, prompt templates
│   ├── providers.py     # LLMConfig + provider selection (anthropic vs openai)
│   └── templates/       # game_review.py
├── game/                # GameAnalyzer (PGN orchestrator)
├── chesscom/            # Chess.com game import: async API client, importer, local PGN storage
├── api/                 # DRF views (async), serializers, URL config
├── ui/                  # Frontend templates and static assets
├── chesscom-review-download/  # Bookmarklet to extract chess.com review data as CSV
├── tests/               # Unit tests (no Stockfish/API required)
│   ├── optimize_classification.py  # Parameter optimizer for chess.com accuracy
│   └── test_games/      # Ground-truth CSVs + Stockfish cache JSONs
```

## Core Interfaces

### EngineService (`chess_engine/service.py`)
Abstract interface. Accepts a FEN + analysis params (depth, multipv, uci_options), returns `EngineResult`.

Implementation:
- **PooledEngineService** — reuses Stockfish processes via EnginePool, applies UCI options per request

Use `get_engine_service()` factory to get the shared singleton instance.

### ContextAssembler (`analysis/context.py`)
Accepts engine output + `chess.Board`, returns enriched context: opening name, tactical patterns, pawn structure, piece activity signals, threat narrative.

### PriorityClassifier (`analysis/priority.py`)
Accepts multi-PV engine output + optional board, returns `PriorityTier` (CRITICAL/TACTICAL/STRATEGIC). Pass the board to enable accurate capture detection in PV lines.

### LLMConfig (`explanation/providers.py`)
Provider-agnostic config. Model names starting with `claude-` → Anthropic; everything else → OpenAI (via the Responses API). Use `make_llm_config(provider, model, api_key, reasoning_effort)` to build one.

### ExplanationGenerator (`explanation/generator.py`)
**Async.** Accepts assembled context + skill level + LLMConfig, calls the chosen provider, returns a 2-3 sentence explanation. Use `get_explainer()` for the singleton.

### GameAnalyzer (`game/analyzer.py`)
Top-level orchestrator: iterates PGN moves, calls engine, assembles context, filters which moves deserve explanations (cp loss threshold + priority escalation), calls the LLM. Skips forced moves and recaptures.

### Chess.com Import (`chesscom/`)
Optional one-time import of a player's chess.com game history into a local folder (`IMPORTED_GAMES_DIR`, default `imported_games/`, gitignored). `ChessComClient` (`client.py`) calls the public API via async httpx with serial requests + 429 retry; `GameImporter` (`importer.py`) yields per-archive progress events for SSE and keeps `index.json` (metadata + idempotency ledger) updated per archive, so interrupted imports resume and completed past months are never re-fetched. Standard-chess games only; variants and PGN-less games are skipped. `storage.py` resolves filenames exclusively through the index (no client-supplied paths). The Game Review page shows a picker for imported games that fills the PGN textarea — the review pipeline itself is unchanged.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/game-review/` | Submit PGN, stream analysis via SSE |
| POST | `/api/position-explorer/` | Analyze single FEN synchronously |
| GET  | `/api/engines/` | List available Stockfish engines |
| POST | `/api/chesscom-import/` | Import a chess.com user's game history, stream progress via SSE |
| GET  | `/api/imported-games/` | List imported players, or games with `?username=` |
| GET  | `/api/imported-games/<username>/<game_id>/` | Fetch one imported game's PGN + metadata |

Per-request LLM overrides come from headers (`X-LLM-Provider`, `X-LLM-Model`, `X-LLM-Reasoning`, `X-Anthropic-Key`, `X-OpenAI-Key`) handled by `ui/middleware.py::LLMMiddleware`, or from body fields (`llm_provider`, `llm_model`, `llm_reasoning_effort`). Body fields take precedence.

## Service Singletons

Services that are expensive to create (engine pool, LLM clients) use module-level singleton factories:

- `chess_engine.service.get_engine_service()` — PooledEngineService
- `explanation.generator.get_explainer()` — ExplanationGenerator (async Anthropic + OpenAI clients)

These are safe to call from async views.

## Important Patterns

### SAN before push
Always compute `board.san(move)` **before** `board.push(move)`. The `san()` method requires the move to be legal in the current board state.

### Async all the way
All LLM calls use `AsyncAnthropic` / `AsyncOpenAI`. All API views are native async Django views. Engine analysis is async via python-chess. Never create manual event loops.

## Tech Stack

- **Django 5.1** + Django REST Framework (async views)
- **python-chess** for UCI interface, board logic, PGN parsing
- **Stockfish 16+** (server-side, process pool via EnginePool)
- **Anthropic Python SDK** (`AsyncAnthropic`) and **OpenAI Python SDK** (`AsyncOpenAI`, Responses API)
- **SQLite** (local dev)

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.sample .env
# Edit .env and add your ANTHROPIC_API_KEY and/or OPENAI_API_KEY

# Run migrations
python manage.py migrate

# Start the dev server (serves ASGI via Daphne — see note below)
python manage.py runserver
# or run uvicorn directly:
uvicorn config.asgi:application --reload
```

> **The app must run under ASGI, never WSGI.** The views are native-async and
> the engine pool / LLM clients are process-lifetime singletons bound to one
> event loop. Plain WSGI serving spins up a fresh event loop per request
> (asgiref runs each async view through `asyncio.run`); the first request
> works, but every request after it hangs because the engine's Stockfish
> subprocess transports are stranded on the now-closed first loop.
>
> Two things keep this from happening:
> - `daphne` is listed **first** in `INSTALLED_APPS`, so `manage.py runserver`
>   serves ASGI (single persistent loop) instead of WSGI. Don't remove it or
>   reorder it below `staticfiles`, or `runserver` reverts to WSGI and hangs.
> - Docker / production run `uvicorn config.asgi:application` directly.

### Docker
```bash
docker compose up
# App at http://localhost:8000
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | One of | Claude API key |
| `OPENAI_API_KEY`    | One of | OpenAI API key (used when an OpenAI model is selected) |
| `DATABASE_URL` | No | DB connection (default: SQLite) |
| `DJANGO_SECRET_KEY` | No | Django secret key (auto-generated in dev) |
| `DEBUG` | No | Debug mode (default: `True` in dev) |
| `STOCKFISH_DIR` | No | Extra dir(s) to scan for Stockfish binaries, `os.pathsep`-separated. `./engines`, `../engines`, and `$PATH` are always scanned. |

## Adding a New Feature

- **Chess logic** → `chess_engine/`, `analysis/`, `explanation/`, or `game/`
- **API endpoint** → `api/views.py` + `api/urls.py`
- **Prompt template** → `explanation/templates/`

**Boundary rule:** if a feature involves chess logic, it goes in core. If it involves user management, billing, or cloud infrastructure, it goes in `chesslens-cloud`.

## Testing

```bash
pytest
pytest tests/test_analysis.py -v
```

Tests use real `python-chess` board objects. Mock `EngineService` and the LLM clients in unit tests. No Stockfish binary or API key needed for the test suite.

## Classification Optimizer

`tests/optimize_classification.py` is a standalone tool for tuning the EP classification model against chess.com's Classification V2 ground truth. It grid-searches over sigmoid constants, EP thresholds, and engine parameters to maximize accuracy.

Key modes:
- `--compare` — print move-by-move comparison with current config (no optimization)
- Default (no flags) — mock mode optimization using chess.com's eval data (fast, no Stockfish)
- `--stockfish` — optimize using real cached Stockfish results
- `--generate-cache` — run Stockfish on all games and save JSON caches
- `--sweep-engine` — sweep depth/multipv to find which engine config matches chess.com best

Ground-truth CSVs in `tests/test_games/` are extracted from chess.com using the bookmarklet (see below). Stockfish cache JSONs live alongside them. Player Elos are stored in `tests/test_games/game_elos.json`.

All tunable classification parameters live in `config/classification.py` (`ClassificationConfig` dataclass). The optimizer reads this config, varies parameters, and reports accuracy against chess.com labels.

**Important:** Always use real Stockfish 16.1 with 1M nodes for classification testing, never mock engine results. Use per-side Elo from `game_elos.json` for accurate sigmoid scaling.

## Chess.com Review Bookmarklet

`chesscom-review-download/` contains a browser bookmarklet that scrapes chess.com's Game Review UI and downloads move-by-move data as a CSV.

Install: paste contents of `bookmarklet.min.txt` as a bookmark URL. Use: click the bookmark on any chess.com Game Review page.

Output CSV columns: `ply, color, move, classification, eval_points, points_difference`. These CSVs are the ground truth for the classification optimizer — drop them into `tests/test_games/`.

If chess.com changes their DOM, update selectors in `bookmarklet.js` and regenerate the minified version (see `chesscom-review-download/README.md`).
