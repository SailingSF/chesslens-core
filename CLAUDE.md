# ChessLens Core — Developer Guide

## Project Overview

`chesslens-core` is the open-source chess analysis library and Django app that lets players run chess.com-style game reviews on their own machine. It combines Stockfish engine analysis with Claude AI narration to classify every move (best, excellent, good, inaccuracy, mistake, blunder, brilliant, great, miss) and explain positions in plain language — the same categories chess.com uses, without requiring a subscription.

**Key design principle: the LLM never calculates — it only narrates.** All ground truth (best moves, scores, tactics) comes from Stockfish and the context assembly layer. Claude receives structured, verified data and is asked only to explain it.

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
| Explanation Layer | `explanation/` | Call Claude API (async) with structured context |

## Directory Structure

```
chesslens-core/
├── config/              # Django project config (settings, urls, asgi, routing)
│   └── classification.py # ClassificationConfig — all tunable EP/sigmoid thresholds
├── chess_engine/        # EngineService interface, PooledEngineService, RemoteEngineService, EnginePool
├── analysis/            # ContextAssembler, PriorityClassifier, ECO lookup, tactical patterns, pawns
│   └── eco_data.json    # ECO opening database (~80 entries, expandable)
├── explanation/         # ExplanationGenerator, CoachNudgeGenerator (async), prompt templates
│   └── templates/       # game_review.py, coach_nudge.py, opening_lab.py
├── game/                # GameAnalyzer, BotService, SessionManager, Django models
├── api/                 # DRF views (async), serializers, URL config
├── ui/                  # Frontend templates and static assets
├── ws/                  # Django Channels WebSocket consumers
├── engine_service/      # Standalone HTTP engine container (for Docker)
├── chesscom-review-download/  # Bookmarklet to extract chess.com review data as CSV
├── tests/               # Unit tests (no Stockfish/API required)
│   ├── optimize_classification.py  # Parameter optimizer for chess.com accuracy
│   └── test_games/      # Ground-truth CSVs + Stockfish cache JSONs
```

## Core Interfaces

### EngineService (`chess_engine/service.py`)
Abstract interface. Accepts a FEN + analysis params (depth, multipv, uci_options), returns `EngineResult`.

Implementations:
- **PooledEngineService** (default) — reuses Stockfish processes via EnginePool, applies UCI options per request
- **RemoteEngineService** — HTTP dispatch to engine container via `ENGINE_URL` (Docker/cloud mode)

Use `get_engine_service()` factory to get the shared singleton instance.

### ContextAssembler (`analysis/context.py`)
Accepts engine output + `chess.Board`, returns enriched context: opening name, tactical patterns, pawn structure, piece activity signals, threat narrative.

### PriorityClassifier (`analysis/priority.py`)
Accepts multi-PV engine output + optional board, returns `PriorityTier` (CRITICAL/TACTICAL/STRATEGIC). Pass the board to enable accurate capture detection in PV lines.

### ExplanationGenerator (`explanation/generator.py`)
**Async.** Accepts assembled context + skill level, calls Claude API via `AsyncAnthropic`, returns 2-3 sentence explanation. Use `get_explainer()` for the singleton.

### CoachNudgeGenerator (`explanation/coach.py`)
**Async.** Like ExplanationGenerator but for live coaching. Returns a single-sentence question. **Never suggests specific moves.** Validates output against move notation regex. Use `get_coach()` for the singleton.

### BotService (`game/bot.py`)
Wraps EngineService with Elo-constrained UCI params (`UCI_LimitStrength`, `UCI_Elo`) + polyglot opening book with configurable deviation rates per Elo band.

### SessionManager (`game/session.py`)
Manages Redis-backed state for active bot match / opening lab sessions. 4-hour TTL, extended on every move.

### GameAnalyzer (`game/analyzer.py`)
Top-level orchestrator: iterates PGN moves, calls engine, assembles context, filters which moves deserve explanations (cp loss threshold + priority escalation), calls LLM. Skips forced moves and recaptures.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/game-review/` | Submit PGN, stream analysis via SSE |
| POST | `/api/position-explorer/` | Analyze single FEN synchronously |
| POST | `/api/bot-match/start/` | Create bot match session, returns session_id |
| POST | `/api/opening-lab/start/` | Create opening lab session, returns session_id |

## WebSocket Endpoints

| Path | Consumer | Description |
|---|---|---|
| `ws/bot-match/<session_id>/` | BotMatchConsumer | Live bot match with coaching |
| `ws/opening-lab/<session_id>/` | OpeningLabConsumer | Opening theory study |

## Service Singletons

Services that are expensive to create (engine pool, LLM client, Redis connections) use module-level singleton factories:

- `chess_engine.service.get_engine_service()` — PooledEngineService or RemoteEngineService
- `explanation.generator.get_explainer()` — AsyncAnthropic-backed ExplanationGenerator
- `explanation.coach.get_coach()` — AsyncAnthropic-backed CoachNudgeGenerator

These are safe to call from async views and WebSocket consumers.

## Important Patterns

### SAN before push
Always compute `board.san(move)` **before** `board.push(move)`. The `san()` method requires the move to be legal in the current board state.

### Async all the way
All LLM calls use `AsyncAnthropic`. All API views are native async Django views. Engine analysis is async via python-chess. Never create manual event loops.

## Tech Stack

- **Django 5.1** + Django REST Framework (async views)
- **Django Channels 4** + Redis channel layer for WebSocket
- **Daphne** ASGI server (in `INSTALLED_APPS` before staticfiles)
- **python-chess** for UCI interface, board logic, PGN parsing
- **Stockfish 16+** (server-side, process pool via EnginePool)
- **Anthropic Python SDK** (`AsyncAnthropic`) for Claude API calls
- **Redis** for channel layer, session state
- **httpx** for async HTTP (RemoteEngineService)
- **SQLite** (local dev), **PostgreSQL + pgvector** (cloud)

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.sample .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run migrations
python manage.py migrate

# Start with WebSocket support (recommended)
daphne -b 0.0.0.0 -p 8000 config.asgi:application

# Or HTTP-only dev server
python manage.py runserver
```

### Docker (full stack with engine + Redis)
```bash
docker compose up
# App at http://localhost:8000
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `REDIS_URL` | No | Redis connection (default: `redis://localhost:6379`) |
| `DATABASE_URL` | No | DB connection (default: SQLite) |
| `ENGINE_URL` | No | Engine service URL (default: local Stockfish subprocess) |
| `DJANGO_SECRET_KEY` | No | Django secret key (auto-generated in dev) |
| `DEBUG` | No | Debug mode (default: `True` in dev) |

## Adding a New Feature

- **Chess logic** → `chess_engine/`, `analysis/`, `explanation/`, or `game/`
- **API endpoint** → `api/views.py` + `api/urls.py`
- **WebSocket event** → `ws/consumers.py`
- **Prompt template** → `explanation/templates/`

**Boundary rule:** if a feature involves chess logic, it goes in core. If it involves user management, billing, or cloud infrastructure, it goes in `chesslens-cloud`.

## Testing

```bash
pytest
pytest tests/test_analysis.py -v
```

Tests use real `python-chess` board objects. Mock `EngineService` and `anthropic` client in unit tests. No Stockfish binary or API key needed for the test suite.

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
