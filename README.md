# ChessLens Core

ChessLens Core is an open-source chess analysis engine that combines [Stockfish](https://stockfishchess.org/) with the [Claude API](https://docs.anthropic.com/en/docs/overview) to explain chess positions in plain language. It powers a Django web application where players can review completed games, practice against Elo-calibrated bots with live coaching, and study openings in a guided lab environment.

The core design principle is that **the LLM never calculates — it only narrates**. Stockfish provides all ground truth (best moves, scores, tactics). Claude receives structured, verified data and is asked only to explain what the engine found, calibrated to the player's skill level.

---

## What It Does

ChessLens Core has four user-facing modes, each served through a REST API or WebSocket connection.

### Game Review

A player submits a completed game as a PGN string. The server walks through every move, runs Stockfish analysis on each position, identifies which moves were mistakes, blunders, or critical moments, and generates a plain-language explanation for each significant move. Results stream back in real time via Server-Sent Events (SSE) so the frontend can display analysis progressively as it's computed.

Not every move gets an explanation. The system skips forced moves (only one legal option), recaptures (likely forced exchanges), and positions where the centipawn loss is below a configurable threshold (default: 30cp). This keeps analysis focused on moments that actually matter.

### Position Explorer

A player submits a single position as a FEN string and gets back a one-shot analysis: the engine's best move, the priority tier of the position, and a natural-language explanation. This is a simple request/response cycle with no streaming.

### Bot Match

A player starts a live game against a bot whose strength is calibrated to a target Elo rating (400–3000). The game runs over a persistent WebSocket connection. On each player move, the server validates the move, generates the bot's response using Elo-constrained Stockfish, and optionally sends a coaching nudge — a single-sentence question that names what's happening on the board and asks the player how they want to respond. The coach never suggests a specific move.

The bot uses a Polyglot opening book for the first 10–15 moves. At lower Elo levels, the bot randomly deviates from book theory to simulate human imprecision (20% at 800 Elo, 10% at 1200, 5% at 1600, 0% at 2000).

Coaching has three verbosity modes chosen at the start of the game:
- **Silent** — no coaching during the game; full review available afterward
- **Nudge** — coaching fires only after significant moments (centipawn loss above threshold)
- **Verbose** — coaching fires after every move

### Opening Lab

A player selects an opening to study and plays through it against a bot that always follows book theory perfectly (zero deviation, full-strength engine beyond the book). When the player deviates from the book continuation, the system explains what the book move achieves, why the deviation is suboptimal in this opening's context, and what positional themes the opening is designed to produce.

---

## How It Works

The system is organized into four layers. Each layer does one job and passes structured data to the next.

### Layer 1: Chess Engine (`chess_engine/`)

The engine layer wraps Stockfish and exposes a single interface: give it a FEN string, get back a list of candidate moves with scores, mate-in distances, and principal variations.

Stockfish processes are expensive to start (neural network weight loading), so the engine uses a **process pool** that keeps multiple Stockfish instances alive and reuses them across requests. The pool size matches the number of available CPU cores. Each analysis request checks out an engine from the pool, runs the analysis, and returns it.

For bot match mode, the same pool is used with per-request UCI option overrides (`UCI_LimitStrength: true`, `UCI_Elo: <target>`). Elo constraints are applied per search and reset after each request so the next caller gets a clean engine.

There are two implementations behind the same interface:
- **PooledEngineService** (default) — local Stockfish subprocess pool, used when running without Docker
- **RemoteEngineService** — sends HTTP requests to a separate engine container, used when `ENGINE_URL` is set (Docker mode)

The factory function `get_engine_service()` returns a shared singleton of whichever implementation the configuration requires.

### Layer 2: Context Assembly (`analysis/`)

Raw engine output (centipawn scores, move sequences) is not enough to generate useful explanations. The context assembly layer enriches each position with named, human-readable features:

- **Opening identification** — looks up the current position against a database of ~80 ECO-coded openings (Sicilian Najdorf, Ruy Lopez, Queen's Gambit Declined, King's Indian, etc.) stored in `analysis/eco_data.json`. Detects when the played move deviates from known book theory.
- **Tactical patterns** — scans the board using python-chess attack maps to identify forks, pins, skewers, discovered attacks, and back-rank weaknesses.
- **Pawn structure** — identifies isolated, doubled, and passed pawns for each side.
- **Piece activity** — detects bishop pairs, rooks on open files, knight outpost candidates, and computes a king danger score based on attacker count in the king zone.
- **Centipawn loss classification** — buckets the difference between the played move and the engine's best move into quality tiers: best (0–10cp), inaccuracy (10–50cp), mistake (50–100cp), blunder (100cp+).
- **Threat narrative** — builds a human-readable description of which pieces are attacked, which are undefended, and which are hanging. This narrative is injected into coaching nudge prompts.

### Layer 3: Priority Classifier (`analysis/priority.py`)

The priority classifier decides what a position demands right now. It produces a tier that governs what the explanation focuses on:

| Tier | Trigger | What the explanation emphasizes |
|---|---|---|
| **CRITICAL** | Score gap ≥ 200cp between best and second-best move, or forced mate in the principal variation | The immediate threat — what must be addressed now |
| **TACTICAL** | Score gap 50–200cp, with captures in the first 3 PV moves | Concrete calculation — the position is sharp and demands specific moves |
| **STRATEGIC** | All top-3 candidate moves within 50cp of each other | Long-term imbalances — pawn structure, piece activity, planning |

This is pure logic derived from engine data. No LLM is involved. The tier is passed downstream to select the right prompt template.

### Layer 4: Explanation Layer (`explanation/`)

The explanation layer calls Claude through the Anthropic API using structured prompt templates. Every call is async (`AsyncAnthropic`) so it doesn't block the event loop during WebSocket games or SSE streaming.

There are three prompt template categories:

- **Game review templates** — used in post-game analysis. Each priority tier has a distinct template: CRITICAL templates use direct, urgent language; TACTICAL templates emphasize move-by-move calculation; STRATEGIC templates foreground pawn structure and long-term planning. Output: 2–3 sentences.

- **Coach nudge templates** — used during live bot matches. These follow strict constraints enforced both in the prompt and by a post-generation validator: the nudge names what's happening on the board (threats, attacked pieces, controlled squares), asks the player a question, and never suggests or hints at a specific move. The validator runs a regex check against the output and rejects any response containing move notation (Nf3, exd5, O-O, etc.). Output: 1 sentence.

- **Opening lab templates** — used when the player deviates from book theory. These explain what the book continuation achieves, why the player's move is suboptimal in this specific opening context, and what positional themes the opening is designed to produce.

All templates are parameterized by **skill level** (beginner, intermediate, advanced), which adjusts the tone and vocabulary:
- **Beginner** — plain language, no jargon, names the specific piece at risk
- **Intermediate** — references named tactical and strategic patterns, light theory
- **Advanced** — references outposts, pawn tension, piece coordination, engine continuations

---

## API Reference

All API endpoints are served under `/api/`. The application also exposes Django admin at `/admin/`.

### `POST /api/game-review/`

Submit a PGN for move-by-move analysis. Returns a stream of Server-Sent Events.

**Request body:**
```json
{
  "pgn": "1. e4 e5 2. Nf3 Nc6 ...",
  "skill_level": "intermediate"
}
```

- `pgn` (required) — complete PGN text of the game
- `skill_level` (optional, default `"intermediate"`) — one of `"beginner"`, `"intermediate"`, `"advanced"`

**Response:** `text/event-stream`. Each event is a JSON object:
```
data: {"move_number": 5, "color": "white", "move_san": "Bxf7+", "cp_loss": 0, "cp_loss_label": "best", "priority_tier": "CRITICAL", "explanation": "..."}

data: {"done": true}
```

Moves that fall below the centipawn loss threshold or are forced/trivial are skipped and do not appear in the stream.

### `POST /api/position-explorer/`

Analyze a single position.

**Request body:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  "skill_level": "intermediate",
  "question": "What should Black do here?"
}
```

- `fen` (required) — position in FEN notation
- `skill_level` (optional, default `"intermediate"`)
- `question` (optional) — natural language question about the position

**Response:**
```json
{
  "explanation": "...",
  "priority_tier": "STRATEGIC",
  "best_move": "e5",
  "pv": ["e5", "Nf3", "Nc6", "Bb5"],
  "cp": 30,
  "opening": "King's Pawn Opening",
  "eco": "B00"
}
```

### `POST /api/bot-match/start/`

Create a new bot match session. Returns a session ID to use with the WebSocket endpoint.

**Request body:**
```json
{
  "elo": 1200,
  "player_color": "white",
  "coaching_mode": "nudge",
  "skill_level": "intermediate",
  "opening": ""
}
```

- `elo` (optional, default `1200`) — bot target rating, 400–3000
- `player_color` (optional, default `"white"`) — `"white"` or `"black"`
- `coaching_mode` (optional, default `"nudge"`) — `"silent"`, `"nudge"`, or `"verbose"`
- `skill_level` (optional, default `"intermediate"`)
- `opening` (optional) — path to a Polyglot opening book file

**Response:**
```json
{
  "session_id": "a1b2c3d4e5f6",
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "player_color": "white"
}
```

If the player chose Black, the response also includes `"first_move"` (the bot's opening move in UCI format) and the updated `"fen"`.

### `POST /api/opening-lab/start/`

Create a new opening lab session.

**Request body:**
```json
{
  "eco_code": "C60",
  "player_color": "white",
  "skill_level": "intermediate"
}
```

- `eco_code` (required) — ECO code of the opening to study (e.g. `"C60"` for Ruy Lopez)
- `player_color` (optional, default `"white"`)
- `skill_level` (optional, default `"intermediate"`)

**Response:**
```json
{
  "session_id": "f6e5d4c3b2a1",
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "opening_eco": "C60",
  "player_color": "white"
}
```

---

## WebSocket Protocol

Both live modes (bot match and opening lab) use WebSocket connections. After creating a session via the REST API, connect to the appropriate WebSocket endpoint with the session ID.

### Bot Match: `ws/bot-match/<session_id>/`

**Client → Server:**
```json
{"type": "player_move", "move": "e2e4"}
{"type": "resign"}
{"type": "request_hint"}
```

- `player_move` — send a move in UCI format. The server validates it, generates the bot's response, and optionally sends a coaching nudge.
- `resign` — end the game by resignation.
- `request_hint` — request a coaching nudge for the current position without making a move.

**Server → Client:**
```json
{"type": "bot_move", "move": "e7e5", "fen": "...", "move_number": 2}
{"type": "coach_nudge", "text": "Their knight now controls the center and threatens your bishop — how do you want to address that?"}
{"type": "game_over", "result": "1-0", "reason": "checkmate"}
{"type": "error", "message": "illegal move"}
```

The bot move and coach nudge arrive as separate messages because they have different latencies. The bot move arrives first (~300ms) so the board updates immediately. The nudge arrives afterward (~800ms).

### Opening Lab: `ws/opening-lab/<session_id>/`

**Client → Server:**
```json
{"type": "player_move", "move": "e2e4"}
```

**Server → Client:**
```json
{"type": "bot_move", "move": "e7e5", "fen": "...", "move_number": 2}
{"type": "theory_deviation", "book_move": "Nf3", "opening": "C60", "explanation": "..."}
{"type": "error", "message": "..."}
```

When the player follows book theory, only the bot's next move is sent. When the player deviates, a `theory_deviation` message explains what the book move achieves and why the deviation is suboptimal.

---

## Data Models

The database stores completed game analyses for history and retrieval.

### SavedGame

| Field | Type | Description |
|---|---|---|
| `pgn` | text | Full PGN of the game |
| `white_player` | string | White player name |
| `black_player` | string | Black player name |
| `result` | string | `"1-0"`, `"0-1"`, or `"1/2-1/2"` |
| `created_at` | datetime | When the game was saved |
| `analyzed_at` | datetime | When analysis completed (null if pending) |
| `skill_level` | string | Skill level used for explanations |

### MoveNote

| Field | Type | Description |
|---|---|---|
| `game` | foreign key | Link to SavedGame |
| `move_number` | integer | Full move number |
| `color` | string | `"white"` or `"black"` |
| `move_san` | string | Move in Standard Algebraic Notation |
| `fen` | text | Position before the move |
| `cp_loss` | integer | Centipawn loss relative to engine best |
| `cp_loss_label` | string | `"best"`, `"inaccuracy"`, `"mistake"`, or `"blunder"` |
| `priority_tier` | string | `"CRITICAL"`, `"TACTICAL"`, or `"STRATEGIC"` |
| `explanation` | text | LLM-generated explanation |

Active game sessions (bot match and opening lab) are stored in Redis, not in the database. Each session has a 4-hour TTL that resets on every move.

---

## Project Structure

```
chesslens-core/
├── config/                    Django project configuration
│   ├── settings.py            Settings, installed apps, channel layers
│   ├── urls.py                HTTP URL routing (admin + api/)
│   ├── asgi.py                ASGI application with WebSocket routing
│   └── routing.py             WebSocket URL patterns
│
├── chess_engine/              Stockfish interface
│   ├── service.py             EngineService interface, PooledEngineService,
│   │                          RemoteEngineService, get_engine_service() factory
│   └── pool.py                EnginePool — async Stockfish process pool
│
├── analysis/                  Position enrichment
│   ├── context.py             ContextAssembler — combines all analysis into AssembledContext
│   ├── priority.py            PriorityClassifier — CRITICAL / TACTICAL / STRATEGIC
│   ├── patterns.py            Tactical pattern detection (fork, pin, skewer, discovered attack, back-rank)
│   ├── pawns.py               Pawn structure analysis (isolated, doubled, passed)
│   ├── eco.py                 ECO opening code lookup
│   └── eco_data.json          Opening database (~80 entries)
│
├── explanation/               LLM narration
│   ├── generator.py           ExplanationGenerator (async) — 2-3 sentence game review explanations
│   ├── coach.py               CoachNudgeGenerator (async) — single-sentence coaching questions
│   └── templates/             Prompt templates
│       ├── game_review.py     Templates per priority tier + skill level
│       ├── coach_nudge.py     Strict no-move-suggestion templates
│       └── opening_lab.py     Theory deviation explanation templates
│
├── game/                      Game orchestration
│   ├── analyzer.py            GameAnalyzer — walks a PGN and orchestrates the full pipeline
│   ├── bot.py                 BotService + OpeningLabBotService — Elo-calibrated move generation
│   ├── session.py             SessionManager — Redis-backed game session state
│   └── models.py              Django models (SavedGame, MoveNote)
│
├── api/                       REST endpoints
│   ├── views.py               Async views: game_review, position_explorer, bot_match_start, opening_lab_start
│   ├── serializers.py         DRF serializers for request validation
│   └── urls.py                URL patterns for all four endpoints
│
├── ws/                        WebSocket handlers
│   └── consumers.py           BotMatchConsumer, OpeningLabConsumer
│
├── engine_service/            Standalone engine container (for Docker)
│   ├── Dockerfile             Installs Stockfish, runs server.py
│   └── server.py              Thin HTTP wrapper: POST /analyze → Stockfish UCI
│
├── tests/
│   └── test_analysis.py       20 unit tests (no Stockfish or API key required)
│
├── docker-compose.yml         Three-container stack: web + engine + redis
├── Dockerfile                 Web container: Django + Daphne + Stockfish
├── requirements.txt           Python dependencies
├── manage.py                  Django management script
├── pytest.ini                 Test configuration
├── .env.sample                Environment variable template
└── Chess_Explainer_Technical_Architecture_v3.md
                               Full technical architecture document
```

---

## Setup

### Prerequisites

- Python 3.12+
- [Stockfish](https://stockfishchess.org/download/) installed and available on your PATH (for local development without Docker)
- Redis running locally on port 6379 (for bot match and opening lab modes)
- An [Anthropic API key](https://console.anthropic.com/)

### Local Development

```bash
# Clone the repository
git clone <repo-url>
cd chesslens-core

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.sample .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# Run database migrations
python manage.py migrate

# Start the server with WebSocket support
daphne -b 0.0.0.0 -p 8000 config.asgi:application
```

The app is now running at `http://localhost:8000`.

For game review and position explorer, you only need Stockfish and an Anthropic API key. For bot match and opening lab, you also need Redis running locally.

### Docker

```bash
# Set your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# Start all three containers
docker compose up
```

This starts:
- **web** — Django app with Daphne on port 8000
- **engine** — Stockfish HTTP service on port 8001
- **redis** — Redis on port 6379

In Docker mode, the web container sends engine requests to the engine container over HTTP rather than spawning local Stockfish processes.

### Running Tests

```bash
pytest
```

All 20 tests run without Stockfish or an API key. They use real `python-chess` board objects with mocked engine results.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Your Claude API key |
| `REDIS_URL` | No | `redis://localhost:6379` | Redis connection URL |
| `ENGINE_URL` | No | — | Engine service URL. When set, uses HTTP dispatch instead of local Stockfish. Set automatically in Docker. |
| `DATABASE_URL` | No | `sqlite:///db.sqlite3` | Database connection |
| `DJANGO_SECRET_KEY` | No | Auto-generated (insecure) | Django secret key. Set this in production. |
| `DEBUG` | No | `True` | Django debug mode |
| `ALLOWED_HOSTS` | No | `localhost,127.0.0.1` | Comma-separated allowed hosts |

---

## Architecture Decisions

**Why Stockfish server-side, not WASM in the browser?** Consistent analysis depth across all clients, the ability to cache evaluations by FEN, and shared engine processes across bot match sessions. Browser WASM engines have variable performance depending on device.

**Why a separate engine container?** Contributors can swap in a different engine (Leela Chess Zero, etc.) by changing the container image without touching application code. Both engines speak UCI, so the same interface works.

**Why Server-Sent Events for game review instead of WebSocket?** Game review is a one-directional stream: the client submits a PGN, and the server pushes results. SSE is simpler than WebSocket for this pattern and doesn't require a persistent bidirectional connection.

**Why Redis for game sessions instead of the database?** Bot match sessions are ephemeral (4-hour TTL), update on every move (high write frequency), and don't need to survive server restarts. Redis hash operations are faster than database writes for this access pattern.

**Why does the coach never suggest moves?** The coaching philosophy is Socratic: name what's happening on the board, ask the player to think. Suggesting moves short-circuits the learning process. This constraint is enforced in the prompt template and validated with a regex on every LLM response.

---

## Relationship to ChessLens Cloud

This repository contains all chess intelligence. A separate private repository (`chesslens-cloud`) imports this as a dependency and adds:

- User authentication and accounts
- Stripe billing and subscription management
- Distributed engine workers with auto-scaling
- PostgreSQL with pgvector for annotated position retrieval
- Usage tracking and rate limiting
- Game history persistence across sessions

The boundary rule is simple: if a feature involves chess logic, it goes in this repo. If it involves user management, billing, or infrastructure, it goes in cloud.
