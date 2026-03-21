# Chess Explainer App — Technical Architecture

*Version 3.0 — March 2026*

---

## 1. Vision

Chess engines are unmatched at finding the best move. Chess coaches are unmatched at explaining why. The Chess Explainer App bridges that gap — combining the computational accuracy of Stockfish with the contextual reasoning of a large language model to give every player a knowledgeable companion that explains positions in plain language.

The app is not a playing engine, a puzzle trainer, or a coaching replacement. It is a **position interpreter**: something that takes the raw output of chess analysis and translates it into human understanding, calibrated to the player's skill level. Beyond post-game review, the app extends into active learning — players can practice against Elo-calibrated bots with live coaching, and drill specific openings in a guided lab environment.

The critical design principle throughout the system is that **the LLM never calculates — it only narrates**. All ground truth (best moves, scores, tactics) comes from the engine and the context layer. The LLM receives structured, verified data and is asked only to explain it. In live coaching modes, the LLM additionally never suggests specific moves — it names threats, highlights attacked squares, and asks the player questions.

---

## 2. Distribution Model

The app ships as two separate codebases with a shared core.

### 2.1 Open Source — `chess-explainer-core`

Licensed MIT or Apache 2.0. Contains all chess analysis logic, the full explanation pipeline, bot match logic, opening lab logic, a working Django app with Channels support, and a Docker Compose stack that runs everything locally. A contributor clones the repo, provides their own Claude API key, runs `docker compose up`, and has a fully functional app at `localhost:8000`.

This repo is the source of truth for all chess intelligence. The paid version imports it as a dependency and never duplicates chess logic.

### 2.2 Paid Cloud — `chess-explainer-cloud`

Private repository. Contains infrastructure configuration, auto-scaling logic, user authentication, billing, caching, and paid-only product features. Imports `chess-explainer-core` as a Python package or Git submodule and wraps it with production infrastructure for multi-user scale.

### 2.3 Boundary Rule

All chess intelligence lives in the core repo: engine interface, context assembly, priority classification, prompt templates, explanation generation, bot move generation, opening lab logic, and coach nudge generation. The cloud repo contains only infrastructure, scaling, and features that depend on persistent multi-user state (accounts, game history, billing, usage tracking). This boundary is enforced by design — if a feature involves chess logic, it goes in core. If it involves user management or infrastructure, it goes in cloud.

---

## 3. Core Architecture

The system is composed of four layers, each with a distinct responsibility. No layer does another's job.

| Layer | Responsibility | Technology |
|---|---|---|
| Chess Engine | Compute best moves, scores, and predicted continuations | Stockfish via UCI protocol (server-side) |
| Context Assembly | Detect tactics, classify openings, read pawn structure, format threat narratives | python-chess + ECO lookup table |
| Priority Classifier | Determine what the position demands right now | Derived from engine multi-PV output |
| Explanation Layer | Narrate the position in human language at the right depth | Claude API (Sonnet) |

### 3.1 Interface Contracts

The four layers communicate through well-defined Python interfaces. These contracts are what allow the open-source and cloud versions to share the same chess logic while swapping infrastructure implementations.

**EngineService** — accepts a FEN string and analysis parameters (depth, multi-PV count, optional UCI overrides such as `UCI_LimitStrength` and `UCI_Elo`), returns structured evaluation data. The open-source version implements this with a local Stockfish subprocess. The cloud version implements it by dispatching to a remote engine worker via a Redis task queue.

**ContextAssembler** — accepts engine output and a `chess.Board` state, returns enriched context including opening name, tactical patterns, pawn structure summary, piece activity signals, and a threat narrative (human-readable description of attacked squares and pieces for use in coaching prompts). Pure Python logic, identical in both versions.

**PriorityClassifier** — accepts multi-PV engine output, returns a priority tier (CRITICAL / TACTICAL / STRATEGIC) and the triggering condition. Pure logic, derived entirely from score deltas between candidate moves.

**ExplanationGenerator** — accepts assembled context and a skill level, returns an LLM-generated explanation. The open-source version calls the Claude API directly with the user's key. The cloud version routes through a managed API key with usage tracking and rate limiting.

**CoachNudgeGenerator** — accepts assembled context with threat narrative and a skill level, returns an LLM-generated coaching question that names threats and asks the player how they want to respond. Never suggests specific moves. Uses a distinct prompt template category from the ExplanationGenerator.

**BotService** — accepts a `chess.Board` state and a bot configuration (target Elo, opening book preference), returns a move. Wraps the EngineService with Elo-constrained UCI parameters and an opening book with configurable deviation rates.

**SessionManager** — manages the state of active bot match and opening lab sessions. Stores current FEN, move history, bot configuration, coaching mode, and player color in Redis. Both versions use the same session logic; only the Redis connection details differ.

**GameAnalyzer** — top-level orchestrator that iterates through a PGN's moves, calls the engine, assembles context, filters which moves deserve explanations, and calls the LLM. Both versions use this same orchestrator; only the underlying service implementations differ.

---

## 4. Data Pipeline

### 4.1 Chess Engine Output

Stockfish communicates via the UCI (Universal Chess Interface) protocol — a text-based stdin/stdout conversation. The engine is run server-side (not in-browser via WASM) to ensure consistent analysis depth, identical behavior across web and mobile clients, and the ability to cache evaluations.

Per position, Stockfish returns:

- **Score in centipawns (cp):** numerical evaluation of the position. 100cp ≈ one pawn of advantage. Negative values favor Black.
- **Mate detection:** if a forced checkmate exists, the engine returns moves-to-mate rather than a centipawn score.
- **Principal Variation (PV):** the engine's predicted best sequence of moves for both sides, extending several moves deep.
- **Multi-PV lines:** when configured (required for priority classification), the top N candidate moves are returned, each with their own score and continuation.
- **Depth:** the number of half-moves (plies) searched, indicating confidence in the evaluation.

The `python-chess` library wraps the UCI protocol and provides a clean Python interface for engine interaction, including async support for non-blocking analysis.

### 4.2 Engine Process Management

Stockfish processes are expensive to start (loading neural network weights, initializing hash tables). The engine service maintains a **process pool** that keeps Stockfish instances alive and reuses them across analysis requests. Each process is configured once at startup with thread count and hash table size, then accepts position after position without restart overhead.

Pool sizing depends on available CPU cores. Each Stockfish process performs best with 1 dedicated thread. A machine with 2 cores runs a pool of 2. Analysis depth is set to 20 plies by default, which takes roughly 0.3–1 second per position on a single core.

For bot match mode, the same engine pool is used but with per-request UCI overrides (`UCI_LimitStrength: true`, `UCI_Elo: <target>`). The pool does not need separate processes for Elo-constrained analysis — Stockfish applies the limit per search, not per process.

### 4.3 Context Assembly

Engine output alone is not enough to generate meaningful explanations. The context layer enriches each position with named, human-readable features:

- **Opening identification:** ECO code lookup maps the current position to a named opening and variation (e.g. B90 = Sicilian Najdorf). Stored as a JSON lookup table in the core repo — roughly 500 ECO codes with ~2,000 named variations. Deviation from known theory is flagged when the played move doesn't match the book continuation.
- **Tactical pattern detection:** a programmatic pass over the board state using `python-chess` piece and attack maps identifies named patterns — forks, pins, skewers, discovered attacks, and back-rank weaknesses.
- **Pawn structure analysis:** isolated, doubled, passed, and backward pawns are computed from the FEN and named explicitly.
- **Piece activity signals:** bishop pair presence, rooks on open files, knight outposts, and king safety indicators are extracted from the position.
- **Centipawn loss classification:** the delta between the played move and the engine's best move is bucketed into quality tiers using standard thresholds: best (0–10cp), inaccuracy (10–50cp), mistake (50–100cp), blunder (100cp+).
- **Threat narrative:** a human-readable description of which squares and pieces are attacked or controlled by each side, formatted for use in coaching nudge prompts. Generated from `python-chess` attack maps by naming specific squares and pieces rather than returning raw bitboard data.

### 4.4 Derived Situational Priority

Rather than relying on a stated player goal, the system derives what the position demands using multi-PV engine output directly. This produces a priority tier that governs what the explanation focuses on.

| Priority Tier | Trigger Condition | Explanation Focus |
|---|---|---|
| **CRITICAL** | Score delta between best and 2nd-best move exceeds 200cp, or mate threat exists in PV | Immediate threat — what must be addressed before anything else |
| **TACTICAL** | Score delta 50–200cp; PV involves captures or material exchange within 3 moves | Concrete calculation — sharp position requiring specific moves |
| **STRATEGIC** | All top-3 moves score within 50cp; position is stable | Long-term imbalances — pawn structure, piece activity, planning |

Priority tier assignment is automatic and fully derived from engine data. No user input is required. The tier determines which prompt template the LLM explanation layer uses.

---

## 5. LLM Explanation Layer

### 5.1 Model Selection

Claude Sonnet is the primary model for explanation generation. It provides the right balance of quality and speed for structured narration tasks. Explanations need to return in 1–2 seconds, and Sonnet handles this comfortably for 2–3 sentence outputs. For bot match coach nudges, latency is even more important — nudges must arrive within ~800ms of the bot's move to feel responsive, and Sonnet's speed for single-sentence outputs meets this target.

### 5.2 Prompt Template System

The system uses multiple prompt template categories, each with priority tier and skill level variants:

**Game review templates** — used in post-game analysis. Each priority tier (CRITICAL / TACTICAL / STRATEGIC) has its own template with different emphasis ordering. CRITICAL templates foreground the threat and demand direct language. TACTICAL templates emphasize concrete calculation and specific moves. STRATEGIC templates foreground pawn structure, piece activity, and long-term planning.

**Coach nudge templates** — used during bot matches and opening lab sessions. These follow a strict constraint: they name threats, highlight attacked squares, and ask the player a question. They never suggest or hint at specific moves. The prompt includes the threat narrative from the context assembler and instructs the LLM to frame its output as a question about what the player wants to do, not what the player should do.

**Opening lab templates** — used when the player deviates from book theory. These explain what the book continuation achieves, why the player's move is suboptimal in this specific opening context, and what positional themes the opening is designed to produce. They reference the ECO name and variation explicitly.

Templates are stored in the database (cloud version) or as configuration files (open-source version) so they can be updated and A/B tested without redeployment.

Each game review prompt includes:

- Current position in FEN notation
- Opening name and variation, with theory deviation flag
- The move that was played and its centipawn loss relative to the engine's best move
- The engine's recommended move and its principal variation (up to 6 moves deep)
- Any tactical patterns detected
- Pawn structure summary
- Priority tier and triggering condition
- Player skill level (beginner / intermediate / advanced)

Each coach nudge prompt includes:

- Current position in FEN notation
- The move just played (by the player or the bot) and its centipawn loss
- Threat narrative: which squares and pieces are now attacked, controlled, or vulnerable
- Tactical patterns detected
- Priority tier
- Player skill level
- Explicit instruction to never suggest a move, only ask a question

The LLM is asked to produce a 2–3 sentence explanation (game review) or a single-sentence coaching question (nudge). It is never asked to find the best move or evaluate the position.

### 5.3 Skill Level Calibration

The same position produces meaningfully different explanations depending on player skill level:

| Level | Game Review Style | Coach Nudge Style |
|---|---|---|
| **Beginner** | Concrete and immediate. Names the piece at risk and what happens if the player doesn't act. No jargon. | Names the specific piece under attack and asks what the player wants to do about it. |
| **Intermediate** | Pattern-naming with light theory. References named tactical and strategic themes. | Names the tactical pattern and the key squares, asks the player to consider the tradeoff. |
| **Advanced** | Strategic and specific. References outposts, pawn tension, piece coordination, and engine continuations. | References positional imbalances and asks how the player wants to manage the tension. |

### 5.4 AI Retrieval Layer (Phase 2)

For enriched explanations, a vector retrieval system can surface relevant annotated positions from a corpus of master games with grandmaster commentary. When a user's game reaches a structurally similar position, the LLM can reference how strong players handled analogous situations.

**Embedding approach:** positions are embedded using a feature vector derived from material balance, pawn structure hash, and piece placement — extracted from FEN. These are not text embeddings; they are chess-specific structural representations.

**Storage:** pgvector extension on PostgreSQL. This keeps the vector store inside the existing database without adding a separate service. The open-source version can use this if the user installs pgvector; the cloud version includes it by default.

**Retrieval scope:** annotated game positions, a pattern library of common strategic themes (minority attack, Greek gift sacrifice, rook lift) with canonical examples. Retrieved context is injected into the LLM prompt as supplementary material, clearly labeled as reference rather than ground truth.

---

## 6. User-Facing Modes

### 6.1 Game Review (Primary Mode)

The user pastes a PGN from any completed game. The app steps through each move, showing the board state and surfacing an explanation at moves where the centipawn loss exceeds a configurable threshold or a priority tier escalates.

**Performance optimization:** not every move requires full depth-20 multi-PV analysis. Forced recaptures, book moves (matched to ECO theory), and positions with only one legal move are detected and skipped, reducing total analysis time by an estimated 30–40% in typical games.

**Explanation throttling:** only moves with centipawn loss exceeding the threshold or priority tier escalation receive LLM explanations. This prevents over-explanation fatigue and keeps API costs proportional to the number of meaningful moments in the game.

### 6.2 Position Explorer

The user sets up any position manually or by FEN input and explores it freely. Natural language questions are answered using the full data pipeline. For this mode, engine analysis runs synchronously (single position, user expects near-instant response) rather than through the async pipeline.

### 6.3 Bot Match

The user plays a live game against an Elo-calibrated bot with optional coaching. This mode introduces persistent, stateful, real-time game sessions — a fundamentally different infrastructure shape from the stateless game review flow.

**Bot configuration:** the player selects a target Elo (e.g. 800, 1200, 1600, 2000) and optionally an opening preference. The bot uses Stockfish with `UCI_LimitStrength` enabled at the target rating. An opening book provides recognizable but Elo-appropriate opening play — at lower levels the bot occasionally deviates from book theory to simulate human imprecision.

**Coaching modes:** the player chooses a coaching verbosity level before the game starts:

- **Silent:** no coaching during the game. The full game review is available afterward.
- **Nudge:** after significant moments (centipawn loss > threshold or priority tier escalation), a coaching question appears. The coach names what changed on the board and asks the player how they want to respond. It never suggests a move.
- **Verbose:** coaching fires after every player move, regardless of quality. Useful for beginners who want constant guidance.

**Session lifecycle:** each active game is a WebSocket connection managed by Django Channels. Game state (current FEN, move history, bot config, coaching mode, player color) is stored in Redis with a 4-hour TTL, extended on every move. If a connection drops, the frontend can restore the session from the PGN history it has locally.

**Communication flow:** the WebSocket is bidirectional. When the player makes a move, the server responds with two messages at different latencies: the bot's responding move arrives in ~300ms (engine evaluation at limited depth is fast), and the coach nudge arrives in ~800ms (LLM generation). The bot move is sent immediately so the board updates, and the nudge streams in afterward.

**Post-game:** when the game ends, the full PGN is submitted to the game review pipeline for comprehensive post-game analysis, giving the player both in-game coaching and detailed review.

### 6.4 Opening Lab

The user selects a named opening to study and plays through it against a bot that follows book theory perfectly. The opening lab is a specialized variant of the bot match mode with two key differences.

**Bot behavior:** the bot always plays the engine's top theoretical move for the selected opening — it does not use Elo constraints or deliberate deviation. This ensures the player practices against correct theory.

**Coaching trigger:** the coach fires on every move that deviates from the book continuation, not on centipawn loss. When the player deviates, the opening lab template explains what the book move achieves, why the deviation is suboptimal in the context of this specific opening, and what positional themes the opening is designed to produce. This creates a tight feedback loop for opening memorization and understanding.

**Drill structure:** the player can choose to play as White or Black in the selected opening. The lab supports repeating the same opening multiple times with variations — when the player deviates at move 8 and reviews the correction, they restart and try to get further into theory the next time. Progress tracking (how deep into each opening the player reaches before deviating) is a planned feature for the cloud version.

### 6.5 Live Tutor (Future — Phase 5)

During an active game against a human opponent, a sidebar panel surfaces the priority tier and a brief explanation after each move. Requires real-time engine evaluation and careful UX to avoid over-explaining. Planned after bot match and opening lab modes are validated, as it shares much of their WebSocket infrastructure.

---

## 7. Tech Stack

### 7.1 Backend

| Component | Technology | Rationale |
|---|---|---|
| Web framework | Django + Django REST Framework | Handles serializers for complex nested responses, solid auth, ORM for game history and user data. Proven experience from the team. |
| Real-time transport | Django Channels + Redis channel layer | WebSocket support for bot match and opening lab sessions. Slots into existing Django project structure with a routing config and consumer classes. |
| ASGI server | Daphne or Uvicorn | Required for Django Channels WebSocket support. Replaces Gunicorn for the web process in both open-source and cloud deployments. |
| Database (open source) | SQLite | Sufficient for single-user local use. Zero configuration. |
| Database (cloud) | PostgreSQL with pgvector extension | Multi-user persistence, analysis history, vector retrieval for annotated positions. Managed via Neon (serverless, scales to zero). |
| Task queue | Redis Queue (rq) | Simpler than Celery — single file for task definitions, no complex configuration. Sufficient for the job pattern (run Stockfish on FEN, return JSON). |
| Cache / message broker / channel layer | Redis | Position evaluation cache (keyed by FEN + depth), job queue broker, real-time analysis progress store, Django Channels channel layer, and session state store for active bot/lab games. |
| Chess logic | python-chess | UCI engine interface, board representation, PGN parsing, move validation, attack map computation. |
| Engine | Stockfish 16+ (server-side binary) | Runs as a subprocess via UCI. Managed by an engine process pool in the engine service. Supports Elo-constrained play via `UCI_LimitStrength`. |

### 7.2 Frontend

| Component | Technology | Rationale |
|---|---|---|
| Web framework | React | Team experience, strong ecosystem for chess UI. |
| Chess board | react-chessboard | Interactive board rendering, drag-and-drop move input, FEN/PGN support. |
| Chess logic (client) | chess.js | Client-side move validation, PGN parsing for immediate UI feedback before server analysis. |
| WebSocket client | Native browser WebSocket API or `reconnecting-websocket` | Connects to Django Channels for bot match and opening lab modes. Handles reconnection on drop. |
| Mobile | React Native | Shares business logic and API layer with web frontend. Board rendering uses a native component. |

### 7.3 AI

| Component | Technology | Rationale |
|---|---|---|
| Explanation model | Claude API (Sonnet) | Right balance of quality and speed for structured narration. 2–3 sentence outputs return in 1–2 seconds. Single-sentence nudges return in ~800ms. |
| Prompt management | Tiered template system | Separate template categories for game review, coach nudge, and opening lab. Each category has priority tier and skill level variants. Stored in DB (cloud) or config files (open source). Independently tunable and A/B testable. |
| Vector retrieval (Phase 2) | pgvector on PostgreSQL | Annotated position retrieval without a separate vector database service. Chess-specific feature vectors, not text embeddings. |

---

## 8. Real-Time Infrastructure — Django Channels

### 8.1 Why WebSockets

Game review and position explorer are stateless request/response flows — the user submits data, the server processes it, and returns results. Bot match and opening lab are fundamentally different: they are **persistent, stateful, real-time game sessions** where moves happen one at a time, the engine responds as the bot, and the coach fires between moves.

The timing is bidirectional: the user moves, the server needs to respond with both a bot move and a coach nudge, and those two things have different latencies. Sending the bot move immediately (~300ms) while the nudge streams in afterward (~800ms) requires a persistent connection, not request/response polling.

### 8.2 Django Channels Integration

Django Channels adds WebSocket support to Django without restructuring the application. It introduces three components:

**ASGI routing** — a top-level `routing.py` that dispatches WebSocket connections to consumer classes based on URL path, alongside the existing HTTP URL configuration.

**Consumer classes** — `BotMatchConsumer` and `OpeningLabConsumer` handle WebSocket lifecycle (connect, receive, disconnect) and coordinate engine calls, context assembly, and LLM nudge generation. These consumers live in the core repo because they orchestrate chess logic.

**Channel layer** — Redis, which is already in the stack, serves as the channel layer backend for Django Channels. No additional infrastructure is needed.

### 8.3 WebSocket Message Protocol

All messages are JSON. The protocol is intentionally simple — no message acknowledgment, no ordering guarantees beyond WebSocket's built-in TCP ordering.

**Client → Server:**

```json
{"type": "player_move", "move": "e2e4", "session_id": "abc123"}
{"type": "resign", "session_id": "abc123"}
{"type": "request_hint", "session_id": "abc123"}
```

**Server → Client:**

```json
{"type": "bot_move", "move": "e7e5", "fen": "...", "move_number": 2}
{"type": "coach_nudge", "text": "Their pawn on e5 controls d4 and f4. What's your plan for those squares?"}
{"type": "game_over", "result": "1-0", "reason": "checkmate"}
{"type": "theory_deviation", "book_move": "Nf3", "opening": "Ruy Lopez", "explanation": "..."}
```

The bot move and coach nudge are sent as separate messages because they have different latencies. The frontend displays the bot move immediately (updating the board) and renders the nudge when it arrives.

### 8.4 Session State Management

Each active bot match or opening lab session is stored in Redis as a hash:

```
Key: game_session:{session_id}
Fields:
  fen: "current position FEN"
  pgn_moves: ["e4", "e5", "Nf3", ...]
  bot_config: {"elo": 1400, "opening": "ruy_lopez", "deviation_rate": 0.10}
  player_color: "white"
  coaching_mode: "nudge"
  skill_level: "intermediate"
  move_count: 12
  mode: "bot_match" | "opening_lab"
  opening_eco: "C60"  (opening lab only)
```

Sessions have a 4-hour TTL, extended on every move. If a session expires or a WebSocket connection drops, the frontend can restore the game from the PGN history it holds locally by initiating a new session with the same move sequence.

### 8.5 ASGI Server Configuration

Django Channels requires an ASGI server instead of WSGI. Both the open-source and cloud versions use Daphne or Uvicorn as the application server. This replaces Gunicorn in the web service start command:

- **Open source:** `daphne -b 0.0.0.0 -p 8000 config.asgi:application`
- **Cloud (Fly.io):** same command, run inside the Django API Fly Machine.

HTTP requests (REST API, static files) and WebSocket connections both route through the same ASGI server. Django Channels dispatches to the appropriate handler based on the protocol.

---

## 9. Bot Service

### 9.1 Elo-Constrained Play

The BotService wraps the existing EngineService with Elo-appropriate configuration. Stockfish natively supports playing at a target rating via UCI parameters (`UCI_LimitStrength: true`, `UCI_Elo: <target>`). This is a per-search setting, not a per-process setting, so the existing engine process pool is reused without modification — the BotService simply passes additional UCI options in the analysis request.

At lower Elo levels, the depth limit is also reduced (e.g. depth 8 at 800 Elo, depth 12 at 1200 Elo) to produce faster responses and more human-like oversights.

### 9.2 Opening Book Integration

The bot uses an opening book for the first 10–15 moves to produce recognizable, human-like opening play. The book is a Polyglot-format binary file (standard format, widely available) loaded via `python-chess`'s `chess.polyglot` reader.

At higher Elo levels (1600+), the bot follows book theory faithfully. At lower levels, the bot has a configurable deviation rate — a probability that it ignores the book move and instead plays the engine's Elo-constrained choice. This simulates human imprecision in the opening:

- 800 Elo: 20% deviation rate (frequently plays off-book)
- 1200 Elo: 10% deviation rate
- 1600 Elo: 5% deviation rate
- 2000 Elo: 0% deviation rate (follows theory)

### 9.3 Opening Lab Bot Variant

In opening lab mode, the bot behaves differently: it always plays the book continuation for the selected opening with zero deviation, regardless of Elo setting. If the position goes beyond the book's coverage, the bot switches to full-strength Stockfish (depth 20, no Elo limit) to continue playing the strongest theoretical moves. The goal is to present the player with correct theory to practice against.

---

## 10. Open Source Deployment

### 10.1 Docker Compose Stack

Three containers, started with a single command.

**web** — Django application with Channels support, serving the REST API, WebSocket endpoints, and bundled React frontend via an ASGI server (Daphne). Handles request routing, analysis orchestration, LLM calls, and real-time bot match / opening lab sessions. In the open-source single-user context, game review analysis can run synchronously with SSE (Server-Sent Events) streaming, eliminating the need for a separate task queue worker.

**engine** — Thin engine service running Stockfish. Separated from the web container so contributors can swap in a different engine (Leela Chess Zero, etc.) by changing the container image without touching application code. Exposes a simple HTTP interface for position analysis requests. Handles both full-strength analysis (game review) and Elo-constrained requests (bot match) through the same interface.

**redis** — Caches analyzed positions, serves as the Django Channels channel layer for WebSocket support, and stores active game session state for bot match and opening lab modes.

### 10.2 Docker Compose Configuration

```yaml
services:
  web:
    build: .
    ports: ["8000:8000"]
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - ENGINE_URL=http://engine:8001
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=sqlite:///db.sqlite3
    depends_on: [engine, redis]
    command: daphne -b 0.0.0.0 -p 8000 config.asgi:application

  engine:
    build: ./engine
    ports: ["8001:8001"]

  redis:
    image: redis:7-alpine
```

### 10.3 Engine Service Dockerfile

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y stockfish && rm -rf /var/lib/apt/lists/*
RUN stockfish <<< "uci" | head -5  # Verify installation

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "server.py"]
```

### 10.4 Contributor Setup

The full setup is four steps documented in the README:

1. Clone the repository.
2. Create a `.env` file with a Claude API key (`ANTHROPIC_API_KEY=sk-...`).
3. Run `docker compose up`.
4. Open `localhost:8000`.

No database migrations, no external service accounts, no build toolchain beyond Docker. Bot match and opening lab work immediately — all modes share the same containers.

---

## 11. Cloud Deployment

### 11.1 Platform — Fly.io

Fly.io is the primary hosting platform for the paid version. It provides per-second billing for compute (critical for bursty engine analysis), a built-in load balancer with WebSocket support, and a machine API for dynamic scaling of engine workers.

### 11.2 Service Topology

**Django API (Fly Machines, 2–3 instances)** — handles user auth, billing, request routing, analysis orchestration, and WebSocket connections for bot match and opening lab sessions. Machines can auto-stop when idle and start on request to minimize cost during quiet periods. Sits behind Fly's built-in load balancer, which supports both HTTP and WebSocket connections.

WebSocket connections require sticky sessions so that a player's bot match stays on the same machine for the duration of the game. Fly's load balancer supports this via connection-based routing — once a WebSocket is established, all messages route to the same machine.

**Engine Workers (Fly Machines, dynamically scaled)** — each machine runs the engine service container with a pool of Stockfish processes (one per CPU core). Workers pull analysis jobs from a shared Redis queue. The Django API monitors queue depth and uses the Fly Machine API to start additional engine machines when the queue grows, and allows them to auto-stop when it empties.

For bot match mode, engine requests are latency-sensitive (~300ms target for bot moves) and bypass the job queue — the Django API calls the engine service directly via HTTP for single-position Elo-constrained analysis. The job queue is used for batch analysis (game review) where latency tolerance is higher.

Machine spec: `performance-2x` (2 dedicated CPU cores, 4GB RAM). Each machine runs 2 Stockfish processes. Cost: ~$0.20/hour per machine, billed per second.

**Redis (Upstash or Fly Redis)** — job queue (via rq), real-time analysis progress store, position evaluation cache, Django Channels channel layer, and session state for active bot/lab games. Upstash's serverless model (pay per command) aligns well with bursty analysis workloads.

**PostgreSQL (Neon)** — user accounts, saved games, analysis history, billing records, and pgvector embeddings. Neon's serverless Postgres scales to zero when unused and auto-scales on demand.

### 11.3 Auto-Scaling Strategy

Engine workers are the primary scaling target. The scaling logic runs in the Django API:

1. When a game review job is submitted, check current queue depth and active worker count.
2. If queue depth exceeds a threshold (e.g. >10 pending jobs) and active workers are below the maximum, call the Fly Machine API to start an additional engine machine.
3. Engine machines pull jobs from the Redis queue until the queue is empty, then auto-stop after a configurable idle timeout (e.g. 60 seconds with no new jobs).
4. A maximum worker count cap prevents runaway scaling.

Bot match and opening lab engine calls do not trigger scaling — they are handled by always-on engine workers via direct HTTP, since each call is a single low-latency position analysis rather than a batch job.

The Django API and database do not need dynamic scaling until the app reaches thousands of concurrent users. However, concurrent WebSocket connections for bot matches do consume memory on Django API machines — each active game holds a connection and session state. At scale, this may require additional API instances.

### 11.4 Adding Additional Chess Engines

The engine service architecture supports multiple chess engines. Each engine type runs as a separate Fly Machine configuration with a different Docker image:

- **Stockfish:** CPU-only, `performance-2x` machines. Fast tactical analysis, primary engine.
- **Leela Chess Zero (lc0):** GPU-accelerated, `gpu` machines. More "human-like" positional evaluation, available as a premium feature.

Both engines speak UCI, so the same `python-chess` interface and engine pool logic applies. The Django API routes to different Redis queues based on engine selection. The core repo's EngineService interface is engine-agnostic — implementations differ only in which binary they launch.

For any future engine that does not speak UCI, a thin adapter translates its output to the same structured JSON schema. The Django backend never knows which engine produced the results.

### 11.5 Estimated Cloud Costs

**Infrastructure (moderate usage, a few hundred active users):**

| Service | Estimated Monthly Cost |
|---|---|
| Django API (2 Fly Machines) | $10–15 |
| Engine Workers (dynamic, avg 2 active) | $10–30 |
| Redis (Upstash) | $5–10 |
| PostgreSQL (Neon) | $10–15 |
| **Total infrastructure** | **$35–70** |

**Per-game-review API cost:** a game review generating ~15 LLM explanations at ~500 tokens each using Claude Sonnet costs approximately $0.02–0.04 in API usage. Engine compute per game review is approximately $0.003.

**Per-bot-match API cost:** a bot match with coaching nudges generates ~10–20 nudges per game at ~200 tokens each. Estimated cost: $0.01–0.02 per game in API usage. Engine compute per bot match is negligible (low-depth Elo-constrained searches).

**Pricing model:** a $10/month subscription covering ~100 game reviews and unlimited bot matches is comfortable margin at these unit economics.

---

## 12. Async Analysis Pipeline

### 12.1 Game Review Flow (Cloud Version)

1. User submits a PGN via the API.
2. Django parses the PGN, extracts the list of positions (one FEN per half-move), and creates an rq job containing the full position list.
3. Django returns a task ID immediately. The frontend begins polling for results.
4. An engine worker picks up the job from the Redis queue.
5. For each position, the worker runs Stockfish multi-PV analysis, computes centipawn loss for the played move, and classifies the priority tier.
6. Each analyzed move is published to Redis immediately (written to a hash keyed by task ID), so the frontend can display partial results as analysis progresses.
7. After all positions are analyzed, the Django API triggers LLM explanation generation for moves that exceed the centipawn loss threshold or have escalated priority tiers.
8. LLM explanations are batched where possible — consecutive moves in the same game phase can be explained in a single prompt to reduce API call count and latency.
9. Explanations are written to Redis as they complete, and the frontend updates in real-time.
10. When all explanations are finished, the task is marked complete and the full analysis is persisted to PostgreSQL.

### 12.2 Game Review Flow (Open Source Version)

1. User submits a PGN via the API.
2. Django opens an SSE (Server-Sent Events) connection and runs analysis synchronously in an async view.
3. For each position, Stockfish analysis runs via the engine service HTTP endpoint, context is assembled, and if the move warrants explanation, the LLM is called.
4. Each completed move is streamed to the frontend as an SSE event.
5. No task queue, no background workers. The analysis runs in the request lifecycle.

### 12.3 Bot Match Flow (Both Versions)

1. Player opens a bot match, selecting Elo, color, coaching mode, and optionally an opening preference. A WebSocket connection is established.
2. Server creates a session in Redis with the game configuration.
3. If the bot plays first (player chose Black), the server immediately generates the bot's opening move and sends it.
4. On each player move:
   a. The server validates the move via `python-chess` and updates the session state in Redis.
   b. The server calls the engine service for the bot's response (Elo-constrained, ~300ms).
   c. The bot move is sent to the client immediately.
   d. In parallel, the server runs context assembly on the new position and, if coaching mode warrants it, generates a coach nudge via the LLM (~800ms).
   e. The nudge is sent to the client as a separate message.
5. On game end (checkmate, resignation, draw), the server sends a game-over message and submits the full PGN to the game review pipeline.

### 12.4 Opening Lab Flow (Both Versions)

1. Player selects an opening (by ECO code or name), chooses a color, and starts the lab. A WebSocket connection is established.
2. Server creates a session in Redis with the opening's book line loaded.
3. On each player move:
   a. The server checks whether the move matches the book continuation for the selected opening.
   b. If the move matches: the bot responds with the next book move. No coaching fires (the player is on track).
   c. If the move deviates: the server generates an opening lab explanation via the LLM (what the book move achieves, why the deviation is suboptimal in this opening context). The bot still responds with its next move (the strongest theoretical reply to the deviation).
4. If the position goes beyond the opening book's coverage, the bot switches to full-strength Stockfish and the mode transitions to standard coach nudge behavior.
5. Progress (deepest move reached in theory before deviating) is tracked per opening per user in the cloud version.

### 12.5 Position Explorer Flow (Both Versions)

1. User submits a FEN and optionally a natural language question.
2. Django calls the engine service synchronously (single position, expected response time <2 seconds).
3. Context assembly and priority classification run inline.
4. LLM explanation is generated and returned in the API response.
5. No async processing needed — this is a single request/response cycle.

### 12.6 Performance Optimizations

**Move filtering:** forced recaptures, book moves (matched to ECO theory), and positions with only one legal move skip full depth-20 multi-PV analysis. Estimated 30–40% reduction in total engine analysis time per game.

**Position caching:** analyzed positions are cached in Redis, keyed by FEN + analysis depth. Identical positions across different games (common in popular openings) return cached results instantly. The cache is especially effective for opening positions, which thousands of games share.

**LLM batching:** consecutive moves in the same game phase (same priority tier, same opening) can be explained in a single LLM prompt, reducing API call count. A game with 15 explanation-worthy moves might require only 5–8 LLM calls rather than 15.

**LLM caching:** identical position + context combinations (which occur frequently in popular openings at the same skill level) can return cached explanations. The cache key is a hash of the assembled context object. Opening lab explanations for common deviations are highly cacheable — many players make the same mistakes in the same openings.

**Bot move latency:** Elo-constrained engine searches at reduced depth return in ~100–300ms, well within interactive response time. These do not use the job queue — they are direct HTTP calls to the engine service.

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| LLM hallucination of chess variations | LLM never generates or validates moves. All move data comes from the engine. LLM only narrates pre-verified facts. |
| Coach nudge accidentally suggests a move | Prompt template explicitly prohibits move suggestions. Output is validated to not contain move notation (e.g. "Nf3", "Bxe5") before sending to client. |
| Over-explanation fatigue | Explanations throttled to moves with meaningful centipawn loss or priority tier escalation. Brevity (2–3 sentences for review, 1 sentence for nudge) enforced in prompt template. Coaching mode selection (silent / nudge / verbose) gives the player control. |
| Tactical detector false positives | Detected patterns are passed to the LLM as context, not asserted as facts. The LLM is instructed to confirm relevance before naming a tactic. |
| Engine worker cost runaway | Maximum worker count cap on auto-scaling. Per-second billing limits exposure. Queue depth monitoring with alerts. |
| Slow game review for long games | Move filtering skips trivial positions. Position caching avoids redundant analysis. LLM batching reduces API calls. Progress streaming keeps the user engaged during analysis. |
| WebSocket connection drops mid-game | Session state persisted in Redis with 4-hour TTL. Frontend holds PGN history locally and can restore the session on reconnect. |
| Concurrent WebSocket memory pressure | Each active bot match holds a WebSocket connection and session state on a Django API machine. At high concurrency, additional API instances are needed. Monitoring connection count per machine triggers scaling. |
| Bot feels unrealistic at low Elo | Opening book with configurable deviation rates produces human-like opening play. Elo-constrained Stockfish produces realistic tactical oversights. Tuning deviation rates per Elo band requires playtesting. |
| Open-source / cloud divergence | All chess logic lives in core repo. Cloud repo only wraps infrastructure. Interface contracts enforce the boundary. |
| Multi-engine inconsistency | All engines return the same structured JSON schema via the EngineService interface. Engine-specific quirks are handled in the adapter layer, not in business logic. |

---

## 14. Open Questions

- Should skill level be set manually by the user, or inferred dynamically from their move quality and question patterns?
- What centipawn loss threshold triggers an explanation in game review mode? Initial setting of 30cp, but this needs tuning — too low produces noise, too high misses important moments.
- How should the app handle positions where engine evaluation is highly volatile (e.g. mutual zugzwang, fortress positions)?
- What is the right idle timeout for auto-stopping engine workers on Fly.io? Too short causes frequent cold starts; too long wastes compute.
- Should the vector retrieval layer (Phase 2) be optional in the open-source version, or should pgvector be a hard dependency?
- What is the right LLM batch size for consecutive move explanations? Batching too many moves in one prompt may reduce explanation quality.
- What centipawn loss threshold triggers a nudge in bot match mode? This may differ from the game review threshold — players in a live game may tolerate fewer interruptions.
- Should the coach nudge include the centipawn loss value, or is that information overwhelming during a live game?
- What is the right opening book format and source? Polyglot books are widely available but vary in quality and depth.
- How should bot Elo deviation rates be calibrated? Initial values (20% at 800, 10% at 1200, 5% at 1600, 0% at 2000) need validation through playtesting.
- Should opening lab support multiple lines per opening (e.g. both the Classical and Sämisch variations of the King's Indian), or one canonical line per ECO code?

---

## 15. Development Roadmap

### Phase 1 — Core Library & Open Source App (4–6 weeks)

Build `chess-explainer-core` with the full analysis pipeline: engine service, context assembly, priority classification, prompt templates, and explanation generation. Ship a working Docker Compose stack with a functional React frontend supporting game review mode. Validate end-to-end for a single user reviewing a completed game.

### Phase 2 — Cloud Infrastructure (3–4 weeks)

Create `chess-explainer-cloud`. Deploy Django API and engine workers on Fly.io. Implement user auth (django-allauth), the distributed engine worker pattern with rq, auto-scaling via Fly Machine API, and position caching. Validate game review for multiple concurrent users.

### Phase 3 — Bot Match & Opening Lab (4–5 weeks)

Add Django Channels and WebSocket infrastructure. Implement BotService with Elo-constrained play and opening book integration. Build CoachNudgeGenerator with question-only prompt templates. Implement SessionManager with Redis-backed state. Build opening lab mode with theory deviation detection. Validate both modes end-to-end in the open-source version, then deploy to cloud.

### Phase 4 — Billing & Product Features (3–4 weeks)

Add Stripe subscription integration, usage tracking per user, game history persistence, opening lab progress tracking, and position explorer mode. This is where the project becomes a product.

### Phase 5 — Advanced Features (ongoing)

Multi-engine support (Leela Chess Zero). Annotated game retrieval via pgvector. Personal mistake pattern tracking and progress analytics. Live tutor mode for active games against human opponents. Skill level auto-detection from move quality analysis. Bot personality variants (aggressive, positional, trappy) beyond simple Elo rating.
