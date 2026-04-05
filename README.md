# ChessLens Core

**Run chess.com-style game reviews on your own computer.** ChessLens Core is an open-source chess analysis engine that uses [Stockfish](https://stockfishchess.org/) and the [Claude API](https://docs.anthropic.com/en/docs/overview) to classify every move in a game — best, excellent, good, inaccuracy, mistake, blunder, brilliant, great, miss — the same categories chess.com uses, with plain-language explanations of what went wrong (or right) and why.

No subscription required. Your games, your hardware, your analysis.

---

## Quick Start

You need three things: **Python**, **Stockfish**, and an **Anthropic API key**.

### 1. Install Python

Download Python 3.12+ from [python.org](https://www.python.org/downloads/) if you don't already have it. You can check by running:

```bash
python --version
```

### 2. Install Stockfish

- **Mac:** `brew install stockfish`
- **Windows:** Download from [stockfishchess.org/download](https://stockfishchess.org/download/), extract it, and add the folder to your PATH
- **Linux:** `sudo apt install stockfish` (Ubuntu/Debian) or `sudo pacman -S stockfish` (Arch)

Verify it's installed:

```bash
stockfish --help
```

### 3. Get an Anthropic API Key

Sign up at [console.anthropic.com](https://console.anthropic.com/) and create an API key. You'll need this for the natural-language explanations. Game review analysis typically costs a few cents per game.

### 4. Set Up ChessLens

```bash
# Clone the repository
git clone https://github.com/SailingSF/chesslens-core.git
cd chesslens-core

# Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configure your API key
cp .env.sample .env
# Open .env in any text editor and replace sk-ant-... with your real key
```

### 5. Run It

```bash
python manage.py migrate
python manage.py runserver
```

Open [http://localhost:8000](http://localhost:8000) in your browser. Paste a PGN from any completed game and get a full move-by-move review with classifications and explanations.

---

## What You Get

### Game Review

Paste a PGN and get a chess.com-style review. Every move is analyzed by Stockfish and classified using an Expected Points model calibrated against chess.com's Classification V2 system. Moves that matter — mistakes, blunders, brilliant finds — get plain-language explanations from Claude, adjusted to your skill level (beginner, intermediate, or advanced).

The system skips forced moves, recaptures, and positions where nothing significant happened, so analysis focuses on the moments that actually affected the game.

### Position Explorer

Paste a FEN to get a one-shot analysis of any position: the engine's best move, a priority assessment (critical/tactical/strategic), and a natural-language explanation.

---

## Additional Tools

### Classification Optimizer

ChessLens includes a parameter optimization tool for tuning move classifications to match chess.com's system as closely as possible. This is useful if you want to experiment with the classification model or validate it against your own games.

The optimizer uses real game data extracted from chess.com reviews (see bookmarklet below) and compares ChessLens classifications against chess.com's ground truth. It supports grid search over sigmoid constants, EP thresholds, and engine parameters.

```bash
# Quick comparison — see how current config matches chess.com
python tests/optimize_classification.py --compare

# Run optimization (uses chess.com eval data, no Stockfish needed)
python tests/optimize_classification.py

# Optimize using real Stockfish analysis
python tests/optimize_classification.py --stockfish

# Sweep engine depth/multipv to find best engine match
python tests/optimize_classification.py --sweep-engine
```

Game CSVs for comparison go in `tests/test_games/`. The repo ships with several pre-extracted games.

### Chess.com Review Bookmarklet

A browser bookmarklet that extracts move-by-move review data from any chess.com Game Review page and downloads it as a CSV. This is how you get ground-truth data for the optimizer — or just export your chess.com reviews for your own records.

**Setup:**

1. Open `chesscom-review-download/bookmarklet.min.txt`
2. Create a new browser bookmark and paste the contents as the URL
3. Navigate to a completed Game Review on chess.com
4. Click the bookmark — a CSV downloads automatically

The CSV includes every move's classification (best, blunder, brilliant, etc.), evaluation, and eval change. See `chesscom-review-download/README.md` for full details.

---

## Bot Match and Opening Lab

ChessLens also includes two live play modes that require a Redis server in addition to Stockfish:

- **Bot Match** — play against an Elo-calibrated bot (400-3000) with optional Socratic coaching that names what's happening on the board without suggesting moves
- **Opening Lab** — study openings against a bot that follows book theory perfectly, with explanations when you deviate

These modes use WebSocket connections. To run them:

```bash
# Install and start Redis (Mac: brew install redis && brew services start redis)
# Then start with WebSocket support:
daphne -b 0.0.0.0 -p 8000 config.asgi:application
```

---

## Docker (Full Stack)

If you prefer Docker, one command starts everything — app, Stockfish engine, and Redis:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env
docker compose up
```

App runs at [http://localhost:8000](http://localhost:8000).

---

## How It Works

The system has four layers:

1. **Chess Engine** (`chess_engine/`) — Stockfish process pool. Reuses engine instances across requests for performance.
2. **Context Assembly** (`analysis/`) — Enriches raw engine output with opening identification, tactical patterns (forks, pins, skewers), pawn structure analysis, and threat narratives.
3. **Priority Classifier** (`analysis/priority.py`) — Pure logic (no LLM). Categorizes each position as CRITICAL, TACTICAL, or STRATEGIC based on score gaps and PV analysis.
4. **Explanation Layer** (`explanation/`) — Calls Claude with structured context to generate natural-language explanations. The LLM never calculates — it only narrates what Stockfish found.

Move classification uses an Expected Points model: Stockfish evaluations are converted to win probabilities via a sigmoid function, and the EP loss (difference between best move and played move win probability) determines the classification. Parameters are calibrated against chess.com's system using Stockfish 16.1 with 1M-node searches.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key for explanations |
| `REDIS_URL` | No | `redis://localhost:6379` | Redis (only needed for bot match / opening lab) |
| `ENGINE_URL` | No | — | Remote engine URL (set automatically in Docker) |
| `DATABASE_URL` | No | `sqlite:///db.sqlite3` | Database connection |
| `DEBUG` | No | `True` | Django debug mode |

---

## Running Tests

```bash
pytest
```

All tests run without Stockfish or an API key — they use real `python-chess` board objects with mocked engine results.

---

## Project Structure

```
chesslens-core/
├── config/                    Django settings, URL routing, ASGI config
├── chess_engine/              Stockfish interface + process pool
├── analysis/                  Context assembly, priority classification, tactics, pawns, openings
├── explanation/               Claude API integration + prompt templates
├── game/                      GameAnalyzer, BotService, SessionManager, models
├── api/                       REST API endpoints (async Django views)
├── ws/                        WebSocket consumers (bot match, opening lab)
├── ui/                        Frontend templates and static assets
├── tests/                     Unit tests + classification optimizer
│   ├── optimize_classification.py   Parameter optimizer for chess.com accuracy
│   └── test_games/            Ground-truth CSVs from chess.com reviews
├── chesscom-review-download/  Bookmarklet for extracting chess.com review data
├── engine_service/            Standalone Stockfish HTTP container (Docker)
├── docker-compose.yml         Full stack: web + engine + redis
└── requirements.txt           Python dependencies
```

---

## Relationship to ChessLens Cloud

This repository contains all chess intelligence. A separate private repository (`chesslens-cloud`) adds production infrastructure: user accounts, billing, distributed engine workers, PostgreSQL, and usage tracking. If a feature involves chess logic, it goes here. If it involves user management or infrastructure, it goes in cloud.
