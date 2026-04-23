# ChessLens Core

**Run chess.com-style game reviews on your own computer.** [ChessLens](https://chesslens.ai) is an open-source chess analysis engine that uses [Stockfish](https://stockfishchess.org/) and LLMs to classify every move in a game — best, excellent, good, inaccuracy, mistake, blunder, brilliant, great, miss — the same categories chess.com uses, with plain-language explanations of what went wrong (or right) and why in order to narrate your reviews and explain why particular moves perform better so that you can improve..

No subscription required. Your games, your hardware, your analysis.

---

## Quick Start

You need **Python 3.12+** and an **Anthropic API key** ([sign up](https://console.anthropic.com/)). Analysis typically costs a few cents per game.

```bash
# 1. Clone and install
git clone https://github.com/SailingSF/chesslens-core.git
cd chesslens-core
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Add your API key
cp .env.sample .env
# Edit .env and replace sk-ant-... with your real key

# 3. Download Stockfish (default: 16.1, same version chess.com uses)
python manage.py install_stockfish

# 4. Run
python manage.py migrate
python manage.py runserver
```

Open [http://localhost:8000](http://localhost:8000), paste a PGN, and get a full move-by-move review.

### Installing Stockfish yourself

`install_stockfish` downloads the official binary into `./engines/`. If you'd rather install it yourself, any of these work — ChessLens discovers binaries from `./engines/`, `$STOCKFISH_DIR`, and your `PATH`:

- **Mac:** `brew install stockfish`
- **Linux:** `sudo apt install stockfish`
- **Windows:** [download](https://stockfishchess.org/download/) and drop the `.exe` into `./engines/`

To run multiple versions side-by-side, drop each binary into `./engines/` (e.g. `stockfish-16.1`, `stockfish-17`). The UI shows a dropdown so you can pick which one to analyze with. Verify discovery with `python manage.py list_engines`.

---

## What You Get

### Game Review
Paste a PGN and get a chess.com-style review. Every move is classified using an Expected Points model calibrated against chess.com's Classification V2 system. Moves that matter get plain-language explanations from Claude, adjusted to your skill level.

### Position Explorer
Paste a FEN for one-shot analysis of any position: top candidate moves, priority assessment, and a natural-language explanation.

---

## How It Works

Four layers:

1. **Chess Engine** (`chess_engine/`) — Stockfish process pool, one per binary version.
2. **Context Assembly** (`analysis/`) — opening ID, tactical patterns, pawn structure, threat narrative.
3. **Priority Classifier** (`analysis/priority.py`) — pure logic; categorizes positions as CRITICAL, TACTICAL, or STRATEGIC.
4. **Explanation Layer** (`explanation/`) — calls Claude with structured context. The LLM never calculates — it only narrates what Stockfish found.

Stockfish evaluations are converted to win probabilities via a sigmoid function, and the EP loss between the best and played move drives classification. Parameters are calibrated against chess.com using Stockfish 16.1 with 1M-node searches.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Claude API key. |
| `STOCKFISH_DIR` | — | Extra directory to scan for Stockfish binaries. |
| `STOCKFISH_DEFAULT_ENGINE` | — | Preferred engine id / version substring (else first 16.1 match). |
| `STOCKFISH_DEFAULT_NODES` | `500000` | Nodes per search. |
| `STOCKFISH_DEFAULT_MULTIPV` | `3` | Candidate lines per search. |
| `STOCKFISH_PV_LENGTH` | — | Truncate PVs to N plies (off by default). |
| `STOCKFISH_PV_END_NODES` | — | Re-evaluate PV endpoints with this node budget. |
| `STOCKFISH_PLAYED_MOVE_NODES` | — | `searchmoves` budget for moves outside top-N. |
| `ENGINE_URL` | — | Remote engine service URL (set automatically in Docker). |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection. |
| `DEBUG` | `True` | Django debug mode. |

---

## Tests

```bash
pytest
```

Unit tests run without Stockfish or an API key.

---

## Docker

```bash
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env
docker compose up
```

Full stack (app + engine container + Redis) at [http://localhost:8000](http://localhost:8000).

---

## Also in this repo

- **`tests/optimize_classification.py`** — parameter optimizer used to tune the classification model against chess.com ground truth. Development tool.
- **`chesscom-review-download/`** — browser bookmarklet that exports chess.com review data as CSV, used to generate optimizer ground truth.
