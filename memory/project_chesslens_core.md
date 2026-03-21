---
name: chesslens-core project context
description: Architecture, module layout, and current status of the chesslens-core open-source chess analysis repo
type: project
---

ChessLens Core is an open-source Django app that combines Stockfish engine analysis with Claude AI narration to explain chess positions.

**Key principle:** LLM never calculates — it only narrates. All chess facts come from Stockfish + context layer.

**Module layout:**
- `chess_engine/` — EngineService interface, PooledEngineService (pool-backed), RemoteEngineService (HTTP), EnginePool, get_engine_service() factory
- `analysis/` — ContextAssembler (with threat narrative, piece activity, theory deviation), PriorityClassifier (accepts board for capture detection), ECOLookup + eco_data.json (~80 openings), tactical patterns (fork, pin, skewer, discovered attack, back-rank), pawn structure
- `explanation/` — ExplanationGenerator (async, AsyncAnthropic), CoachNudgeGenerator (async, move notation validator), prompt templates (game_review, coach_nudge, opening_lab), singleton factories
- `game/` — GameAnalyzer (PGN orchestrator, skips forced moves + recaptures), BotService + OpeningLabBotService (UCI_LimitStrength wired through), SessionManager (Redis), Django models (SavedGame, MoveNote)
- `api/` — Native async DRF views: game-review (SSE), position-explorer, bot-match/start, opening-lab/start
- `ws/` — Django Channels consumers: BotMatchConsumer, OpeningLabConsumer (SAN-before-push, shared services)
- `config/` — Django settings (daphne in INSTALLED_APPS), asgi.py (Channels routing), routing.py

**Status (2026-03-16):** Architecture review completed. All 20 tests passing. Major fixes applied:
- Engine uses PooledEngineService (reuses Stockfish processes) instead of per-call subprocess
- UCI options (Elo constraints) wired through to engine
- LLM calls are fully async (AsyncAnthropic)
- API views are native async Django views
- SAN-before-push bug fixed in all consumers
- Bot match + opening lab session start endpoints added
- RemoteEngineService added for Docker mode
- Context assembly stubs implemented (threat narrative, piece activity, theory deviation)

**Remaining Phase 1 work:**
- Expand eco_data.json (currently ~80 entries, spec calls for ~500 ECO codes / 2,000 variations)
- Refine knight outpost detection (currently rank-based heuristic, needs pawn support check)
- Position caching in Redis (by FEN + depth) — deferred for now
- LLM response caching — deferred for now
- Build React frontend

**Why:** Open-source MIT-licensed core that the private chesslens-cloud repo will import as a dependency.
