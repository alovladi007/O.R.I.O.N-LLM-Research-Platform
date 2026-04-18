# Changelog

All notable changes to the ORION platform are tracked here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project aims
for [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Phase 0 — Repository hygiene (in progress)

- **Session 0.1 (merged `48ec361`)** — consolidated four parallel FastAPI
  entry points down to the canonical `src.api.app:app` on port 8002. Mock
  variants `simple_api.py` and `src/api/app_dev.py` deleted. Marketing demos
  moved to `examples/marketing/`. Pydantic v1 → v2 migration of
  `src/api/config.py`, deferred OpenTelemetry imports, added missing runtime
  deps. `metadata` column attribute renamed to `extra_metadata` across
  five SQLAlchemy models (DB column name preserved).
- **Session 0.2 (merged `b7ba6df`)** — removed ten legacy `src/*` packages
  (`core`, `candidate_generation`, `knowledge_graph`, `rag`, `data_ingest`,
  `simulation`, `experimental_design`, `evaluation`, `feedback_loop`, `ui`).
  Neo4j fully dropped from docker-compose, config, k8s, CI, and requirements.
  Five monolithic `orion_*.py` root-level dumps deleted (~12k lines).
  `config/config.yaml`, `requirements-minimal.txt`, `setup.py`, and one empty
  folder removed. `orion-ui` compose service (built from a nonexistent
  Dockerfile) replaced with `orion-frontend` pointing at the real Next.js
  Dockerfile. Net: **20,352 lines deleted, 247 added.**
- **Session 0.3 (this commit)** — docs triage. ~30 root-level markdown
  files reorganised: status / session-implementation / CI-fix reports moved
  to `docs/history/`; real user-facing guides moved to `docs/guides/`. Root
  kept to README, QUICK_START, DEPLOYMENT, BACKEND_QUICKSTART, CHANGELOG,
  ROADMAP_PROMPTS. README rewritten to match reality.

## [0.1.0-prototype] — 2026-04-17 (pre-Phase 0)

Baseline snapshot at the start of the roadmap. This tag marks what the
platform looked like before Phase 0 began: production-grade scaffolding
(Docker compose with 13 services, Alembic migrations, CI/CD, JWT auth,
Celery infrastructure) wrapped around a research prototype with
~24 `TODO`s, minimal test coverage, and numerous stubs.

### Known characteristics at 0.1.0-prototype

- Canonical `src/api/app:app` had never successfully imported due to
  Pydantic v1 syntax, an eager `src/__init__.py`, and an SQLAlchemy
  reserved-name collision on a `metadata` column.
- Four parallel FastAPI entry points (`app.py`, `app_dev.py`, `simple_api.py`,
  `demo_app.py`).
- Ten pre-refactor `src/*` packages untouched by the canonical API.
- Docker-compose referenced a nonexistent `docker/Dockerfile.ui`.
- Routers return mock data despite real implementations existing in
  `backend/common/*`.
- Only 2 test files (~878 lines) against ~14k Python files.
- CORS allowed `["*"]`, `.env` was tracked in git, demo passwords in
  `.env.example`.

See [ROADMAP_PROMPTS.md](./ROADMAP_PROMPTS.md) for the 14-phase plan to
bring the platform to production-ready state.
