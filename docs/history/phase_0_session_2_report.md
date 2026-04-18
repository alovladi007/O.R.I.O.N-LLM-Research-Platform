# Phase 0 / Session 0.2 — Legacy `src/` cleanup + Neo4j removal

**Branch:** `phase-0-session-2-legacy-src-cleanup`
**Date:** 2026-04-17

## Scope

Remove or promote the pre-refactor `src/*` packages that are not part of the
canonical ORION backend (`src/api` + `backend/common` + `src/worker`), and
collapse the Neo4j dependency — unused in the canonical code but present in
docker-compose, config, k8s manifests, CI, and requirements.

## Usage audit table

| Package | External imports found | Decision |
|---|---|---|
| `src/core` | only legacy + `examples/quick_start.py` | **deleted** |
| `src/candidate_generation` | only `src.core.orion_system` | **deleted** |
| `src/knowledge_graph` | only `src.core.orion_system` | **deleted** |
| `src/rag` | only `src.core.orion_system` | **deleted** |
| `src/data_ingest` | only `src.core.orion_system` (empty pkg) | **deleted** |
| `src/simulation` | only `src.core.orion_system` (empty pkg) | **deleted** |
| `src/experimental_design` | only `src.core.orion_system` (empty pkg) | **deleted** |
| `src/evaluation` | none | **deleted** |
| `src/feedback_loop` | only `src.core.orion_system` (empty pkg) | **deleted** |
| `src/ui` (Streamlit) | none | **deleted** |
| `src/worker` | `src/api/routers/jobs.py` | **kept** (real Celery infra) |
| `src/api` | canonical | **kept** |

Ten legacy packages removed. Total `src/` minus `api/` + `worker/`.

## Neo4j removed

Neo4j was only referenced by the legacy `src/knowledge_graph` (deleted above).
The canonical API and `backend/common/*` don't import the `neo4j` package at
all. Removed from:

- `docker-compose.yml` — `neo4j` service, volumes, `depends_on` on `orion-api`
  and `orion-worker`, `NEO4J_URI` env var on `orion-api`.
- `src/api/config.py` — `neo4j_uri` / `neo4j_user` / `neo4j_password` fields.
- `requirements.txt` — `neo4j>=5.0.0`.
- `pyproject.toml` — `neo4j = "^5.15.0"`.
- `k8s/base/api-deployment.yaml` — `NEO4J_URI` / `NEO4J_PASSWORD` env refs.
- `.github/workflows/ci-cd.yml` — `neo4j` service in CI, env vars.
- `.env.example` — `NEO4J_*` block.

## Other legacy artifacts removed

- **Five monolithic `orion_*.py` dumps at repo root** (~400 KB, not imported
  by anything): `orion_advanced_features.py`, `orion_core_modules.py`,
  `orion_integration_deployment.py`, `orion_testing_optimization (1).py`,
  `orion_usage_examples.py`. These appear to be pre-refactor design sketches
  or AI-generated scaffolds; nothing in the canonical code references them.
- **`config/config.yaml`** — 319-line YAML config with no readers anywhere.
  The canonical app reads env via pydantic-settings. Phase 13 will reinstate
  a structured config file if needed, properly wired.
- **`index.html`** at repo root — moved to `examples/marketing/index.html`
  with an HTML comment explaining the move.
- **`setup.py`** — duplicates `pyproject.toml`'s poetry config and references
  a nonexistent `orion.cli:main` console script. Deleted.
- **`requirements-minimal.txt`** — Streamlit-era bootstrap (6 lines:
  streamlit, plotly, pandas, numpy, python-dotenv, pyyaml). Obsolete.
- **Empty folder `O.R.I.O.N-LLM-Research-Platform/`** at root — removed.
- **`test_engines.py`** at root — real smoke test for `backend/common/engines`;
  moved to `scripts/test_engines_smoke.py`. Session 0.5 builds the proper
  `tests/` harness.
- **`examples/quick_start.py`** — imported `src.core.ORIONSystem` which is
  now gone. Deleted.

## Dependency hygiene

### `requirements.txt`
Removed (bundled with dead packages): `streamlit`, `plotly`, `asyncio-nest`,
`neo4j`, `pymongo`, `rdflib`, `owlready2`, `faiss-cpu`, `sentence-transformers`,
`langchain-community`. Added `anthropic` (Phase 7 agent). Plotly stays in
`pyproject.toml` because the frontend / Jupyter can still use it.

### `pyproject.toml`
Heavy restructure — moved from an unorganised ~100-line dependency dump to
grouped blocks (API, DB, workers, observability, ML, scientific, utilities,
deploy) with trimmed set. Dev deps moved to `[tool.poetry.group.dev]`.
Removed: `neo4j`, `gradio`, `streamlit`, `pymongo`/`motor`/`beanie` (Mongo
unused), `spacy`/`nltk`/`gensim` (RAG NLP), `chembl-webresource-client`/
`pubchempy`/`biopython` (legacy data ingest), `rdflib`/`owlready2`,
`faiss-cpu`, `sentence-transformers`, `arrow`/`pendulum`/`schedule`/
`apscheduler` (unused utility grab-bag), `grafana-api`, `marshmallow`,
`ray`/`dask` (premature — Phase 6 can reintroduce if needed).

## docker-compose changes

- `neo4j` service: removed.
- `orion-ui` (Streamlit, built from a nonexistent `docker/Dockerfile.ui`):
  replaced with `orion-frontend` that builds the existing
  `docker/Dockerfile.frontend` (the real Next.js UI) and exposes port 3002.
- `orion-worker`: `command` corrected from `celery -A orion.tasks` (nonexistent
  module) to `celery -A src.worker.celery_app` which actually exists.
- `ORION_CONFIG=/app/config/config.yaml` env var removed (file deleted).

## What still doesn't work

`src.api.app` still cannot be imported — same blocker as end of Session 0.1:
`src/api/models/__init__.py` imports five files that don't exist
(`workflow.py`, `embedding.py`, `provenance.py`, `experiment.py`,
`orchestrator.py`). That's Session 1.2 territory.

Also newly surfaced (pre-existing, not introduced by this session):
`backend/common/engines/lammps.py` imports `ExecutionResult` from `.base`
which doesn't export it. Noted for Phase 4 / LAMMPS work.

## Files changed summary

```
49 files changed, 103 insertions(+), 20,352 deletions(-)
```

| Category | Files deleted | Lines removed |
|---|---|---|
| Legacy `src/*` packages | ~20 files | ~8,000 |
| Root-level `orion_*.py` monoliths | 5 files | ~12,000 |
| `config/config.yaml` | 1 | 319 |
| `requirements-minimal.txt`, `setup.py`, empty dir | 3 | ~95 |

## Acceptance tests — state vs roadmap

| Roadmap check | Status | Notes |
|---|---|---|
| `python -c "import src.api.app"` works | ❌ still blocked by missing model files (Session 1.2) |
| `docker-compose.yml` has no orphan services | ✅ | Neo4j gone, `orion-ui` replaced with `orion-frontend` pointing at real Dockerfile |
| `MIGRATION_GUIDE.md` updated with "what moved" table | ⏳ | Deferred to Session 0.3 (docs triage) |
| No residual `from src.core` / legacy-pkg imports in tree | ✅ | confirmed by grep |
| No residual Neo4j / rdflib / owlready2 refs outside comments | ✅ | only mentions are this report and pyproject comments |

## Decision log

- **Neo4j fully removed, not feature-flagged.** Roadmap Session 0.2 offered
  both options; picked removal because the knowledge-graph work wasn't on the
  roadmap and no canonical code used it. Can always be reintroduced cleanly
  in a future phase.
- **`setup.py` deleted instead of kept.** Poetry `pyproject.toml` is the
  source of truth. Keeping both invites drift.
- **`sdk/python/` left alone.** Separate Python SDK package with its own
  `setup.py`. Out of scope for this session.
- **Plotly kept** in pyproject despite dropping from requirements.txt — useful
  for Jupyter sidecar and optional visualization; low install cost.
