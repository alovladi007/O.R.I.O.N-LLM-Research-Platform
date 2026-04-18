# Phase 0 / Session 0.1 — Entry-point consolidation

**Branch:** `phase-0-session-1-entrypoint-consolidation`
**Date:** 2026-04-17

## Scope delivered (strictly in-scope per roadmap)

- Deleted mock API variants that duplicated the canonical app:
  - `simple_api.py` (root-level mock, port 8002)
  - `src/api/app_dev.py` (no-DB mock with hardcoded structures)
- Moved marketing demos out of the root into `examples/marketing/`:
  - `demo_app.py` → `examples/marketing/streamlit_demo.py` (Streamlit marketing UI)
  - `run_demo.py` → `examples/marketing/landing_page_demo.py` (stdlib HTTP landing page)
  - Each got a top-of-file docstring marking it as non-backend and explaining the move.
- `Makefile`'s `make dev` now runs `uvicorn src.api.app:app --reload --port 8002`
  (canonical entry point; 8002 matches recent commits and the frontend origin).
- Rewrote `BACKEND_QUICKSTART.md` to reference only the canonical path.
- Added `.claude/` to `.gitignore`.

## Scope expanded — unavoidable blockers on the acceptance test

The roadmap's acceptance test for 0.1 is:
> `uvicorn src.api.app:app --port 8002` starts without import errors.

Running the smoke test uncovered that the canonical app had **never** successfully
imported in this codebase. Three distinct blockers, each fixed below, were required
just to get the import tree resolving:

### 1. Legacy `src/__init__.py` eager-imported broken modules

Top-level `src/__init__.py` contained `from .core import ORIONSystem, ...`, which
eagerly imported `src.knowledge_graph.schema`, which had a Python 3.10 dataclass
ordering bug (`non-default argument 'process_type' follows default argument`).
That bug is in the legacy pre-refactor code earmarked for removal in Session 0.2.

**Fix:** blanked `src/__init__.py` to a minimal package init. The legacy modules
still exist on disk and still have bugs — Session 0.2 will remove or migrate them.

### 2. `src/api/app.py` imported OpenTelemetry Jaeger exporter at module top level

Import path `from opentelemetry.exporter.jaeger import JaegerExporter` failed on
current `opentelemetry-exporter-jaeger` (moved to `.thrift` and `.proto_grpc`
subpackages). Because the import was at module top level, it failed even when
`enable_tracing=False`.

**Fix:** deferred all OpenTelemetry imports into the `if settings.enable_tracing`
branch inside the lifespan handler. Also changed the default of `enable_tracing`
to `False` — tracing is now opt-in (Phase 10 sets this up properly).

### 3. `src/api/config.py` used Pydantic v1 syntax, but Pydantic v2 is installed

`BaseSettings` moved to the separate `pydantic-settings` package; `@validator` →
`@field_validator`; `Field(env=...)` → `Field(alias=...)`; and pydantic-settings v2
tries to JSON-decode `List[str]` fields from the env, which breaks on plain CSV
values like `CORS_ORIGINS=http://a,http://b`.

**Fix:** full Pydantic v2 migration of `config.py`, including a custom
`_CsvEnvSource` / `_CsvDotEnvSource` pair registered via
`settings_customise_sources` that skips JSON decoding for the two CSV-list
fields (`cors_origins`, `elasticsearch_hosts`). This preserves the existing
`.env` format without requiring users to rewrite their env files.

## Scope expanded a second time — SQLAlchemy reserved-name bug (user-approved)

After unblocking the imports, the next failure was:
`sqlalchemy.exc.InvalidRequestError: Attribute name 'metadata' is reserved`

Seven `metadata: Mapped[Optional[dict]]` columns across 5 model files collided
with SQLAlchemy's reserved `Base.metadata`. With user approval (Option A of the
pause-and-ask), renamed each column attribute to `extra_metadata` while keeping
the DB column name as `"metadata"` via the positional `mapped_column("metadata",
...)` argument. No Alembic migration needed. Updated the `to_dict()` serializers
to call `self.extra_metadata` but emit `"metadata"` as the JSON key, so the
external API is unchanged.

Files touched:
- `src/api/models/material.py`
- `src/api/models/structure.py`
- `src/api/models/simulation.py` (2 occurrences)
- `src/api/models/predicted_properties.py`
- `src/api/models/campaign.py` (2 occurrences)

## Dependencies added to `requirements.txt`

The canonical `app.py` imports were not fully declared. Added:

- `prometheus-fastapi-instrumentator>=6.1.0`
- `slowapi>=0.1.9`
- `opentelemetry-api>=1.20.0`
- `opentelemetry-sdk>=1.20.0`
- `opentelemetry-exporter-jaeger>=1.20.0`
- `opentelemetry-instrumentation-fastapi>=0.41b0`

## Acceptance test — actual state

| Roadmap check | Status | Notes |
|---|---|---|
| `uvicorn src.api.app:app --port 8002` starts without import errors | ❌ still fails | Blocked by `src/api/models/__init__.py` importing non-existent `workflow.py`, `embedding.py`, `provenance.py`, `experiment.py`, `orchestrator.py`. This is Session 1.2 (data model) territory. |
| `curl http://localhost:8002/healthz` returns 200 | ⏸ not runnable until import works | |
| `rg -n "simple_api\|standalone_api\|demo_app" --type py` returns zero hits except examples/ | ✅ passes | Only `examples/marketing/streamlit_demo.py` matches |
| `pytest tests/ -x` still passes | ⚠ not run | tests/ has import-time failures too — Session 0.5 establishes a real harness |

## Known blockers passed to Session 0.2 / 1.2

1. **`src/api/models/__init__.py` imports five non-existent modules:**
   `workflow.py`, `embedding.py`, `provenance.py`, `experiment.py`, `orchestrator.py`.
   Either the files were never created or they were deleted without cleaning
   the package init. Session 1.2 must reconcile (`__init__.py` ↔ actual files).
2. **Legacy `src/core/`, `src/knowledge_graph/`, `src/rag/`, etc.** still exist
   and have their own bugs (Python 3.10 dataclass ordering, missing deps).
   Session 0.2 removes or migrates them.
3. **`src/api/database.py` uses `|` union syntax** which requires Python ≥3.10.
   `pyproject.toml` declares `python = "^3.10"` but `requirements-minimal.txt`
   and CI may use 3.9 — audit in Session 0.5.

## Files changed summary

```
 .gitignore                                    |  +3
 BACKEND_QUICKSTART.md                         |  rewritten
 Makefile                                      |  dev target port 8000→8002
 requirements.txt                              |  +7 deps
 simple_api.py                                 |  deleted (-111 lines)
 src/api/app_dev.py                            |  deleted (-340 lines)
 demo_app.py → examples/marketing/streamlit_demo.py            | moved + header
 run_demo.py → examples/marketing/landing_page_demo.py         | moved + header
 src/__init__.py                               |  blanked of legacy imports
 src/api/app.py                                |  tracing imports deferred
 src/api/config.py                              |  full Pydantic v1→v2 migration
 src/api/models/{material,structure,simulation,predicted_properties,campaign}.py
                                               |  metadata → extra_metadata
 docs/history/phase_0_session_1_report.md      |  this report
```

## Decision log

- **Kept DB column named `"metadata"`** (not `extra_metadata`) — avoids a
  migration for data that may already exist in anyone's local dev DB.
- **Did not attempt to implement missing `workflow.py` / `embedding.py` / etc.**
  — that's scope for Session 1.2 data-model work. Stopping here preserves the
  per-session branch discipline the user requested.
- **Default `enable_tracing` flipped False** — tracing requires a Jaeger agent
  running; off by default prevents the app from trying to connect on startup.
  Phase 10 turns it back on correctly.
