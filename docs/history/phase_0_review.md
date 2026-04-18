# Phase 0 Review

**Date:** 2026-04-17
**Scope:** Verify every Phase 0 roadmap acceptance test against the as-merged
state of `main`.

## Commits in Phase 0

```
5078fbd  Session 0.5 — baseline test harness (closes Phase 0)
e8878c7  Session 0.4 — CORS, secrets, .env hygiene
7812bdc  Session 0.3 — docs triage
b7ba6df  Session 0.2 — remove legacy src/ packages, drop Neo4j
48ec361  Session 0.1 — consolidate entry points + bootstrap fixes
```

All five merged cleanly to `main` via fast-forward; per-session branches
deleted locally and on origin. Each has a report in `docs/history/`.

---

## Acceptance-test verification

### Session 0.1 — entry-point consolidation

| Roadmap check | Verified? | Evidence |
|---|---|---|
| `uvicorn src.api.app:app --port 8002` starts without import errors | **❌ deferred** | Tracked as the Session 1.2 blocker (stale `src/api/models/__init__.py` imports). Watchdog xfail test in `tests/test_smoke.py::test_health_endpoint_returns_200`. |
| `curl http://localhost:8002/healthz` returns 200 | ❌ deferred | Same blocker |
| `rg -n "simple_api\|standalone_api\|demo_app" --type py` → zero hits outside `examples/` | ✅ | `tests/test_smoke.py::TestRepoLayout::test_legacy_entry_points_are_gone` |
| `pytest tests/ -x` still passes | ✅ | 18 pass / 1 skip / 1 xfail, full suite green |

### Session 0.2 — legacy `src/` cleanup

| Roadmap check | Verified? | Evidence |
|---|---|---|
| `python -c "import src.api.app"` works | **❌ deferred** | Session 1.2 |
| `docker-compose.yml` has no orphan services | ✅ | Neo4j removed; `orion-frontend` builds the real Next.js Dockerfile |
| `MIGRATION_GUIDE.md` updated with "what moved" table | ✅ | `docs/history/phase_0_session_2_report.md` includes the complete move table |

**Additional 0.2 invariants now locked in by tests:**
- `test_legacy_src_packages_are_gone` — ensures `src/core`, `src/rag`, etc. don't reappear.
- `test_no_neo4j_refs_in_tracked_config` — ensures Neo4j stays out of compose, config, requirements, pyproject, `.env.example`.

### Session 0.3 — docs triage

| Roadmap check | Verified? | Evidence |
|---|---|---|
| `ls *.md` at root returns ≤ 6 files | ✅ | 6 exactly: README, QUICK_START, BACKEND_QUICKSTART, DEPLOYMENT, CHANGELOG, ROADMAP_PROMPTS. Test: `test_root_markdown_count_within_budget`. |
| README quickstart works when followed literally | 🟡 partial | Everything up to `make dev` works; `src.api.app` import still blocked (Session 1.2). Noted in README's "What works today" section. |
| No section in README claims a stubbed feature | ✅ | README has an explicit honest-status callout and a "What works today" table separating merged from still-stubbed |

### Session 0.4 — CORS, secrets, `.env` hygiene

| Roadmap check | Verified? | Evidence |
|---|---|---|
| `git ls-files .env` returns nothing | ✅ | Test at commit: `git ls-files .env` → empty |
| Start with `CORS_ORIGINS="*"` + `ORION_ENV=production` fails fast with descriptive error | ✅ | `test_wildcard_origin_rejected_in_prod` |
| Cross-origin OPTIONS from `http://evil.com` does not return `Access-Control-Allow-Origin` | 🟡 | Can't run end-to-end until Session 1.2 lets the app start. Config-level invariant that guarantees this behavior is verified. |
| **Extra, not in roadmap:** short `JWT_SECRET_KEY` rejected in prod | ✅ | `test_short_secret_rejected_in_prod` |
| **Extra:** demo DB/Redis passwords rejected in prod | ✅ | `test_demo_db_password_rejected_in_prod`, `test_demo_redis_password_rejected_in_prod` |
| **Extra:** `cors_allow_methods='*'` always rejected | ✅ | `test_wildcard_methods_rejected_always` |
| **Extra:** docker-compose `${VAR:?err}` enforces env-provided secrets | ✅ | All hardcoded demo passwords replaced with interpolation |

### Session 0.5 — baseline test harness

| Roadmap check | Verified? | Evidence |
|---|---|---|
| `pytest -q` prints ≥ 6 passing tests | ✅ | **18 passed**, 1 skipped, 1 xfailed |
| `pytest --cov=src --cov=backend --cov-report=term-missing` runs and prints a number | ✅ | Baseline = **12%** |
| CI job "tests" passes on a fresh PR | 🟡 | Workflow written, lint non-blocking by design. Will verify once a PR hits the workflow. |

---

## Roadmap additions that landed in Phase 0

Things the roadmap didn't explicitly require but that proved necessary:

1. **Pydantic v1 → v2 migration of `src/api/config.py`** (Session 0.1) — needed just to import the canonical app at all.
2. **Custom CSV-aware env source** (Session 0.1) — pydantic-settings v2 otherwise JSON-decodes `List[str]` env values.
3. **`metadata` → `extra_metadata` attribute rename** across 5 SQLAlchemy models (Session 0.1) — collided with `Base.metadata`.
4. **`src/__init__.py` and `src/api/__init__.py` blanked of eager imports** (Sessions 0.1 + 0.5) — legacy chains dragged in broken code.
5. **`Settings.environment` default flipped to `"development"`** (Session 0.5) — fail-safe vs. fail-unsafe when `ORION_ENV` is unset.
6. **Missing runtime deps added to `requirements.txt`** (Session 0.1) — `prometheus-fastapi-instrumentator`, `slowapi`, `opentelemetry-*`.
7. **`docker-compose.yml` service list reconciled** (Session 0.2) — `orion-ui` was built from a nonexistent Dockerfile; `orion-worker` pointed at a nonexistent Celery app. Both fixed.
8. **Legacy CI/security workflows stashed** (Session 0.5) — the 375-line aspirational `ci-cd.yml` references staging/prod K8s deploys, Trivy, Snyk, k6, and test directories that don't exist; stashed as `*.disabled-until-phase-{11,13}` for correct revival later.

---

## Known blockers passed forward to Phase 1

1. **`src/api/models/__init__.py` stale imports** — references five nonexistent modules (`workflow.py`, `embedding.py`, `provenance.py`, `experiment.py`, `orchestrator.py`). This is the single biggest blocker; Session 1.2 resolves it. Watchdog: `tests/test_smoke.py::test_health_endpoint_returns_200` (xfail).
2. **`metadata` → `extra_metadata` DB-column rename** — attribute renamed in Session 0.1 but the underlying DB column is still `"metadata"` via `mapped_column("metadata", ...)`. Session 1.2 must add an Alembic migration to rename the column itself (user directive from 2026-04-17).
3. **`backend/common/engines/lammps.py`** imports `ExecutionResult` from `.base` which isn't exported. Phase 4.
4. **`backend/common/ml/*`, `backend/common/provenance/*`** import from model submodules that don't exist. Unblocks naturally once Session 1.2 fills in the missing model files.
5. **Frontend API calls likely still point at port 8000**, not the canonical 8002. Session 9.1 audit will fix.

---

## Lines of code delta, Phase 0

| Category | Added | Removed |
|---|---|---|
| Session 0.1 | 479 | 782 |
| Session 0.2 | 247 | 20,352 |
| Session 0.3 | 485 | 481 |
| Session 0.4 | 516 | 164 |
| Session 0.5 | 929 | 47 |
| **Total** | **2,656** | **21,826** |

**Net −19,170 lines.** A 10x signal-to-noise improvement before any science
work begins.

---

## Ready for Phase 1?

Yes, with one open branch-level item: **the canonical FastAPI app still
cannot import.** That's the first thing Session 1.2 must fix. Session 1.1
(structures router wiring) technically depends on the app being importable
to test end-to-end, so I'll likely need to either:

- (a) execute Session 1.2 first, out of roadmap order, to unblock 1.1, or
- (b) execute 1.1's code changes and rely on module-level unit tests,
  deferring the E2E HTTP tests to after 1.2.

Recommendation: **(a) — swap the order.** Session 1.2 (data model) is smaller
and cleaner than 1.1 (router wiring + validation) and removes the xfail
watchdog. Then 1.1 lands with full HTTP-level tests.

Session 1.2's scope expands slightly to also include the `metadata` →
`extra_metadata` DB-column rename with an Alembic migration (user directive).

**Phase 0 complete.** On to Phase 1.
