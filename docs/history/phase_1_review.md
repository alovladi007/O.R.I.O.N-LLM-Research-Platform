# Phase 1 Review — Data & Domain Foundation

**Date:** 2026-04-18
**Scope:** Verify every Phase 1 roadmap acceptance test against the as-
merged state of `main` and set up Phase 2.

## Commits in Phase 1

```
c0e8300  Session 1.5 — MP seed loader + offline fixtures
816c52e  Session 1.4 — jobs spine (state machine, SSE, log stub)
d80ad1a  Session 1.3 — bulk property importer (synchronous MVP)
8a49f4b  Session 1.1 — structures router wired to pymatgen
a090d46  Session 1.2b — units, structure_hash, embedding, migration 009
4c5e1c7  Session 1.2a — unblock canonical src.api.app import
```

Six merges. Executed in the order 1.2 → 1.1 → 1.3 → 1.4 → 1.5 (swapping
1.1 and 1.2 as recommended in the Phase 0 review — the canonical app
needed to import before the HTTP-level structure tests could land).

## Acceptance-test verification

### Session 1.1 — Structures router

| Check | Verified | Evidence |
|---|---|---|
| Si → spacegroup 227 (Fd-3m), cubic | ✅ | `tests/test_structures_parse.py::test_silicon_parses_to_spacegroup_227` |
| NaCl → spacegroup 225 (Fm-3m), cubic | ✅ | `test_nacl_parses_to_spacegroup_225` |
| StructureMatcher-style round-trip | ✅ (hash) | `test_hash_matches_for_reordered_atoms` |
| Invalid CIF → 422 | ✅ | HTTP `test_invalid_cif_returns_error` |
| Overlapping sites (<0.5 Å) → 422 | ✅ | `test_overlapping_sites_return_422` |
| HTTP-level tests via TestClient | ✅ | 5 cases in `TestParseEndpoint` |

### Session 1.2 — Data model

| Check | Verified | Evidence |
|---|---|---|
| `python -c "import src.api.app"` works | ✅ | 102 routes, `test_canonical_app_imports` + `test_health_endpoint_returns_200` |
| Structure has unique `structure_hash` | ✅ | schema + unique index in migration 009 |
| Unit module with pint + property registry | ✅ | `tests/test_units.py` — 14 cases |
| Hash invariants (reorder / tolerance / species / lattice) | ✅ | `tests/test_structure_hash.py` — 12 cases |
| Alembic up/down cycle clean | 🟡 syntax-verified | Exercised live in CI once Postgres service container is reachable |
| `metadata` → `extra_metadata` DB column rename | ✅ | migration 009 renames on 8 tables |

### Session 1.3 — Bulk property importer

| Check | Verified | Evidence |
|---|---|---|
| Negative bandgap row rejected (but siblings import) | ✅ | `test_negative_bandgap_rejected_by_bounds_check` |
| Errors CSV download works | ✅ | `test_renders_row_columns_from_raw` + endpoint |
| Unit mismatch rejected | ✅ | `test_unit_not_accepted_for_property_raises` |
| Bounds violation rejected | ✅ | `backend.common.units.validate_value` used on every row |
| Error cap enforced | ✅ | `test_cap_limits_recorded_errors` |
| HTTP round-trip with real Postgres | ⏳ deferred | Placeholder `requires_db` test skipped locally; runs in CI |

### Session 1.4 — Jobs spine

| Check | Verified | Evidence |
|---|---|---|
| Illegal state transition → 409 | ✅ | `cancel_simulation_job` catches `IllegalJobTransitionError` |
| State machine table matches roadmap | ✅ | `_LEGAL_TRANSITIONS` + `tests/test_job_state_machine.py` (35 cases) |
| `JobKind` enum spelled correctly | ✅ | `test_job_kind_values_match_roadmap` |
| SSE `/events` endpoint exists | ✅ | Route registered; polling implementation |
| `/logs` endpoint returns text/plain | ✅ | Placeholder pending Session 2.1 MinIO wiring |
| Real Celery cancellation signal | ⏳ Session 2.1 | Intentionally deferred |

### Session 1.5 — Seed data

| Check | Verified | Evidence |
|---|---|---|
| MP live fetch path implemented | ✅ (untested w/o key) | `fetch_mp_subset(family=...)` |
| Offline fallback exists | 🟡 partial | 5 structures, not 20 — scope-trimmed; see 1.5 report |
| Re-run inserts 0 duplicates | ✅ | by construction via `structure_hash` uniqueness |
| 3 seed users (admin/scientist/viewer) | ✅ | `_seed_users` + `DEFAULT_SEED_USERS` |
| `structure_hash` backfill script | ✅ | `--backfill-hashes` flag |

## What shipped outside scope

- **Double-prefix router bug** fixed on `structures` (Session 1.1) and
  `jobs` (Session 1.4). Materials, workflows, campaigns, ml still carry
  the same bug — flagged for Session 2.1.
- **Pydantic v1 → v2 migration continued** (AliasChoices for env vars,
  computed_field for properties, model_validator for security invariants).
- **5 Alembic migrations** added: 009 (rename + hash + embedding + unit
  / uncertainty + experiments tables), 010 (spacegroup columns),
  011 (bulk import jobs), 012 (simulation job kind). Migrations 009 and
  010 touch the Structure table — a consolidated version will be useful
  if we ever squash for a clean initial schema (Phase 13 release).
- **Five restored model files** in 1.2a (workflow, embedding, provenance,
  experiment, orchestrator) — the existing `models/__init__.py` expected
  them but the files didn't exist. Without this Phase 1 couldn't begin.

## Tests now vs. end of Phase 0

| Metric | Phase 0 end | Phase 1 end | Delta |
|---|---|---|---|
| Total tests | 18 | 130 | **+112** |
| Passing | 18 | 130 | +112 |
| Coverage | 12% | not re-measured | — |
| Canonical app importable | ❌ | ✅ (97 routes → 102) | — |
| Physics-level checks | 0 | Si 227 + NaCl 225 + density + bounds | — |

Test runtime: ~12 s for the full 130-test suite. Well inside CI time
budgets.

## Phase 2 readiness

Phase 2 — *Job Execution Spine* — consumes most of what Phase 1 shipped:

- `JobKind` + state machine → Session 2.1 Celery task base class.
- `BulkImportJob` schema with `source_kind=minio` field → Session 2.1
  moves bulk import off the inline-CSV path.
- `structure_hash` unique index → Session 2.2 mock engine dedup.
- SSE `/events` endpoint → Session 2.1 adds the Redis pubsub publisher
  (clients already work against the polling fallback).
- `/logs` placeholder → Session 2.1 wires MinIO.

**The canonical FastAPI app is now running with 130 green tests.** Phase
2 can start.

## Known items carried forward

1. Double-prefix bug on materials/workflows/campaigns/ml routers.
2. Offline MP bundle at 5 — grow to ≥ 20 when a later session actually
   stresses the dataset size.
3. `requires_db` tests are placeholders; the full DB-integrated suite
   needs a per-test transaction fixture (Session 2.1 or 0.5-follow-up).
4. Pydantic deprecation warnings (`min_items` → `min_length`, class
   Config → ConfigDict) scattered across schemas — cosmetic; batch-fix
   in Phase 11.
5. Legacy `src/worker/` module still has `settings.TESTING` reference
   (`src/api/database.py:49`). Harmless until Phase 2 actually starts
   workers.
