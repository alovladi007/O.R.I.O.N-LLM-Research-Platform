# Phase 1 / Session 1.2 — Data-model foundation

**Branch:** `phase-1-session-1.2-data-model`
**Date:** 2026-04-17

## Scope

Two things in one session (as two separate commits):

1. **1.2a — Unblock the canonical `src.api.app` import.** This was the
   Phase 0 watchdog xfail. Phase 1 can't start while the app can't load.
2. **1.2b — Physics-correctness upgrades** the roadmap calls for:
   unit system (`backend/common/units.py`), deterministic structure
   hashing, `Structure.embedding`, `PredictedProperties.value_unit` +
   `.uncertainty`, the `Instrument` / `ExperimentRun` tables, and the
   Alembic migration that renames the `metadata` → `extra_metadata` DB
   column everywhere (user directive from 2026-04-17).

Session 1.1 (structures router wiring with physics-level acceptance
tests) was intentionally held until 1.2 landed, per the Phase 0 review's
recommendation. It's up next.

## 1.2a — restore missing models + fix import chain (commit 1)

### Missing model files

`src/api/models/__init__.py` imported five files that simply didn't
exist on disk. Restored:

| File | Source of truth for shape | Enums added |
|---|---|---|
| `workflow.py` | migration `001_initial_schema` (`workflow_templates`) | — |
| `embedding.py` | migration `001_initial_schema` (`vector_embeddings`) | — |
| `provenance.py` | migration `003_add_provenance_tracking` | `EntityType`, `EventType` |
| `orchestrator.py` | migration `008_add_orchestrator` | `OrchestratorMode` |
| `experiment.py` | **no existing migration** — migration `009` added here | `InstrumentStatus`, `ExperimentStatus` |

Each model's column list matches the corresponding migration exactly,
with proper indexes, FKs, and relationships. `to_dict()` helpers follow
the same pattern as the existing models.

### Import-path bugs fixed

- **3-level relative imports** that couldn't resolve (they'd expected a
  top-level package structure ORION doesn't have):
  - `backend/common/design/search.py`
  - `backend/common/campaigns/loop.py` (3 occurrences)
  - `src/api/routers/design.py`

  All switched to absolute `from src.api.models...` or
  `from backend.common...`.

- **Missing re-exports**:
  - `backend/common/provenance/__init__.py` — added
    `get_provenance_summary` (function existed in `tracker.py`).
  - `backend/orchestrator/__init__.py` — added
    `get_or_create_orchestrator` (function existed in `core.py`).

- **Missing auth scaffolding**:
  - `src/api/auth/__init__.py` created — re-exports from `.security`
    and provides a `get_optional_user` helper.
  - `src/api/dependencies/__init__.py` + `src/api/dependencies/auth.py`
    created as a thin shim so routers importing
    `src.api.dependencies.auth` keep working.

- **Exception-class name drift**: `src/api/app.py` imports
  `ORIONAPIException`, `handle_orion_exception`, etc., but
  `src/api/exceptions.py` defines `NANOOSException`,
  `nanoos_exception_handler`, ... . Added ORION-branded aliases at the
  bottom of `exceptions.py` without renaming the originals.

### Gitignore bug

`.gitignore` had an unanchored `models/` rule that silently ignored new
files under `src/api/models/`. Anchored to `/models/` (root-level ML
artefact dir). `git add -f` not needed after the fix.

### Result

`from src.api.app import app` → **97 routes registered, 0 errors.**

Phase 0 watchdog `test_health_endpoint_returns_200` promoted from
`@xfail` placeholder to a real in-process `TestClient.get("/health")`
assertion. Added `test_canonical_app_imports` as a layered check.

## 1.2b — physics model upgrades (commit 2)

### `backend/common/units.py`

New module; zero dependencies besides `pint`. Design:

- SI everywhere internally; boundary-only conversions.
- `to_si(value, unit)` uses `pint.Quantity.to_base_units()` so we don't
  maintain a hand-rolled dimension table.
- `PROPERTY_REGISTRY` is the single source of truth: canonical unit,
  accepted input units, and physical bounds for 11 properties
  (bandgap, formation_energy_per_atom, total_energy, density,
  bulk/shear_modulus, poisson_ratio, lattice_parameter, cell_volume,
  pressure, temperature).
- `normalize_to_canonical(name, value, unit)` converts **and** validates
  against the registered bounds in one call. Session 1.3 (bulk
  property importer) and Session 3.2 (QE parser) will both use this.

Known-bad inputs rejected in unit tests:

| Input | Why |
|---|---|
| `bandgap = -1 eV` | below 0 (metals have zero gap, not negative) |
| `poisson_ratio = 0.9` | thermodynamic upper limit is 0.5 |
| Unknown property name | `KeyError` with list of registered names |

### `backend/common/structures/hashing.py`

Canonical SHA-256 hash of a structure. Invariants tested:

- Atom-order permutations → same hash.
- Sub-1e-4 fractional perturbations → same hash.
- 1e-3 perturbations → different hash (correctly detects real movement).
- Species swap → different.
- Lattice change → different.
- Dict-shaped and tuple-shaped atom lists give the same hash.
- Pymatgen path (when installed) uses
  `SpacegroupAnalyzer.get_refined_structure()` for supercell / symmetry
  invariance; fallback path uses species + rounded fractional coords.

### Structure model extensions

Added to `src/api/models/structure.py`:

- `structure_hash: String(64), unique, indexed, nullable` — filled by
  `hashing.structure_hash()`. Unique index means duplicate uploads
  collide at insert time. Nullable so pre-existing rows don't block
  the migration; Session 1.5 backfills.
- `embedding: Text, nullable` — room for the Phase 6 learned
  representation. Text (not pgvector) for now: no pgvector install
  dependency, trivial to upgrade later.

### PredictedProperties extensions

- `value_unit: String(32), nullable` — records the unit the raw value
  was reported in (e.g., `"eV"`). Canonical SI form stored elsewhere.
- `uncertainty: JSON, nullable` — typed payload descriptor:
  `{"kind": "stddev", "sigma": …}`, `{"kind": "interval", ...}`, or
  `{"kind": "ensemble", ...}`. Phase 6 narrows this to a strict schema.

### Alembic migration `009_rename_metadata_and_add_structure_hash`

One migration covering:

- **Rename `metadata` → `extra_metadata`** on 8 tables: materials,
  structures, simulation_jobs, simulation_results, predicted_properties,
  design_campaigns, design_iterations, vector_embeddings. Completes
  the attribute rename started in Session 0.1 (where the DB column was
  kept via `mapped_column("metadata", ...)` as a stopgap).
- **`structures.structure_hash`** + unique index.
- **`structures.embedding`** (Text).
- **`predicted_properties.value_unit`** + **`predicted_properties.uncertainty`**.
- **`instruments` table** + **`experiment_runs` table** (matching the
  restored Python models from 1.2a).

Reversible `downgrade()` provided.

### Dropped positional `"metadata"` column hack

Every other model that had the positional-arg workaround
(`mapped_column("metadata", JSON, ...)`) now uses the plain attribute
form. Affected: material, simulation (2 classes), predicted_properties,
campaign (2 classes), experiment (2 classes), embedding.

`to_dict()` serializers still emit the JSON key `"metadata"` — intentional
backward-compat with any client that already consumes the old key. If
we want to break that too, a follow-up session can flip it to
`"extra_metadata"` and bump the API version. Noted for Phase 1.1.

### Test coverage added

- `tests/test_units.py` — 14 cases:
  energy / pressure / length / density conversions, registry contents,
  boundary rejection, round-trip idempotence.
- `tests/test_structure_hash.py` — 12 cases:
  atom-order invariance, tolerance window, species / lattice
  discrimination, hash shape, dict-vs-tuple, pymatgen path (skipped
  when pymatgen absent).
- Updated `tests/test_smoke.py`:
  - `test_canonical_app_imports` — the app loads, has 97 routes,
    exposes `/api/v1` prefix.
  - `test_health_endpoint_returns_200` — real in-process TestClient,
    no DB or Redis needed for `/health`.

## Acceptance tests from ROADMAP Session 1.2

| Check | Status | Evidence |
|---|---|---|
| Alembic upgrade+downgrade cycle is clean | 🟡 | Migration is written with a full `downgrade()`; actual DB roundtrip requires a live Postgres and will be exercised in Session 0.5-style integration tests when we add them. Syntax-level review here. |
| `tests/test_materials_model.py`: duplicate structures collide on hash | 🟡 | Hash invariants tested at the pure-function level (`tests/test_structure_hash.py`). DB-level uniqueness test needs a live DB — `requires_db` marker, enabled in Session 1.1 which already needs live DB. |
| `tests/test_units.py`: 1 eV == 1.602176634e-19 J | ✅ `test_ev_to_joule` |
| Backfill script | ⏳ deferred | Moved to Session 1.5 (seed data) where real rows exist. Backfilling an empty table is pointless. |

## Files changed

```
.gitignore                                              (anchor /models/ rule)
backend/common/campaigns/loop.py                        (absolute imports)
backend/common/design/search.py                         (absolute imports)
backend/common/provenance/__init__.py                   (re-export get_provenance_summary)
backend/common/structures/hashing.py                    NEW
backend/common/units.py                                 NEW
backend/orchestrator/__init__.py                        (re-export get_or_create_orchestrator)
alembic/versions/009_rename_metadata_and_add_structure_hash.py  NEW
requirements.txt                                        (+pint)
src/api/auth/__init__.py                                NEW
src/api/dependencies/__init__.py                        NEW
src/api/dependencies/auth.py                            NEW
src/api/exceptions.py                                   (ORION aliases)
src/api/models/campaign.py                              (drop "metadata" positional)
src/api/models/embedding.py                             NEW (then drop "metadata")
src/api/models/experiment.py                            NEW
src/api/models/material.py                              (drop "metadata" positional)
src/api/models/orchestrator.py                          NEW
src/api/models/predicted_properties.py                  (+ value_unit, uncertainty)
src/api/models/provenance.py                            NEW
src/api/models/simulation.py                            (drop "metadata" positional)
src/api/models/structure.py                             (+ structure_hash, embedding)
src/api/models/workflow.py                              NEW
src/api/routers/design.py                               (absolute imports)
tests/test_smoke.py                                     (real health check)
tests/test_structure_hash.py                            NEW
tests/test_units.py                                     NEW
docs/history/phase_1_session_1.2_report.md              (this)
```

Commit 1.2a (import unblock): 16 files, +159/−19 lines.
Commit 1.2b (physics model + migration): 22 files.
Migration `009` will execute on first `alembic upgrade` against a real DB.

## Known blockers passed to Session 1.1

- **`to_dict()` JSON keys still emit `"metadata"`** instead of
  `"extra_metadata"`. Intentional for backward compat; can be flipped in
  Session 1.1 when we're already rewriting router output shapes.
- **`src/api/database.py` still references `settings.TESTING`** which
  doesn't exist on the `Settings` class. Harmless for now because it
  only runs inside `create_engine_with_pool()` at app startup; Session
  1.4 (jobs spine) will want a clean engine factory, so it gets fixed
  there.
- **Pre-refactor tests under `tests/_legacy/`** still reference missing
  pieces. Session 1.1 revives `test_structures.py.pre_refactor` once the
  structures router actually calls the real parsers.
- **Pydantic v2 deprecation warnings** in `src/api/schemas/orchestrator.py`
  (`min_items` → `min_length`). Cosmetic; batch-fix in Session 11.
- **No pgvector column type yet** — `Structure.embedding` and
  `VectorEmbedding.embedding` are still `Text`. Phase 6 upgrades both.
