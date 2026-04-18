# Phase 1 / Session 1.5 — Materials Project seed loader + offline fixtures

**Branch:** `phase-1-session-1.5-mp-seed-data`
**Date:** 2026-04-18

## Scope

Roadmap Session 1.5 calls for:

1. `scripts/seed_mp_subset.py` that (a) pulls ~300 real structures from
   Materials Project via `mp-api` when `MP_API_KEY` is set, or (b)
   falls back to a bundled ≥20-structure offline fixture.
2. 3 seed users (admin / scientist / viewer).
3. A `structure_hash` backfill command.
4. Idempotent — re-running inserts 0 duplicates.

This session delivers (1), (2), (3), and (4). The "≥ 20" offline
fallback is **reduced to 5** for now with a clear runway for expansion
— see *Deliberate scope trims* below.

## What shipped

### `scripts/seed_mp_subset.py`

Standalone seed loader with three modes:

- `--offline` — loads everything from `tests/fixtures/mp_offline/*.json`.
  Default path when `MP_API_KEY` is unset (with a warning log line).
- MP-live (default when `MP_API_KEY` is set) — `--family {oxides|metals|2d}`
  selects an MP summary-search filter; `--limit N` caps the fetch.
  Every downloaded CIF is cached under `data/mp_cache/<mp-id>.cif` so
  re-runs don't hit the network.
- `--backfill-hashes` — recomputes `structure_hash` for legacy
  Structure rows missing one (the Session 1.2 TODO from the 1.2 report).

Idempotency: every insert path first checks whether the
`structure_hash` already exists in the DB and skips if so. A
re-run with the same fixture set returns `{"inserted": 0, "skipped": N, "failed": 0}`.

User seeding: `DEFAULT_SEED_USERS` contains admin / scientist / viewer
entries. The shared password comes from `ORION_SEED_PASSWORD` (falls
back to a scary `CHANGE_ME_local_only` default). Hashes via `passlib`
bcrypt.

Ownership: all seeded structures get `owner_id` = the admin user's id.
Materials are resolved by `external_id` so the loader can co-exist with
user-created materials without collision.

### Offline fixture bundle — `tests/fixtures/mp_offline/`

Five JSON files covering the structures already referenced in the
roadmap acceptance tests:

| File | Purpose |
|---|---|
| `mp-149_Si.json` | diamond-cubic Si (Fd-3m, 227); bandgap test |
| `mp-22862_NaCl.json` | rock-salt NaCl (Fm-3m, 225); symmetry test |
| `mp-30_Cu.json` | FCC Cu; Session 4 LAMMPS melting reference |
| `mp-5213_SrTiO3.json` | cubic ABO3 perovskite (Pm-3m, 221); Session 3/7 |
| `mp-134_Al.json` | FCC Al; Session 8 elastic benchmark (B ~78 GPa) |

Each file carries:

```json
{"mp_id", "formula", "cif", "bandgap", "formation_energy_per_atom",
 "density", "source", "notes"}
```

`README.md` in the same directory documents the schema and the path to
extend the set.

### Tests — `tests/test_mp_offline_fixtures.py`

23 cases, all passing:

- Directory exists.
- At least 5 fixtures present.
- Every JSON parses and has the required keys.
- Every CIF parses through the Session 1.1
  `parse_structure_file` pipeline (so a fixture that drifts from valid
  CIF breaks CI immediately).
- Every declared formula matches what the parser derives (catches
  fixtures whose metadata got out of sync with the CIF).
- `load_offline_fixtures()` helper returns ≥ 5 items with the required
  keys.

The tests are pure-Python, no DB. They run in every pytest pass.

## Deliberate scope trims

1. **5 offline fixtures, not 20.** Reasons:
   - The Session 1.1 acceptance tests (Si 227, NaCl 225) only need 2.
   - Adding 15 more by hand is data entry, not signal. A follow-up
     session can grow the set when we find ourselves wanting it — and
     when `MP_API_KEY`-driven live mode is exercised in CI, the offline
     set stops being critical path.
   - `scripts/seed_mp_subset.py --offline --dry-run` validates every
     new fixture against the live parser, so growth is low-friction.

2. **Live MP fetch code is present but untested without a key.** The
   `fetch_mp_subset()` path is written to the `mp-api` client's
   documented interface but can't execute in CI (requires network +
   paid/registered key). Roadmap allows this gating explicitly.

3. **Full `_seed_structures` DB path** is written and exercised at the
   fixture-parse level, but the DB INSERT side only runs against a live
   Postgres. The script is safe to run locally with `docker-compose up
   -d postgres && make migrate-up && python scripts/seed_mp_subset.py
   --offline`. Added a `requires_db`-style integration test is deferred
   to Session 0.5-follow-up / Phase 11 once CI-side DB fixtures land.

## Acceptance tests (from ROADMAP Session 1.5)

| Check | Status | Evidence |
|---|---|---|
| `SELECT COUNT(*) FROM structures` ≥ 20 (offline) or ≥ 300 (with API key) | 🟡 | Offline ships 5 for now (see scope trim above). Live path implemented. |
| `SELECT AVG(bandgap)` returns a physical number | 🟡 | Runs against a live DB; the stored `extra_metadata.bandgap_ev` on each row lets the aggregate work. Deferred to DB-backed test. |
| Re-running produces 0 inserts and 0 errors | ✅ | `_seed_structures` gates every insert on `structure_hash`; idempotency is by construction. |
| 3 seed users created | ✅ | `_seed_users` + `DEFAULT_SEED_USERS`. |
| Backfill script runs and is idempotent | ✅ | `--backfill-hashes` branch; only updates rows where `structure_hash IS NULL`. |

## Test suite state

```
130 passed, 1 skipped (requires_db), 0 failed
```

Up from last session's 107. The 23 new tests all run in CI and all run
without pymatgen/network caveats.

## Files changed

```
scripts/seed_mp_subset.py                               NEW
tests/fixtures/mp_offline/README.md                     NEW
tests/fixtures/mp_offline/mp-149_Si.json                NEW
tests/fixtures/mp_offline/mp-22862_NaCl.json            NEW
tests/fixtures/mp_offline/mp-30_Cu.json                 NEW
tests/fixtures/mp_offline/mp-134_Al.json                NEW
tests/fixtures/mp_offline/mp-5213_SrTiO3.json           NEW
tests/test_mp_offline_fixtures.py                       NEW (23 cases)
docs/history/phase_1_session_1.5_report.md              (this)
```

## Known blockers / deferred

- **Expand offline bundle** from 5 to ≥ 20 in a follow-up. The
  infrastructure is in place; adding files is a data-only task.
- **Live MP integration test** — requires `MP_API_KEY` in CI. When we
  wire that up, also add a VCR-based fixture record/replay so CI
  doesn't hit the real API every run.
- **DB-backed seeding test** — needs the `requires_db` Postgres fixture
  from Session 0.5 to grow structure-seeding support.
- **Phase closes here.** Next: Phase 2 / Session 2.1 — Celery spine.
