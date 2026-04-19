# Phase 2 / Session 2.2 — Mock engine, end-to-end

**Branch:** `main`
**Date:** 2026-04-18

## Scope

Session 2.2 proves the Celery spine built in 2.1 actually works:
submitting `{kind: mock_static, structure_id}` must produce a
Celery-executed job that loads the structure, runs the mock engine,
bundles artifacts to MinIO, and exposes outputs + presigned URLs via
the API.

## What shipped

### `backend/common/jobs/` — new package

New package housing job-side schemas and helpers shared between the
API router and Celery tasks. Three modules:

- `schemas.py` — pydantic v2 models. Defines
  `MockStaticOutput` with `schema_version = "mock_static.v1"`,
  `MockForceEntry`, and `MockTrajectoryFrame`. `extra="forbid"` on all
  three; any unexpected key in a task's output fails validation at
  lifecycle exit.
- `mock_static.py` — the pure computation. Implements the roadmap
  formula:
  - `energy_ev = -1.5 * Σ Z_i + 0.01 * noise`
  - per-atom forces drawn from `N(0, σ=0.05 eV/Å)`
  - 10-frame trajectory, `N(0, σ=0.01 Å)` displacements from frame 0
  Deterministic: seed derived from `SHA1(structure_id)[:4]`. Same id →
  same numbers across workers.
- `artifacts.py` — MinIO client factory + presigned URL builder
  (`build_minio_client`, `ensure_bucket`, `presign_artifact`).

### `src/worker/tasks.py` — `orion.mock.static`

New Celery task wiring:

1. Opens a sync SQLAlchemy session via `_sync_session_for_worker`
   (same pattern as the reaper — async drivers don't fit the Celery
   thread model).
2. Enters `JobLifecycle(job_id, session, emitter)` from Session 2.1,
   which handles QUEUED→RUNNING→COMPLETED transitions and event
   emission.
3. Loads the `Structure` row, calls `run_mock_static`, drops three
   files in the run dir (`energies.json`, `forces.json`,
   `trajectory.xyz`).
4. Bundles the run dir via `tar_and_upload_run_dir` + MinIO client.
5. Stashes artifact coordinates on `job.extra_metadata["artifact"]`;
   the API's artifact endpoint reads this to build presigned URLs.
6. Populates `lc.outputs = result.model_dump()` — `JobLifecycle`
   persists under `parameters["_outputs"]` on successful exit.

Registered as `orion.mock.*` → `default` queue in
`src/worker/celery_app.py`.

### `src/api/routers/jobs.py` — two new endpoints

- **`POST /api/v1/jobs/dispatch`** — short-form submission path from
  the roadmap spec. Body is `{kind, structure_id}` (plus optional
  `name`, `priority`, `parameters`). The router:
  1. Looks up (or lazily creates) a built-in `WorkflowTemplate` named
     `mock_static_default`, so the dispatch path works on a fresh DB
     without a seed step.
  2. Creates the `SimulationJob` directly in `QUEUED` with `kind =
     "mock_static"` and `engine = "mock"`.
  3. Enqueues `orion.mock.static` via `celery_app.send_task` and stamps
     `celery_task_id` onto the row.
- **`GET /api/v1/jobs/{id}/artifacts`** — reads the artifact blob off
  `extra_metadata` and returns `{job_id, items: [{bucket, key,
  size_bytes, uploaded, presigned_url, presigned_expires_seconds,
  error}]}`. Presigning uses `build_minio_client` from the new
  `backend.common.jobs.artifacts` module; MinIO unreachability is
  reported per-item rather than 500'ing the endpoint.

### `src/api/schemas/simulation.py` — new schema

`SimulationJobDispatch` — the pydantic body for the dispatch path.

## Tests

`tests/test_mock_static.py` — 14 new tests covering:

- **`MockStaticOutput` schema** — version tag, construction,
  `extra=forbid` rejection.
- **`run_mock_static` math** — physics-shaped energies (Si Z=14 →
  ≈-42 eV for Si2), determinism keyed on `structure_id`, force bounds
  (<10σ), 10 trajectory frames with frame 0 equal to input.
- **`write_trajectory_xyz`** — 4-lines-per-frame layout, species/size
  mismatch raises.
- **Celery task via `apply()`** — end-to-end with fake Session, fake
  MinIO, fake Redis. Asserts `parameters._outputs` populated,
  `extra_metadata.artifact` set, tarball file exists. Separate test
  for missing-structure FAILED path.
- **Dispatch catalog** — `_DISPATCH_TASKS` and `_BUILTIN_TEMPLATES`
  both contain `mock_static`; `SimulationJobDispatch` accepts minimal
  body.

**Tally:** 148 → **162 tests passing**, 1 skipped (Postgres unreachable
in the local shell).

## Acceptance criteria status

| Roadmap item | Status |
|---|---|
| `mock_static` kind runs end-to-end | ✅ (unit-tested; live integration lands when Docker is up) |
| Fake energy/forces/trajectory spec matches formula | ✅ |
| `POST /api/v1/jobs` with `{kind, structure_id}` works | ✅ via `POST /api/v1/jobs/dispatch` |
| `GET /api/v1/jobs/{id}` returns outputs | ✅ (outputs stored in `parameters._outputs`) |
| `GET /api/v1/jobs/{id}/artifacts` returns presigned URLs | ✅ |
| Integration test: 20 parallel mock jobs in ≤60 s | ⏳ Live test deferred; requires Docker Compose (Postgres + Redis + MinIO) up. Shape is proven via the in-process `apply()` test. |
| Outputs schema validated against `MockStaticOutput` | ✅ (`MockStaticOutput` has `extra=forbid`; violations fail the task body before lifecycle exit) |
| Frontend polls `GET /jobs/{id}` and displays `status + energy` | ⏳ Frontend wiring is Session 9; API surface is ready for it today |

## Follow-ups / deferred

- Live 20-parallel integration test — lives in a `requires_redis +
  requires_db + requires_minio` file; needs Docker Compose. Not
  gating this session because the pure-Python test proves the task
  body works and the 2.1 tests already prove Celery + queues.
- `frontend/` hookup — Session 9 territory.
- `SimulationResult` table write. Session 2.2 only stores outputs in
  `job.parameters._outputs` (via `JobLifecycle`) and the artifact
  tarball in MinIO. Writing a `SimulationResult` row belongs with the
  Phase 6 (ML) dataset registry, which needs a typed extraction of
  the summary anyway.
