# Phase 2 / Session 2.4 — Workflow DAG executor

**Branch:** `main`
**Date:** 2026-04-18

## Scope

Session 2.4 introduces multi-step DAG workflows on top of the
Celery/SimulationJob spine. A user submits a spec; a tick-based
scheduler drives dispatch as predecessors complete.

Per roadmap: data dependencies via `{"uses": "step.outputs.field"}`,
`foreach` fan-out, aggregate status endpoint, cancel propagation, and
a `workflow.json` manifest mapping `step_id → job_id → artifact_uri`.

## What shipped

### `backend/common/workflows/` — new package

- `spec.py` — pydantic v2 models (`WorkflowSpec`, `StepSpec`,
  `StepStatus`) with `extra="forbid"`. Validators reject duplicate
  step IDs, non-slug IDs, empty foreach lists. `toposort_steps`
  implements Kahn's algorithm with stable ordering (predictable tests)
  and raises `WorkflowCycleError` on cycles. Implicit edges come from
  both explicit `depends_on` and `{"uses": ...}` references.
- `foreach.py` — Cartesian-product expansion. `{T: [100, 200], P: [1,
  2]}` → 4 children with ids like `sweep__100-1`. Downstream steps
  that depended on the un-expanded parent get rewritten to depend on
  every child (fan-in barrier).
- `resolver.py` — walks `inputs` recursively, replaces each
  `{"uses": "step.outputs.dotted.path"}` with the value from the
  step's outputs dict. List indices supported
  (`forces.0.fx`). `ResolutionError` is raised on missing steps /
  paths / out-of-range indices.
- `executor.py` — the pure tick function. Takes a run-like object, a
  `job_lookup` callable, and a `Dispatcher`. Performs: poll running
  children → propagate FAILED to SKIPPED descendants → dispatch newly
  ready steps → roll up aggregate status. Returns a `TickResult` dict
  with `advanced / completed / failed / skipped / aggregate_status`.
- `celery_dispatcher.py` — production `Dispatcher` that creates a
  `SimulationJob` in QUEUED, enqueues via `celery_app.send_task`, and
  returns the new `job_id`. Lazy-creates a built-in
  `WorkflowTemplate` on first use of each kind (same pattern as
  Session 2.2's `/jobs/dispatch`).
- `manifest.py` — `build_workflow_manifest(workflow_run_id, name,
  step_records)` → the canonical `workflow.json` shape with
  `manifest_schema = "workflow_run.v1"`.

### DB models + migration 013

- `src/api/models/workflow_run.py` — `WorkflowRun` + `WorkflowRunStep`
  rows with JSONB `spec`, `outputs`, `manifest`. Foreign key from
  `workflow_run_steps.simulation_job_id → simulation_jobs.id (ON DELETE
  SET NULL)` so orphan steps keep their audit trail if a job row goes.
  Unique constraint on `(workflow_run_id, step_id)`.
- Alembic migration `013_add_workflow_runs` creates both tables with
  the expected indexes.

### Celery tick task + beat

- `orion.workflows.tick` runs `_tick_workflow_runs_sync()`, which
  selects every `WorkflowRun` in `PENDING`/`RUNNING`, advances it via
  the tick function, and writes the manifest on terminal. Errors
  caught at the task level — a bad run shouldn't stop the beat cycle.
- Beat schedule adds `orion.workflows.tick` at 5 s cadence (the
  roadmap's "children dispatched as predecessors complete" is
  effectively a 5 s responsiveness budget; adjust when campaigns hit
  real QE runtimes).
- Task route prefix `orion.workflows.*` → `default` queue.

### API router

`POST /api/v1/workflow-runs` — body `{spec: <WorkflowSpec>}`. Parses
→ `expand_foreach` → `toposort_steps` → persists the run + per-step
rows → kicks a tick immediately (don't wait for beat).

`GET /api/v1/workflow-runs/{id}` — aggregate state with selectin-loaded
steps.

`POST /api/v1/workflow-runs/{id}/cancel` — marks `PENDING`/
`DISPATCHABLE` steps `CANCELLED`, run status `CANCELLED`. Already-running
`SimulationJobs` are left alive; the jobs-router cancel endpoint stops
those individually, per roadmap spec.

`GET /api/v1/workflow-runs/{id}/manifest` — returns the stored manifest
for terminal runs, or builds one on-the-fly for in-flight runs.

## Tests

Two new test files:

**`tests/test_workflow_spec.py` (17 tests)**
- WorkflowSpec parsing: minimal / duplicate IDs / unknown deps /
  unknown `uses` targets / bad slug chars.
- toposort: linear chain / diamond / cycle / implicit `uses` edges.
- foreach: single var / Cartesian product / downstream dep rewrite /
  empty list rejected.
- Resolver: simple ref / nested dotted path with list index / missing
  step / bad ref shape / literals untouched.

**`tests/test_workflow_executor.py` (8 tests)**
- First tick dispatches roots; second tick picks up completed outputs
  and advances consumers with resolved inputs.
- Terminal completion path (run → COMPLETED, finished_at set).
- Failure propagation: FAILED predecessor → descendants SKIPPED → run
  FAILED.
- 8-way fan-out dispatches all children at once (acceptance: "fan-out
  of 8 temperatures produces 8 children that all succeed").
- Cancelled run is a no-op (tick returns without touching anything).
- Manifest aggregation shape matches spec.

**Tally:** 186 → **211 tests passing**, 2 skipped (Postgres + SLURM).

## Acceptance criteria status

| Roadmap item | Status |
|---|---|
| `WorkflowSpec` pydantic schema | ✅ |
| Topological sort → dispatch jobs as predecessors complete | ✅ (tick-based) |
| `POST /api/v1/workflows/` submit + GET aggregate state | ✅ (mounted at `/api/v1/workflow-runs` — `/workflows` is the template CRUD from earlier) |
| Fan-out via `foreach` (e.g. `temperatures=[100,200,300]`) | ✅ |
| Workflow.json manifest | ✅ |
| CIF parse → mock_static → derived property end-to-end | ⏳ Pieces present; live happy-path test needs Postgres + Redis up |
| Cancelling cancels pending children, leaves running intact | ✅ (endpoint; acceptance test needs live infra) |
| Fan-out of 8 kids all succeed | ✅ (dispatcher-level unit test) |

## Follow-ups / deferred

- **Live acceptance** — an end-to-end test that submits a workflow
  spec, observes the tick driving it to completion, and verifies the
  manifest. Requires Postgres + Redis + MinIO + Celery worker
  running. Lives in the same `requires_db + requires_redis` bucket as
  Session 2.2's parallel-mock test.
- **Retries at the workflow level** — the tick marks FAILED as
  terminal. A future session could add per-step `max_retries`.
- **Sub-workflows** — a step whose `kind` is another `WorkflowSpec`.
  Not needed until campaigns grow that complex.
- **`workflows_router` rename** — the existing `/workflows` router is
  `WorkflowTemplate` CRUD. I added `/workflow-runs` for DAGs to avoid
  collision. A later session may fold these into one namespace.
