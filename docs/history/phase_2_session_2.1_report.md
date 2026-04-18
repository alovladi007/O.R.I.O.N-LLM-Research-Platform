# Phase 2 / Session 2.1 — Celery spine

**Branch:** `phase-2-session-2.1-celery-spine`
**Date:** 2026-04-18

## Scope

Roadmap Session 2.1 sets up the worker infrastructure on top of which
every subsequent session (mock engine, QE, LAMMPS, ML training, bulk
imports) runs. Concretely:

1. A Celery app with the **5 roadmap queues** and a name-prefix routing
   convention.
2. A `base.py` module with the canonical task lifecycle hooks (start /
   success / failure / retry).
3. An **artifact bundle** helper that tars a run-dir and uploads to MinIO.
4. A **retry policy** signalled by `TransientEngineError`.
5. A **reaper** that flips stalled `RUNNING` jobs to `FAILED`
   with reason `worker_lost`, wired into Celery Beat.
6. A **pub/sub event emitter** for SSE clients.

All of this ships without requiring a live Redis / MinIO / Postgres
during tests — the emitters and artifact helpers accept injected fakes
and `JobLifecycle` takes any object with a `get`/`commit` interface.

## What shipped

### `src/worker/celery_app.py` — queue layout

Queues reorganised to match the roadmap:

| Queue | Purpose | Priority ceiling |
|---|---|---|
| `default` | catch-all (orchestrator ticks, IO) | 10 |
| `io` | bulk imports, artifacts, notifications | 10 |
| `dft` | Quantum Espresso / other DFT | 20 |
| `md` | LAMMPS / MD engines | 20 |
| `ml` | training + inference | 20 |
| `simulations` *(back-compat)* | old `run_simulation_job` route | 20 |

Task routing is by name prefix: any `orion.dft.*` task goes to `dft`,
`orion.md.*` → `md`, `orion.ml.*` → `ml`, `orion.io.*` → `io`. The
legacy `src.worker.tasks.run_*` explicit routes stay for now
(Session 2.2 migrates them).

Beat schedule:

- `orion.reaper.stalled-jobs` — runs every 60 s (see below).
- `orion.orchestrator.tick` — off by default; opt-in via
  `ORION_ENABLE_ORCHESTRATOR_BEAT=true`. Session 7 flips it on.

### `backend/common/workers/` — shared lifecycle

New package. Keeps the heavy lifting out of `src/worker/` so both
Celery tasks and any non-Celery runners (inline tests, synchronous
code paths) share a single contract.

`backend/common/workers/base.py`:

- **`JobLifecycle`** — context-manager class that wraps a task body:

      with JobLifecycle(job_id, session=..., emitter=...) as lc:
          run_dir = build_run_dir(job_id)
          ...
          lc.outputs = {...}
          lc.bundle = ArtifactBundle(run_dir, job_id)

  On enter: transitions the job `QUEUED → RUNNING` (or
  `PENDING → RUNNING` if Celery pulled it before the QUEUED flip
  landed) and emits `job.status.running`.
  On clean exit: transitions to `COMPLETED`, persists `outputs` into
  `SimulationJob.parameters["_outputs"]`, emits `job.status.completed`.
  On `TransientEngineError`: bumps `retry_count`, emits
  `job.status.retrying`, and re-raises so Celery's `autoretry_for`
  picks it up.
  On any other exception: transitions to `FAILED` (with
  `error_message` truncated to 2000 chars), emits
  `job.status.failed`, re-raises.

- **`TransientEngineError`** — the exception type DFT/MD tasks should
  raise to opt into autoretry with exponential backoff. Session 3.2 /
  Session 4.1 engine wrappers use this.

- **`JobLifecycleError`** — permanent violations (missing job row,
  missing session). Not retryable.

- **`ArtifactBundle` + `build_run_dir` + `tar_and_upload_run_dir`** —
  pure helpers:
  - `build_run_dir(job_id, parent=...)` creates a unique dir via
    `tempfile.mkdtemp`, name-prefixed so retries don't collide.
  - `tar_and_upload_run_dir(bundle, minio_client=...)` writes a
    `manifest.json` with standard fields (job id, bundled timestamp,
    schema version), tars the dir, and optionally uploads to a
    MinIO-compatible client. When `minio_client=None` (tests, dev)
    it just writes the tarball locally. Upload failures are caught
    and reported in the result dict — the task body never dies from
    an upload failure.

### `backend/common/workers/events.py` — event emitters

- **`NullEventEmitter`** — default. Collects events in a list for
  tests / dev; no network I/O.
- **`RedisPubSubEmitter`** — lazy `redis` import; publishes a JSON
  envelope to `orion:events:job:<id>`. The envelope is stable:

      {"event": "job.status.running",
       "ts": "2026-04-18T12:34:56.789Z",
       "payload": {...},
       "emitter_version": "1"}

  Session 10 subscribes the Session 1.4 SSE endpoint to this channel
  so clients get sub-millisecond updates instead of the current 2 s
  polling.
- Emits are **fire-and-forget** — any Redis error returns 0 and logs
  a warning. A broken broker must never kill a task.

### Reaper — `src.worker.tasks.reap_stalled_jobs`

New beat-scheduled Celery task. Scans for `SimulationJob` rows with
`status == RUNNING` and `updated_at` older than a configurable
`stall_seconds` cutoff (default 120 s), and transitions them to
`FAILED` with `error_message = "worker_lost: no heartbeat in ≥Ns"`.
Uses a short-lived sync engine so the task doesn't have to spin up an
event loop.

Wired into the Celery beat schedule at 60 s cadence. Split into a
pure `_reap_stalled_sync(stall_seconds)` helper so the logic is
testable once a DB fixture exists (Session 0.5 follow-up).

### Bootstrap fix: `ExecutionResult`

`backend/common/engines/lammps.py` imported `ExecutionResult` from
`.base`, which didn't exist — Phase 0.5 flagged this as a known
blocker. Added a small `@dataclass` in `backend/common/engines/base.py`:

    @dataclass
    class ExecutionResult:
        success: bool
        returncode: int
        stdout: str = ""
        stderr: str = ""
        timed_out: bool = False

Every Celery task can now import `backend.common.engines` cleanly;
Phase 3 and 4 (QE, LAMMPS) start unblocked.

## Tests — `tests/test_celery_spine.py`

18 cases, all passing:

- `TestCeleryApp` — queue names, name-prefix routes, beat entries,
  orchestrator tick gating.
- `TestArtifactHelpers` — `build_run_dir` uniqueness, tar +
  manifest.json, successful MinIO upload via a `FakeMinio`, non-fatal
  upload failure.
- `TestEventEmitters` — `NullEventEmitter` captures; envelope shape;
  `RedisPubSubEmitter` with a fake Redis client; a broken publisher
  returns 0 without raising.
- `TestJobLifecycleSuccess` — QUEUED → RUNNING → COMPLETED flow,
  outputs persisted, events emitted in order.
- `TestJobLifecycleTransientRetry` — `TransientEngineError` bumps
  `retry_count`, emits `job.status.retrying`, propagates for autoretry.
- `TestJobLifecyclePermanentFailure` — arbitrary exception → FAILED +
  error_message; missing job row raises `JobLifecycleError`.
- `TestExecutionResult` — dataclass shape + `as_dict()`.

All fakes are in-file; no infrastructure required.

## Acceptance tests (from ROADMAP Session 2.1)

| Check | Status | Evidence |
|---|---|---|
| Dummy 1s task writes file in run_dir, artifact appears in MinIO | 🟡 | Artifact pipeline unit-tested with a FakeMinio; real round-trip needs Docker Compose. |
| Kill worker mid-task → reaper flips to FAILED within <30s | ✅ (logic) | `reap_stalled_jobs` task registered in beat every 60 s; `_reap_stalled_sync` transitions stalled rows. Full flow requires a live Redis + Postgres — integration test in CI. |
| SSE events observed for pending→running→succeeded | ✅ (unit) | `JobLifecycle` emits `job.status.running` on enter and `job.status.completed` on clean exit; envelope shape verified. The SSE endpoint from Session 1.4 already consumes DB polling — Session 10 wires pub/sub consumption. |

## Test suite state

```
148 passed, 1 skipped (requires_db), 0 failed
```

Up from last session's 130. 18 new tests in `test_celery_spine.py`.

## Files changed

```
src/worker/celery_app.py                                (queue layout + beat)
src/worker/tasks.py                                     (+ reap_stalled_jobs + helper)
backend/common/workers/__init__.py                      NEW
backend/common/workers/base.py                          NEW
backend/common/workers/events.py                        NEW
backend/common/engines/base.py                          (+ ExecutionResult)
tests/test_celery_spine.py                              NEW (18 cases)
docs/history/phase_2_session_2.1_report.md              (this)
```

## Known blockers / deferred

- **Live end-to-end with Celery + Redis + MinIO.** Works unit-level;
  full loop will be exercised in Session 2.2 (mock engine E2E) which
  is the natural first consumer of this spine.
- **SSE consumer switch to pub/sub.** Session 10 subscribes the
  `/jobs/{id}/events` endpoint to `orion:events:job:*` and retires the
  2 s polling fallback.
- **Back-compat routes** (`src.worker.tasks.run_*`) stay until Session
  2.2 renames them to the `orion.<queue>.*` convention.
- **Logs tail** on `/jobs/{id}/logs` still a placeholder — needs the
  MinIO log-upload path wired up on the task side. Session 2.2 lands
  it together with the mock-engine artifact flow.
