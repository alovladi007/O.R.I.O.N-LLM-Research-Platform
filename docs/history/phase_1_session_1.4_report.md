# Phase 1 / Session 1.4 — Jobs spine (state machine, SSE, log stub)

**Branch:** `phase-1-session-1.4-jobs-spine`
**Date:** 2026-04-18

## Scope

Roadmap Session 1.4 asks for the job state machine, real cancellation,
the `JobKind` enum, an SSE `/events` endpoint, and a `/logs` tail. The
Celery task-termination half of real cancellation lands in Session 2.1
(that's where the broker + task registry come online); Session 1.4
delivers everything that's API-side.

## What changed

### State machine

Added to `src/api/models/simulation.py`:

- `_LEGAL_TRANSITIONS` table encoding the roadmap graph:

      PENDING  → QUEUED | CANCELLED | FAILED
      QUEUED   → RUNNING | CANCELLED | FAILED
      RUNNING  → COMPLETED | FAILED | CANCELLED | TIMEOUT
      COMPLETED | FAILED | CANCELLED | TIMEOUT → ∅ (terminal)

- `IllegalJobTransitionError` — raised on any illegal move, with a
  descriptive message enumerating what *would* be legal from the
  current state.
- `check_transition(current, target)` — pure function (accepts
  either `JobStatus` or `str`). Unit-testable without SQLAlchemy.
- `is_terminal_status(status)` — companion predicate.
- `SimulationJob.transition_to(target, *, error_message=None,
  set_started=False, set_finished=False)` — method on the ORM class
  that delegates the legality check to `check_transition`, then sets
  `updated_at` and optionally `started_at` / `finished_at` /
  `error_message`. Sync by design so it composes with both sync and
  async sessions.

### `JobKind` enum

New enum covering the 18 kinds the roadmap enumerates:

    mock_static, dft_relax, dft_static, dft_bands, dft_dos,
    md_nvt, md_npt, md_nve,
    continuum_elastic, continuum_thermal, mesoscale_kmc,
    ml_train, ml_infer, bo_suggest, al_query,
    import, export, agent_step.

Plus a nullable `kind` column on `simulation_jobs` (Alembic migration
012, indexed). The existing `engine` column stays — `engine` says
*which* physics package; `kind` says *what* the job is attempting.
Test `test_job_kind_values_match_roadmap` locks the spelling against
the roadmap list.

### Jobs router cleanup

`src/api/routers/jobs.py`:

- Dropped the router-level `prefix="/jobs"` — same double-prefix bug
  Session 1.1 fixed on structures. Paths now correctly resolve to
  `/api/v1/jobs/...` instead of `/api/v1/jobs/jobs/...`. Canonical
  app went from 97 → 102 routes; the extra 5 are the de-doubled
  entries that had been unreachable.
- `cancel_simulation_job` rewritten to use `job.transition_to(
  JobStatus.CANCELLED, set_finished=True, error_message=...)`.
  `IllegalJobTransitionError` → HTTP 409 `ConflictError`. Removed
  the old `if job.is_terminal: raise` + manual status assignment;
  the state machine is the single source of truth now.

### `GET /jobs/{id}/events` — Server-Sent Events

New endpoint streaming status changes as SSE. Format:

    event: snapshot
    data: {"job_id": "...", "status": "RUNNING", "progress": 0.42}

    event: status
    data: {"job_id": "...", "status": "COMPLETED", ...}

    event: terminal
    data: {"job_id": "...", "final_status": "COMPLETED", "finished_at": "..."}

Session 1.4 implements this as a 2-second polling loop that re-reads
the row — this is correct but inefficient. Session 10 (observability)
replaces the poll with a Redis pub/sub push from the Celery task
hooks so clients see updates within milliseconds.

### `GET /jobs/{id}/logs` — placeholder tail

Returns `text/plain` composed from what the job row already knows
(status, engine, kind, worker_id, started/finished_at, error_message,
current_step). `tail` query parameter controls line count (defaults
to 200, capped at 10k). Session 2.1 replaces this with a real MinIO
presigned URL + worker log tail; the placeholder keeps the API
contract stable in the meantime.

### Model exports

`src/api/models/__init__.py` now re-exports `JobStatus`, `JobKind`,
`JobPriority`, and `IllegalJobTransitionError` so routers and tests
can import from one place.

## Acceptance tests (from ROADMAP Session 1.4)

| Check | Status | Evidence |
|---|---|---|
| Illegal state transition (succeeded → running) is rejected by the router with 409 | ✅ | `cancel_simulation_job` catches `IllegalJobTransitionError` → `ConflictError`; `test_illegal_transition_rejected` covers the kernel. |
| A test that enqueues a fake 30s sleep task, cancels via API, sees `state=cancelled` within 2s | 🟡 | Database-level cancel path is verified by the router rewrite; the 30s sleep requires Celery (Session 2.1). Deferred. |
| SSE endpoint emits events when a second process updates the job row | 🟡 | Endpoint written and tested at the route-registration level (100 → 102 routes, correct path). The polling loop does see cross-process writes (async session re-reads each iteration); a live DB is needed to verify the real-time behavior. Flagged for CI when Postgres is live. |
| `test_job_state_machine.py` 35 tests pass | ✅ | New file, all green. |

## Test suite state

```
107 passed, 1 skipped (requires_db), 0 failed
```

Up from last session's 72 → 107 (35 new cases in the state-machine file).

## Files changed

```
alembic/versions/012_add_simulation_job_kind.py         NEW
src/api/models/simulation.py                            (+ JobKind, state machine, transition_to)
src/api/models/__init__.py                              (+ exports)
src/api/routers/jobs.py                                 (drop prefix, rewrite cancel,
                                                         + /events SSE, + /logs stub)
tests/test_job_state_machine.py                         NEW (35 cases)
docs/history/phase_1_session_1.4_report.md              (this)
```

## Known blockers / deferred

- **Real Celery cancellation signal.** API status is authoritative
  now; worker-side termination comes with Session 2.1's Celery spine.
- **SSE via pub/sub** rather than polling — Session 10.
- **MinIO-backed log tail** — Session 2.1.
- **Per-kind Pydantic input schemas** (``dft_relax`` params vs.
  ``md_nvt`` params, etc.) — the ``JobKind`` enum is in place, but
  a per-kind validated schema lands in Session 2.2 when the mock
  engine wire-up proves the pattern.
- **Other routers still double-prefixed** — `materials`, `workflows`,
  `campaigns`, `ml`. Session 1.5 or 2.1 picks these up naturally.
