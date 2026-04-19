# Phase 2 / Session 2.3 — Execution backends (local + SLURM)

**Branch:** `main`
**Date:** 2026-04-18

## Scope

Session 2.3 introduces the execution backend abstraction so that
engine runners (QE, LAMMPS, and future binaries) don't shell out to
`subprocess` directly. One interface — two implementations — picked
per job.

Per roadmap:

```python
class ExecutionBackend(Protocol):
    async def submit(self, cmd, run_dir, resources) -> SubmissionHandle
    async def poll(self, handle) -> JobState
    async def cancel(self, handle) -> None
    async def fetch_artifacts(self, handle, dest_dir) -> Path
```

## What shipped

### `backend/common/execution/` — rewritten

The old `session-27`-era module (ABC with `nodes/cores_per_node/walltime_hours`
and sync-only API) had no external imports. Session 2.3 rewrote it
to match the roadmap.

- `base.py`
  - `JobState` enum — `pending | running | completed | failed | cancelled`
    with `is_terminal` property.
  - `Resources` dataclass — `cpus / gpus / memory_gb / walltime_minutes
    / queue / account / env`. The `queue`/`account` fields are
    SLURM-only but live on the shared type so engine code doesn't
    branch.
  - `SubmissionHandle` — `backend_kind + external_id + run_dir +
    submitted_at_epoch + meta`. Serializable, so a restarted worker
    can still `poll` a handle it didn't submit itself.
  - `TimedOut` exception.
  - `ExecutionBackend` async Protocol.

- `local.py` — `LocalBackend`
  - `asyncio.create_subprocess_exec` with stdout/stderr streaming to
    `run_dir/stdout.txt` and `run_dir/stderr.txt`.
  - `cancel` walks the full process tree via `psutil`, SIGTERM then
    SIGKILL after a `cancel_grace_seconds` window (default 3 s) so
    engine scripts that spawn child MPI processes get cleaned up.
  - `poll` checks the asyncio handle first, falls back to `psutil.Process`
    when the handle isn't in-cache (after a worker restart).
  - `TimedOut` surfaces when `walltime_minutes` has elapsed.
  - `sync_execute(backend, cmd, run_dir, resources)` wraps
    submit → poll-loop → terminal under `asyncio.run` for callers that
    aren't yet async (QE/LAMMPS).

- `slurm.py` — `SlurmBackend`
  - `submit` renders a batch script with `#SBATCH` directives
    (cpus-per-task, gres=gpu:N, mem=NG, time=HH:MM:00, partition,
    account) and invokes `sbatch` locally **or** via SSH when `host`
    is configured.
  - SSH mode uses `asyncssh` — imported lazily, marked as an optional
    dep (not in `requirements.txt`). Throws a clear error if missing.
  - `poll` hits `squeue -j <id> -h -o %T` first; empty stdout means
    the job is no longer queued so we fall through to
    `sacct -j <id> --format=State,ExitCode -n -P` for the terminal
    verdict.
  - `cancel` → `scancel`.
  - `fetch_artifacts` is a no-op in local-submit mode (files are on
    the shared filesystem); remote mode pulls via SFTP over the same
    asyncssh connection.
  - State translators (`_translate_squeue_state`,
    `_translate_sacct_state`) map SLURM states to `JobState`, including
    `CANCELLED by <uid>` → `JobState.CANCELLED`.
  - `SlurmSubmitError` for failed `sbatch` submissions.

- `get_execution_backend(kind)` factory — reads
  `settings.slurm_host / slurm_user / slurm_key_path / slurm_partition`
  for SLURM; falls back to `LocalBackend` on any config error so a
  busted env var doesn't wedge the worker.

### `src/api/config.py`

New env-var-keyed fields:

- `ORION_SLURM_HOST`  → `settings.slurm_host`
- `ORION_SLURM_USER`  → `settings.slurm_user`
- `ORION_SLURM_KEY_PATH` → `settings.slurm_key_path`
- `ORION_SLURM_PARTITION` → `settings.slurm_partition`

All optional. Unset → local sbatch (if on a cluster submit node) or
pure local execution.

### Engine wiring

- `SimulationEngine.execute_command(cmd, run_dir, execution_kind=...)`
  helper on the base class. Uses `get_execution_backend` +
  `sync_execute`, reads `run_dir/stdout.txt` + `run_dir/stderr.txt`
  after the process terminates, and returns the familiar
  `ExecutionResult`. Engines pick `execution_kind` from
  `parameters['execution']['kind']` (default `"local"`).
- `QuantumEspressoEngine._run_real` — replaced the inline
  `subprocess.Popen([qe_executable, "-in", input])` with
  `self.execute_command(...)`. Legacy contract preserved: the captured
  stdout still gets written to `self.output_file` so `_parse_output`
  reads the same file it always did.
- `LAMMPSEngine.execute` — replaced `subprocess.run` with
  `self.execute_command(...)`. The `-log` flag stays so LAMMPS writes
  its own log alongside the backend's `stdout.txt`.

## Tests

`tests/test_execution_backends.py` — 25 tests (24 passing, 1 skipped
on `requires_slurm`):

- **Resources / JobState** — walltime math, terminal predicate.
- **Factory** — `"local"` / `"slurm"` / `None` / unknown.
- **LocalBackend real subprocess** — echo succeeds, `exit 7` fails.
- **LocalBackend cancel** — spawns real `sleep 60`, cancels, uses
  `psutil.Process(pid).is_running()` to verify the PID is gone.
  `poll` returns `CANCELLED` afterwards.
- **LocalBackend walltime** — monkey-patches `submitted_at_epoch` to
  force a past deadline and asserts `TimedOut`.
- **SlurmBackend script rendering** — `#SBATCH` directives match
  resources, explicit queue/account override the backend defaults.
- **SlurmBackend submit parsing** — monkey-patched `_run_slurm_cmd`
  fakes sbatch stdout; asserts external_id extracted.
  `SlurmSubmitError` path covered.
- **SlurmBackend state translators** — squeue + sacct state maps
  (including `CANCELLED by 12345` sacct variant).
- **SlurmBackend poll flow** — squeue reports RUNNING; later sacct
  reports COMPLETED after queue drains.
- **Engine execute_command** — success + failure round-trip through
  LocalBackend, verifying `ExecutionResult.stdout` is populated from
  the on-disk file.

**Live SLURM test** (`test_live_slurm_hostname`) submits `hostname`
against a real cluster via SSH when `ORION_SLURM_HOST` is set; skipped
otherwise.

**Tally:** 162 → **186 tests passing**, 2 skipped (Postgres unreachable,
SLURM not configured).

## Acceptance criteria status

| Roadmap item | Status |
|---|---|
| `ExecutionBackend` Protocol with submit/poll/cancel/fetch_artifacts | ✅ |
| `Resources` dataclass (cpus/gpus/memory_gb/walltime_minutes/queue/account) | ✅ |
| `LocalBackend` (subprocess + psutil) | ✅ |
| `SlurmBackend` (sbatch/squeue/scancel + SSH via asyncssh when remote) | ✅ |
| Engine runners route through the backend (QE + LAMMPS) | ✅ |
| Config: `ORION_SLURM_HOST/USER/KEY_PATH/PARTITION` | ✅ |
| Unit test with fake subprocess for LocalBackend | ✅ (real subprocess used since shell is available; same contract) |
| `@pytest.mark.requires_slurm` live test | ✅ |
| Cancel kills OS process, `ps` verifies | ✅ (`psutil` verifies) |

## Follow-ups / deferred

- `asyncssh` is not in `requirements.txt` — added to the error
  message for remote SLURM mode. When the first remote deployment
  lands, move it to `requirements-hpc.txt` or similar extra.
- QE/LAMMPS `parameters["execution"]["cpus"/"walltime_minutes"]` plumbing
  is wired but not yet exposed through `WorkflowTemplate.default_resources`.
  Session 2.4 or Phase 3 can surface these in the template UI.
- `fetch_artifacts` for remote SLURM is implemented but untested
  (no remote cluster in CI).
