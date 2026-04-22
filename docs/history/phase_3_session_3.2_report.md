# Phase 3 / Session 3.2 — Run QE and parse outputs

**Branch:** `main`
**Date:** 2026-04-21

## Scope

Session 3.2 closes the QE execution loop: take a rendered pw.x input
(from Session 3.1), run the binary through the execution backend
(Session 2.3), and parse the resulting output into structured Python
that can feed `{"uses": "step.outputs.energy_ev"}` resolution in the
Session 2.4 workflow executor.

## What shipped

### `backend/common/engines/qe_run/` — new package

Two modules, both pure Python, both defensive against partial output:

- `output.py` — `parse_pw_output(source)` where *source* is either a
  file path or raw text. Extracts:
  - **Energy** — the `!    total energy` marker (final, in Ry) +
    per-iteration SCF history. Reports both Ry and eV
    (`RY_TO_EV = 13.605693...`).
  - **Forces** — per-atom Cartesian components (Ry/Bohr from pw.x,
    converted to eV/Å). Picks the *last* force block so relax runs
    report post-relaxation forces, not pre. Atom indices normalised
    to 0-based.
  - **Stress tensor** — 3×3 in kbar from pw.x → GPa. Derives
    hydrostatic pressure as `-trace/3`.
  - **Convergence** — `ConvergenceStatus` enum: CONVERGED,
    UNCONVERGED, ERRORED (explicit `%%%` banner), UNKNOWN. The enum
    bubbles up to the Celery task + workflow executor.
  - **Counts** — `n_atoms`, `n_species`, `n_electrons`,
    `n_scf_iterations`, `scf_accuracy_ry`, `wall_time_seconds`.
  - **Error messages** — one entry per `Error in routine ...` line,
    so crashed runs still give the user something to act on.
  - Never raises unless the output is truly empty
    (`PWOutputParseError`). Missing sections → `None` / empty list.

- `runner.py` — `run_pw(rendered, run_dir, ...)` orchestrator:
  1. Stages UPFs from the registry's source dir into
     `run_dir/pseudos/` (symlinks would be faster; we copy so the run
     dir is self-contained for archival).
  2. Rewrites `pseudo_dir` in the input text to the local staged
     path so the execution backend container / SLURM worker doesn't
     need access to the original SSSP directory.
  3. Invokes `pw.x -in <input>` via the execution backend (Session
     2.3's `sync_execute`). Resources come from caller's `cpus` +
     `walltime_minutes`.
  4. Parses `run_dir/stdout.txt` regardless of exit code —
     `pw.x` often writes useful partial output before crashing, and
     the parser reports `convergence=ERRORED` for those cases.
  5. Returns a `PWRunResult` with `success`, `stage` (`ok` /
     `unconverged` / `nonzero_exit` / `parse_failed`), `returncode`,
     and the structured `PWOutput`.

### Celery task: `orion.dft.static`

Added to `src/worker/tasks.py` — mirrors the Session 2.2 `mock_static`
pattern:

1. Sync session via `_sync_session_for_worker`.
2. `JobLifecycle` context manager (Session 2.1 handles the job state
   machine + events).
3. New helper `_load_structure_for_qe(session, structure_id)` returns
   `{lattice, species, frac_coords}` in the shape
   `generate_pw_input` expects.
4. Loads `job.parameters` → builds `QEInputParams` (filters to known
   fields so arbitrary user keys don't blow up pydantic).
5. `PseudopotentialRegistry` pulls from `QE_PSEUDO_DIR` env (default:
   `~/orion/pseudos/SSSP_1.3.0_PBE_efficiency/`).
6. `run_pw` with `QE_EXECUTABLE` env (default: `pw.x` on PATH).
7. On failure, `RuntimeError` propagates so `JobLifecycle` marks the
   SimulationJob FAILED with a useful error message. On success,
   `lc.outputs = run_result.output.as_dict()` persists to
   `parameters._outputs` for `{"uses": ...}` resolution.
8. Tarballs `run_dir` (including `pseudos/`, `stdout.txt`,
   `stderr.txt`, `si.in`) → MinIO. Artifact blob on
   `job.extra_metadata.artifact` for `GET /jobs/{id}/artifacts`.

### Dispatcher wiring

`dft_static` kind registered in three places:
- `src/api/routers/jobs.py:_DISPATCH_TASKS` → task name.
- `src/api/routers/jobs.py:_BUILTIN_TEMPLATES` → auto-created
  `WorkflowTemplate` with `engine="qe"`, category "static", defaults
  for `calculation`/`occupations`/`smearing`/`degauss`.
- `backend/common/workflows/celery_dispatcher.py:_KIND_TO_TASK` →
  Session 2.4 workflow executor can now dispatch `dft_static` steps.

Which means a workflow spec like:
```yaml
steps:
  - id: si_scf
    kind: dft_static
    structure_id: <uuid>
  - id: reduce_energy
    kind: mock_static
    inputs:
      baseline: {uses: si_scf.outputs.energy_ev}
```
already dispatches end-to-end via the Session 2.4 tick loop — no new
workflow wiring needed.

## Tests

### Output parser — `tests/test_qe_output_parser.py` (18 tests)

Golden fixtures under `tests/fixtures/pw_outputs/`:
- `si_scf_converged.out` — SCF with 3 iterations, 2-atom forces,
  3×3 stress, wall time, all counts.
- `si_scf_unconverged.out` — 200 iterations, `NOT achieved` marker.
- `error_wrong_pseudo.out` — `%%%%` banner + `Error in routine readpp`.

Covers: energy Ry→eV conversion, SCF history extraction,
convergence-status mapping, force Ry/Bohr→eV/Å conversion and 1-based
→ 0-based indexing, stress kbar→GPa and pressure derivation, counts
parsing, wall-time extraction, `as_dict()` shape (contract for
workflow-resolver `{"uses": ...}` consumers), empty-input raises,
text-vs-path polymorphism, minimal output doesn't raise on missing
sections, errored outputs still report the error_messages list.

### Runner — `tests/test_qe_runner.py` (7 tests + 1 live)

Uses a `_make_fake_pw()` shell stub that emits a golden fixture
instead of running real pw.x. Covers:
- UPFs copied into `run_dir/pseudos/`.
- Input text has the `pseudo_dir` rewritten to the staged path.
- Missing UPF in source dir → `PWRunError`.
- Converged run → `success=True, stage="ok"`.
- Unconverged run (exit 0, SCF failed) → `success=False, stage="unconverged"`.
- Non-zero exit → `success=False, stage="nonzero_exit"`.
- `species_hint` labels parsed force entries.
- **Live smoke test** (skipped unless `ORION_PWX_PATH` is set + SSSP
  installed): generates a 2-atom Si input, runs real `pw.x`,
  asserts energy is in the right order of magnitude and convergence
  reached. This is the roadmap's 3.2 acceptance test.

### Full suite

**236 → 261 passing**, 3 skipped:
- Postgres unreachable (test_bulk_property_import).
- SLURM unreachable (test_execution_backends).
- Live pw.x not installed (test_qe_runner).

## Acceptance criteria status

| Roadmap item | Status |
|---|---|
| Run QE binary on a rendered input | ✅ (shape verified via fake-`pw.x` fixture tests) |
| Parse total energy, forces, stress, convergence | ✅ |
| Unit tests covering golden outputs | ✅ (18 tests, 3 fixtures) |
| Live smoke test: Si SCF end-to-end | ⏳ Test exists + runs when `ORION_PWX_PATH` is set; binary install pending (conda install stuck, Homebrew has no QE formula, building from source) |
| Integrate with workflow DAG | ✅ (dispatcher tables + workflow executor registered `dft_static`) |

## Deferred / known issues

- **Live `pw.x` install.** Three attempts today:
  - `brew install quantum-espresso` → no formula.
  - `conda install -c conda-forge qe -y` → stuck indefinitely in
    "Solving environment" retries against pkgs.anaconda.org; the
    install process eats 99% CPU and never finishes. Anaconda's
    `anaconda-auth` plugin also fails with a pydantic import error.
  - `brew install gcc open-mpi fftw cmake` + source build from
    `~/Downloads/qe-7.5/` is in progress. Exec path is ready; once
    the binary exists, setting `ORION_PWX_PATH=<absolute path to
    pw.x>` unskips the live test.
- **Force species labeling** is done by `species_hint` passed from the
  caller. We could also extract from the input file's
  `ATOMIC_SPECIES` block — added if parser edge cases surface.
- **Non-local force contribution** — parser stops at the
  "The non-local contrib" line, matching the total-force block. A
  future session can extract the decomposition if campaigns need it.
- **Relax / MD output** — pw.x writes multiple SCF blocks per
  ionic step; the parser's `_parse_forces` picks the last block
  (post-relax). Ionic-step-by-ionic-step history is deferred to
  Session 3.3 (relax workflow).
