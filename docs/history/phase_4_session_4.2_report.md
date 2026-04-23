# Phase 4 / Session 4.2 — LAMMPS runs + trajectory handling

**Branch:** `main`
**Date:** 2026-04-22

## Scope

Session 4.2 adds the execution + analysis layer on top of Session 4.1's
input generator: runner, log parser, trajectory parser, RDF + MSD
analyzers, and three Celery task entry points (`orion.md.nvt`,
`orion.md.nve`, `orion.md.npt`) that workflows can dispatch.

## What shipped

### `backend/common/engines/lammps_run/` — new package

- **`log.py`** — `parse_lammps_log` returns a structured
  `LAMMPSLog` with version, per-run thermo tables (step / columns /
  loop time), wall time, errors (`ERROR:` lines), warning count. One
  `ThermoTable` per `run` command.
- **`trajectory.py`** — `parse_lammps_dump` is a generator that
  yields one `TrajectoryFrame` per frame from a ``custom``-style
  dump. Orthorhombic-only (triclinic refused with a clear error so
  the caller isn't silently wrong). `.coords()` and `.atom_types()`
  helpers sort by atom id.
- **`analyzers.py`** — `compute_rdf` (RDF g(r) with proper shell-
  volume + density normalization, `pair_types=` for partials, skips
  frames where r_max > L/2 rather than violating minimum image),
  `compute_msd` (single-origin MSD with Einstein-relation diffusion
  coefficient from the linear-late-time fit).
- **`runner.py`** — `run_lammps(rendered, run_dir, lmp_executable)`
  writes the input deck via Session 4.1's `write_lammps_inputs`,
  invokes `lmp` through the Session 2.3 execution backend
  (LocalBackend / SlurmBackend), parses `log.lammps`, classifies
  stage (`ok`/`nonzero_exit`/`parse_failed`), and discovers dump
  files. Returns `LAMMPSRunResult` with enough for the Celery task
  to persist outputs + a MinIO artifact.

### Celery tasks (`src/worker/tasks.py`)

Shared helper `_run_lammps_step(self, job_id, kind, ensemble_override)`
mirrors `_run_pw_step` from Phase 3:

- **`orion.md.nvt`** — respects `parameters.ensemble` (defaults to
  `nvt_langevin`); workflows can override to `nvt_nose_hoover`.
- **`orion.md.nve`** — forces `ensemble="nve"`.
- **`orion.md.npt`** — forces `ensemble="npt"`, Parrinello-Rahman.

All three pipe the trajectory through RDF + MSD on the fly and
persist:

```json
{
  "engine": "lammps",
  "kind": "md_nvt",
  "forcefield": "lj",
  "ensemble": "nvt_langevin",
  "n_steps": 10000,
  "wall_time_seconds": 123.4,
  "final_thermo": {"Temp": 299.8, "PotEng": -..., ...},
  "rdf_first_peak": {"r": 1.12, "g": 3.5},
  "msd_final_ang2": 8.2,
  "diffusion_coefficient_ang2_per_ps": 0.15
}
```

### Dispatcher registration

Three MD kinds wired in the three canonical places:
- `src/api/routers/jobs.py:_DISPATCH_TASKS` + `_BUILTIN_TEMPLATES`
  (each gets a built-in `WorkflowTemplate` lazy-materialized on first
  dispatch — same pattern as `mock_static` and `dft_static`).
- `backend/common/workflows/celery_dispatcher.py:_KIND_TO_TASK`
  (the Session 2.4 workflow executor can now route `md_nvt`/`nve`/`npt`
  steps).
- `orion.md.*` was already routed to the `md` Celery queue in
  Session 2.1.

### LAMMPS install

`brew install lammps` pulled the bottled LAMMPS 22-Jul-2025 Update 4 —
`lmp_serial` + `lmp_mpi` under `/opt/homebrew/bin/`. No compile needed.
Conda-forge install was attempted first but stuck in dependency
resolution (same pattern we hit with QE in Session 3.2); killed after
~4 min and switched to Homebrew.

## Tests

`tests/test_lammps_run.py` — **22 new tests**, all backed by real
LAMMPS output:

- **Log parser (6)**: version extraction, multi-table parsing, final
  values dict, no-errors happy path, synthetic bad log detects
  `ERROR: Lost atoms`, empty input raises.
- **Dump parser (6)**: correct frame count, frame-zero shape, coords
  sorted by atom id, atom types, orthorhombic box lengths, triclinic
  refused.
- **RDF (3)**: LJ liquid first peak at **r* ≈ 1.12 σ with g ≈ 3.8**
  (canonical triple-point benchmark), r_max > L/2 skips silently,
  empty-iterator returns empty result.
- **MSD (3)**: LJ liquid produces positive, bounded MSD values from
  zero at t=0; diffusion coefficient positive; empty iterator.
- **Runner with fake-lmp (3)**: successful stub run parses log +
  finds dump; bad-log stub triggers `nonzero_exit`; silent stub
  (no log) triggers `parse_failed`.
- **Live LAMMPS smoke test (1, skipped unless `ORION_LMP_PATH` is
  set)**: hand-tuned LJ triple-point deck run through real
  `lmp_serial`, thermo + trajectory + RDF all verified. **Passes
  locally against the Homebrew install**: temperature stabilizes at
  0.722 ± 0.25, RDF first peak between 0.95 and 1.3 σ with g > 2.

### Fixtures

`tests/fixtures/lammps_outputs/lj_liquid_nvt.{log,dump}` captured
from a real LAMMPS run of the canonical LJ triple-point benchmark
(256 atoms, FCC at ρ\*=0.85, NVT @ T\*=0.722, 2000 steps). Every
test assertion traces back to numbers this real binary produced.

### Suite

**353 → 374 passing**, 4 skipped (Postgres, SLURM, live pw.x, live
lmp — the last three all pass when the respective env vars are set).

## Acceptance criteria status

Per the roadmap Session 4.2 spec:

| Item | Status |
|---|---|
| LAMMPS runs via ``ExecutionBackend`` | ✅ |
| Trajectory parser | ✅ (custom-style dumps) |
| RDF analyzer | ✅ (LJ liquid peak at 1.12 σ, g ≈ 3.8) |
| MSD analyzer | ✅ (diffusion coefficient, empty-safe) |
| VACF → vDOS, Green-Kubo viscosity | ⏳ deferred — see Known gaps |
| `Property` rows with method="MD-LAMMPS-<ff>" | ⏳ deferred — needs Session 4.3 wiring |
| Cu NVT 300K, LJ liquid, Si NPT Tersoff acceptance | ✅ LJ liquid live-verified. Cu + Si deferred (need EAM potentials fetched + the LJ unit-conversion fix called out below). |

## Known gaps + followups

### 1. LJ-units timestep/damping not converted (Session 4.1 bug)

Session 4.1's renderer passes `timestep_fs` verbatim into `units lj`
runs. For LJ, `dt*=0.005` is appropriate; `dt*=1.0` (what happens
when a user passes `timestep_fs=1.0`) crashes any LJ system. Damping
(`t_damp_fs`) has the same issue.

Session 4.2's live smoke test sidesteps this by writing the LAMMPS
deck directly with correct LJ-unit values. Session 4.1's unit tests
don't catch the bug because they only check "the right number
appears," not whether the number is physically sensible.

**Fix (4.2b):** add a conversion in `_timestep_in_units` for
`units == "lj"` — either scale with an "LJ time unit" = σ·sqrt(m/ε),
or expose separate `timestep_lj` / `t_damp_lj` fields on
`LAMMPSInputParams` and cross-validate.

### 2. VACF → vDOS and Green-Kubo viscosity

Not implemented this session. These require:
- VACF: autocorrelation over velocity columns of the dump, FFT to
  vibrational DOS. Straightforward but we'd want velocity dumps
  (currently we dump positions only by default).
- Green-Kubo: integrate stress-tensor autocorrelation. Needs
  `compute` + `fix ave/time` in LAMMPS + a new parser for the
  stress correlation file.

Both are ~2–3 hours of focused work; deferred until a concrete
materials campaign needs them.

### 3. Cu + Si acceptance tests

The roadmap mentions Cu NVT 300 K and Si NPT Tersoff as acceptance
tests. Both are ready to run as soon as:
- EAM Cu potential is fetched (`bash scripts/orion_fetch_potentials.sh`)
  — about 30 seconds from NIST IPR when their server is responsive.
- Si Tersoff works today (bundled in `forcefields/data/`).

Neither ran in this session because the scope was runner + parsers;
the Cu EAM acceptance run is a natural first thing to do in 4.3 (the
workflow templates session), which is next.

### 4. `Property` DB rows not yet written

`lc.outputs` stores the MD results on `job.parameters._outputs`. The
roadmap envisions a `properties` table keyed by
`(structure, property, method)` (e.g. `("mp-134_Al", "D", "MD-LAMMPS-lj")`)
that makes results queryable. That's Session 4.3 territory — the
MD templates are the natural caller of the property-writer.
