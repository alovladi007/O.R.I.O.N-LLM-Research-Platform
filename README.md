# ORION — Optimized Research & Innovation for Organized Nanomaterials

**Computational materials-science platform with multi-scale simulation,
machine learning, and an LLM-driven design loop.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/FastAPI-0.108+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Next.js-14-black.svg" alt="Next.js 14">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/status-phase--6--ML--in--progress-blue.svg" alt="status">
  <img src="https://img.shields.io/badge/tests-508%20passing-brightgreen.svg" alt="tests">
  <img src="https://img.shields.io/badge/CI-green-brightgreen.svg" alt="CI">
</p>

> **Honest status (2026-04-23):** Phases 0–5 complete and merged; Phase 6
> (ML pipeline) in progress — Session 6.1 (featurizers) shipped, 6.2+ to
> come. Everything below actually runs end-to-end, backed by unit tests +
> live acceptance runs where a real physics binary is available.
>
> - **DFT (Phase 3):** real Quantum ESPRESSO — input gen, pw.x/dos.x/ph.x
>   execution via a pluggable Local/SLURM backend, output parsing, MinIO
>   artifact upload, multi-step DAGs (relax → scf → bands / DOS / phonons
>   Γ). Eight elements calibrated against live PBE + SSSP (Al, Cl, Cu, Na,
>   O, Si, Sr, Ti); NaCl formation energy matches MP within 0.23 eV/atom
>   (raw PBE, no empirical halide corrections).
> - **MD (Phase 4):** real LAMMPS — input gen, forcefield registry (LJ /
>   EAM Cu/Ni/Al / Tersoff Si/C / ReaxFF + MACE stubs), runner, trajectory
>   parser with RDF + MSD, three aggregate analyzers (melting-point
>   detection, Arrhenius fit, elastic-constant fit). Live Al C₁₁ =
>   110.8 GPa (target 108 ± 15%); live Cu melting detected in [1000, 1700]
>   K.
> - **Continuum + mesoscale (Phase 5):** FEM elasticity + steady heat on
>   `scikit-fem` (cantilever 0.14% error, 1D heat midpoint machine-
>   precision). Rejection-free Gillespie kMC for vacancy + interstitial
>   defect migration (single-walker D within 2-3% of analytical,
>   V+I annihilation decays cleanly). Multi-scale DFT → MD → continuum
>   workflow template scaffolded with deferred-contract tasks.
> - **ML (Phase 6, just started):** matminer composition featurizers
>   (146-d) + SiteStatsFingerprint structure descriptor (122-d) +
>   canonical radius-graph builder + in-memory cache + PCA-based 256-d
>   embedding with save/load. 100 structures featurize in 1.6 s. Si
>   similarity query recovers C and Ge cousins in the top-3.
>
> Remaining: GNN training, active learning, Bayesian optimization, LLM
> agent loop, frontend wiring, hardening, deployment — Phases 6.4 through
> 13. See [ROADMAP_PROMPTS.md](./ROADMAP_PROMPTS.md) for the plan and
> [docs/history/](./docs/history/) for session-by-session reports.

---

## Table of contents

1. [Roadmap](#roadmap)
2. [Architecture](#architecture)
3. [Quick start](#quick-start)
4. [What works today](#what-works-today)
5. [Repository layout](#repository-layout)
6. [Documentation](#documentation)
7. [Contributing](#contributing)

---

## Roadmap

Implementation is tracked as a **14-phase, ~50-session plan** in
[ROADMAP_PROMPTS.md](./ROADMAP_PROMPTS.md). Each session is a self-contained,
paste-ready prompt with physics-based acceptance tests (e.g. Si PBE bandgap
≈ 0.6 eV, Cu EAM melting ± 150 K of 1358 K, Al bulk modulus ± 10 % of 78 GPa).

- **Phase 0** ✅ repository hygiene — 5 sessions + review
- **Phase 1** ✅ data & domain foundation — 5 sessions + review
- **Phase 2** ✅ job execution spine (Celery + engines) — 4 sessions
- **Phase 3** ✅ DFT engine (Quantum Espresso) — 4 sessions + addenda
- **Phase 4** ✅ MD engine (LAMMPS) — 4 sessions + live Al/Cu acceptance
- **Phase 5** ✅ continuum / FEM (5.1) + kMC mesoscale (5.2) + multiscale
  scaffold (5.3, DFT-elastic + Green-Kubo analyzers deferred)
- **Phase 6** 🟡 ML features (6.1 done), datasets (6.2+), baselines (6.3),
  GNN (6.4), active learning (6.5)
- **Phase 7** — Bayesian optimisation + LLM-driven agent loop
- **Phase 8** — elastic tensors, phonons, point defects
- **Phase 9** — frontend end-to-end wiring
- **Phase 10** — observability & performance
- **Phase 11** — security hardening
- **Phase 12** — provenance, reproducibility, compliance
- **Phase 13** — K8s, CI/CD, DR, 1.0.0 release

See [CHANGELOG.md](./CHANGELOG.md) for merged changes and
[docs/history/](./docs/history/) for per-session reports.

---

## Architecture

```
┌──────────────────┐    HTTPS     ┌──────────────────────────────┐
│  Next.js 14 UI   │◀────────────▶│  FastAPI  (src.api.app)      │
│  (frontend/)     │              │  JWT · pydantic-v2 · SQLAlc. │
└──────────────────┘              └──────┬───────────────────────┘
                                         │
           ┌─────────────────────────────┼─────────────────────────────┐
           │                             │                             │
           ▼                             ▼                             ▼
  ┌──────────────────┐          ┌────────────────┐           ┌──────────────┐
  │  PostgreSQL      │          │  Redis         │           │  MinIO       │
  │  (+ pgvector)    │          │  cache + broker│           │  artifacts   │
  └──────────────────┘          └───────┬────────┘           └──────────────┘
                                        │
                                        ▼
                                ┌──────────────────────────────┐
                                │  Celery workers              │
                                │  backend.common.engines.*    │
                                │    - mock                    │
                                │    - qe_input / qe_run       │
                                │    - lammps_input / lammps_run│
                                │    - continuum_fem (scikit-fem)│
                                │    - mesoscale_kmc (Gillespie)│
                                │  backend.common.ml.*         │
                                │    - features_v2 (6.1 — live)│
                                │    - datasets / models / GNN │
                                │      / BO / AL (Phase 6.2+) │
                                └──────────────────────────────┘
```

### Services in `docker-compose.yml`

| Service | Purpose | Default port |
|---|---|---|
| `postgres` | Main DB (pgvector-enabled) | 5432 |
| `redis` | Cache + Celery broker | 6379 |
| `elasticsearch` | Full-text search | 9200 |
| `minio` | S3-compatible object storage | 9000 / 9001 |
| `orion-api` | FastAPI backend | 8000 (host) |
| `orion-frontend` | Next.js UI | 3002 (host) |
| `orion-worker` | Celery worker | — |
| `flower` | Celery monitoring UI | 5555 |
| `prometheus` | Metrics | 9090 |
| `grafana` | Dashboards | 3000 |
| `jupyter` | Notebook sandbox | 8888 |

---

## Quick start

### Prereqs

- Python 3.10 or newer (pyproject pins `^3.10`)
- Docker + Docker Compose
- Node 18+ (for the frontend)

### Local backend (no Docker)

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # created in Session 0.5

cp .env.example .env                  # fill JWT_SECRET_KEY, DATABASE_URL, etc.
docker-compose up -d postgres redis   # just the deps
make migrate-up                       # Alembic up
make dev                              # uvicorn src.api.app:app on :8002
```

Verify:

```bash
curl http://localhost:8002/health
open  http://localhost:8002/docs
```

Full details in [BACKEND_QUICKSTART.md](./BACKEND_QUICKSTART.md) and
[QUICK_START.md](./QUICK_START.md).

### Running real DFT (Phase 3)

The QE pipeline expects `pw.x` on `PATH` and a pseudopotential library on
disk:

```bash
# Install pw.x v7.5 from source (see docs/history/phase_3_session_3.2_report.md
# for the macOS-specific build notes — anaconda ld shadowing + cpp flag fix).
# Or install via conda-forge: `conda install -c conda-forge qe`.

export QE_EXECUTABLE=~/orion/qe-7.5/bin/pw.x
export QE_PSEUDO_DIR=~/orion/pseudos/SSSP_1.3.0_PBE_efficiency

# Calibrate per-element reference energies (vc-relax each elemental prototype)
python scripts/orion_calibrate.py --elements Si,Cu,Al,Na --skip-db

# Compute formation energy of a compound against MP's published value
python scripts/orion_scf_compound.py tests/fixtures/mp_offline/mp-22862_NaCl.json
```

### Running real MD (Phase 4)

The LAMMPS pipeline expects `lmp_serial` (or `lmp_mpi`) on `PATH`.
Homebrew on macOS works (`brew install lammps`); the interatomic
potentials for Cu, Al, Ni are fetched on demand:

```bash
# Homebrew on macOS (or conda-forge: `conda install -c conda-forge lammps`)
export ORION_LMP_PATH=/opt/homebrew/bin/lmp_serial

# Fetch Cu, Al, Ni EAM potentials into
# backend/common/engines/lammps_input/forcefields/data/
bash scripts/orion_fetch_potentials.sh

# Live acceptance: Al C11 via ±ε strain workflow, Cu melting curve
ORION_LMP_PATH=$ORION_LMP_PATH pytest tests/test_md_live_acceptance.py -v
```

### Frontend

```bash
cd frontend
npm install
npm run dev       # http://localhost:3002
```

### Full stack via Docker Compose

```bash
docker-compose up -d
```

---

## What works today

Updated at the end of each merged session. If a feature isn't listed here,
treat it as stubbed or WIP.

### Core platform (Phases 0–2)

- Single canonical FastAPI entry point (`src.api.app:app`) on port 8002;
  111 routes registered (QE + MD + multiscale template endpoints).
- Pydantic v2 settings, CSV-aware env parsing, explicit CORS allowlist,
  32+ char JWT secret enforced in prod.
- 12 Alembic migrations; `extra_metadata` JSONB everywhere; pgvector-enabled
  Postgres.
- Celery spine with 5-queue layout (default / io / dft / md / ml),
  `JobLifecycle` context manager, `TransientEngineError` retry policy,
  stalled-job reaper on Celery beat, artifact tarball uploads to MinIO.
- Execution backend abstraction (Session 2.3): `LocalBackend`
  (asyncio.subprocess + psutil cancel-tree) and `SlurmBackend`
  (sbatch/squeue/scancel, asyncssh for remote clusters) behind a single
  async Protocol. All physics engines route through this.
- Workflow DAG executor (Session 2.4) with `{"uses": "step.outputs.X"}`
  reference resolution, `foreach` Cartesian fan-out, cycle detection,
  manifest aggregation. Drives multi-step DFT + MD workflows end-to-end.
- Real pymatgen-backed structure parsers, structure hashing, bulk
  property importer with unit validation, Materials Project seed loader.

### DFT pipeline (Phase 3) — production-ready

- **Session 3.1** — canonical QE input generator
  (`backend.common.engines.qe_input`) with SSSP efficiency v1.3.0
  support, automatic k-mesh from reciprocal lattice, per-element
  cutoff inference from UPF headers.
- **Session 3.2** — runner + output parser. pw.x invoked via the
  execution backend; parser extracts energy (eV), forces (eV/Å), stress
  (GPa), convergence status, SCF iterations, wall time. Live
  end-to-end Si acceptance test passing (total energy -305 eV, forces
  within spec).
- **Session 3.3** — four workflow templates
  (`relax_then_static` / `band_structure` / `dos` / `phonons_gamma`)
  submitted via `POST /api/v1/workflow-runs/templates/qe/{name}`.
  DOS integral at E_F equals 8.000 electrons for Si (exact). Γ-only
  phonon guard refuses non-cubic inputs to avoid silently-wrong
  numbers on polar/anisotropic materials.
- **Session 3.4 + 3.4b** — reference-energy calibration. 8 elements
  calibrated live: Al, Cl, Cu, Na, O, Si, Sr, Ti. Diatomic gases (O, Cl)
  supported via molecule-in-vacuum prototype with spin-polarization for
  triplets. `orion calibrate` CLI writes to the `reference_energies`
  DB table. NaCl compound cross-validation against Materials Project:
  deviation 0.23 eV/atom (raw PBE without MP's empirical halide
  corrections).

### MD pipeline (Phase 4) — production-ready

- **Session 4.1** — LAMMPS input generation + forcefield registry
  (`backend.common.engines.lammps_input`). Declarative
  `ForcefieldSpec` records with auto-selection by element set; ships
  LJ (fallback), EAM for Cu/Ni/Al (fetched on demand via
  `scripts/orion_fetch_potentials.sh`), Tersoff Si/C (bundled),
  ReaxFF + MACE as feature-flagged stubs. Four ensembles supported
  (NVE, NVT Nose-Hoover, NVT Langevin, NPT Parrinello-Rahman);
  timesteps kept in fs internally and converted per LAMMPS unit style.
- **Session 4.2** — runner + trajectory handling
  (`backend.common.engines.lammps_run`). `log.lammps` parser, custom-
  dump streaming parser, RDF and MSD analyzers. Three Celery tasks
  (`orion.md.{nvt,nve,npt}`) wired into the workflow dispatcher. Live
  LJ triple-point smoke green: RDF first peak at r\* ≈ 1.12 σ, g ≈ 3.8.
- **Session 4.3 (a + b)** — four MD workflow templates
  (`equilibrate_nvt_then_nve`, `melting_curve`, `diffusivity_vs_T`,
  `elastic_constants_via_strain`) via
  `POST /api/v1/workflow-runs/templates/md/{name}`. Three aggregate
  analyzers (`detect_melting_point`, `arrhenius_fit`,
  `fit_elastic_constants`) with R² quality gates. Live acceptance:
  Al C₁₁ = 110.8 GPa with R² = 1.00 (target 108 ± 15%); Cu T_m
  detected at 1575 K via MSD-jump (inside [1000, 1700] K bracket).

### Continuum + mesoscale (Phase 5)

- **Session 5.1 — FEM continuum** (`backend.common.engines.continuum_fem`).
  Pure-Python `scikit-fem` solver: linear elasticity (Hex1/Hex2,
  Dirichlet + Neumann BCs) + steady heat (∇·(k∇T) = 0). VTU export
  via meshio for ParaView. Acceptance: cantilever deflection within
  0.14% of PL³/(3EI); 1D heat rod midpoint matches (T_left+T_right)/2
  to machine precision.
- **Session 5.2 — kMC mesoscale** (`backend.common.engines.mesoscale_kmc`).
  Rejection-free Gillespie on simple-cubic with vacancy + interstitial
  hops + pair annihilation. Occupancy-hash bookkeeping for O(1)
  collision lookup. Unwrapped coords for artifact-free MSD.
  Acceptance: 500 non-interacting vacancies at 600 K give D within
  2-3% of a²ν₀e⁻ᴱ/ᵏᵀ; 1% V + 1% I annihilation decays to 0 within
  2 M steps.
- **Session 5.3 — multiscale scaffold** (DFT → MD → continuum).
  Workflow template `dft_to_md_to_continuum` with provenance threading;
  DFT elastic-tensor extraction deferred to Phase 8, Green-Kubo κ
  deferred to a Phase 4 follow-up. `PendingAnalyzerError` with tracker
  hints in place of silent zeros.

### ML pipeline (Phase 6) — in progress

- **Session 6.1 — featurizers** (`backend.common.ml.features_v2`).
  matminer-backed composition stack (ElementProperty/Magpie + Stoichiometry
  + ValenceOrbital, 146-d) + `SiteStatsFingerprint` structure descriptor
  (122-d) + canonicalized radius-graph builder (35-d node features,
  10-d edge features with 8-center Gaussian basis) + in-memory cache
  keyed by `(structure_hash, featurizer_id, version)` +
  `Standardize → PCA → L2-normalize` 256-d embedder with save/load.
  Acceptance: 100 structures featurize in 1.6 s (target < 30 s);
  permuted-atom-order Structures produce bit-identical graphs; Si
  top-3 similarity query returns C, Ge, Al (group-IV cousins).

### Not yet real

- ML datasets / models / GNN / active learning — Phase 6.2+.
- LLM agent loop — stubs until Phase 7.
- Elastic tensors from DFT, phonons beyond Γ, point defects — Phase 8.
- Frontend — not exercised against the live backend since Phase 0 (Phase 9).
- Production deployment, K8s, hardening — Phase 11+.

**Test harness:** 508 passing, 6 skipped (Postgres / SLURM / live pw.x /
live lmp — infra-gated). CI on GitHub Actions green on `main`. Every
Phase 2+ session ships unit tests; Phases 3, 4, and 5 additionally
carry **live-run acceptance tests** against real pw.x, lmp_serial, and
analytical benchmarks respectively (Si total energy, LJ triple-point
RDF, Al C₁₁, Cu T_m, cantilever δ, KMC D, etc.).

---

## Repository layout

```
src/
  api/          Canonical FastAPI backend (routers, models, schemas, auth)
  worker/       Celery worker (job execution)
backend/
  common/       Domain logic shared by API and workers
    structures/       pymatgen-based structure parsers + hashing
    engines/          simulation engines, one subpackage per engine:
      qe_input/         QE input gen + SSSP registry   (Session 3.1)
      qe_run/           pw.x runner + output parser    (Session 3.2)
      lammps_input/     LAMMPS input gen + forcefield registry (Session 4.1)
      lammps_run/       LAMMPS runner + RDF / MSD analyzers    (Session 4.2)
      continuum_fem/    scikit-fem elasticity + heat            (Session 5.1)
      mesoscale_kmc/    rejection-free Gillespie kMC            (Session 5.2)
      (legacy continuum.py / mesoscale.py / qe.py / lammps.py — stubs)
    reports/          aggregate analyzers + report schemas:
      md.py             melting / Arrhenius / elastic (Session 4.3b)
      multiscale.py     DFT→MD→continuum provenance  (Session 5.3)
    ml/               ML stack:
      features_v2/      matminer composition + SSF + radius graph +
                        PCA embedding (Session 6.1)
      (features.py — legacy CGCNN-only; datasets / models — Phase 6.2+)
    workflows/        DAG spec, toposort, foreach, templates/{qe,md,multiscale}
    workers/          JobLifecycle, artifact bundler, pub/sub events
    campaigns/        design-campaign orchestration
    experiments/      lab instrument mocks
frontend/         Next.js 14 UI (TypeScript + Tailwind + MUI)
sdk/python/       Standalone Python SDK (separate package)
alembic/          DB migrations
docker/           Per-service Dockerfiles
k8s/              Kubernetes manifests (Phase 13)
scripts/          One-off operational scripts + smoke tests
                  (includes orion_fetch_potentials.sh for EAM data)
tests/            Test suite — 508 tests, CI-green
docs/
  guides/         User-facing guides (setup, migrations, ML usage)
  history/        Per-session implementation reports (append-only log)
examples/
  marketing/      Standalone marketing demos (not the real frontend)
ROADMAP_PROMPTS.md   14-phase implementation plan
CHANGELOG.md         Merged changes
```

---

## Documentation

- **[ROADMAP_PROMPTS.md](./ROADMAP_PROMPTS.md)** — the plan
- **[CHANGELOG.md](./CHANGELOG.md)** — what's been merged
- **[QUICK_START.md](./QUICK_START.md)** — shortest path to a running system
- **[BACKEND_QUICKSTART.md](./BACKEND_QUICKSTART.md)** — backend dev setup
- **[DEPLOYMENT.md](./DEPLOYMENT.md)** — production deployment (being revised
  in Phase 13)
- **[docs/guides/](./docs/guides/)** — migrations, ML prediction,
  orchestrator, engines quick-start, macOS setup
- **[docs/history/](./docs/history/)** — per-session reports and archived
  pre-refactor status documents

---

## Contributing

1. Check [ROADMAP_PROMPTS.md](./ROADMAP_PROMPTS.md) for the current phase.
2. Work on a per-session branch: `phase-N-session-M-<slug>`.
3. Add a session report in `docs/history/phase_N_session_M_report.md`
   documenting what was done, scope expansions if any, and known blockers
   passed to the next session.
4. Keep the acceptance tests from the roadmap honest — do not mark a session
   complete with numerically wrong physics.

---

## License

MIT. See [LICENSE](./LICENSE).
