# ORION — Optimized Research & Innovation for Organized Nanomaterials

**Computational materials-science platform with multi-scale simulation,
machine learning, and an LLM-driven design loop.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/FastAPI-0.108+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Next.js-14-black.svg" alt="Next.js 14">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/status-phase--4--MD--in--progress-blue.svg" alt="status">
  <img src="https://img.shields.io/badge/tests-353%20passing-brightgreen.svg" alt="tests">
</p>

> **Honest status (2026-04-22):** Phases 0–3 complete and merged. Phase 4
> (LAMMPS MD) in progress — Session 4.1 done, 4.2+ in flight. The job
> execution spine runs real Quantum ESPRESSO calculations end-to-end: input
> generation, pw.x/dos.x/ph.x execution via the pluggable Local/SLURM
> backend, output parsing, MinIO artifact upload, and multi-step workflow
> DAGs (relax → scf → bands / DOS / phonons). Eight elements have real
> reference energies stored (Al, Cl, Cu, Na, O, Si, Sr, Ti) from live PBE +
> SSSP runs; a compound cross-validation against Materials Project's NaCl
> formation energy agrees to within 0.23 eV/atom — in the expected PBE band
> once MP's empirical halide corrections are accounted for. ML, continuum,
> mesoscale engines, and the LLM agent loop remain stubs — those are
> Phases 5–8. See [ROADMAP_PROMPTS.md](./ROADMAP_PROMPTS.md) for the plan
> and [docs/history/](./docs/history/) for session-by-session reports.

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
- **Phase 4** 🟡 MD engine (LAMMPS) — 4.1 done, 4.2+ in flight
- **Phase 5** — continuum / FEM + kMC mesoscale
- **Phase 6** — ML features, datasets, baselines, GNN, active learning
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
                                │    - quantum_espresso (QE)   │
                                │    - LAMMPS (MD)             │
                                │    - continuum (FEM)         │
                                │    - mesoscale (kMC)         │
                                │  backend.common.ml.*         │
                                │    - CGCNN-like GNN          │
                                │    - Bayesian opt            │
                                │    - active learning         │
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
  109 routes registered.
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
  manifest aggregation. Drives multi-step DFT workflows end-to-end.
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

### MD pipeline (Phase 4) — in progress

- **Session 4.1** — LAMMPS input generation + forcefield registry
  (`backend.common.engines.lammps_input`). Declarative
  `ForcefieldSpec` records with auto-selection by element set; ships
  LJ (fallback), EAM for Cu/Ni/Al (fetched on demand via
  `scripts/orion_fetch_potentials.sh`), Tersoff Si/C (bundled),
  ReaxFF + MACE as feature-flagged stubs. Four ensembles supported
  (NVE, NVT Nose-Hoover, NVT Langevin, NPT Parrinello-Rahman);
  timesteps kept in fs internally and converted per LAMMPS unit style.
- **Session 4.2+** — runner, trajectory parsers (RDF, MSD, VACF,
  Green-Kubo), MD workflow templates. In flight.

### Not yet real

- ML routers (`/api/v1/ml/*`) — still mocks until Phase 6.
- Continuum / mesoscale engines — still mocks until Phase 5.
- LLM agent loop — stubs until Phase 7.
- Frontend — not exercised against the live backend since Phase 0 (Phase 9).
- Production deployment, K8s, hardening — Phase 11+.

**Test harness:** 353 passing, 3 skipped (Postgres / SLURM / live
pw.x — infra-gated). Every Phase 2 and Phase 3 session has unit tests;
Phase 3 additionally has **live-run acceptance tests** against real
pw.x/dos.x/ph.x binaries (~30 tests backed by real QE output
fixtures, not synthetics).

---

## Repository layout

```
src/
  api/          Canonical FastAPI backend (routers, models, schemas, auth)
  worker/       Celery worker (job execution)
backend/
  common/       Domain logic shared by API and workers
    structures/   pymatgen-based structure parsers
    engines/      mock / QE / LAMMPS / continuum / mesoscale
    ml/           features, datasets, GNN, BO, active learning
    campaigns/    design-campaign orchestration
    experiments/  lab instrument mocks
frontend/         Next.js 14 UI (TypeScript + Tailwind + MUI)
sdk/python/       Standalone Python SDK (separate package)
alembic/          DB migrations
docker/           Per-service Dockerfiles
k8s/              Kubernetes manifests (Phase 13)
scripts/          One-off operational scripts + smoke tests
tests/            Test suite (Session 0.5 will grow this)
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
