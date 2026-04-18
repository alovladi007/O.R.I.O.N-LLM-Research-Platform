# ORION — Optimized Research & Innovation for Organized Nanomaterials

**Computational materials-science platform with multi-scale simulation,
machine learning, and an LLM-driven design loop.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/FastAPI-0.108+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Next.js-14-black.svg" alt="Next.js 14">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/status-prototype--in--refactor-orange.svg" alt="status">
</p>

> **Honest status:** ORION today is a **prototype with production-grade
> scaffolding**. Infrastructure (Docker Compose, Alembic, JWT auth, Celery
> queues, CI/CD skeleton) is real. Most science-facing API endpoints are still
> stubs while the canonical backend (`src.api.app`) is being brought online.
> Real implementations exist in `backend/common/` (pymatgen structure parsers,
> engine wrappers for QE / LAMMPS / mock, GNN / BO / active-learning modules);
> the work-in-flight is wiring them into the routers, covering them with tests,
> and validating them against known-physics benchmarks. See
> [ROADMAP_PROMPTS.md](./ROADMAP_PROMPTS.md) for the 14-phase plan and
> [docs/history/](./docs/history/) for session-by-session implementation
> reports.

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

- **Phase 0** (in progress) — repository hygiene
- **Phase 1** — data & domain foundation
- **Phase 2** — job execution spine (Celery + engines)
- **Phase 3** — DFT engine (Quantum Espresso)
- **Phase 4** — MD engine (LAMMPS)
- **Phase 5** — continuum / FEM + kMC mesoscale
- **Phase 6** — ML features, datasets, baselines, GNN, active learning
- **Phase 7** — Bayesian optimisation + LLM-driven agent loop
- **Phase 8** — elastic tensors, phonons, point defects
- **Phase 9** — frontend end-to-end wiring
- **Phase 10** — observability & performance
- **Phase 11** — security hardening
- **Phase 12** — provenance, reproducibility, compliance
- **Phase 13** — K8s, CI/CD, DR, 1.0.0 release

See [CHANGELOG.md](./CHANGELOG.md) for merged changes.

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

Updated as each Phase 0 session lands. If a feature isn't listed here, treat
it as stubbed or WIP.

**Merged (Phase 0 Sessions 0.1 + 0.2):**

- Single canonical FastAPI entry point (`src.api.app:app`) on port 8002
- Pydantic v2 settings with CSV-aware env parsing for `CORS_ORIGINS`
- Pre-refactor `src/*` packages and Neo4j removed — cleaner dependency graph
- Docker Compose services reconciled (`orion-frontend` now builds the real
  Next.js Dockerfile; `orion-worker` points at the real Celery app)

**Still broken end-to-end (tracked in roadmap):**

- `src/api/app:app` import still blocked by stale imports in
  `src/api/models/__init__.py` — Session 1.2.
- Structure parse/export API routes return mock data even though
  `backend/common/structures/` has real pymatgen parsers — Session 1.1.
- ML training submission, job cancellation, agent campaign loop — Sessions
  1.4, 6.4, 7.3.
- QE / LAMMPS engines wired but not benchmark-validated — Phase 3, Phase 4.
- Only 2 test files at Phase 0 start; Session 0.5 builds the real harness.

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
