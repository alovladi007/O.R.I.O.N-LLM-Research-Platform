# ORION — Quick Start

Shortest path from a fresh clone to something running. Honest version.

> For the full story and roadmap, read [README.md](./README.md) and
> [ROADMAP_PROMPTS.md](./ROADMAP_PROMPTS.md). For backend dev specifics, read
> [BACKEND_QUICKSTART.md](./BACKEND_QUICKSTART.md).

## Prereqs

- Python ≥ 3.10
- Docker + Docker Compose
- Node ≥ 18 (frontend only)

## 1. Clone and configure

```bash
git clone https://github.com/alovladi007/O.R.I.O.N-LLM-Research-Platform.git
cd O.R.I.O.N-LLM-Research-Platform

cp .env.example .env
# Edit .env and set at least:
#   DATABASE_URL, REDIS_URL, JWT_SECRET_KEY (≥32 chars), CORS_ORIGINS
```

## 2. Start infra (DB + Redis only — minimal footprint)

```bash
docker-compose up -d postgres redis
make migrate-up        # alembic upgrade head
```

## 3. Run the backend locally

```bash
pip install -r requirements.txt
make dev               # uvicorn src.api.app:app --reload --port 8002
```

Smoke test:

```bash
curl http://localhost:8002/health      # should return 200
open  http://localhost:8002/docs       # OpenAPI UI
```

## 4. Run the frontend

```bash
cd frontend
npm install
npm run dev            # http://localhost:3002
```

## 5. Or: full stack via Docker Compose

```bash
docker-compose up -d   # brings up postgres, redis, minio, elasticsearch,
                       # prometheus, grafana, api, worker, frontend, jupyter
```

Services and their ports are listed in [README.md](./README.md#services-in-docker-composeyml).

## Troubleshooting

- **`src.api.app` won't import** — expected during Phase 0; the canonical app
  still has stale model imports that Session 1.2 resolves. Check
  [CHANGELOG.md](./CHANGELOG.md) to see which session you're on.
- **CORS rejected in browser** — add your origin to `CORS_ORIGINS` in `.env`
  (comma-separated).
- **Port 8002 in use** — `lsof -ti :8002 | xargs kill` or change the port in
  the `make dev` target.
- **macOS specifics** — see [docs/guides/MACOS_SETUP.md](./docs/guides/MACOS_SETUP.md).
