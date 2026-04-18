# ORION Backend — Quick Start

**Canonical entry point:** `src.api.app:app` on **port 8002**.

No other entry points exist. `src/api/app_dev.py`, `simple_api.py`, and the root-level
`demo_app.py` / `run_demo.py` were removed or moved to `examples/marketing/` in
Phase 0 / Session 0.1 (entry-point consolidation).

---

## 1. Minimal local dev (no Docker)

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt   # created in Session 0.5

# Start the API (Postgres + Redis must be reachable per DATABASE_URL / REDIS_URL)
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8002
```

or via `make`:

```bash
make dev
```

Verify:

```bash
curl http://localhost:8002/health
open http://localhost:8002/docs
```

## 2. Full stack via Docker Compose

```bash
docker-compose up -d
# API exposed on host port 8000 (inside the container it also runs on 8000).
# Local dev without Docker uses 8002 to avoid clashing with the containerised API.
```

(Port mapping for Compose is unified to 8002 in Phase 13 / Session 13.1.)

## 3. Required environment

Copy `.env.example` to `.env` and fill in:

```bash
DATABASE_URL=postgresql+asyncpg://orion:<password>@localhost:5432/orion_db
REDIS_URL=redis://:<password>@localhost:6379/0
JWT_SECRET_KEY=<≥32 random bytes, generate with `python -c 'import secrets;print(secrets.token_urlsafe(32))'`>
ORION_ENV=dev
CORS_ORIGINS=http://localhost:3000,http://localhost:3002
```

Startup will fail fast if `JWT_SECRET_KEY` is missing or shorter than 32 chars, or if
`CORS_ORIGINS` is `*` outside `ORION_ENV=dev` (enforced in Session 0.4).

## 4. Frontend connection

Frontend is expected at `http://localhost:3002`. CORS is pre-wired for that origin.

## 5. Endpoints

See `http://localhost:8002/docs` for the auto-generated OpenAPI UI. Current routers:

- `/healthz`, `/readyz`
- `/api/v1/auth/*`
- `/api/v1/materials/*`, `/api/v1/structures/*`, `/api/v1/workflows/*`, `/api/v1/jobs/*`
- `/api/v1/ml/*`, `/api/v1/design/*`, `/api/v1/provenance/*`, `/api/v1/campaigns/*`
- `/api/v1/mesoscale/*`, `/api/v1/continuum/*`
- `/api/v1/orchestrator/*`, `/api/v1/agent/*`

**Note:** Many endpoints still return stubbed data — see
[ROADMAP_PROMPTS.md](./ROADMAP_PROMPTS.md) Phases 1–8 for the work to make each real.

## 6. Troubleshooting

- **`ImportError: cannot import name ...` on startup** → likely a dependency is missing.
  Run `pip install -r requirements.txt`.
- **DB connection refused** → start Postgres locally or run `docker-compose up -d postgres redis`.
- **CORS rejected in browser** → add your origin to `CORS_ORIGINS` in `.env`.
- **Port 8002 already in use** → `lsof -ti :8002 | xargs kill` or change the port in `make dev`.
