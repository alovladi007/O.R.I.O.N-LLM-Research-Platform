# Phase 0 / Session 0.4 — CORS, secrets, .env hygiene

**Branch:** `phase-0-session-4-cors-secrets-env`
**Date:** 2026-04-17

## Scope

Fix the security posture gaps noted at roadmap review time: `.env` tracked in
git, CORS allowing `["*"]`, demo passwords baked into `.env.example` and
`docker-compose.yml`, no startup validation of JWT secret length or DB/Redis
default-password use.

## `.env` untracked, history verified clean

- `git ls-files .env` showed it was tracked despite being listed in
  `.gitignore` (tracked *before* it was ignored).
- Ran `git log --all -p -- .env`. Only one commit ever touched it
  (`92ecbcb`, "Add comprehensive access guide and environment configuration").
  Content at that commit was development placeholders + CORS origins — **no
  real secrets**. No git-history rewrite required.
- `git rm --cached .env` — file remains on disk locally, but is not tracked
  going forward.
- `.gitignore` already included `.env` from Session 0.1; keeping that.

## CORS hardened in `src/api/config.py`

### Defaults tightened

| Field | Before | After |
|---|---|---|
| `cors_allow_methods` | `["*"]` | `["GET","POST","PUT","PATCH","DELETE","OPTIONS"]` |
| `cors_allow_headers` | `["*"]` | `Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With, X-Request-ID` |
| `cors_origins` (default) | 5 localhost origins including :3001 | 3 canonical local dev origins (:3000, :3002, :8002) |

### Always-on invariants (every environment)

- `cors_allow_methods` may not contain `*`.
- `cors_allow_headers` may not contain `*`.

CORS wildcards are incompatible with `allow_credentials=True` per the CORS
spec — browsers silently drop the response — so we reject at startup.

### Non-dev invariants (`ORION_ENV != "development"`)

- `cors_origins` may not contain `*`.
- `JWT_SECRET_KEY` (or legacy `SECRET_KEY` alias) must be ≥32 characters.
- `DATABASE_URL` may not contain the known demo substrings
  (`orion_secure_pwd`, `your_secure_postgres_password`).
- `REDIS_URL` may not contain the known demo substrings
  (`orion_redis_pwd`, `your_secure_redis_password`).

Enforced by a `@model_validator(mode="after")` on `Settings`, with a
`_ensure_no_wildcard` helper that produces a descriptive error message.

### Env variable aliases

- `ORION_ENV` is now the canonical environment variable;
  `ENVIRONMENT` is accepted as a backcompat alias via
  `AliasChoices("ORION_ENV", "ENVIRONMENT")`.
- `JWT_SECRET_KEY` is now the canonical JWT secret variable; `SECRET_KEY`
  is accepted as a backcompat alias.

### CSV-list parser unified

The four list-typed settings (`cors_origins`, `cors_allow_methods`,
`cors_allow_headers`, `elasticsearch_hosts`) all flow through one
`parse_csv_list` `@field_validator(mode="before")`, plus the existing
`_CsvEnvSource` / `_CsvDotEnvSource` that disables JSON-decoding of these
fields. Removed the duplicated `parse_cors_origins` /
`parse_elasticsearch_hosts` validators.

## Demo credentials purged from tracked configs

### `docker-compose.yml`

Replaced hardcoded `orion_secure_pwd`, `orion_redis_pwd`, `orion_minio_pwd`,
`orion_grafana_pwd`, `orion_jupyter_token` with `${VAR:?err}` env
interpolation against `.env`:

- `POSTGRES_PASSWORD`, `POSTGRES_USER`, `POSTGRES_DB`
- `REDIS_PASSWORD` (used by both `redis` service `--requirepass` and the
  `CELERY_*` URLs on `orion-worker` and `flower`)
- `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`
- `GRAFANA_PASSWORD` (soft default `admin` for local dev)
- `JUPYTER_TOKEN` (soft default for local dev)

`${VAR:?msg}` aborts compose startup with a clear error if the variable is
unset — catches the "forgot to copy .env.example" case at the right moment.

### `.env.example`

Rewrote: structured sections (Application, Security, CORS, Database,
Redis, Elasticsearch, MinIO, LLM providers, Materials Project,
Observability, Email, Celery, Engines, Cloud storage). Every
previously-hardcoded demo password replaced with `CHANGE_ME_*` markers.
Top-of-file preamble explains the startup invariants so users can't
accidentally deploy with placeholder values.

Added new entries needed by the compose overhaul:
`POSTGRES_USER/PASSWORD/DB`, `REDIS_PASSWORD`, `GRAFANA_PASSWORD`.

## `docs/SECURITY.md` created

New top-level security doc covering:
- Secret model (env-based dev, secret-manager backed prod).
- Rotation policy table (JWT, DB, Redis, MinIO, API keys, signed URLs).
- Complete list of startup invariants with their enforcement triggers.
- CORS configuration matrix.
- Known gaps with roadmap session references.
- Vulnerability reporting placeholder.

## Acceptance tests — results

| Roadmap check | Status |
|---|---|
| `git ls-files .env` returns nothing | ✅ |
| Starting with `CORS_ORIGINS="*"` + `ORION_ENV=production` fails fast with a descriptive error | ✅ verified with an inline Python test |
| A cross-origin OPTIONS from `http://evil.com` does not return `Access-Control-Allow-Origin` | 🟡 can't run without the app importing (Session 1.2 blocker); the config-level enforcement that backs this is verified |
| Secret < 32 chars in prod rejected at startup | ✅ verified |
| `DATABASE_URL` containing `orion_secure_pwd` rejected in prod | ✅ verified |

## Decision log

- **Didn't rewrite git history** for `.env`. The one commit that added it
  contained only placeholders and development CORS origins — no real
  credential exposure. History rewrites are destructive on shared repos; a
  `git rm --cached` is sufficient here. Noted in `docs/SECURITY.md`.
- **Development mode intentionally relaxed.** `ORION_ENV=development`
  bypasses the origin-wildcard and short-secret rejections so local runs
  don't need a populated `.env`. A warning is logged instead.
- **Kept `SECRET_KEY` alias** for the JWT secret even though `JWT_SECRET_KEY`
  is the new canonical name. Existing deployments reading the old env var
  won't break silently.
- **Compose uses `${VAR:?err}` instead of soft defaults** for sensitive
  values. Prefer loud failure over silent "you accidentally deployed the
  demo password".
- **Grafana admin password defaulted to `admin` (soft)** — Grafana is not
  externally exposed in the default compose setup, so local-dev can
  tolerate a weak default; Phase 13 tightens this.

## Residual known blockers (unchanged)

- `src.api.app` import still blocked by `src/api/models/__init__.py`
  stale imports (`workflow.py`, `embedding.py`, `provenance.py`,
  `experiment.py`, `orchestrator.py`). Session 1.2.
- `backend/common/engines/lammps.py` imports `ExecutionResult` from
  `.base` which isn't exported. Phase 4 concern.

These are pre-existing; Session 0.4 introduced no new breakage.
