# Security

Baseline security posture as of Phase 0 / Session 0.4. Phases 11 (hardening)
and 13 (deployment) extend this.

---

## Secret model

**Everything stays in `.env`, never in git.**

- `.env` is ignored by `.gitignore` and was untracked (without history
  rewrite) in Session 0.4. A scan of the one historical commit that added
  `.env` (`92ecbcb`) confirmed only development placeholders — no real
  credentials — so the git history is considered clean.
- `.env.example` lists every variable the app reads. **Only placeholders**
  (`CHANGE_ME_…`) appear there; no real secrets.
- Application code reads secrets via pydantic-settings in
  [`src/api/config.py`](../src/api/config.py). Compose reads them via
  `${VAR:?error}` interpolation from the same `.env`.

### Production must use a real secret manager

`.env` is acceptable for dev and CI; **not** for production. Target in
Phase 11 / 13: back the `Settings` loader with AWS Secrets Manager,
HashiCorp Vault, or GCP Secret Manager via a pluggable adapter. Until then,
production deployments should inject secrets via their platform's native
mechanism (K8s Secrets with SealedSecrets / External-Secrets Operator, or
equivalent) — *not* by shipping `.env` files.

### Rotation policy

| Secret | Rotation cadence | Who |
|---|---|---|
| `JWT_SECRET_KEY` | Every 90 days (sooner on incident) | Phase 11.3 adds `/admin/rotate-key` |
| Database passwords | Every 180 days | Ops |
| Redis password | Every 180 days | Ops |
| MinIO root credentials | Every 180 days | Ops |
| API keys (OpenAI, Anthropic, MP) | Per provider policy | Platform owner |
| Signed MinIO URLs | TTL ≤ 1 hour (Phase 11.3) | App |

On rotation: update the secret manager, redeploy, and invalidate any
outstanding tokens. For JWT specifically, Phase 11.3 implements a dual-key
window so live sessions don't drop on rotation.

---

## Invariants enforced at startup

The canonical [`Settings`](../src/api/config.py) model rejects insecure
configurations before the app starts serving requests. These are enforced
**always**:

- `cors_allow_methods` may not contain `*`.
- `cors_allow_headers` may not contain `*`.

CORS + credentials wildcards are incompatible with the CORS spec — browsers
silently drop the response — so the reject-at-startup behaviour avoids
confusing downstream failures.

These are enforced **only when `ORION_ENV != "development"`** (production,
staging, testing):

- `cors_origins` may not contain `*`.
- `JWT_SECRET_KEY` (or legacy `SECRET_KEY`) must be ≥32 characters.
- `DATABASE_URL` may not contain the known demo substrings
  (`orion_secure_pwd`, `your_secure_postgres_password`).
- `REDIS_URL` may not contain the known demo substrings
  (`orion_redis_pwd`, `your_secure_redis_password`).

Development mode logs a warning but continues — we want local runs to
work even with `CORS_ORIGINS=*` and a generated ephemeral JWT.

### Generating a real JWT secret

```bash
python -c 'import secrets; print(secrets.token_urlsafe(48))'
```

---

## CORS

| Setting | Dev default | Prod requirement |
|---|---|---|
| `cors_origins` | `http://localhost:{3000,3002,8002}` | explicit origin list |
| `cors_allow_credentials` | `True` | `True` |
| `cors_allow_methods` | `GET,POST,PUT,PATCH,DELETE,OPTIONS` | explicit list |
| `cors_allow_headers` | `Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With, X-Request-ID` | explicit list |

All lists are comma-separated in env. Pydantic-settings is patched with a
custom `EnvSettingsSource` ([`_CsvEnvSource`](../src/api/config.py)) so plain
CSV values parse cleanly without JSON encoding.

---

## Known gaps (tracked in [ROADMAP_PROMPTS.md](../ROADMAP_PROMPTS.md))

- **AuthZ depth** — roles are defined but per-object ownership checks are
  incomplete. Phase 11.1.
- **Rate limits** — currently IP-based via slowapi; user-scoped limits
  land in Phase 11.2.
- **File upload handling** — size cap / MIME sniffing / sandboxed parsing
  land in Phase 11.2.
- **Secret manager integration** — Phase 11.3.
- **Provenance & reproducibility** — Phase 12.

## Reporting a vulnerability

Email: _(to be filled in by repo owner)_. Do not open a public issue for
anything exploitable.
