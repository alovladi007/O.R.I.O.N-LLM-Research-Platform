#!/usr/bin/env bash
# Phase 9 / Session 9.1 — generate frontend/src/types/api.generated.ts
# from the live FastAPI openapi schema.
#
# Usage
# -----
#
#   # Local: backend running on the canonical 8002 port.
#   cd frontend && npm run gen:api
#
#   # Custom backend URL (CI / docker compose / staging):
#   OPENAPI_URL=http://localhost:8888/openapi.json npm run gen:api
#
# The generated file IS committed (so the frontend CI doesn't depend
# on a live backend). Re-run this script + commit whenever the
# backend's openapi changes — typically alongside the session that
# added / removed / renamed endpoints.
#
# Notes
# -----
#
# - openapi-typescript is a devDep; do not rely on it being installed
#   in production images.
# - We run it via npx so Pre-commit hooks / CI lockfile-installs
#   don't need a global binary.

set -euo pipefail

OPENAPI_URL="${OPENAPI_URL:-http://localhost:8002/openapi.json}"
OUT="src/types/api.generated.ts"

# Confirm we're at frontend/ — the npm script runs us there but
# direct invocation may not.
if [[ ! -f package.json ]] || ! grep -q '"name": "orion-platform-frontend"' package.json; then
    echo "error: must be run from frontend/ — got $(pwd)" >&2
    exit 1
fi

echo "→ fetching openapi schema from ${OPENAPI_URL}"
echo "→ writing TypeScript types to ${OUT}"

npx --yes openapi-typescript "${OPENAPI_URL}" -o "${OUT}"

echo "✓ wrote $(wc -l < "${OUT}") lines to ${OUT}"
echo "  remember to commit ${OUT} so the frontend CI doesn't need a live backend"
