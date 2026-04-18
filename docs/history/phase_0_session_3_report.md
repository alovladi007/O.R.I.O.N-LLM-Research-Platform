# Phase 0 / Session 0.3 — Docs triage

**Branch:** `phase-0-session-3-docs-triage`
**Date:** 2026-04-17

## Scope

The repo root had **32 markdown files + a shell script + a legacy .txt**,
most of them contradicting each other (SESSION_5 through SESSIONS_21-28
implementation logs, three different "Complete / Status / Summary" docs,
two CI-CD fix writeups, multiple GitHub push helpers). Roadmap Session 0.3
targets ≤ 6 root markdown files with a rewritten README that matches reality.

## Moves

### Moved to `docs/history/`

**Session / implementation logs:**
- `SESSION_5_IMPLEMENTATION.md`, `SESSION_5_SUMMARY.txt`
- `SESSION_6_IMPLEMENTATION.md`, `SESSION_6_SUMMARY.md`
- `SESSION_30_IMPLEMENTATION.md`
- `SESSIONS_7-9_IMPLEMENTATION.md`
- `SESSIONS_10-12_IMPLEMENTATION.md`, `SESSIONS_10-12_INTEGRATION_COMPLETE.md`
- `SESSIONS_13-30_INTEGRATION_COMPLETE.md`
- `SESSIONS_14-17_IMPLEMENTATION.md`
- `SESSIONS_18-20_IMPLEMENTATION.md`
- `SESSIONS_21-28_IMPLEMENTATION.md`

**Status / summary docs (aspirational, not current):**
- `ACCESS_GUIDE.md`, `FRONTEND_STATUS.md`, `ADVANCED_FEATURES.md`
- `COMPLETE_IMPLEMENTATION_SUMMARY.md`, `IMPLEMENTATION_STATUS.md`
- `SUMMARY.md`

**CI-fix and git housekeeping write-ups (one-time):**
- `CI-CD_FIX_COMPLETE.md`, `CI-CD_FIX_FINAL.md`
- `CLEANUP_BRANCHES.md`
- `GITHUB_PUSH_INSTRUCTIONS.md`, `PUSH_TO_GITHUB.md`

Plus `docs/history/README.md` added as an index explaining these are frozen
historical docs.

### Moved to `docs/guides/` (real, current user-facing guides)

- `MACOS_SETUP.md`
- `MIGRATION_GUIDE.md`
- `ML_PREDICTION_GUIDE.md`
- `ORCHESTRATOR_DEPLOYMENT_GUIDE.md`
- `QUICKSTART_ENGINES.md`

Plus `docs/guides/README.md` added as an index.

### Deleted

- `push-to-github.sh` — one-shot GitHub initialization script. Repo is
  already on GitHub; dead code.

## Root-level markdown — final state

```
$ ls *.md
BACKEND_QUICKSTART.md
CHANGELOG.md
DEPLOYMENT.md
QUICK_START.md
README.md
ROADMAP_PROMPTS.md
```

Exactly 6 — matches the roadmap acceptance test (≤ 6 root md).

## CHANGELOG.md created

Seeded with:
- An `[Unreleased]` section listing Session 0.1, 0.2, and 0.3 with commit
  hashes and one-line summaries.
- A `[0.1.0-prototype] — 2026-04-17` baseline entry describing what the
  repo looked like at the start of the roadmap (four entry points, latent
  Pydantic v1 / SQLAlchemy collision / Neo4j cruft, 20k lines of dead code).

Keep-a-Changelog format, follows SemVer intent.

## README.md rewrite

Replaced the aspirational "NANO-OS" framing with honest ORION state:

- Top-of-file "honest status" callout stating that it is a prototype with
  production-grade scaffolding and where real code lives vs. stubs.
- Table of contents.
- New phase-labeled roadmap section with Phase 0 / 1 / ... / 13 summary.
- New architecture diagram (ASCII) showing the API → workers → DBs fan-out.
- New "Services in docker-compose.yml" table (matches the post-Session-0.2
  compose with `orion-frontend` in place of the broken `orion-ui`).
- Separate "Quick start" and "What works today" sections; the latter
  distinguishes merged vs. still-stubbed behaviour and names specific
  roadmap sessions.
- New repository layout tree.
- Updated Documentation section pointing at `docs/guides/` and
  `docs/history/`.
- Removed fabricated badges and marketing language.

## QUICK_START.md rewrite

Previous file claimed "ALL SERVICES RUNNING ✅" and listed endpoints that
return mock data. Rewrote it as an actual quickstart — prereqs, env setup,
`make migrate-up`, `make dev`, smoke-check, frontend instructions, and a
short troubleshooting block that admits `src.api.app` won't import until
Session 1.2.

## Acceptance tests

| Roadmap check | Status | Notes |
|---|---|---|
| `ls *.md` at root returns ≤ 6 files | ✅ | Exactly 6 |
| README quickstart works when followed literally | 🟡 | Works up to `make dev`; `src.api.app` import still blocked by Session 1.2. Documented in "What works today" and in QUICK_START troubleshooting. |
| No section in README claims a feature that is stubbed | ✅ | "What works today" explicitly separates merged vs. still-stubbed, with session references. |

## Files changed summary

```
30 renames (md files into docs/history/ and docs/guides/)
4  new: CHANGELOG.md, docs/history/README.md, docs/guides/README.md, this report
3  rewritten: README.md, QUICK_START.md, docs/history/... (new report)
1  deleted: push-to-github.sh
```

## Decision log

- **Kept `BACKEND_QUICKSTART.md` at root** though it overlaps with
  `QUICK_START.md`. Deferred merging them to Session 0.5 (test harness
  session) since that's where the dev-setup story will solidify.
- **Didn't touch `DEPLOYMENT.md` content** — Phase 13 rewrites it properly
  with K8s manifests and CI/CD finalized. Left as-is at root.
- **`docs/history/README.md` explicitly warns the archived documents have
  aspirational / outdated claims.** This preserves the historical record
  without misleading future readers into treating pre-refactor status docs
  as current.
