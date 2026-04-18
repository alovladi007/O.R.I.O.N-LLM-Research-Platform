# docs/history — append-only implementation log

This directory holds **historical documents**: per-session reports for every
roadmap session that has been completed, plus archived pre-refactor status
documents and CI-fix write-ups.

Files here are **frozen** — they describe what was true at the time they were
written, not necessarily what is true now. Do not edit past reports to reflect
later changes; instead, write a new report in the corresponding session and
cross-reference.

## Per-session reports (Phase 0 onward)

- `phase_0_session_1_report.md` — entry-point consolidation + bootstrap fixes
  (merged `48ec361`).
- `phase_0_session_2_report.md` — legacy `src/*` cleanup + Neo4j removal
  (merged `b7ba6df`).
- `phase_0_session_3_report.md` — docs triage (this session).

## Pre-Phase-0 archived documents

The large backlog of `SESSION_*`, `SESSIONS_*`, `CI-CD_FIX_*`,
`IMPLEMENTATION_STATUS`, `COMPLETE_IMPLEMENTATION_SUMMARY`, `ACCESS_GUIDE`,
`FRONTEND_STATUS`, `SUMMARY`, `ADVANCED_FEATURES`, `CLEANUP_BRANCHES`,
`PUSH_TO_GITHUB`, and `GITHUB_PUSH_INSTRUCTIONS` files captured in this
directory describe the platform's state *before* the refactor roadmap began.
Many of their feature claims are aspirational or outdated — use them as
historical context, not as current specs. The canonical current state lives
in [CHANGELOG.md](../../CHANGELOG.md) and the roadmap session reports.
