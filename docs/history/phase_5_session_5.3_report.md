# Phase 5 / Session 5.3 — sequential multiscale (DFT → MD → continuum) scaffold

**Branch:** `main`
**Date:** 2026-04-22

## Headline

Session 5.3 ships the multiscale coupling **scaffold** — DAG,
provenance threading, report schema, endpoint — but leaves the two
upstream physics analyzers as `PendingAnalyzerError` stubs per the
roadmap's explicit deferred-dependency note. This is the Session
4.3a pattern (scaffold + loud deferred contract), not 4.3b
(implementations arrive).

- **DAG**: `dft_to_md_to_continuum` workflow template with three
  steps wired by `{"uses": "step.outputs.X"}` references. Toposort
  + foreach expansion work; structural validation passes.
- **Deferred**: `dft_elastic` (Phase 8), `md_green_kubo_thermal`
  (Phase 4 follow-up), `continuum_thermomechanical` (Session 5.3b,
  depends on both upstream). All three have Celery tasks registered
  that raise `PendingAnalyzerError` at execution.
- **Provenance**: `ProvenanceLink` threads `(job_id, step_id, kind,
  workflow_run_id)` through each step's outputs; the report
  surfaces links for every step that produced a `job_id`.
- **Report**: `MultiscaleReport` with `pending_analyzers` list that
  spells out *what's missing + which phase ships it* — a user can
  read the report without consulting the roadmap.
- **Endpoint**: `POST /api/v1/workflow-runs/templates/multiscale/{template_name}`
  mirrors the QE + MD endpoints.

Tests: 462 → 482 passing (+20 new), 6 infra/live skips unchanged.

## Explicit framing (honest accounting)

Following the Session 3.3 lesson — distinguish "scaffolded" from
"complete". The roadmap acceptance for 5.3 is:

> End-to-end run on Si produces a continuum solution whose material
> parameters trace back (via provenance, Phase 12) to specific
> DFT/MD jobs.

We meet **the provenance-threading half** (the report carries
provenance links that would trace back once real jobs ran) but we
**don't meet the end-to-end-on-Si half** — the two upstream
analyzers don't exist yet. A user who submits
`dft_to_md_to_continuum` today sees:

1. Workflow record is created.
2. First step `dft_elastic` dispatches.
3. Celery task raises `PendingAnalyzerError("dft_elastic ... Phase 8")`.
4. Workflow status: FAILED. Step output carries the tracker hint.

That's the loud-not-silent failure mode the 4.3a pattern mandates,
and it matches the roadmap's own "Phase 8 — deferred dependency
noted" text.

## What shipped

### `backend/common/reports/multiscale.py`

- `ProvenanceLink(job_id, step_id, kind, workflow_run_id=None)` —
  pydantic model. Phase 12 (full provenance graph with content
  hashes + timestamps) will extend this; 5.3 ships the minimum
  fields the acceptance test checks.
- `MultiscaleReport(report_schema="multiscale_report.v1", ...)` —
  the user-facing report schema. Elastic tensor, thermal
  conductivity, continuum max displacement / von Mises / max T —
  each paired with its own `ProvenanceLink`. A `pending_analyzers`
  list records which deferrals fired during build.
- `extract_dft_elastic_tensor(step_outputs)` — raises
  `PendingAnalyzerError("extract_dft_elastic_tensor",
  tracker="Phase 8 — DFT elastic tensor via QE ±ε strain runs")`.
- `extract_md_thermal_conductivity(step_outputs)` — raises
  `PendingAnalyzerError(..., tracker="Phase 4 follow-up — Green-Kubo
  κ from heat-flux ACF")`.
- `build_multiscale_report(step_outputs, workflow_run_id)` —
  assembles a `MultiscaleReport` tolerantly. Empty input raises
  `AnalyzerInputError`; missing step ids leave fields `None`;
  deferred-analyzer failures are caught and recorded in
  `pending_analyzers` rather than propagating.
- Step-id constants `STEP_ID_DFT_ELASTIC`, `STEP_ID_MD_THERMAL`,
  `STEP_ID_CONTINUUM` — single source of truth shared between
  the report module and the workflow-template builder.

### `backend/common/workflows/templates/multiscale.py`

`dft_to_md_to_continuum_spec(structure_id, **overrides) →
WorkflowSpec` with three `StepSpec`s:

| step_id | kind | parameters |
|---|---|---|
| `dft_elastic` | `dft_elastic` | `strain_magnitude=0.005` |
| `md_thermal` | `md_green_kubo_thermal` | `temperature_k=300.0`, `duration_ps=200.0`, `_after_dft: {uses: dft_elastic.outputs.status}` |
| `continuum_thermomechanical` | `continuum_thermomechanical` | `length_{x,y,z}_m`, `elastic_tensor_voigt_gpa: {uses: dft_elastic.outputs.elastic_tensor_voigt_gpa}`, `thermal_conductivity_w_per_m_k: {uses: md_thermal.outputs.thermal_conductivity_w_per_m_k}` |

Plus `SPEC_BUILDERS = {"dft_to_md_to_continuum": dft_to_md_to_continuum_spec}`
so the endpoint can look it up by name.

### Deferred Celery tasks — `src/worker/tasks.py`

Three new tasks, all raising `PendingAnalyzerError` via a shared
`_raise_pending(job_id, kind, analyzer, tracker)` helper that wires
up the `JobLifecycle` bookkeeping so the workflow tick correctly
propagates FAILED state upstream:

- `orion.dft.elastic` — tracker `"Phase 8"`.
- `orion.md.green_kubo_thermal` — tracker `"Phase 4 follow-up"`.
- `orion.continuum.thermomechanical` — tracker
  `"Session 5.3b (depends on Phase 8 + Phase 4 follow-up)"`.

### Registry updates

- `backend.common.workflows.celery_dispatcher._KIND_TO_TASK` — +3
  kinds.
- `src.api.routers.jobs._DISPATCH_TASKS` — +3 kinds (must stay in
  sync with the above; a new test verifies).
- `src.api.routers.jobs._BUILTIN_TEMPLATES` — +3 lazy-materialized
  templates so the dispatch path works on a fresh DB.

### Endpoint

`POST /api/v1/workflow-runs/templates/multiscale/{template_name}` —
copy of the QE / MD handlers. Validates the template name against
`SPEC_BUILDERS`, validates the structure exists, toposort-validates
the expanded spec, persists `WorkflowRun` + `WorkflowRunStep`
records, kicks the workflow tick.

## Tests — `tests/test_multiscale_template.py`

20 new tests across five classes:

- `TestDftToMdToContinuumDAG` — step IDs / kinds, toposort order
  respects the DFT → MD → continuum dependency, continuum step's
  `uses` references both upstream outputs, default parameters
  sensible, `SPEC_BUILDERS` exposes the one template.
- `TestDeferredAnalyzers` — DFT analyzer raises with Phase 8
  tracker, MD analyzer raises with Phase 4 + "Green-Kubo" in the
  message, `PendingAnalyzerError` is still a
  `NotImplementedError` subclass.
- `TestMultiscaleReportSchema` — minimal report validates, extra
  fields forbidden, `ProvenanceLink` round-trips through
  `model_dump` / reconstruction.
- `TestBuildMultiscaleReport` — empty input raises
  `AnalyzerInputError`, partial input populates provenance +
  `pending_analyzers` correctly, missing `job_id` yields no
  provenance link.
- `TestDispatcherWiring` — all three kinds are in both dispatch
  tables and builtin templates, task names match across tables,
  endpoint is registered, existing QE / MD endpoints still
  registered.

Full suite: **462 → 482 passed, 6 skipped** (infra / live gates
unchanged).

## Known gaps / followups

### 1. Session 5.3b — run the scaffolded tasks for real

The moment Phase 8 ships `dft_elastic` and the Phase 4 follow-up
ships Green-Kubo κ, the `continuum_thermomechanical` task becomes
runnable. The linear-elasticity + steady-heat solvers already
exist (Session 5.1); 5.3b wires the thermoelastic coupling
(`σ = C:ε − α·ΔT·I`) on top. Estimate: ~3 hours once both upstream
inputs are live.

### 2. DFT relax as a sub-step of `dft_elastic`

The roadmap's Phase 8 spec wraps a relax + strain sweep inside a
single step. Session 5.3 just gives a single `dft_elastic` kind;
Phase 8 will likely decompose it into a relax sub-DAG +
foreach-over-strains pattern like Session 4.3a's elastic
constants workflow. The interface `MultiscaleReport` sees is the
same (single elastic tensor output) regardless.

### 3. Phase 12 provenance graph

`ProvenanceLink` ships the three fields the multiscale report
needs; Phase 12's full graph will add content hashes, timestamps,
container / software-version triples, and a graph-query API.
`MultiscaleReport.*_provenance` stays forward-compatible — Phase 12
can either extend `ProvenanceLink` or return a richer subclass.

### 4. Thermal expansion α_T

The `continuum_thermomechanical` task needs a thermal-expansion
tensor to couple C_ij and ΔT. Session 5.3 doesn't plumb α_T
through the DAG. Either (a) add an `α_T` field to the DFT elastic
analyzer's outputs (quasi-harmonic DFT gives this), or (b) MD
extracts it separately. Decision deferred to 5.3b.

### 5. Legacy `continuum.py` / `mesoscale.py` stubs

Still in place (noted in 5.1 and 5.2 reports). The canonical FEM
path is `continuum_fem`; the canonical kMC path is `mesoscale_kmc`.
Neither legacy stub participates in the multiscale DAG.

## Phase 5 status

5.1 (FEM continuum) + 5.2 (kMC mesoscale) + 5.3 (multiscale scaffold)
all merged. Phase 5 is done as-scoped — the 5.3b fill-in waits on
Phase 8 + the Phase 4 Green-Kubo follow-up.

**Next session per roadmap:** Phase 6 / Session 6.1 (Featurizers for
ML pipeline).
