# Phase 4 / Session 4.3 — MD workflow templates (4.3a)

**Branch:** `main`
**Date:** 2026-04-22

## Headline (honest framing, lessons from 3.3 applied)

The roadmap's Session 4.3 lists **four MD workflow templates**:
`equilibrate_nvt_then_nve`, `melting_curve`, `diffusivity_vs_T`,
`elastic_constants_via_strain`. Each pairs a workflow DAG with a
post-analyzer (RDF/MSD on the NVE leg, melting-point detection,
Arrhenius fit, C_ij solver respectively).

This session ships **one template fully** (`equilibrate_nvt_then_nve`)
and **three templates partially** (DAG + report-schema scaffolding +
explicit `PendingAnalyzerError` stubs for the post-analyzers). All
four DAGs dispatch correctly through the Session 2.4 workflow
executor; users running `melting_curve` today get the per-temperature
MD outputs, but the melting-point detection that turns those into a
single number lands in **Session 4.3b**.

This split is **deliberate and explicitly framed**, in contrast to
Session 3.3's original framing that called bands/phonons "deferred"
in a footnote. Each deferred analyzer raises a custom
`PendingAnalyzerError` (subclass of `NotImplementedError`) carrying
the `Session 4.3b` tracker hint. No silent zeros. No misleading
"complete" claim.

## What shipped

### `backend/common/workflows/templates/md.py` — new module

Four `WorkflowSpec` builders + a `SPEC_BUILDERS` dict:

- **`equilibrate_nvt_then_nve_spec(structure_id, ...)`** — 2-step DAG:
  `md_nvt` (Langevin warm-up, sparse dumps) → `md_nve` (production,
  dense dumps for RDF/MSD). The NVE step depends on the NVT step via
  a `{"uses": ...}` reference so the executor orders them. **Production-ready.**
- **`melting_curve_spec(structure_id, temperatures_k=(...))`** —
  fan-out DAG via `foreach`: N independent NPT runs at N
  temperatures. Default sweep is 800–1600 K per the roadmap. The
  melting-point detection (`detect_melting_point`) currently raises
  `PendingAnalyzerError`; the per-temperature MD runs themselves
  execute correctly.
- **`diffusivity_vs_t_spec(structure_id, temperatures_k=(...))`** —
  fan-out DAG: N independent NVE runs at N temperatures with dense
  dumps for MSD quality. Arrhenius fit (`arrhenius_fit`) deferred to
  4.3b.
- **`elastic_constants_via_strain_spec(structure_id, ...)`** — six
  ±0.5% diagonal-strain MD runs. C_ij solver (`fit_elastic_constants`)
  deferred to 4.3b. Note: shear deformations + the `change_box`
  prepend that actually applies the strain are **also** 4.3b — for
  now the strain values flow into outputs as marker fields, but the
  current MD runs all use the unstrained input cell. **The DAG ships
  to lock in the API surface; the live science needs 4.3b.**

### `backend/common/reports/` — new package

- `MDReport` base pydantic model with the common fields
  (`workflow_run_id`, `step_outputs`, timestamps, schema tag).
- Three specializations that the four templates would emit when their
  analyzers land: `MeltingCurveReport`, `ArrheniusReport`,
  `ElasticConstantsReport`.
- `PendingAnalyzerError(NotImplementedError)` carries the analyzer
  name + the "Session 4.3b" tracker. Inherits from
  `NotImplementedError` so callers that catch the standard exception
  still see it.
- Three deferred analyzer functions
  (`detect_melting_point`, `arrhenius_fit`, `fit_elastic_constants`)
  raise it. Each carries a docstring describing the algorithm 4.3b
  will implement, so the next session has the design in front of it.

### API endpoint

`POST /api/v1/workflow-runs/templates/md/{template_name}` —
mirrors the QE template endpoint from Session 3.3. Validates the
template name, looks up the structure, builds the DAG, persists +
kicks the workflow tick.

Routes: 109 → 110.

## Tests

`tests/test_md_workflow_templates.py` — **24 new tests**:

- **`equilibrate_nvt_then_nve` (5 tests)** — two-step DAG, kinds
  correct, temperature/duration parameters propagated, production
  step dumps more densely than equilibrate.
- **`melting_curve_spec` (2)** — foreach expansion produces N
  children with correct temperatures; default ensemble is NPT.
- **`diffusivity_vs_t_spec` (2)** — uses NVE; dump_every=100
  (dense for MSD).
- **`elastic_constants_via_strain_spec` (2)** — six steps with
  marker fields; strains balanced (3 negative, 3 positive of same
  magnitude).
- **MDReports (4)** — base + three specializations all validate.
- **Deferred analyzers (4)** — each raises `PendingAnalyzerError`
  with `4.3b` in the message; class is a `NotImplementedError`
  subclass.
- **Dispatcher registration (4)** — md_nvt/nve/npt in jobs router +
  workflow executor; MD endpoint registered; QE endpoint still
  registered (regression check).
- **`SPEC_BUILDERS` table (1)** — exactly the four expected names.

**Suite: 374 → 398 passing, 4 skipped.**

## Acceptance criteria status

Per the roadmap Session 4.3 spec:

| Item | Status |
|---|---|
| `equilibrate_nvt_then_nve` template | ✅ ships fully |
| `melting_curve` template | ⚠️ DAG ships; melting-detection deferred to 4.3b |
| `diffusivity_vs_T` template | ⚠️ DAG ships; Arrhenius fitter deferred to 4.3b |
| `elastic_constants_via_strain` template | ⚠️ DAG ships; strain application + C_ij solver deferred to 4.3b |
| Each template produces a Report object | ✅ schemas in place; 3 of 4 analyzers raise PendingAnalyzerError |
| Cu melting curve T_m within ±150 K of 1358 K | ❌ deferred — needs detect_melting_point + EAM Cu fetched (`bash scripts/orion_fetch_potentials.sh`) |
| Li Arrhenius fit R² > 0.95 | ❌ deferred — needs Li forcefield (LJ doesn't capture this) |
| Al elastic C_11 within 15% of 108 GPa | ❌ deferred — needs `change_box` prepend + C_ij solver |

## What's actually trustworthy today

Use this template if you want a working result this session:

```python
from backend.common.workflows.templates.md import equilibrate_nvt_then_nve_spec
spec = equilibrate_nvt_then_nve_spec(
    structure_id="<your-structure-uuid>",
    temperature_k=300.0,
    equilibrate_ps=20.0,
    production_ps=20.0,
)
# Submit via POST /api/v1/workflow-runs/templates/md/equilibrate_nvt_then_nve
#   ?structure_id=...
```

The two MD jobs run; the NVE leg's RDF first peak + MSD + diffusion
coefficient land on `step_outputs["production"]` (where the workflow
manifest aggregates them). For Cu EAM + Si Tersoff, fetch the
respective potentials via `bash scripts/orion_fetch_potentials.sh`
(EAM only; Tersoff is bundled).

## Session 4.3b ticket (explicit)

The three deferred analyzers + the strain-application path. Estimated
1–3 hours total:

- **`detect_melting_point`** (~45 min): sort step_outputs by T;
  compute dMSD/dT and dE_total/dT; find the temperature where both
  show a discontinuity; cross-check with RDF first-peak height.
  Return T_m at the midpoint with a confidence level.
- **`arrhenius_fit`** (~30 min): linear least-squares on
  (1/T, ln D); compute E_a (eV), prefactor D_0, R².
- **`fit_elastic_constants` + `change_box` prepend** (~1.5 h): wire
  a strain-application option into `LAMMPSInputParams` (extra
  `change_box` line at the top of the deck); extract per-step stress
  from `final_thermo['Press']`; solve Hooke's law σ = C ε for the
  three diagonal C_ii.
- **Live acceptance tests for Cu / Li / Al** — once the analyzers
  exist, run them against the roadmap's reference materials.

These are all bounded, well-specified tasks. None is research; all
are implementations of standard methods.
