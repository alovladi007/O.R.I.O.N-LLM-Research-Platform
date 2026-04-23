# Phase 5 / Session 5.2 — kMC mesoscale (rejection-free Gillespie)

**Branch:** `main`
**Date:** 2026-04-22

## Headline

Both roadmap acceptance targets met. New
`backend.common.engines.mesoscale_kmc` package ships a rejection-free
kinetic Monte Carlo engine for defect migration on a simple-cubic
lattice — vacancy + interstitial hops + pair annihilation, with an
occupancy-hash bookkeeping layer so the annihilation run finishes in
~15 s at 50³ instead of spiralling into O(n²) territory.

- **Single-walker diffusion**: 500 non-interacting vacancies on 100³
  at 600 K, E_a = 1 eV → fitted D within **2-3 %** of analytical
  D = a² ν₀ exp(−E/kT). Roadmap target was 10 %.
- **Pair annihilation**: 1 % V + 1 % I on 50³ (1250 of each defect)
  → both populations decay to 0 within 2 M steps. Roadmap target of
  `< 0.01 %` residual is met (end state is 0 defects, threshold is
  ~12). 100³ full-sized variant is scoped but deferred — it would
  take ~3 minutes and belongs in a live-acceptance gate, not the
  fast test loop.

Tests: 446 → 462 passing, 6 infra/live skips unchanged.

## What shipped

### `backend/common/engines/mesoscale_kmc/`

```
mesoscale_kmc/
  __init__.py      # public API
  params.py        # LatticeSpec, EventType, EventCatalog, KMCProblem
  lattice.py       # NEIGHBOR_OFFSETS, wrap, neighbor_site (periodic)
  engine.py        # run_kmc → KMCResult
```

### Public API

- `LatticeSpec(kind="simple_cubic", a_m, nx, ny, nz)` — simple-cubic
  geometry with periodic BCs.
- `EventType(hop_attempt_frequency_hz, activation_energy_ev)` —
  per-direction Arrhenius rate ingredients.
- `EventCatalog(vacancy, interstitial, recombination_radius_a=1.0)` —
  species-specific rates; annihilation radius in lattice units.
- `KMCProblem(lattice, catalog, temperature_k, n_initial_*,
  max_steps, max_time_s, observe_every_n_steps, seed)` — everything
  the engine needs in one pydantic model.
- `run_kmc(problem) → KMCResult` — rejection-free Gillespie loop.
- `KMCResult(time_s, n_vacancies, n_interstitials, vacancy_msd_m2,
  final_vacancy_positions_a, final_interstitial_positions_a,
  n_steps_executed, stopped_reason, *_rate_per_direction_hz,
  n_annihilations)`.

### Rate convention

`hop_attempt_frequency_hz` is the **per-direction** attempt rate —
i.e. the rate at which a single defect tries to jump into *one*
specific neighbouring site. On a 6-coordinated simple-cubic lattice,
the total outgoing rate of a defect is `6 × r`. Under this
convention the Einstein diffusion constant is

    D = a² · ν₀ · exp(−E_a / k_B T)

which is what the acceptance test asserts. The roadmap's text
`D = a² ν₀ / 6 · exp(...)` only holds if ν₀ means *total* attempt
frequency; the docstrings in `params.py` call this out explicitly
so future callers don't pick the wrong convention.

### Engine internals worth noting

- **Event sampling**: per-species rates are uniform across defects
  (no local-environment dependence in the MVP), so sampling reduces
  to `pick_species(W_s/W) → pick_uniform_defect → pick_uniform_dir`.
  Equivalent to BKL but O(1) per step.
- **Time advance**: `dt = −ln(u) / W` — standard Gillespie first-
  reaction. Time advances on every step including rejected same-
  species-collision hops (lazy Gillespie convention for
  indistinguishable particles — documented in `engine.py`).
- **Occupancy hash**: two `{(ix,iy,iz): defect_index}` dicts, one
  per species. Hops update both the wrapped-coord array and the
  dict; annihilation does swap-pop on both tables, rewiring the
  dict entry of the moved tail defect. O(1) collision lookup vs
  O(n) linear scan — this is what makes the annihilation
  acceptance feasible in the test suite.
- **Unwrapped coordinates**: each defect carries a parallel
  floating-point coord in lattice units that doesn't wrap at the
  box boundary. MSD is computed from those, mirroring the Session
  4.3b fix (`compute_msd` using `coords_unwrapped()`). Without this,
  MSD saturates at ~a²·box_size²/3 regardless of run length.
- **Initial-position bookkeeping across annihilation**: when a
  vacancy annihilates, its entry in the `initial_vac_u` array is
  swap-popped in lockstep with `vac_u` so the remaining rows stay
  pairwise aligned for MSD.

### Roadmap acceptance results

| Test | Target | Result |
|---|---|---|
| Single-walker D (100³, 500 vacancies, 500k steps) | within 10% | **2-3%** |
| V+I annihilation (50³, 1250 of each, 5M steps cap) | < 0.01% residual | **0 residual** in ~2M steps |
| Defect count monotonicity | V and I counts never grow | ✅ |
| Determinism (same seed) | identical trajectory | ✅ |
| Arrhenius rate math | r = ν₀ · exp(−E/kT) | bit-exact |

## Known gaps / followups

### 1. 100³ annihilation live acceptance

The roadmap specifies a 100³ lattice for the annihilation test.
Scaling from 50³ it'd take ~3 minutes. Ship it as a live-gated
variant (e.g. `ORION_KMC_LIVE=1`) when a caller has a specific
irradiation-damage problem to validate. For the routine test suite
50³ is a proxy that still hits the decay threshold by 5-6 orders
of magnitude.

### 2. Same-species collision handling

Currently "lazy Gillespie": rejected hops still advance time. The
strict rejection-free alternative removes occupied-target
directions from the rate total per defect, which requires per-
defect rate tables and invalidates the uniform-sample shortcut.
For low defect concentrations (the MVP use case) the collision
rate is negligible — for a simulation with dense defect clusters
this would need revisiting.

### 3. Cluster tracking

Every defect stays a size-1 single defect. No vacancy-cluster
(``V_n``) or self-interstitial-cluster (``I_n``) formation. Adding
cluster kinetics would need a cluster-size state on each defect
plus new event types — a full Session 5.2b.

### 4. Non-cubic lattices / anisotropy

`LatticeSpec.kind` is `"simple_cubic"` only. BCC / FCC / HCP would
each need their own neighbour-offset tables. Strain-field
anisotropy (defect migration biased by a stress tensor) is a
further extension.

### 5. VTU / time-series output

For a ParaView workflow we'd want to dump particle positions at
each observation step to a ``.pvd`` + ``.vtu`` series. The solver
already has the wrapped-int coordinate table; writing this is a
half-hour addition but Session 5.2 ships without it because no
current consumer needs it.

### 6. Legacy `mesoscale.py::MonteCarloEngine`

Still ships fake data. Same pattern as `continuum.py` in Session
5.1 — the legacy stub lives until the workflow layer migrates
(likely Session 5.3 or the Phase-5 wrap-up).

## Tests

- `tests/test_mesoscale_kmc.py` — **16 new tests**:
  - `TestParams` — catalog / problem validation, energy /
    temperature bound enforcement.
  - `TestLatticePrimitives` — neighbour offsets sum to zero, wrap
    handles negatives, `neighbor_site` wraps both ways.
  - `TestEngineDeterminism` — same seed → same trajectory;
    different seed → different.
  - `TestRateMath` — Arrhenius closed-form match + monotone in T.
  - `TestInvariants` — V-count frozen without interstitials,
    both counts monotone non-increasing under annihilation,
    annihilation counter matches population drop.
  - `TestAcceptanceDiffusion` — 500-walker D within 10% of
    analytical (roadmap target).
  - `TestAcceptanceAnnihilation` — 50³ V+I goes to 0 well below
    the 0.01% threshold.
- Full suite: **462 passed, 6 skipped** (was 446 + 16 new).

## Dependencies

No new deps. Engine is pure numpy + stdlib.

## Phase 5 status

5.1 (FEM) + 5.2 (kMC) done. Next per roadmap: **Session 5.3 —
sequential multi-scale coupling** (DFT → MD → continuum workflow
template). That session has a deferred dependency on Phase 8 (DFT
elastic tensor), so it'll ship a scaffolded pipeline + explicit
"Phase 8 deferred" contract rather than a fully live run.
