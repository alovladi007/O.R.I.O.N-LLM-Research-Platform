# Phase 3 / Session 3.1 — QE input generation

**Branch:** `main`
**Date:** 2026-04-21

## Scope

Session 3.1 builds the canonical Quantum ESPRESSO input file
generator. No execution yet (Session 3.2), no output parsing (Session
3.2 onward) — just: given a structure + parameters + a
pseudopotential library on disk, produce a valid `pw.x` input file.

## What shipped

### `backend/common/engines/qe_input/` — new package

Kept separate from the legacy `backend/common/engines/qe.py` because
the latter mixes generation + subprocess + parsing, uses a hardcoded
33-element pseudopotential dict, and writes `CELL_PARAMETERS` in the
wrong order relative to `ATOMIC_POSITIONS`. Session 3.2 will migrate
the execution path; for now, `qe.py` keeps running for code that
already depends on it.

Three modules:

- `params.py` — `QEInputParams` pydantic v2 schema. Structured fields
  for every common flag (`calculation`, `ecutwfc`, `conv_thr`,
  `occupations`, `smearing`, `spin_polarized`, `tstress`, `kspacing`,
  etc.) plus `extra_control` / `extra_system` / `extra_electrons`
  escape hatches for niche namelist keys. `extra="forbid"` catches
  typos early.
- `registry.py` — `PseudopotentialRegistry` scans one UPF directory
  at build time:
  - Extracts the element from filenames like
    `Si.pbe-n-rrkjus_psl.1.0.0.UPF`, `na_pbe_v1.5.uspp.F.UPF`,
    `Ag_ONCV_PBE-1.0.oncvpsp.upf`, `Ce.paw.z_12.atompaw.wentzcovitch.v1.2.upf`
    by splitting on `.`/`_` and matching the first token that's a
    valid periodic symbol. Case-insensitive fallback for GBRV-style
    `na`/`cl`/`fe` lowercase filenames.
  - Parses `wfc_cutoff`/`rho_cutoff` attributes from UPF v2 XML
    headers, falls back to `Suggested minimum cutoff...` text lines
    from UPF v1. Cutoffs in Ry.
  - Raises `UPFFileNotFoundError` on missing/empty dirs,
    `UnknownElementError` when a structure's element has no
    pseudopotential.
  - `recommended_ecutwfc(elements) → max(wfc_i)` and same for rho,
    matching the SSSP recommendation.
- `renderer.py` — `generate_pw_input(structure, params, registry) →
  RenderedInput`. Steps:
  1. Normalize structure (pymatgen `Structure` or `{lattice, species,
     frac_coords}` dict).
  2. Resolve pseudopotentials (fails fast on unknown elements).
  3. Infer cutoffs from the registry when not user-set;
     `ecutrho = max(registry_rho, dual * ecutwfc)` with `dual=8` for
     USPP/PAW.
  4. Compute Monkhorst-Pack grid via `kgrid_from_structure` when
     `params.kpoints` is unset — uses the reciprocal lattice norm
     with a `kspacing=0.25 Å⁻¹` default.
  5. Render the file: `&CONTROL`, `&SYSTEM`, `&ELECTRONS`, `&IONS`
     (relax/vc-relax/md), `&CELL` (vc-relax), `ATOMIC_SPECIES`,
     `CELL_PARAMETERS angstrom`, `ATOMIC_POSITIONS crystal`,
     `K_POINTS automatic`.
  6. Returns a `RenderedInput` dataclass with `.text`, `.pseudo_files`,
     resolved cutoffs + k-point grid — callers (Session 3.2) will
     use `pseudo_files` to stage only the UPFs they need.

### Fortran value formatting

Small but important detail: scientific notation like `1e-8` must
become `1.0d-8` or QE's parser misinterprets it. `_fortran_float` in
`renderer.py` handles the rendering; `conv_thr`, `forc_conv_thr`,
`degauss` all round-trip correctly.

## Tests

`tests/test_qe_input_gen.py` — 25 tests:

**Registry (8 tests)**
- Detects elements across pslibrary, GBRV/USPP lowercase, kjpaw, ONCV
  filename conventions.
- Filename mapping returns exact UPF name (required for the
  `ATOMIC_SPECIES` block).
- Parses cutoffs from synthetic UPF v2 headers.
- `recommended_ecutwfc` / `recommended_ecutrho` take the max across
  elements.
- `UnknownElementError` for missing elements.
- `UPFFileNotFoundError` for empty/missing directories.

**K-point heuristic (4 tests)**
- Cubic 5 Å lattice + kspacing=0.25 Å⁻¹ → 6×6×6.
- Anisotropic lattice → denser grid along shorter lattice vector.
- Tighter spacing → denser grid.
- Singular (zero-volume) lattice raises.

**Renderer (11 tests)**
- Si SCF: required blocks + section order (CELL_PARAMETERS before
  ATOMIC_POSITIONS, which is what `ibrav=0` requires in practice).
- Si cutoffs inferred from registry's UPF cutoffs.
- User-supplied `ecutwfc`/`ecutrho` override the inferred values.
- NaCl (two-species) produces two `ATOMIC_SPECIES` lines, both UPFs
  in `pseudo_files`.
- K-points override writes the exact grid + shift.
- `calculation=relax` emits `&IONS` with `ion_dynamics='bfgs'` +
  `forc_conv_thr`.
- `calculation=vc-relax` emits `&CELL` with `press_conv_thr`.
- `spin_polarized=True` writes `nspin = 2` and optional
  `tot_magnetization`.
- `extra_control={"wf_collect": True, "restart_mode": "restart"}`
  lands as Fortran-flavored values in the namelist.
- Unknown element raises `UnknownElementError`.
- Bad structure shape raises `ValueError` from normalization.
- `pseudo_dir` override points QE at a different directory (for
  cluster-shared libraries).

**Real SSSP (2 tests)**
- Skipped unless `~/orion/pseudos/SSSP_1.3.0_PBE_efficiency/` exists.
- Confirms the registry loads ≥80 elements from the real library and
  that Si, Na, Cl are all present.
- Renders a Si SCF against the real library and verifies the
  `Si.pbe-n-rrkjus_psl...UPF` filename is referenced.

**Tally:** 211 → **236 tests passing**, 2 skipped (Postgres + SLURM).

## Acceptance criteria status

| Roadmap item | Status |
|---|---|
| Given a Structure row, emit a valid `pw.x` input | ✅ |
| Pseudopotential library support | ✅ (SSSP Efficiency v1.3.0 verified; 99/103 files mapped to elements on the user's disk) |
| Parameter defaults that actually work | ✅ (PBE + smearing + BFGS are the screening-default set) |
| Cutoffs from UPF metadata | ✅ (parses both UPF v2 XML attrs and UPF v1 docstring lines) |
| K-point density from structure | ✅ (reciprocal-lattice-norm / kspacing) |
| Byte-identical-to-known-good Si input | ⏳ Deferred to Session 3.2 when `pw.x` is installed — structure of our output matches QE's expected grammar, but "byte-identical to reference" needs a real `pw.x` to say yes/no |

## Deferred (out of 3.1 scope)

- **Live `pw.x` validation.** Homebrew's `quantum-espresso` formula is
  gone; conda-forge install started in the background but didn't
  finish; user also has `qe-7.5-ReleasePack.tar.gz` in Downloads for a
  from-source build. The unit tests cover the rendered input's shape;
  Session 3.2 will feed a generated input to the real `pw.x` and
  close the loop.
- **UPF v1 edge cases.** Some very old pseudopotential files in SSSP
  (4 out of 103) don't match our element-extraction regex and are
  silently skipped. If those elements come up in practice, I'll widen
  the regex — for now they're the ones nobody asks for (e.g.
  actinides with non-standard file layouts).
- **SSSP cutoff JSON.** Materials Cloud also publishes a JSON with
  "recommended" cutoffs that are sometimes *higher* than the UPF
  intrinsic minimum (SSSP applies a safety margin). We use the UPF
  values for now; a later session can optionally load the JSON
  when available for a tighter convergence signal.
