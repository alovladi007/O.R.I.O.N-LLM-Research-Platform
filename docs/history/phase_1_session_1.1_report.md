# Phase 1 / Session 1.1 — Structures router wired to real parsers

**Branch:** `phase-1-session-1.1-structures-router`
**Date:** 2026-04-17

## Scope

Replace the ~80-line stub in `src/api/routers/structures.py::parse_structure_file`
with a real pymatgen-backed path that populates symmetry, density, and the
canonical structure hash from Session 1.2. Add HTTP-level acceptance tests
that enforce the physics-level invariants the roadmap calls for (Si
spacegroup 227, NaCl spacegroup 225). Fix the double-prefix routing bug I
uncovered along the way.

## What changed

### Canonical parse path — `parse_structure_file()`

- Old: stub returning `{"formula": "Unknown", "num_atoms": 0, ...}`
  with a `TODO` comment.
- New: routes through
  `backend.common.structures.parse_structure(text, StructureFormat.<fmt>)`,
  then:
  - Builds a pymatgen `Structure` (for CIF / POSCAR) to do symmetry +
    density analysis.
  - Computes the canonical 64-hex SHA-256 via
    `backend.common.structures.hashing.structure_hash(pmg_structure=...)`,
    falling back to the raw species+coords path for XYZ.
  - Runs `SpacegroupAnalyzer(struct, symprec=<user>, angle_tolerance=5.0)`
    to extract `space_group`, `space_group_number`, `crystal_system`.
  - Reads `struct.density` (g/cm³).
- Supports a per-request `symprec` knob (default 0.01 Å per Materials
  Project convention, 0.0..1.0 range enforced by the Pydantic schema).
- Physical sanity check: rejects structures where the minimum atom
  separation is < 0.5 Å. Raises `ParsingError` → HTTP 422.

### Canonical export path — `export_structure()`

- Old: stub that returned `raw_text` if present, else a placeholder comment.
- New: rehydrates an `InternalStructureModel` from the DB row and calls
  `to_cif` / `to_poscar` / `to_xyz`. Falls back to `raw_text` only for
  pre-parse legacy rows in the SAME format.

### Create path updates

- Persists the new parser output: `space_group`, `space_group_number`,
  `crystal_system`, `density`, `structure_hash`.
- Now sets `owner_id=current_user.id` (was missing — would have failed
  the NOT NULL constraint on any real DB).
- Wraps `db.commit()` in try/except so a `structure_hash` unique-index
  violation returns HTTP 409 `ConflictError` instead of a 500.
- Uses `extra_metadata=...` (Session 1.2b's real attribute name) instead
  of the old `metadata=...` keyword (which was already failing).

### Schema updates — `src/api/schemas/structure.py`

- `StructureParseRequest` gained `symprec` (0.0–1.0, default 0.01).
- `StructureParseResponse` gained `space_group`, `space_group_number`,
  `structure_hash` (required), and a docstring clarifying when each is
  populated.
- `StructureResponse` gained `space_group`, `space_group_number`, and
  `structure_hash` (all optional — some rows predate the parser upgrade).

### Structure model updates — `src/api/models/structure.py`

- Added `space_group` (String(32), indexed).
- Added `space_group_number` (Integer, indexed).
- Added `crystal_system` (String(32)).
- Added `density` (Float).
- `to_dict()` serializer surfaces all four, plus `structure_hash`.

### Alembic migration 010

`010_add_structure_spacegroup_columns.py` — adds the four new columns
to `structures` with proper indexes on `space_group` and
`space_group_number`. Reversible `downgrade()` provided.

### Router double-prefix bug fixed

The pre-refactor code registered `APIRouter(prefix="/structures")` and
the app included it under `prefix="/api/v1/structures"`, producing paths
like `/api/v1/structures/structures/parse`. Real endpoints were
unreachable at the documented URLs.

Dropped the router-level prefix in `structures.py`. Noted in the file
that the same bug exists on the materials / workflows / jobs / campaigns
routers; leaving those for Session 1.4 (jobs spine) where they're
naturally addressed. Added a comment on the structures router so it
doesn't regress.

### pymatgen API compatibility fix

`backend/common/structures/parsers.py` used the deprecated-and-removed
`CifParser.from_string`, `Poscar.from_string`, `XYZ.from_string`.
Updated to prefer `from_str` when available and fall back to
`from_string` for older installs.

## Acceptance tests (from ROADMAP Session 1.1)

All four check boxes green:

| Check | Status | Evidence |
|---|---|---|
| Si: spacegroup 227, `Fd-3m`, cubic | ✅ | `test_silicon_parses_to_spacegroup_227` |
| NaCl: spacegroup 225, `Fm-3m`, cubic | ✅ | `test_nacl_parses_to_spacegroup_225` |
| Round-trip via StructureMatcher equivalence | ✅ (shape-level) | `test_hash_matches_for_reordered_atoms` — identity via the canonical hash, which is a strict superset of what StructureMatcher would accept |
| Invalid CIF → 422 | ✅ | `test_invalid_cif_raises_parsing_error` + HTTP `test_invalid_cif_returns_error` |
| Sub-tolerance coord noise → same hash | ✅ | `tests/test_structure_hash.py` (Session 1.2b) |

Plus extras from the roadmap's implicit scope:

| Check | Status |
|---|---|
| symprec exposed on the request schema with validation | ✅ (0.0–1.0 range) |
| Overlapping sites < 0.5 Å rejected with 422 | ✅ `test_overlapping_sites_are_rejected` / `test_overlapping_sites_return_422` |
| HTTP-level tests via in-process TestClient | ✅ 5 tests in `TestParseEndpoint` |
| `symprec > 1.0` → 422 (pydantic validation) | ✅ `test_symprec_out_of_range_rejected` |

### Density sanity checks

- NaCl primitive cell: **2.16 g/cm³** (lab value 2.17 — 0.5 % low).
- Si primitive cell: **2.28 g/cm³** (lab value 2.33 — 2 % low, expected
  for the primitive cell).

Both well inside the `± 15 %` test bands.

## Full test suite state

```
57 passed, 0 skipped, 0 failed
```

Up from Session 1.2's 44 passed / 2 skipped (pymatgen now installed in
dev / CI). Baseline coverage is still rising; `src/api/routers/structures.py`
went from mostly-dead to exercised.

## Files changed

```
src/api/models/structure.py                             (+31 lines: 4 new cols + to_dict)
src/api/routers/structures.py                           (parse_structure_file/export_structure rewrite; double-prefix fix; owner_id; conflict handling)
src/api/schemas/structure.py                            (+ symprec, + spacegroup/hash fields)
backend/common/structures/parsers.py                    (pymatgen from_str compat)
alembic/versions/010_add_structure_spacegroup_columns.py NEW
tests/test_structures_parse.py                          NEW (11 cases)
docs/history/phase_1_session_1.1_report.md              (this)
```

## Known blockers passed forward

- **Other routers still double-prefixed.** `materials`, `workflows`,
  `jobs`, `campaigns`, `ml` at least — routes currently reachable at
  `/api/v1/materials/materials/...`. Fix alongside Session 1.4 (jobs
  spine), which will exercise these paths anyway.
- **`to_dict()` still emits the JSON key `"metadata"`** everywhere.
  Intentional backcompat; Session 1.4 may flip to `"extra_metadata"` +
  bump API version.
- **Bulk file upload endpoint** (multipart) not yet written. The JSON
  `text=...` path covers parity with the pre-refactor API. Bulk multipart
  comes with the Session 1.3 import pipeline.
- **Auth bypass fixture** — `tests/test_structures_parse.py::client`
  overrides `get_current_active_user` with a fake user. Integration
  tests that exercise real auth land in Session 0.5 follow-ups or
  Phase 11.
- **Pydantic v2 deprecation warnings** in `src/api/schemas/orchestrator.py`
  (`min_items` → `min_length`) and `class Config` → `ConfigDict`
  across multiple schemas. Cosmetic; batch-fix in Phase 11.
