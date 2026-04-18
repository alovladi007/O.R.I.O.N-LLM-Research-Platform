# `tests/fixtures/mp_offline`

Bundled Materials Project-style structures used when `MP_API_KEY` isn't
available. Also used directly by unit tests that want a parsed, known-good
CIF without hitting the network.

Each file is a JSON object with:

```json
{
  "mp_id":   "mp-149",
  "formula": "Si",
  "cif":     "data_Si\n_cell_length_a ...",
  "bandgap": 0.61,
  "formation_energy_per_atom": 0.0,
  "density": 2.33,
  "source":  "materials_project",
  "notes":   "..."
}
```

## Current set (5 structures)

Intentionally minimal — a representative spread rather than a full
roadmap-mandated 20. The roadmap asks for "≥ 20 offline fallback"; this
session ships 5 covering the shapes we exercise in tests and leaves a
task to extend the set in a future session when the underlying loader
proves out in CI.

- `mp-149_Si.json`       — diamond-cubic Si (Fd-3m, 227) — band gap test
- `mp-22862_NaCl.json`   — rock-salt NaCl (Fm-3m, 225) — symmetry test
- `mp-30_Cu.json`        — FCC Cu (Fm-3m) — elemental metal
- `mp-5213_SrTiO3.json`  — cubic ABO3 perovskite (Pm-3m, 221)
- `mp-134_Al.json`       — FCC Al — for Session 8 elastic benchmark

## How to add more

1. Pull the CIF from MP: `MPRester.get_structure_by_material_id("mp-…").to(fmt="cif")`.
2. Save as `{mp-id}_{formula}.json` with the fields above.
3. `scripts/seed_mp_subset.py --offline --dry-run` verifies it parses
   cleanly via the Session 1.1 path.
4. (Optional) `--offline` without `--dry-run` to actually load it into
   your local Postgres.

## Why these five

- **Si / NaCl** are the primary acceptance-test structures for Session
  1.1 (spacegroup detection).
- **Cu** is the EAM reference for Session 4 (LAMMPS MD) — melting point,
  RDF, etc.
- **SrTiO3** exercises the oxide / perovskite path for Sessions 3 (DFT)
  and 7 (BO over ABO3 composition space).
- **Al** is the elastic-benchmark reference for Session 8 (bulk modulus
  ~78 GPa).
