"""orion_scf_compound — run an SCF on an MP-structure CIF and record total energy.

Used by Session 3.4b to produce the compound fixture for cross-validation
against MP formation energies. Reads one of the `tests/fixtures/mp_offline/*.json`
files, parses its CIF, runs pw.x in SCF mode, and writes the result into
`tests/fixtures/calibration/compound_scf_pbe_sssp_efficiency_1.3.0.json`.

Example::

    python scripts/orion_scf_compound.py \\
        tests/fixtures/mp_offline/mp-22862_NaCl.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_DEFAULT_OUTPUT = (
    _REPO_ROOT / "tests" / "fixtures" / "calibration"
    / "compound_scf_pbe_sssp_efficiency_1.3.0.json"
)


def _parse_mp_fixture(fixture_path: Path) -> Dict[str, Any]:
    """Load an MP fixture and parse its CIF into a QE-shaped struct dict."""
    fixture = json.loads(fixture_path.read_text())
    cif = fixture["cif"]

    # Use the same pymatgen path as Session 1.1.
    from pymatgen.io.cif import CifParser
    import io

    parser = CifParser(io.StringIO(cif))
    structures = parser.parse_structures(primitive=True)
    if not structures:
        raise ValueError(f"No structures parsed from CIF in {fixture_path}")
    structure = structures[0]
    qe_struct = {
        "lattice": [list(row) for row in structure.lattice.matrix],
        "species": [str(s) for s in structure.species],
        "frac_coords": [list(c) for c in structure.frac_coords],
    }
    return {
        "mp_id": fixture["mp_id"],
        "formula": fixture["formula"],
        "formation_energy_per_atom": fixture.get("formation_energy_per_atom"),
        "qe_struct": qe_struct,
    }


def _merge_into_output(
    output_path: Path, record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    existing: List[Dict[str, Any]] = []
    if output_path.is_file():
        existing = json.loads(output_path.read_text())
    # Replace any record with the same mp_id + formula.
    filtered = [
        r for r in existing
        if not (r.get("mp_id") == record.get("mp_id")
                and r.get("formula") == record.get("formula"))
    ]
    filtered.append(record)
    return filtered


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="orion_scf_compound",
        description="Run pw.x SCF on an MP-fixture compound and record the energy.",
    )
    parser.add_argument(
        "fixture", type=Path,
        help="Path to an MP fixture JSON (with 'cif' + 'formula').",
    )
    parser.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT,
        help="Where to append the result. Replaces any existing entry for the same mp_id.",
    )
    parser.add_argument(
        "--qe-executable",
        default=os.environ.get("QE_EXECUTABLE", "pw.x"),
    )
    parser.add_argument(
        "--pseudo-dir",
        default=os.environ.get(
            "QE_PSEUDO_DIR",
            str(Path.home() / "orion" / "pseudos" / "SSSP_1.3.0_PBE_efficiency"),
        ),
    )
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--walltime-minutes", type=int, default=30)
    args = parser.parse_args(argv)

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    fixture_rec = _parse_mp_fixture(args.fixture)
    print(f"Running SCF on {fixture_rec['formula']} ({fixture_rec['mp_id']})")

    from backend.common.calibration import run_compound_scf

    result = run_compound_scf(
        fixture_rec["qe_struct"],
        formula=fixture_rec["formula"],
        qe_executable=args.qe_executable,
        pseudo_dir=args.pseudo_dir,
        cpus=args.cpus,
        walltime_minutes=args.walltime_minutes,
    )

    record = {
        "mp_id": fixture_rec["mp_id"],
        "formula": fixture_rec["formula"],
        "total_energy_ev": result.total_energy_ev,
        "species": result.species,
        "n_atoms": result.n_atoms,
        "formation_energy_per_atom_mp": fixture_rec["formation_energy_per_atom"],
        "extra_metadata": result.extra_metadata,
    }
    merged = _merge_into_output(args.output, record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2))
    print(f"Wrote {args.output}")
    print(f"  total_energy_ev = {result.total_energy_ev:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
