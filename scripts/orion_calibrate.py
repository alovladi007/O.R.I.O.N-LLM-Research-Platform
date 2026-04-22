"""orion calibrate — run calibration for one or more elements.

Example::

    # Calibrate all supported elements against PBE + SSSP_efficiency_1.3.0.
    python scripts/orion_calibrate.py --functional PBE \\
        --pseudos SSSP_efficiency_1.3.0

    # Just Si, or a comma-separated subset:
    python scripts/orion_calibrate.py --elements Si,Cu,Al

    # Re-run Si even if it's already in the table:
    python scripts/orion_calibrate.py --elements Si --force

Idempotency rule: if a (element, functional, pseudo_family) row already
exists in ``reference_energies``, the CLI skips the element unless
``--force`` is set. This is the Session 3.4 roadmap requirement.

Binary + pseudo path are controlled by env vars so the CLI works on
a cluster login node with `QE_EXECUTABLE` pointing at a module-loaded
pw.x:

    QE_EXECUTABLE=pw.x
    QE_PSEUDO_DIR=~/orion/pseudos/SSSP_1.3.0_PBE_efficiency
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Allow `python scripts/orion_calibrate.py ...` without PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="orion calibrate",
        description="Compute per-element reference energies for formation-energy calcs.",
    )
    p.add_argument(
        "--functional",
        default="PBE",
        help="XC functional label to record (default: PBE).",
    )
    p.add_argument(
        "--pseudos",
        default="SSSP_efficiency_1.3.0",
        help="Pseudopotential family label to record.",
    )
    p.add_argument(
        "--elements",
        default=None,
        help=(
            "Comma-separated list of elements to calibrate (e.g. 'Si,Cu,Al'). "
            "Default: all supported elements."
        ),
    )
    p.add_argument(
        "--pseudo-dir",
        default=os.environ.get(
            "QE_PSEUDO_DIR",
            str(Path.home() / "orion" / "pseudos" / "SSSP_1.3.0_PBE_efficiency"),
        ),
        help="Directory containing UPF files (defaults to $QE_PSEUDO_DIR).",
    )
    p.add_argument(
        "--qe-executable",
        default=os.environ.get("QE_EXECUTABLE", "pw.x"),
        help="Path to pw.x binary (defaults to $QE_EXECUTABLE or 'pw.x' on PATH).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute elements that are already in reference_energies.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan (elements to compute) and exit.",
    )
    p.add_argument(
        "--cpus",
        type=int,
        default=int(os.environ.get("ORION_CAL_CPUS", "1")),
    )
    p.add_argument(
        "--walltime-minutes",
        type=int,
        default=30,
        help="Per-element walltime ceiling.",
    )
    p.add_argument(
        "--skip-db",
        action="store_true",
        help=(
            "Run calibrations but don't write to the DB — print results as "
            "JSON. Useful when Postgres isn't up."
        ),
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
    )
    return p.parse_args(argv)


def _resolve_element_list(
    requested: Optional[str], all_supported: List[str],
) -> List[str]:
    if requested is None:
        return all_supported
    elements = [e.strip() for e in requested.split(",") if e.strip()]
    unsupported = [e for e in elements if e not in all_supported]
    if unsupported:
        raise SystemExit(
            f"Unsupported elements: {unsupported}. Supported: {all_supported}"
        )
    return elements


def _existing_references(session, functional: str, pseudo_family: str) -> set:
    """Return set of elements already calibrated for this (functional, pseudos)."""
    from sqlalchemy import select
    from src.api.models import ReferenceEnergy

    stmt = (
        select(ReferenceEnergy.element)
        .where(ReferenceEnergy.functional == functional)
        .where(ReferenceEnergy.pseudo_family == pseudo_family)
    )
    return {row[0] for row in session.execute(stmt).all()}


def _persist(session, cal_result, *, force: bool) -> str:
    """Insert or update a CalibrationResult in reference_energies.

    Returns 'inserted' / 'updated' / 'skipped'.
    """
    from sqlalchemy import select
    from src.api.models import ReferenceEnergy

    stmt = (
        select(ReferenceEnergy)
        .where(ReferenceEnergy.element == cal_result.element)
        .where(ReferenceEnergy.functional == cal_result.functional)
        .where(ReferenceEnergy.pseudo_family == cal_result.pseudo_family)
    )
    existing = session.execute(stmt).scalar_one_or_none()
    if existing is not None and not force:
        return "skipped"

    if existing is None:
        row = ReferenceEnergy(
            element=cal_result.element,
            functional=cal_result.functional,
            pseudo_family=cal_result.pseudo_family,
            energy_per_atom_ev=cal_result.energy_per_atom_ev,
            n_atoms_in_reference_cell=cal_result.n_atoms_in_reference_cell,
            reference_prototype=cal_result.reference_prototype,
            relaxed_a_ang=cal_result.relaxed_a_ang,
            extra_metadata=cal_result.extra_metadata,
        )
        session.add(row)
        session.commit()
        return "inserted"

    existing.energy_per_atom_ev = cal_result.energy_per_atom_ev
    existing.n_atoms_in_reference_cell = cal_result.n_atoms_in_reference_cell
    existing.reference_prototype = cal_result.reference_prototype
    existing.relaxed_a_ang = cal_result.relaxed_a_ang
    existing.extra_metadata = cal_result.extra_metadata
    session.commit()
    return "updated"


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from backend.common.calibration import (
        run_element_calibration,
        supported_elements,
    )

    requested_elements = _resolve_element_list(args.elements, supported_elements())

    already: set = set()
    session = None
    if not args.skip_db:
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker

            from src.api.config import settings

            sync_url = settings.database_url.replace("+asyncpg", "")
            engine = create_engine(sync_url, pool_pre_ping=True, future=True)
            Session = sessionmaker(engine, expire_on_commit=False, future=True)
            session = Session()
            already = _existing_references(session, args.functional, args.pseudos)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Could not reach Postgres ({exc}); falling back to --skip-db")
            args.skip_db = True

    to_compute = [
        e for e in requested_elements
        if args.force or e not in already
    ]
    skipped = [e for e in requested_elements if e not in to_compute]

    print(f"Plan: {len(to_compute)} to compute, {len(skipped)} to skip (already calibrated).")
    print(f"  To compute: {to_compute}")
    if skipped:
        print(f"  Skipped:    {skipped}")

    if args.dry_run:
        return 0

    results = []
    failed = []
    for i, element in enumerate(to_compute, 1):
        print(f"\n[{i}/{len(to_compute)}] calibrating {element} ...")
        t0 = time.time()
        try:
            cal_result = run_element_calibration(
                element,
                qe_executable=args.qe_executable,
                pseudo_dir=args.pseudo_dir,
                functional=args.functional,
                pseudo_family=args.pseudos,
                cpus=args.cpus,
                walltime_minutes=args.walltime_minutes,
            )
        except Exception as exc:  # noqa: BLE001
            dt = time.time() - t0
            print(f"  FAILED after {dt:.1f}s: {exc}")
            failed.append((element, str(exc)))
            continue
        dt = time.time() - t0
        print(
            f"  OK in {dt:.1f}s: {cal_result.energy_per_atom_ev:.4f} eV/atom "
            f"(prototype={cal_result.reference_prototype}, "
            f"a_relaxed={cal_result.relaxed_a_ang:.3f} Å)"
        )
        results.append(cal_result)

        if not args.skip_db and session is not None:
            outcome = _persist(session, cal_result, force=args.force)
            print(f"  DB: {outcome}")

    print("\n" + "=" * 60)
    print(f"Done: {len(results)} calibrated, {len(failed)} failed, {len(skipped)} skipped.")
    if failed:
        print("Failures:")
        for element, msg in failed:
            print(f"  {element}: {msg[:100]}")
    if args.skip_db:
        import json

        print("\nResults (JSON, because --skip-db):")
        print(json.dumps(
            [
                {
                    "element": r.element,
                    "functional": r.functional,
                    "pseudo_family": r.pseudo_family,
                    "energy_per_atom_ev": r.energy_per_atom_ev,
                    "n_atoms_in_reference_cell": r.n_atoms_in_reference_cell,
                    "reference_prototype": r.reference_prototype,
                    "relaxed_a_ang": r.relaxed_a_ang,
                    "extra_metadata": r.extra_metadata,
                }
                for r in results
            ],
            indent=2,
        ))

    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
