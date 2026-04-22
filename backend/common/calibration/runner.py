"""Calibration runner: run vc-relax → scf for one element, store reference.

This is the pure-compute engine. The CLI (``orion calibrate``) and
workflow entry point both call it. It's synchronous and returns a
result record; the caller decides whether to persist to the DB, wrap
in a Celery task, etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_FUNCTIONAL = "PBE"
DEFAULT_PSEUDO_FAMILY = "SSSP_efficiency_1.3.0"


@dataclass
class CalibrationResult:
    """One completed element calibration."""

    element: str
    functional: str
    pseudo_family: str
    energy_per_atom_ev: float
    n_atoms_in_reference_cell: int
    reference_prototype: str
    relaxed_a_ang: Optional[float]
    run_dir: Path
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


def run_element_calibration(
    element: str,
    *,
    qe_executable: str,
    pseudo_dir: Path | str,
    functional: str = DEFAULT_FUNCTIONAL,
    pseudo_family: str = DEFAULT_PSEUDO_FAMILY,
    a_override: Optional[float] = None,
    vacuum_box_ang: Optional[float] = None,
    run_dir_parent: Optional[Path] = None,
    cpus: int = 1,
    walltime_minutes: int = 30,
) -> CalibrationResult:
    """Relax + SCF the elemental reference cell for *element* and return the result.

    Two modes, selected by prototype:

    - **Bulk crystals** (``diamond_cubic`` / ``fcc`` / ``bcc`` / ``hcp``):
      ``vc-relax`` at the automatic k-mesh derived from the lattice.
    - **Diatomic gases** (``molecule_in_vacuum``): ``relax`` in a
      fixed 15 Å cubic box, Γ-only k-sampling, spin-polarized when the
      ground state is a triplet (O₂ notably).

    The caller persists the :class:`CalibrationResult` to the
    ``reference_energies`` table (or handles UniqueConstraint collisions
    for idempotency).
    """
    from backend.common.engines.qe_input import (
        PseudopotentialRegistry,
        QEInputParams,
        generate_pw_input,
    )
    from backend.common.engines.qe_run import run_pw
    from backend.common.workers import build_run_dir

    from .references import build_elemental_reference_cell, is_triplet_diatomic

    ref_cell = build_elemental_reference_cell(
        element, a_override=a_override, vacuum_box_ang=vacuum_box_ang,
    )
    qe_struct = ref_cell.as_qe_struct()

    registry = PseudopotentialRegistry(Path(pseudo_dir).expanduser())

    # Prototype-specific QE parameters.
    if ref_cell.prototype == "molecule_in_vacuum":
        # Relax (not vc-relax) so the 15 Å box stays 15 Å.
        # Γ-only k-sampling — there's no periodicity to sample.
        # Triplet diatomics (O₂) need spin polarization + fixed
        # tot_magnetization = 2.
        triplet = is_triplet_diatomic(element)
        qe_params = QEInputParams(
            prefix=f"cal_{element.lower()}",
            calculation="relax",
            occupations="smearing",
            smearing="gauss",
            degauss=0.01,
            kpoints=(1, 1, 1),
            spin_polarized=triplet,
            tot_magnetization=2.0 if triplet else None,
        )
    else:
        qe_params = QEInputParams(
            prefix=f"cal_{element.lower()}",
            calculation="vc-relax",
            occupations="smearing",
            smearing="gauss",
            degauss=0.01,
        )
    rendered = generate_pw_input(qe_struct, qe_params, registry)

    run_dir = build_run_dir(f"calibration-{element}", parent=run_dir_parent)
    pw_result = run_pw(
        rendered,
        run_dir,
        qe_executable=qe_executable,
        pseudo_src_dir=registry.pseudo_dir,
        cpus=cpus,
        walltime_minutes=walltime_minutes,
        species_hint=qe_struct["species"],
    )
    if not pw_result.success:
        raise RuntimeError(
            f"Calibration for {element} failed at stage={pw_result.stage}: "
            f"{pw_result.error_message}"
        )

    output = pw_result.output
    if output is None or output.energy is None:
        raise RuntimeError(
            f"Calibration for {element}: pw.x succeeded but no total "
            "energy in output."
        )

    energy_total = output.energy.total_ev
    n_atoms_out = output.n_atoms or ref_cell.n_atoms
    energy_per_atom = energy_total / n_atoms_out

    relaxed_a_ang: Optional[float] = None
    if output.relaxed is not None:
        # For diamond_cubic / fcc stored as primitive → conventional via sqrt(2).
        # For bcc primitive → conventional via 2/sqrt(3). For hcp, we store
        # the a-axis of the conventional cell directly. For
        # molecule_in_vacuum, we store the relaxed *bond length*
        # (distance between the two atoms) — a more useful reference
        # number than the box edge.
        import math
        prim_a = output.relaxed.a_lattice_const_ang
        if ref_cell.prototype in ("diamond_cubic", "fcc"):
            relaxed_a_ang = prim_a * math.sqrt(2)
        elif ref_cell.prototype == "bcc":
            relaxed_a_ang = prim_a * 2 / math.sqrt(3)
        elif ref_cell.prototype == "hcp":
            relaxed_a_ang = prim_a
        elif ref_cell.prototype == "molecule_in_vacuum":
            coords = output.relaxed.cart_coords_ang
            if len(coords) >= 2:
                dx = coords[1][0] - coords[0][0]
                dy = coords[1][1] - coords[0][1]
                dz = coords[1][2] - coords[0][2]
                relaxed_a_ang = math.sqrt(dx * dx + dy * dy + dz * dz)
        else:
            relaxed_a_ang = prim_a

    return CalibrationResult(
        element=element,
        functional=functional,
        pseudo_family=pseudo_family,
        energy_per_atom_ev=energy_per_atom,
        n_atoms_in_reference_cell=n_atoms_out,
        reference_prototype=ref_cell.prototype,
        relaxed_a_ang=relaxed_a_ang,
        run_dir=run_dir,
        extra_metadata={
            "ecutwfc_ry": rendered.ecutwfc_ry,
            "ecutrho_ry": rendered.ecutrho_ry,
            "kpoints": [
                rendered.kpoints.nk1, rendered.kpoints.nk2, rendered.kpoints.nk3,
            ],
            "pseudo_dir": str(registry.pseudo_dir),
            "n_scf_iterations": output.n_scf_iterations,
            "wall_time_seconds": output.wall_time_seconds,
            "seed_a_ang": ref_cell.a_conv_ang,
        },
    )
