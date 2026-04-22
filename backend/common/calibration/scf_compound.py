"""One-shot SCF on a compound structure — helper for cross-validation.

The calibration CLI runs vc-relax on *elemental* reference cells; the
cross-validation pipeline also needs plain SCF on *compounds* (at MP's
relaxed geometry) to compute formation energies. This module is the
thin wrapper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CompoundSCFResult:
    """SCF result for one compound."""

    formula: str
    species: List[str]
    total_energy_ev: float
    n_atoms: int
    run_dir: Path
    extra_metadata: Dict[str, Any]


def run_compound_scf(
    qe_struct: Dict[str, Any],
    *,
    formula: str,
    qe_executable: str,
    pseudo_dir: Path | str,
    cpus: int = 1,
    walltime_minutes: int = 30,
    kpoints: Optional[tuple[int, int, int]] = None,
    spin_polarized: bool = False,
    tot_magnetization: Optional[float] = None,
    run_dir_parent: Optional[Path] = None,
) -> CompoundSCFResult:
    """Run pw.x in SCF mode on *qe_struct* and return the total energy.

    The structure is taken verbatim (no vc-relax) — the caller supplies
    MP's relaxed geometry or pre-relaxes via ``run_element_calibration``.
    """
    from backend.common.engines.qe_input import (
        PseudopotentialRegistry,
        QEInputParams,
        generate_pw_input,
    )
    from backend.common.engines.qe_run import run_pw
    from backend.common.workers import build_run_dir

    registry = PseudopotentialRegistry(Path(pseudo_dir).expanduser())
    qe_params = QEInputParams(
        prefix=f"cmp_{formula.lower().replace(' ', '')}",
        calculation="scf",
        occupations="smearing",
        smearing="gauss",
        degauss=0.01,
        kpoints=kpoints,
        spin_polarized=spin_polarized,
        tot_magnetization=tot_magnetization,
    )
    rendered = generate_pw_input(qe_struct, qe_params, registry)
    run_dir = build_run_dir(f"compound-{formula.lower()}", parent=run_dir_parent)
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
            f"Compound SCF for {formula} failed at stage={pw_result.stage}: "
            f"{pw_result.error_message}"
        )
    output = pw_result.output
    if output is None or output.energy is None:
        raise RuntimeError(
            f"Compound SCF for {formula}: no total energy in output."
        )
    return CompoundSCFResult(
        formula=formula,
        species=list(qe_struct["species"]),
        total_energy_ev=output.energy.total_ev,
        n_atoms=output.n_atoms or len(qe_struct["species"]),
        run_dir=run_dir,
        extra_metadata={
            "ecutwfc_ry": rendered.ecutwfc_ry,
            "ecutrho_ry": rendered.ecutrho_ry,
            "kpoints": [
                rendered.kpoints.nk1, rendered.kpoints.nk2, rendered.kpoints.nk3,
            ],
            "n_scf_iterations": output.n_scf_iterations,
            "wall_time_seconds": output.wall_time_seconds,
        },
    )
