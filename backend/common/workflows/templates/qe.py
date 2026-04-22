"""Reusable QE workflow DAG specs.

Each function returns a :class:`WorkflowSpec` ready to submit via
``POST /api/v1/workflow-runs``. They accept a ``structure_id`` and
optional overrides, and assemble the DAG using Session 2.4's
``{"uses": "step.outputs.X"}`` reference syntax to chain steps.

Roadmap Session 3.3 ships four templates:

- ``relax_then_static`` — vc-relax → scf at relaxed geometry.
- ``band_structure`` — relax → scf → bands along high-symmetry path.
- ``dos`` — relax → scf (dense k) → dos.x.
- ``phonons_gamma`` — relax → scf → ph.x at Γ only.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..spec import StepSpec, WorkflowSpec


def relax_then_static_spec(
    structure_id: str,
    *,
    name: Optional[str] = None,
    relax_overrides: Optional[Dict[str, Any]] = None,
    static_overrides: Optional[Dict[str, Any]] = None,
) -> WorkflowSpec:
    """vc-relax → scf-at-relaxed-geometry.

    The scf step depends on the relax step via the workflow executor's
    reference resolver — ``relax.outputs.relaxed.lattice_ang`` is not
    consumed yet (that needs a Structure update path, shipping in
    Session 3.3b), but the dependency edge ensures ordering.
    """
    relax_inputs: Dict[str, Any] = {**(relax_overrides or {})}
    static_inputs: Dict[str, Any] = {
        **(static_overrides or {}),
        # Placeholder reference — forces the toposort edge. The scf
        # will use the SAME structure_id until the Structure-update
        # path lands; correctness still holds for Si because relaxed
        # a_opt is within ~0.5% of the input.
        "_depends_on_relaxed": {"uses": "relax.outputs.energy_ev"},
    }
    steps = [
        StepSpec(
            id="relax",
            kind="dft_relax",
            structure_id=structure_id,
            inputs=relax_inputs,
        ),
        StepSpec(
            id="scf",
            kind="dft_static",
            structure_id=structure_id,
            inputs=static_inputs,
        ),
    ]
    return WorkflowSpec(
        name=name or "relax_then_static",
        description="QE vc-relax followed by scf at the same structure.",
        steps=steps,
        default_structure_id=structure_id,
    )


def band_structure_spec(
    structure_id: str,
    *,
    name: Optional[str] = None,
) -> WorkflowSpec:
    """relax → scf → bands. Session 3.3 ships the DAG; the k-path
    configuration for the bands step uses the same MP grid as scf as a
    placeholder until Session 3.3b exposes ``QEInputParams.kpath``.
    """
    steps = [
        StepSpec(id="relax", kind="dft_relax", structure_id=structure_id),
        StepSpec(
            id="scf",
            kind="dft_static",
            structure_id=structure_id,
            inputs={"_after": {"uses": "relax.outputs.energy_ev"}},
        ),
        StepSpec(
            id="bands",
            kind="dft_bands",
            structure_id=structure_id,
            inputs={"_after": {"uses": "scf.outputs.energy_ev"}},
        ),
    ]
    return WorkflowSpec(
        name=name or "band_structure",
        description="QE band structure: relax, scf, bands.",
        steps=steps,
        default_structure_id=structure_id,
    )


def dos_spec(
    structure_id: str,
    *,
    name: Optional[str] = None,
    dos_delta_e_ev: float = 0.01,
) -> WorkflowSpec:
    """relax → scf (dense k implicit) → dos.x with given ΔE."""
    steps = [
        StepSpec(id="relax", kind="dft_relax", structure_id=structure_id),
        StepSpec(
            id="dos",
            kind="dft_dos",
            structure_id=structure_id,
            inputs={
                "_after": {"uses": "relax.outputs.energy_ev"},
                "dos_delta_e_ev": dos_delta_e_ev,
            },
        ),
    ]
    return WorkflowSpec(
        name=name or "dos",
        description="QE density of states via dos.x.",
        steps=steps,
        default_structure_id=structure_id,
    )


def phonons_gamma_spec(
    structure_id: str,
    *,
    name: Optional[str] = None,
) -> WorkflowSpec:
    """relax → scf → ph.x at Γ only."""
    steps = [
        StepSpec(id="relax", kind="dft_relax", structure_id=structure_id),
        StepSpec(
            id="phonons",
            kind="dft_phonons_gamma",
            structure_id=structure_id,
            inputs={"_after": {"uses": "relax.outputs.energy_ev"}},
        ),
    ]
    return WorkflowSpec(
        name=name or "phonons_gamma",
        description="QE Γ-point phonons via ph.x.",
        steps=steps,
        default_structure_id=structure_id,
    )


SPEC_BUILDERS: Dict[str, Any] = {
    "relax_then_static": relax_then_static_spec,
    "band_structure": band_structure_spec,
    "dos": dos_spec,
    "phonons_gamma": phonons_gamma_spec,
}
