"""Sequential-multiscale workflow templates (Phase 5 / Session 5.3).

One template today: ``dft_to_md_to_continuum``. Three steps, each
dependent on the previous via ``{"uses": "step.outputs.X"}``
references, so the Session 2.4 workflow tick orders them correctly.

Session 5.3 is deliberately scaffolded, not fully live. Two of the
three steps run kinds that raise :class:`PendingAnalyzerError` at
dispatch:

- ``dft_elastic`` — Phase 8 will ship the 6 ±strain DFT runs + C_ij
  fit. Until then the task raises.
- ``md_green_kubo_thermal`` — Green-Kubo thermal conductivity is a
  deferred Phase-4 follow-up (flagged in the 4.2 report alongside
  VACF → vDOS). Until then the task raises.

The third step ``continuum_thermomechanical`` also raises today
because its inputs come from the pending upstream tasks — but the
pattern stays honest: the DAG is submittable, the failures are loud
at execution time, the provenance wiring + report schema let a
caller see exactly what's missing without reading the roadmap.

Pattern inheritance
-------------------

This is the Session 4.3a pattern, not 4.3b:

- 4.3a: DAG ships, aggregate analyzer raises
  ``PendingAnalyzerError``. User submitting a DAG today gets real
  MD per-step outputs; the aggregate "melting point" answer doesn't
  land until 4.3b.
- 5.3: DAG ships, the *step kinds themselves* raise
  ``PendingAnalyzerError``. User submitting gets a run that fails
  loudly at the first step with a tracker-hint pointing at Phase 8.

The interface contract (step IDs, input shapes, report fields) is
locked so 5.3b / Phase 8 can fill in without rewiring consumers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend.common.reports.multiscale import (
    STEP_ID_CONTINUUM,
    STEP_ID_DFT_ELASTIC,
    STEP_ID_MD_THERMAL,
)

from ..spec import StepSpec, WorkflowSpec


def dft_to_md_to_continuum_spec(
    structure_id: str,
    *,
    name: Optional[str] = None,
    # Phase 8 will compute the elastic tensor from 6 ±ε DFT runs at
    # this magnitude. Phase 3 relax should precede — we skip the
    # explicit relax step in Session 5.3 (not in scope), Phase 8
    # will add it as a sub-DAG.
    dft_strain_magnitude: float = 0.005,
    # MD temperature for the Green-Kubo conductivity calculation.
    md_temperature_k: float = 300.0,
    md_duration_ps: float = 200.0,
    # Continuum geometry — the roadmap's acceptance case uses a
    # Si-like box. Defaults are placeholder numbers; callers
    # override as needed.
    continuum_length_x_m: float = 1.0e-3,
    continuum_length_y_m: float = 1.0e-3,
    continuum_length_z_m: float = 1.0e-3,
) -> WorkflowSpec:
    """DFT → MD → continuum coupling DAG (sequential multiscale).

    Step layout
    -----------

    1. ``dft_elastic`` (kind ``dft_elastic``) — computes the 6×6
       Voigt C_ij tensor via ±ε strain DFT runs. **Pending Phase 8.**
    2. ``md_thermal`` (kind ``md_green_kubo_thermal``) — computes
       κ(T) via Green-Kubo. **Pending Phase 4 follow-up.**
       Depends on the DFT step only for ordering (Phase 8
       relaxed geometry is a natural MD starting point); the MD
       run doesn't consume the elastic tensor.
    3. ``continuum_thermomechanical`` (kind
       ``continuum_thermomechanical``) — thermoelastic FEM run
       using C_ij from step 1 and κ from step 2.

    Parameters are documented in the caller's API schema; defaults
    here favour silicon at 300 K with a 1 mm cube continuum domain.
    """
    dft_step = StepSpec(
        id=STEP_ID_DFT_ELASTIC,
        kind="dft_elastic",
        structure_id=structure_id,
        inputs={
            "strain_magnitude": dft_strain_magnitude,
        },
    )
    md_step = StepSpec(
        id=STEP_ID_MD_THERMAL,
        kind="md_green_kubo_thermal",
        structure_id=structure_id,
        inputs={
            "temperature_k": md_temperature_k,
            "duration_ps": md_duration_ps,
            # An ordering edge only — the MD run doesn't numerically
            # consume the elastic tensor. A future revision could
            # feed it in for cross-checks.
            "_after_dft": {
                "uses": f"{STEP_ID_DFT_ELASTIC}.outputs.status",
            },
        },
    )
    continuum_step = StepSpec(
        id=STEP_ID_CONTINUUM,
        kind="continuum_thermomechanical",
        structure_id=structure_id,
        inputs={
            "length_x_m": continuum_length_x_m,
            "length_y_m": continuum_length_y_m,
            "length_z_m": continuum_length_z_m,
            # Material parameters are resolved at dispatch time from
            # the ``uses`` references. The continuum Celery task
            # will fail loudly today (PendingAnalyzerError from
            # upstream) but the reference wiring is in place for
            # when 5.3b / Phase 8 fills in the numbers.
            "elastic_tensor_voigt_gpa": {
                "uses": f"{STEP_ID_DFT_ELASTIC}.outputs.elastic_tensor_voigt_gpa",
            },
            "thermal_conductivity_w_per_m_k": {
                "uses": f"{STEP_ID_MD_THERMAL}.outputs.thermal_conductivity_w_per_m_k",
            },
        },
    )

    return WorkflowSpec(
        name=name or "dft_to_md_to_continuum",
        description=(
            "Sequential multiscale DAG: DFT elastic tensor → MD Green-Kubo "
            "thermal conductivity → thermomechanical continuum FEM. "
            "Session 5.3 scaffolds the DAG + report schema + provenance; "
            "DFT step is pending Phase 8, MD step is pending the Phase 4 "
            "Green-Kubo follow-up. The continuum step inherits pending "
            "status until both upstream analyzers land."
        ),
        steps=[dft_step, md_step, continuum_step],
        default_structure_id=structure_id,
    )


SPEC_BUILDERS: Dict[str, Any] = {
    "dft_to_md_to_continuum": dft_to_md_to_continuum_spec,
}
