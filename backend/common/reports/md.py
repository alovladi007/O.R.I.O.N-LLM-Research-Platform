"""MD-specific Report models + post-analyzers.

Base + four specializations:

- :class:`MDReport` — the common fields (run id, step outputs).
- :class:`MeltingCurveReport` — T, E, D tables + detected T_m.
- :class:`ArrheniusReport` — D(T) + linear fit of ln D vs 1/T.
- :class:`ElasticConstantsReport` — stress-vs-strain + C_ij.

The three specialized analyzers for melting, Arrhenius, and elastic
raise :class:`PendingAnalyzerError` today. Their DAG templates in
:mod:`backend.common.workflows.templates.md` still dispatch the MD
runs; a user who inspects the per-step outputs gets usable physics
numbers. The aggregate analysis lands in Session 4.3b.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PendingAnalyzerError(NotImplementedError):
    """Raised by post-analyzers deferred to Session 4.3b.

    Carries the 4.3b tracker hint so callers don't guess when the
    feature lands.
    """

    def __init__(self, analyzer: str, tracker: str = "Session 4.3b"):
        self.analyzer = analyzer
        self.tracker = tracker
        super().__init__(
            f"{analyzer} is not yet implemented — {tracker}. The workflow "
            "DAG ran correctly; inspect per-step outputs for raw values."
        )


# ---------------------------------------------------------------------------
# Base MD report
# ---------------------------------------------------------------------------


class MDReport(BaseModel):
    """Common fields shared by all MD aggregate reports."""

    model_config = ConfigDict(extra="forbid")

    report_schema: str = "md_report.v1"
    workflow_run_id: str
    name: str
    n_steps: int
    # Per-step outputs, in topological order, as produced by
    # _run_lammps_step: {step_id: {...}}.
    step_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Melting curve
# ---------------------------------------------------------------------------


class MeltingCurveReport(MDReport):
    """Temperature sweep with melting-point detection."""

    report_schema: str = "melting_curve_report.v1"
    temperatures_k: List[float] = Field(default_factory=list)
    total_energies_ev: List[Optional[float]] = Field(default_factory=list)
    msd_final_ang2: List[Optional[float]] = Field(default_factory=list)
    diffusion_coefficients: List[Optional[float]] = Field(default_factory=list)
    detected_melting_point_k: Optional[float] = None
    detection_confidence: Optional[str] = None   # "low" | "medium" | "high"


def detect_melting_point(step_outputs: Dict[str, Dict[str, Any]]) -> MeltingCurveReport:
    """Detect T_m from an MSD-jump + enthalpy-discontinuity heuristic.

    The algorithm (per the Session 4.3 roadmap spec) is:
    1. Sort step outputs by temperature.
    2. Compute dMSD/dT and dE/dT; find the temperature where both
       show a discontinuity.
    3. Cross-check with RDF first-peak height (drops on melting).
    4. Return T_m at the midpoint of the transition with a
       confidence level.

    **Session 4.3a:** raises :class:`PendingAnalyzerError`. The MD
    runs still executed correctly; users can extract the raw tables
    from ``step_outputs`` manually.
    """
    raise PendingAnalyzerError("detect_melting_point")


# ---------------------------------------------------------------------------
# Arrhenius fit
# ---------------------------------------------------------------------------


class ArrheniusReport(MDReport):
    """D(T) at multiple temperatures with Arrhenius fit parameters.

    The Arrhenius model is::

        D(T) = D_0 * exp(-E_a / k_B T)

    Linearized: ``ln D = ln D_0 - (E_a / k_B) * (1/T)``, fit by least
    squares. Reports both E_a in eV and R² of the linear fit.
    """

    report_schema: str = "arrhenius_report.v1"
    temperatures_k: List[float] = Field(default_factory=list)
    diffusion_coefficients: List[float] = Field(default_factory=list)
    activation_energy_ev: Optional[float] = None
    prefactor_ang2_per_ps: Optional[float] = None
    r_squared: Optional[float] = None


def arrhenius_fit(step_outputs: Dict[str, Dict[str, Any]]) -> ArrheniusReport:
    """Fit ``ln D vs 1/T`` to extract activation energy.

    **Session 4.3a:** raises :class:`PendingAnalyzerError`. The MD
    runs still executed; users can pull (T, D) pairs from
    ``step_outputs`` and run scipy.optimize themselves.
    """
    raise PendingAnalyzerError("arrhenius_fit")


# ---------------------------------------------------------------------------
# Elastic constants
# ---------------------------------------------------------------------------


class ElasticConstantsReport(MDReport):
    """Diagonal C_ij from ±ε strain sweep (cubic-system subset).

    The full tensor requires six shear deformations on top of the six
    diagonal ones shipped here. The Session 4.3 MVP (per roadmap) is
    the diagonal-only subset, sufficient for cubic materials' C_11 /
    C_22 / C_33.
    """

    report_schema: str = "elastic_constants_report.v1"
    strains_voigt: List[float] = Field(default_factory=list)
    stresses_gpa: List[Optional[float]] = Field(default_factory=list)
    c11_gpa: Optional[float] = None
    c22_gpa: Optional[float] = None
    c33_gpa: Optional[float] = None


def fit_elastic_constants(step_outputs: Dict[str, Dict[str, Any]]) -> ElasticConstantsReport:
    """Solve σ = C ε for the diagonal elements of the stiffness tensor.

    **Session 4.3a:** raises :class:`PendingAnalyzerError`. Users can
    extract stress values from ``step_outputs[*]['final_thermo']['Press']``
    and do the fit themselves.
    """
    raise PendingAnalyzerError("fit_elastic_constants")
