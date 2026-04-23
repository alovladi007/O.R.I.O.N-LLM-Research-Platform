"""Multiscale (DFT → MD → continuum) report + deferred analyzers.

Session 5.3 ships the *scaffolding*: workflow DAG, provenance thread,
report schema, plus two deferred analyzers that raise
:class:`PendingAnalyzerError`. The downstream data (DFT elastic
tensor, MD thermal conductivity via Green-Kubo) both come from
analyzers that other phases haven't shipped yet; 5.3 locks the
interface contract so those phases can fill in without reshaping
the pipeline.

The pattern mirrors Session 4.3a (DAG ships, post-analyzers raise),
not 4.3b (implementations arrive). A 5.3b follow-up will arrive
naturally once Phase 8 (DFT elastic) ships — the MD Green-Kubo
half was already flagged as deferred in the Phase 4.2 report.

Public API
----------

- :class:`ProvenanceLink` — (job_id, step_id, kind) triple linking a
  material-parameter value back to the upstream job that produced it.
- :class:`MultiscaleReport` — ties DFT + MD + continuum outputs
  together with full provenance.
- :func:`extract_dft_elastic_tensor` — placeholder; raises.
- :func:`extract_md_thermal_conductivity` — placeholder; raises.
- :func:`build_multiscale_report` — assembles a report from
  ``step_outputs`` (loud on empty input like the Session 4.3b
  analyzers).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .md import AnalyzerInputError, PendingAnalyzerError


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class ProvenanceLink(BaseModel):
    """Immutable pointer from a derived material value back to its source job.

    Phase 12 introduces a proper provenance graph with timestamps +
    content hashes. Session 5.3 ships the minimum three fields the
    multiscale workflow actually needs so the acceptance test can
    trace a continuum material parameter back to the specific DFT /
    MD job that produced it.

    Fields
    ------
    job_id
        The Celery job UUID (string-form) that produced the value.
    step_id
        The workflow-step id in the parent workflow run — e.g.
        ``"dft_elastic"`` or ``"md_thermal"``.
    kind
        The engine kind (``"dft_elastic"`` / ``"md_green_kubo_thermal"``
        / ``"continuum_thermomechanical"``) so consumers don't have
        to consult the step spec to know what ran.
    workflow_run_id
        The workflow-run UUID that ``step_id`` lives in. Optional —
        unit tests constructing a report inline can omit it.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    step_id: str
    kind: str
    workflow_run_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class MultiscaleReport(BaseModel):
    """DFT → MD → continuum coupling result with provenance.

    All three sets of material parameters have a matching
    :class:`ProvenanceLink`. In Session 5.3 the DFT and MD lists
    are filled in as ``None`` (the analyzers that produce them are
    deferred); the continuum field follows suit because its inputs
    are missing.

    Users who inspect the report today get a loud signal that the
    pipeline *ran* — DAG dispatched, steps completed in some form —
    but the physics numbers they'd feed into a design decision are
    all ``None`` with a ``pending_analyzers`` list that names the
    specific follow-ups.
    """

    model_config = ConfigDict(extra="forbid")

    report_schema: str = "multiscale_report.v1"
    workflow_run_id: str = ""
    name: str = "dft_to_md_to_continuum"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # DFT outputs (Phase 8 dependency)
    elastic_tensor_voigt_gpa: Optional[List[List[float]]] = None
    elastic_tensor_provenance: Optional[ProvenanceLink] = None

    # MD outputs (Phase 4 Green-Kubo follow-up)
    thermal_conductivity_w_per_m_k: Optional[float] = None
    thermal_conductivity_temperature_k: Optional[float] = None
    thermal_conductivity_provenance: Optional[ProvenanceLink] = None

    # Continuum outputs (depends on both of the above)
    continuum_max_displacement_m: Optional[float] = None
    continuum_max_von_mises_pa: Optional[float] = None
    continuum_max_temperature_k: Optional[float] = None
    continuum_provenance: Optional[ProvenanceLink] = None

    # Raw step_outputs, for downstream debugging / custom analysis.
    step_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Explicit list of analyzers that are deferred — makes the
    # "what's missing?" question answerable from the report itself
    # rather than requiring readers to know the roadmap.
    pending_analyzers: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Deferred analyzers
# ---------------------------------------------------------------------------


def extract_dft_elastic_tensor(step_outputs: Dict[str, Any]) -> List[List[float]]:
    """Read a 6×6 Voigt elastic tensor (GPa) from a ``dft_elastic`` step.

    **Session 5.3:** raises :class:`PendingAnalyzerError`. The DFT
    elastic-tensor computation (6 ±strain DFT runs → stress-vs-strain
    fit → C_ij) lives in Phase 8. The signature + return shape are
    locked here so the multiscale template doesn't shift when
    Phase 8 lands.
    """
    raise PendingAnalyzerError(
        "extract_dft_elastic_tensor",
        tracker="Phase 8 — DFT elastic tensor via QE ±ε strain runs",
    )


def extract_md_thermal_conductivity(step_outputs: Dict[str, Any]) -> float:
    """Read thermal conductivity κ (W/m·K) from an MD Green-Kubo step.

    **Session 5.3:** raises :class:`PendingAnalyzerError`. The
    Green-Kubo heat-flux autocorrelation + time integral lives in a
    Phase 4 follow-up (flagged as deferred in the Phase 4.2 report
    alongside VACF→vDOS).
    """
    raise PendingAnalyzerError(
        "extract_md_thermal_conductivity",
        tracker="Phase 4 follow-up — Green-Kubo κ from heat-flux ACF",
    )


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


# Step-id conventions used by the multiscale template. Importing this
# constant from here (rather than redefining it in the template
# module) keeps the analyzer ↔ template contract in one place.
STEP_ID_DFT_ELASTIC = "dft_elastic"
STEP_ID_MD_THERMAL = "md_thermal"
STEP_ID_CONTINUUM = "continuum_thermomechanical"


def build_multiscale_report(
    step_outputs: Dict[str, Dict[str, Any]],
    *,
    workflow_run_id: str = "",
) -> MultiscaleReport:
    """Assemble a :class:`MultiscaleReport` from a run's ``step_outputs``.

    Policy
    ------
    - Empty ``step_outputs`` → :class:`AnalyzerInputError` (mirrors
      the Session 4.3b analyzer contract — loud on structural failure).
    - Missing step ids (e.g. only DFT completed) → the corresponding
      fields stay ``None`` and the ``pending_analyzers`` list grows.
    - Deferred analyzers (elastic tensor, κ) are called wrapped in
      ``try / except PendingAnalyzerError`` so the report still
      returns; the caller sees ``pending_analyzers`` spelling out
      what's missing.
    """
    if not step_outputs:
        raise AnalyzerInputError("build_multiscale_report: step_outputs is empty")

    report = MultiscaleReport(
        workflow_run_id=workflow_run_id,
        step_outputs=dict(step_outputs),
    )

    # DFT elastic tensor — deferred.
    if STEP_ID_DFT_ELASTIC in step_outputs:
        dft_out = step_outputs[STEP_ID_DFT_ELASTIC]
        try:
            report.elastic_tensor_voigt_gpa = extract_dft_elastic_tensor(dft_out)
        except PendingAnalyzerError as exc:
            report.pending_analyzers.append(str(exc))
        report.elastic_tensor_provenance = _provenance_from_outputs(
            dft_out, STEP_ID_DFT_ELASTIC, workflow_run_id,
        )

    # MD thermal conductivity — deferred.
    if STEP_ID_MD_THERMAL in step_outputs:
        md_out = step_outputs[STEP_ID_MD_THERMAL]
        try:
            report.thermal_conductivity_w_per_m_k = (
                extract_md_thermal_conductivity(md_out)
            )
        except PendingAnalyzerError as exc:
            report.pending_analyzers.append(str(exc))
        report.thermal_conductivity_temperature_k = md_out.get("temperature_k")
        report.thermal_conductivity_provenance = _provenance_from_outputs(
            md_out, STEP_ID_MD_THERMAL, workflow_run_id,
        )

    # Continuum — depends on the above two. If both upstream analyzers
    # are pending, the continuum step's outputs are also pending (the
    # Celery task raises on dispatch). We still emit a provenance
    # link + a pending marker.
    if STEP_ID_CONTINUUM in step_outputs:
        cont_out = step_outputs[STEP_ID_CONTINUUM]
        report.continuum_max_displacement_m = cont_out.get("max_displacement_m")
        report.continuum_max_von_mises_pa = cont_out.get("max_von_mises_pa")
        report.continuum_max_temperature_k = cont_out.get("max_temperature_k")
        report.continuum_provenance = _provenance_from_outputs(
            cont_out, STEP_ID_CONTINUUM, workflow_run_id,
        )
        if cont_out.get("pending"):
            report.pending_analyzers.append(
                f"continuum_thermomechanical pending: {cont_out['pending']}"
            )

    return report


def _provenance_from_outputs(
    step_out: Dict[str, Any], step_id: str, workflow_run_id: str,
) -> Optional[ProvenanceLink]:
    """Return a :class:`ProvenanceLink` if the step outputs carry a job_id.

    The MD / continuum runners stamp ``job_id`` + ``kind`` into their
    output dicts (see ``src.worker.tasks._run_lammps_step``); for
    deferred tasks that raise at dispatch there's no job_id yet, so
    we return ``None``.
    """
    job_id = step_out.get("job_id")
    kind = step_out.get("kind")
    if not job_id or not kind:
        return None
    return ProvenanceLink(
        job_id=str(job_id),
        step_id=step_id,
        kind=str(kind),
        workflow_run_id=workflow_run_id or None,
    )
