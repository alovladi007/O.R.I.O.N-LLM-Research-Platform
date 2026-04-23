"""Report objects for workflow-run aggregate analysis.

Phase 4 / Session 4.3 introduces this package. A Report is the
frontend-renderable JSON that summarizes a completed workflow run:
melting point, diffusion activation energy, elastic tensor, etc.

Design
------

- :class:`MDReport` is the base pydantic model with the common fields
  (workflow_run_id, per-step outputs, timestamps). Specialized
  reports extend it with domain-specific fields.
- Each specialized analyzer (e.g. :func:`detect_melting_point`,
  :func:`arrhenius_fit`, :func:`fit_elastic_constants`) takes a list
  of per-step output dicts and returns a populated specialized
  report.
- Three of the four Session 4.3 specialized analyzers raise
  :class:`NotImplementedError` with a clear 4.3b pointer — the DAG
  templates (:mod:`backend.common.workflows.templates.md`) dispatch
  today; the aggregate analysis lands in the follow-up.
"""

from .md import (
    ArrheniusReport,
    ElasticConstantsReport,
    MDReport,
    MeltingCurveReport,
    PendingAnalyzerError,
    arrhenius_fit,
    detect_melting_point,
    fit_elastic_constants,
)

__all__ = [
    "ArrheniusReport",
    "ElasticConstantsReport",
    "MDReport",
    "MeltingCurveReport",
    "PendingAnalyzerError",
    "arrhenius_fit",
    "detect_melting_point",
    "fit_elastic_constants",
]
