"""Tests for Phase 5 / Session 5.3 — sequential-multiscale workflow.

Session 5.3 ships the *scaffolding* for the ``dft_to_md_to_continuum``
workflow; the three deferred Celery tasks raise
:class:`~backend.common.reports.PendingAnalyzerError` rather than
running real physics. These tests therefore cover:

1. DAG shape — three steps, correct kinds, topological order, ``uses``
   references wired between stages.
2. Deferred analyzers — ``extract_dft_elastic_tensor`` and
   ``extract_md_thermal_conductivity`` raise with the expected
   tracker-hint content.
3. Report schema — :class:`MultiscaleReport` round-trips through
   pydantic; :class:`ProvenanceLink` validates.
4. ``build_multiscale_report`` — empty input → ``AnalyzerInputError``;
   partial input → sensible provenance + ``pending_analyzers`` entries.
5. Dispatcher + endpoint registration — every new kind is wired in
   the three canonical tables (`_KIND_TO_TASK`, `_DISPATCH_TASKS`,
   `_BUILTIN_TEMPLATES`), and the ``/templates/multiscale/{name}``
   route exists alongside the QE + MD ones.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


_STRUCT_ID = "00000000-0000-0000-0000-0000000000de"


# ---------------------------------------------------------------------------
# DAG shape
# ---------------------------------------------------------------------------


class TestDftToMdToContinuumDAG:
    def test_three_steps_with_expected_ids_and_kinds(self):
        from backend.common.workflows.templates.multiscale import (
            dft_to_md_to_continuum_spec,
        )

        spec = dft_to_md_to_continuum_spec(_STRUCT_ID)
        by_id = {s.id: s for s in spec.steps}
        assert set(by_id) == {
            "dft_elastic", "md_thermal", "continuum_thermomechanical",
        }
        assert by_id["dft_elastic"].kind == "dft_elastic"
        assert by_id["md_thermal"].kind == "md_green_kubo_thermal"
        assert by_id["continuum_thermomechanical"].kind == "continuum_thermomechanical"

    def test_topological_order_respects_data_flow(self):
        """DFT runs before MD before continuum (implicit via ``uses``)."""
        from backend.common.workflows import toposort_steps
        from backend.common.workflows.templates.multiscale import (
            dft_to_md_to_continuum_spec,
        )

        spec = dft_to_md_to_continuum_spec(_STRUCT_ID)
        order = toposort_steps(spec)
        assert order.index("dft_elastic") < order.index("md_thermal")
        assert order.index("md_thermal") < order.index("continuum_thermomechanical")

    def test_continuum_references_both_upstream_outputs(self):
        """The continuum step's inputs ``uses`` both upstream outputs."""
        from backend.common.workflows.templates.multiscale import (
            dft_to_md_to_continuum_spec,
        )

        spec = dft_to_md_to_continuum_spec(_STRUCT_ID)
        cont = next(s for s in spec.steps if s.id == "continuum_thermomechanical")
        assert cont.inputs["elastic_tensor_voigt_gpa"] == {
            "uses": "dft_elastic.outputs.elastic_tensor_voigt_gpa"
        }
        assert cont.inputs["thermal_conductivity_w_per_m_k"] == {
            "uses": "md_thermal.outputs.thermal_conductivity_w_per_m_k"
        }

    def test_default_parameters_reasonable(self):
        from backend.common.workflows.templates.multiscale import (
            dft_to_md_to_continuum_spec,
        )

        spec = dft_to_md_to_continuum_spec(_STRUCT_ID)
        dft = next(s for s in spec.steps if s.id == "dft_elastic")
        md = next(s for s in spec.steps if s.id == "md_thermal")
        cont = next(s for s in spec.steps if s.id == "continuum_thermomechanical")
        assert dft.inputs["strain_magnitude"] == 0.005
        assert md.inputs["temperature_k"] == 300.0
        assert md.inputs["duration_ps"] == 200.0
        # Continuum geometry defaults to a 1 mm cube.
        assert cont.inputs["length_x_m"] == 1.0e-3

    def test_spec_builders_table_contains_one_template(self):
        from backend.common.workflows.templates.multiscale import SPEC_BUILDERS

        assert set(SPEC_BUILDERS) == {"dft_to_md_to_continuum"}


# ---------------------------------------------------------------------------
# Deferred analyzers
# ---------------------------------------------------------------------------


class TestDeferredAnalyzers:
    def test_extract_dft_elastic_tensor_raises_with_phase_8_tracker(self):
        from backend.common.reports import (
            PendingAnalyzerError, extract_dft_elastic_tensor,
        )

        with pytest.raises(PendingAnalyzerError) as exc_info:
            extract_dft_elastic_tensor({})
        assert "Phase 8" in str(exc_info.value)
        assert exc_info.value.analyzer == "extract_dft_elastic_tensor"

    def test_extract_md_thermal_conductivity_raises_with_phase_4_tracker(self):
        from backend.common.reports import (
            PendingAnalyzerError, extract_md_thermal_conductivity,
        )

        with pytest.raises(PendingAnalyzerError) as exc_info:
            extract_md_thermal_conductivity({})
        assert "Phase 4" in str(exc_info.value)
        # The tracker-hint text also calls out Green-Kubo so future
        # readers don't have to grep the roadmap.
        assert "Green-Kubo" in str(exc_info.value)

    def test_both_remain_NotImplementedError_subclasses(self):
        from backend.common.reports import PendingAnalyzerError

        assert issubclass(PendingAnalyzerError, NotImplementedError)


# ---------------------------------------------------------------------------
# Report schema + builder
# ---------------------------------------------------------------------------


class TestMultiscaleReportSchema:
    def test_minimal_report_validates(self):
        from backend.common.reports import MultiscaleReport

        r = MultiscaleReport(workflow_run_id="abc", name="smoke")
        assert r.report_schema == "multiscale_report.v1"
        assert r.elastic_tensor_voigt_gpa is None
        assert r.thermal_conductivity_w_per_m_k is None
        assert r.pending_analyzers == []

    def test_extra_fields_forbidden(self):
        from backend.common.reports import MultiscaleReport

        with pytest.raises(Exception):
            MultiscaleReport(workflow_run_id="abc", bogus_field=1)

    def test_provenance_link_validates(self):
        from backend.common.reports import ProvenanceLink

        link = ProvenanceLink(
            job_id="job-123", step_id="dft_elastic", kind="dft_elastic",
            workflow_run_id="wfr-1",
        )
        assert link.job_id == "job-123"
        # Round-trip through JSON.
        dumped = link.model_dump()
        restored = ProvenanceLink(**dumped)
        assert restored == link


class TestBuildMultiscaleReport:
    def test_empty_step_outputs_raises(self):
        from backend.common.reports import (
            AnalyzerInputError, build_multiscale_report,
        )

        with pytest.raises(AnalyzerInputError):
            build_multiscale_report({})

    def test_dft_only_outputs_yield_pending_for_other_two(self):
        from backend.common.reports import build_multiscale_report

        r = build_multiscale_report({
            "dft_elastic": {"job_id": "job-dft", "kind": "dft_elastic"},
        })
        # Elastic tensor is still None because the analyzer is deferred,
        # but the provenance link is populated.
        assert r.elastic_tensor_voigt_gpa is None
        assert r.elastic_tensor_provenance is not None
        assert r.elastic_tensor_provenance.job_id == "job-dft"
        assert r.thermal_conductivity_provenance is None
        # pending_analyzers lists the deferred DFT analyzer.
        assert any("extract_dft_elastic_tensor" in m for m in r.pending_analyzers)

    def test_all_three_outputs_yield_full_provenance(self):
        from backend.common.reports import build_multiscale_report

        r = build_multiscale_report({
            "dft_elastic": {"job_id": "job-dft", "kind": "dft_elastic"},
            "md_thermal": {
                "job_id": "job-md", "kind": "md_green_kubo_thermal",
                "temperature_k": 300.0,
            },
            "continuum_thermomechanical": {
                "job_id": "job-cont", "kind": "continuum_thermomechanical",
                "pending": "upstream analyzers deferred",
            },
        })
        assert r.elastic_tensor_provenance.job_id == "job-dft"
        assert r.thermal_conductivity_provenance.job_id == "job-md"
        assert r.thermal_conductivity_temperature_k == 300.0
        assert r.continuum_provenance.job_id == "job-cont"
        # Two deferred analyzers + one continuum-pending marker.
        assert len(r.pending_analyzers) == 3

    def test_missing_job_id_yields_no_provenance(self):
        """An outputs dict without ``job_id`` produces no ProvenanceLink."""
        from backend.common.reports import build_multiscale_report

        r = build_multiscale_report({
            "dft_elastic": {"kind": "dft_elastic"},  # no job_id
        })
        assert r.elastic_tensor_provenance is None


# ---------------------------------------------------------------------------
# Dispatcher + endpoint registration
# ---------------------------------------------------------------------------


class TestDispatcherWiring:
    def test_multiscale_kinds_in_workflow_dispatcher(self):
        from backend.common.workflows.celery_dispatcher import _KIND_TO_TASK

        for kind in (
            "dft_elastic",
            "md_green_kubo_thermal",
            "continuum_thermomechanical",
        ):
            assert kind in _KIND_TO_TASK, (
                f"{kind} not registered in _KIND_TO_TASK"
            )

    def test_multiscale_kinds_in_jobs_dispatch(self):
        from src.api.routers.jobs import _BUILTIN_TEMPLATES, _DISPATCH_TASKS

        for kind in (
            "dft_elastic",
            "md_green_kubo_thermal",
            "continuum_thermomechanical",
        ):
            assert kind in _DISPATCH_TASKS
            assert kind in _BUILTIN_TEMPLATES

    def test_dispatch_task_names_match_across_tables(self):
        """The two dispatch tables must agree on task names."""
        from backend.common.workflows.celery_dispatcher import _KIND_TO_TASK
        from src.api.routers.jobs import _DISPATCH_TASKS

        for kind in (
            "dft_elastic",
            "md_green_kubo_thermal",
            "continuum_thermomechanical",
        ):
            assert _KIND_TO_TASK[kind] == _DISPATCH_TASKS[kind]

    def test_multiscale_template_endpoint_registered(self):
        from src.api.app import app

        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert (
            "/api/v1/workflow-runs/templates/multiscale/{template_name}" in paths
        )

    def test_existing_endpoints_untouched(self):
        """Adding the multiscale endpoint didn't remove QE / MD ones."""
        from src.api.app import app

        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/api/v1/workflow-runs/templates/qe/{template_name}" in paths
        assert "/api/v1/workflow-runs/templates/md/{template_name}" in paths
