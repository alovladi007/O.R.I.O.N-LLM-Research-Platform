"""Tests for Phase 4 / Session 4.3 — MD workflow templates + reports.

Coverage:

1. ``equilibrate_nvt_then_nve_spec`` — DAG shape, step order,
   parameter wiring.
2. Three deferred templates — ``melting_curve_spec``,
   ``diffusivity_vs_t_spec``, ``elastic_constants_via_strain_spec``:
   the DAG itself is correct (toposort, foreach expansion, step
   counts) even though the post-analyzer is deferred.
3. Three deferred analyzers raise :class:`PendingAnalyzerError` with
   the 4.3b pointer — the failure mode is loud, not silent.
4. ``MeltingCurveReport`` / ``ArrheniusReport`` / ``ElasticConstantsReport``
   pydantic shape validates.
5. Dispatcher registration — md_nvt / md_nve / md_npt all wired in
   the three canonical places (jobs router, builtin templates,
   workflow executor).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


_STRUCT_ID = "00000000-0000-0000-0000-000000000abc"


# ---------------------------------------------------------------------------
# equilibrate_nvt_then_nve — full template
# ---------------------------------------------------------------------------


class TestEquilibrateNvtThenNve:
    def test_two_step_dag(self):
        from backend.common.workflows import toposort_steps
        from backend.common.workflows.templates.md import (
            equilibrate_nvt_then_nve_spec,
        )

        spec = equilibrate_nvt_then_nve_spec(_STRUCT_ID)
        assert spec.name == "equilibrate_nvt_then_nve"
        assert len(spec.steps) == 2
        assert {s.id for s in spec.steps} == {"equilibrate", "production"}
        order = toposort_steps(spec)
        assert order == ["equilibrate", "production"]

    def test_kinds(self):
        from backend.common.workflows.templates.md import (
            equilibrate_nvt_then_nve_spec,
        )

        spec = equilibrate_nvt_then_nve_spec(_STRUCT_ID)
        eq = next(s for s in spec.steps if s.id == "equilibrate")
        prod = next(s for s in spec.steps if s.id == "production")
        assert eq.kind == "md_nvt"
        assert prod.kind == "md_nve"

    def test_temperature_propagated(self):
        from backend.common.workflows.templates.md import (
            equilibrate_nvt_then_nve_spec,
        )

        spec = equilibrate_nvt_then_nve_spec(_STRUCT_ID, temperature_k=500.0)
        for step in spec.steps:
            assert step.inputs["temperature_k"] == 500.0

    def test_durations_propagated(self):
        from backend.common.workflows.templates.md import (
            equilibrate_nvt_then_nve_spec,
        )

        spec = equilibrate_nvt_then_nve_spec(
            _STRUCT_ID, equilibrate_ps=15.0, production_ps=30.0,
        )
        eq = next(s for s in spec.steps if s.id == "equilibrate")
        prod = next(s for s in spec.steps if s.id == "production")
        assert eq.inputs["duration_ps"] == 15.0
        assert prod.inputs["duration_ps"] == 30.0

    def test_production_dumps_more_densely_than_equilibrate(self):
        from backend.common.workflows.templates.md import (
            equilibrate_nvt_then_nve_spec,
        )

        spec = equilibrate_nvt_then_nve_spec(_STRUCT_ID)
        eq = next(s for s in spec.steps if s.id == "equilibrate")
        prod = next(s for s in spec.steps if s.id == "production")
        assert prod.inputs["dump_every"] < eq.inputs["dump_every"]


# ---------------------------------------------------------------------------
# Deferred-analyzer templates: DAGs ship, analyzers don't
# ---------------------------------------------------------------------------


class TestMeltingCurveSpec:
    def test_temperature_sweep_expands_via_foreach(self):
        from backend.common.workflows import expand_foreach
        from backend.common.workflows.templates.md import melting_curve_spec

        spec = melting_curve_spec(
            _STRUCT_ID, temperatures_k=(800, 900, 1000),
        )
        assert len(spec.steps) == 1  # before expansion
        expanded = expand_foreach(spec)
        assert len(expanded.steps) == 3
        # Each child carries one of the temperatures.
        temps = sorted(s.inputs["temperature_k"] for s in expanded.steps)
        assert temps == [800.0, 900.0, 1000.0]

    def test_default_uses_npt(self):
        from backend.common.workflows.templates.md import melting_curve_spec

        spec = melting_curve_spec(_STRUCT_ID)
        assert spec.steps[0].kind == "md_npt"


class TestDiffusivityVsTSpec:
    def test_arrhenius_sweep_uses_nve(self):
        from backend.common.workflows.templates.md import diffusivity_vs_t_spec

        spec = diffusivity_vs_t_spec(_STRUCT_ID, temperatures_k=(400, 500))
        assert spec.steps[0].kind == "md_nve"

    def test_dense_dumps_for_msd(self):
        from backend.common.workflows.templates.md import diffusivity_vs_t_spec

        spec = diffusivity_vs_t_spec(_STRUCT_ID)
        # 100 == relatively dense; needed for accurate D from MSD.
        assert spec.steps[0].inputs["dump_every"] == 100


class TestElasticConstantsSpec:
    def test_six_diagonal_strain_steps(self):
        from backend.common.workflows.templates.md import (
            elastic_constants_via_strain_spec,
        )

        spec = elastic_constants_via_strain_spec(_STRUCT_ID)
        # ±ε on each of three diagonal axes = 6 steps total.
        assert len(spec.steps) == 6
        # 4.3b field names — same semantics, renamed to match the
        # fit_elastic_constants analyzer's contract.
        for step in spec.steps:
            assert "strain_voigt" in step.inputs
            assert "strain_value" in step.inputs

    def test_strain_values_balanced(self):
        from backend.common.workflows.templates.md import (
            elastic_constants_via_strain_spec,
        )

        spec = elastic_constants_via_strain_spec(_STRUCT_ID)
        strains = sorted(s.inputs["strain_value"] for s in spec.steps)
        # Three negative, three positive at the same magnitudes.
        assert strains == [-0.005, -0.005, -0.005, 0.005, 0.005, 0.005]

    def test_emits_change_box_extra_commands(self):
        """4.3b: each step carries ``change_box`` lines that apply its strain."""
        from backend.common.workflows.templates.md import (
            elastic_constants_via_strain_spec,
        )

        spec = elastic_constants_via_strain_spec(_STRUCT_ID)
        for step in spec.steps:
            extra = step.inputs.get("extra_commands")
            assert extra is not None and isinstance(extra, list)
            assert any("change_box" in line for line in extra)

    def test_stress_thermo_columns_requested(self):
        """4.3b: analyzer needs per-component stress → Pxx/Pyy/Pzz in thermo."""
        from backend.common.engines.lammps_input import THERMO_COLUMNS_STRESS
        from backend.common.workflows.templates.md import (
            elastic_constants_via_strain_spec,
        )

        spec = elastic_constants_via_strain_spec(_STRUCT_ID)
        for step in spec.steps:
            assert step.inputs["thermo_columns"] == THERMO_COLUMNS_STRESS


# ---------------------------------------------------------------------------
# Reports: pydantic shape + deferred-analyzer error contract
# ---------------------------------------------------------------------------


class TestMDReports:
    def test_md_report_base_validates(self):
        from backend.common.reports import MDReport

        r = MDReport(workflow_run_id="abc", name="x", n_steps=2)
        assert r.report_schema == "md_report.v1"
        assert r.step_outputs == {}

    def test_melting_curve_report_validates(self):
        from backend.common.reports import MeltingCurveReport

        r = MeltingCurveReport(
            workflow_run_id="abc", name="x", n_steps=3,
            temperatures_k=[800, 900, 1000],
            total_energies_ev=[None, None, None],
        )
        assert r.report_schema == "melting_curve_report.v1"
        assert r.detected_melting_point_k is None

    def test_arrhenius_report_validates(self):
        from backend.common.reports import ArrheniusReport

        r = ArrheniusReport(
            workflow_run_id="abc", name="x", n_steps=5,
            temperatures_k=[400, 500, 600, 700, 800],
            diffusion_coefficients=[0.01, 0.05, 0.1, 0.2, 0.4],
        )
        assert r.report_schema == "arrhenius_report.v1"
        assert r.activation_energy_ev is None  # not yet fit

    def test_elastic_report_validates(self):
        from backend.common.reports import ElasticConstantsReport

        r = ElasticConstantsReport(
            workflow_run_id="abc", name="x", n_steps=6,
        )
        assert r.report_schema == "elastic_constants_report.v1"


class TestAnalyzerStructuralFailures:
    """Session 4.3a raised PendingAnalyzerError from the three aggregate
    analyzers. Session 4.3b implements them — the raise-on-empty
    contract moves to AnalyzerInputError, which signals *bad input*
    rather than *unimplemented*. PendingAnalyzerError is still
    exported for future deferred analyzers (VACF→vDOS, Green-Kubo).
    """

    def test_detect_melting_point_rejects_empty(self):
        from backend.common.reports import AnalyzerInputError, detect_melting_point

        with pytest.raises(AnalyzerInputError):
            detect_melting_point({})

    def test_arrhenius_fit_rejects_empty(self):
        from backend.common.reports import AnalyzerInputError, arrhenius_fit

        with pytest.raises(AnalyzerInputError):
            arrhenius_fit({})

    def test_fit_elastic_constants_rejects_empty(self):
        from backend.common.reports import AnalyzerInputError, fit_elastic_constants

        with pytest.raises(AnalyzerInputError):
            fit_elastic_constants({})

    def test_pending_analyzer_error_still_exported(self):
        """Future-deferred analyzers re-use this sentinel; make sure the
        export didn't disappear when 4.3b implemented the three 4.3a
        stubs."""
        from backend.common.reports import PendingAnalyzerError

        assert issubclass(PendingAnalyzerError, NotImplementedError)
        # Round-trip: the message still carries a tracker hint.
        err = PendingAnalyzerError("vacf_vdos", tracker="Phase 8")
        assert "vacf_vdos" in str(err)
        assert "Phase 8" in str(err)


# ---------------------------------------------------------------------------
# Dispatcher registration sanity
# ---------------------------------------------------------------------------


class TestMDDispatcherRegistration:
    def test_md_kinds_in_jobs_dispatch(self):
        from src.api.routers.jobs import _BUILTIN_TEMPLATES, _DISPATCH_TASKS

        for kind in ("md_nvt", "md_nve", "md_npt"):
            assert kind in _DISPATCH_TASKS
            assert kind in _BUILTIN_TEMPLATES

    def test_md_kinds_in_workflow_executor(self):
        from backend.common.workflows.celery_dispatcher import _KIND_TO_TASK

        for kind in ("md_nvt", "md_nve", "md_npt"):
            assert kind in _KIND_TO_TASK

    def test_md_template_endpoint_registered(self):
        from src.api.app import app

        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/api/v1/workflow-runs/templates/md/{template_name}" in paths

    def test_qe_template_endpoint_still_registered(self):
        """Adding the MD endpoint must not have removed the QE one."""
        from src.api.app import app

        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/api/v1/workflow-runs/templates/qe/{template_name}" in paths


# ---------------------------------------------------------------------------
# SPEC_BUILDERS exposes what the endpoint expects
# ---------------------------------------------------------------------------


class TestSpecBuildersTable:
    def test_md_spec_builders_complete(self):
        from backend.common.workflows.templates.md import SPEC_BUILDERS

        expected = {
            "equilibrate_nvt_then_nve",
            "melting_curve",
            "diffusivity_vs_T",
            "elastic_constants_via_strain",
        }
        assert set(SPEC_BUILDERS) == expected
        for name, builder in SPEC_BUILDERS.items():
            spec = builder(_STRUCT_ID)
            assert spec.name in {
                name, name,  # name may differ from key e.g. "diffusivity_vs_T"
            } or spec.name.startswith(name)
