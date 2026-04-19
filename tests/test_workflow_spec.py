"""Tests for Session 2.4 — WorkflowSpec parsing, toposort, foreach, resolver.

No DB, no Celery, no FastAPI. Pure in-memory.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# WorkflowSpec parsing
# ---------------------------------------------------------------------------


class TestWorkflowSpec:
    def test_minimal_spec_validates(self):
        from backend.common.workflows import WorkflowSpec

        spec = WorkflowSpec.model_validate(
            {
                "name": "minimal",
                "steps": [{"id": "a", "kind": "mock_static"}],
            }
        )
        assert spec.steps[0].id == "a"
        assert spec.steps[0].kind == "mock_static"

    def test_duplicate_step_ids_rejected(self):
        from pydantic import ValidationError

        from backend.common.workflows import WorkflowSpec

        with pytest.raises(ValidationError):
            WorkflowSpec.model_validate(
                {
                    "name": "dup",
                    "steps": [
                        {"id": "a", "kind": "mock_static"},
                        {"id": "a", "kind": "mock_static"},
                    ],
                }
            )

    def test_unknown_depends_on_rejected_on_validate(self):
        from backend.common.workflows import WorkflowSpec, WorkflowSpecError

        spec = WorkflowSpec.model_validate(
            {
                "name": "bad",
                "steps": [
                    {"id": "a", "kind": "mock_static", "depends_on": ["ghost"]},
                ],
            }
        )
        with pytest.raises(WorkflowSpecError):
            spec.validate_dependencies()

    def test_unknown_uses_target_rejected(self):
        from backend.common.workflows import WorkflowSpec, WorkflowSpecError

        spec = WorkflowSpec.model_validate(
            {
                "name": "ref",
                "steps": [
                    {
                        "id": "consume",
                        "kind": "mock_static",
                        "inputs": {"energy": {"uses": "ghost.outputs.energy_ev"}},
                    }
                ],
            }
        )
        with pytest.raises(WorkflowSpecError):
            spec.validate_dependencies()

    def test_step_id_forbidden_chars_rejected(self):
        from pydantic import ValidationError

        from backend.common.workflows import WorkflowSpec

        with pytest.raises(ValidationError):
            WorkflowSpec.model_validate(
                {
                    "name": "bad-id",
                    "steps": [{"id": "has spaces", "kind": "mock_static"}],
                }
            )


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


class TestToposortSteps:
    def test_linear_chain(self):
        from backend.common.workflows import WorkflowSpec, toposort_steps

        spec = WorkflowSpec.model_validate(
            {
                "name": "chain",
                "steps": [
                    {"id": "c", "kind": "mock_static", "depends_on": ["b"]},
                    {"id": "a", "kind": "mock_static"},
                    {"id": "b", "kind": "mock_static", "depends_on": ["a"]},
                ],
            }
        )
        order = toposort_steps(spec)
        assert order == ["a", "b", "c"]

    def test_diamond(self):
        from backend.common.workflows import WorkflowSpec, toposort_steps

        spec = WorkflowSpec.model_validate(
            {
                "name": "diamond",
                "steps": [
                    {"id": "d", "kind": "mock_static", "depends_on": ["b", "c"]},
                    {"id": "c", "kind": "mock_static", "depends_on": ["a"]},
                    {"id": "b", "kind": "mock_static", "depends_on": ["a"]},
                    {"id": "a", "kind": "mock_static"},
                ],
            }
        )
        order = toposort_steps(spec)
        assert order[0] == "a"
        assert order[-1] == "d"
        assert set(order[1:3]) == {"b", "c"}

    def test_cycle_raises(self):
        from backend.common.workflows import WorkflowCycleError, WorkflowSpec, toposort_steps

        spec = WorkflowSpec.model_validate(
            {
                "name": "cycle",
                "steps": [
                    {"id": "a", "kind": "mock_static", "depends_on": ["b"]},
                    {"id": "b", "kind": "mock_static", "depends_on": ["a"]},
                ],
            }
        )
        with pytest.raises(WorkflowCycleError):
            toposort_steps(spec)

    def test_implicit_deps_via_uses(self):
        """A `uses` reference implies a topological edge."""
        from backend.common.workflows import WorkflowSpec, toposort_steps

        spec = WorkflowSpec.model_validate(
            {
                "name": "implicit",
                "steps": [
                    {"id": "producer", "kind": "mock_static"},
                    {
                        "id": "consumer",
                        "kind": "mock_static",
                        "inputs": {"e": {"uses": "producer.outputs.energy_ev"}},
                    },
                ],
            }
        )
        order = toposort_steps(spec)
        assert order == ["producer", "consumer"]


# ---------------------------------------------------------------------------
# Foreach expansion
# ---------------------------------------------------------------------------


class TestForeach:
    def test_single_variable_expands(self):
        from backend.common.workflows import WorkflowSpec, expand_foreach

        spec = WorkflowSpec.model_validate(
            {
                "name": "sweep",
                "steps": [
                    {
                        "id": "sweep",
                        "kind": "mock_static",
                        "foreach": {"temperature": [100, 200, 300]},
                    }
                ],
            }
        )
        expanded = expand_foreach(spec)
        assert len(expanded.steps) == 3
        ids = [s.id for s in expanded.steps]
        assert set(ids) == {"sweep__100", "sweep__200", "sweep__300"}
        temps = sorted(s.inputs["temperature"] for s in expanded.steps)
        assert temps == [100, 200, 300]

    def test_multi_variable_cartesian(self):
        from backend.common.workflows import WorkflowSpec, expand_foreach

        spec = WorkflowSpec.model_validate(
            {
                "name": "grid",
                "steps": [
                    {
                        "id": "grid",
                        "kind": "mock_static",
                        "foreach": {"T": [100, 200], "P": [1, 2]},
                    }
                ],
            }
        )
        expanded = expand_foreach(spec)
        assert len(expanded.steps) == 4
        combos = sorted((s.inputs["T"], s.inputs["P"]) for s in expanded.steps)
        assert combos == [(100, 1), (100, 2), (200, 1), (200, 2)]

    def test_downstream_deps_rewritten(self):
        """A step that depends on the un-expanded parent depends on *all* children."""
        from backend.common.workflows import WorkflowSpec, expand_foreach, toposort_steps

        spec = WorkflowSpec.model_validate(
            {
                "name": "fanout-aggregate",
                "steps": [
                    {
                        "id": "sweep",
                        "kind": "mock_static",
                        "foreach": {"T": [100, 200, 300]},
                    },
                    {
                        "id": "aggregate",
                        "kind": "mock_static",
                        "depends_on": ["sweep"],
                    },
                ],
            }
        )
        expanded = expand_foreach(spec)
        agg = next(s for s in expanded.steps if s.id == "aggregate")
        assert set(agg.depends_on) == {"sweep__100", "sweep__200", "sweep__300"}
        # Toposort the expanded spec — aggregate must be last.
        order = toposort_steps(expanded)
        assert order[-1] == "aggregate"

    def test_empty_foreach_list_rejected(self):
        from pydantic import ValidationError

        from backend.common.workflows import WorkflowSpec

        with pytest.raises(ValidationError):
            WorkflowSpec.model_validate(
                {
                    "name": "empty",
                    "steps": [
                        {
                            "id": "sweep",
                            "kind": "mock_static",
                            "foreach": {"T": []},
                        }
                    ],
                }
            )


# ---------------------------------------------------------------------------
# Reference resolver
# ---------------------------------------------------------------------------


class TestResolver:
    def test_simple_ref(self):
        from backend.common.workflows import resolve_references

        outputs = {"step_a": {"energy_ev": -42.0}}
        got = resolve_references(
            {"e": {"uses": "step_a.outputs.energy_ev"}},
            outputs,
        )
        assert got == {"e": -42.0}

    def test_nested_dotted_path(self):
        from backend.common.workflows import resolve_references

        outputs = {"s": {"forces": [{"fx": 0.01}, {"fx": 0.02}]}}
        got = resolve_references(
            {"f0": {"uses": "s.outputs.forces.0.fx"}},
            outputs,
        )
        assert got == {"f0": 0.01}

    def test_missing_step_raises_resolution_error(self):
        from backend.common.workflows import ResolutionError, resolve_references

        with pytest.raises(ResolutionError):
            resolve_references(
                {"x": {"uses": "ghost.outputs.anything"}},
                outputs_by_step={},
            )

    def test_bad_ref_shape_raises(self):
        from backend.common.workflows import ResolutionError, resolve_references

        with pytest.raises(ResolutionError):
            resolve_references(
                {"x": {"uses": "step_without_outputs_keyword"}},
                outputs_by_step={"step_without_outputs_keyword": {}},
            )

    def test_literals_pass_through(self):
        from backend.common.workflows import resolve_references

        got = resolve_references({"a": 1, "b": [2, 3], "c": "hello"}, {})
        assert got == {"a": 1, "b": [2, 3], "c": "hello"}
