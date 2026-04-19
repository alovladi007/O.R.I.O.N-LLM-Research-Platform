"""Tests for Session 2.4 — workflow DAG executor (tick loop).

Exercises the ``backend.common.workflows.tick`` function against
in-memory run + step objects and a fake dispatcher. No DB, no Celery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeStep:
    step_id: str
    kind: str
    topo_index: int
    spec: Dict[str, Any]
    status: str = "PENDING"
    simulation_job_id: Optional[str] = None
    outputs: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class FakeRun:
    owner_id: str
    spec: Dict[str, Any]
    steps: List[FakeStep]
    status: str = "PENDING"
    started_at: Any = None
    finished_at: Any = None


@dataclass
class FakeJob:
    id: str
    status: str = "RUNNING"
    parameters: Dict[str, Any] = field(default_factory=dict)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class FakeDispatcher:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self._next = 1

    def dispatch(self, *, step_id, kind, structure_id, inputs, owner_id):
        jid = f"job-{self._next}"
        self._next += 1
        self.calls.append(
            {
                "step_id": step_id,
                "kind": kind,
                "structure_id": structure_id,
                "inputs": inputs,
                "owner_id": owner_id,
                "job_id": jid,
            }
        )
        return jid


def _build_run(spec_dict):
    """Build a FakeRun + FakeStep list from a spec dict, with topo indices."""
    from backend.common.workflows import WorkflowSpec, toposort_steps

    spec = WorkflowSpec.model_validate(spec_dict)
    order = toposort_steps(spec)
    topo_idx = {sid: i for i, sid in enumerate(order)}
    steps = [
        FakeStep(
            step_id=s.id,
            kind=s.kind,
            topo_index=topo_idx[s.id],
            spec=s.model_dump(),
        )
        for s in spec.steps
    ]
    return FakeRun(
        owner_id="owner-1",
        spec=spec.model_dump() | {"default_structure_id": "struct-42"},
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Tick behavior
# ---------------------------------------------------------------------------


class TestTickDispatchReadySteps:
    def test_first_tick_dispatches_roots(self):
        from backend.common.workflows import tick

        run = _build_run(
            {
                "name": "chain",
                "steps": [
                    {"id": "a", "kind": "mock_static"},
                    {
                        "id": "b",
                        "kind": "mock_static",
                        "inputs": {"e": {"uses": "a.outputs.energy_ev"}},
                    },
                ],
            }
        )
        dispatcher = FakeDispatcher()
        result = tick(run, job_lookup=lambda _jid: None, dispatcher=dispatcher)

        # Only "a" is root-ready; "b" depends on a.
        assert [c["step_id"] for c in dispatcher.calls] == ["a"]
        assert result.advanced == ["a"]
        a = next(s for s in run.steps if s.step_id == "a")
        b = next(s for s in run.steps if s.step_id == "b")
        assert a.status == "RUNNING"
        assert a.simulation_job_id == "job-1"
        assert b.status == "PENDING"
        assert run.status == "RUNNING"

    def test_second_tick_picks_up_completed_outputs_and_advances(self):
        from backend.common.workflows import tick

        run = _build_run(
            {
                "name": "chain",
                "steps": [
                    {"id": "a", "kind": "mock_static"},
                    {
                        "id": "b",
                        "kind": "mock_static",
                        "inputs": {"e": {"uses": "a.outputs.energy_ev"}},
                    },
                ],
            }
        )
        dispatcher = FakeDispatcher()
        # Tick 1: dispatch a
        tick(run, job_lookup=lambda _jid: None, dispatcher=dispatcher)

        a_job_id = next(s for s in run.steps if s.step_id == "a").simulation_job_id
        a_job = FakeJob(
            id=a_job_id,
            status="COMPLETED",
            parameters={"_outputs": {"energy_ev": -42.0}},
        )

        def lookup(jid):
            return a_job if jid == a_job_id else None

        result = tick(run, job_lookup=lookup, dispatcher=dispatcher)
        # a completed, b dispatched with resolved input
        assert result.completed == ["a"]
        assert result.advanced == ["b"]
        b_call = [c for c in dispatcher.calls if c["step_id"] == "b"][0]
        assert b_call["inputs"] == {"e": -42.0}
        b = next(s for s in run.steps if s.step_id == "b")
        assert b.status == "RUNNING"

    def test_terminal_completion(self):
        from backend.common.workflows import tick

        run = _build_run(
            {
                "name": "single",
                "steps": [{"id": "a", "kind": "mock_static"}],
            }
        )
        dispatcher = FakeDispatcher()
        tick(run, job_lookup=lambda _jid: None, dispatcher=dispatcher)

        a_job_id = run.steps[0].simulation_job_id
        a_job = FakeJob(
            id=a_job_id,
            status="COMPLETED",
            parameters={"_outputs": {"energy_ev": -1.0}},
        )
        result = tick(
            run,
            job_lookup=lambda jid: a_job if jid == a_job_id else None,
            dispatcher=dispatcher,
        )
        assert run.status == "COMPLETED"
        assert result.aggregate_status == "COMPLETED"
        assert run.finished_at is not None


class TestTickFailurePropagation:
    def test_failed_predecessor_skips_descendants(self):
        from backend.common.workflows import tick

        run = _build_run(
            {
                "name": "fail-chain",
                "steps": [
                    {"id": "a", "kind": "mock_static"},
                    {"id": "b", "kind": "mock_static", "depends_on": ["a"]},
                    {"id": "c", "kind": "mock_static", "depends_on": ["b"]},
                ],
            }
        )
        dispatcher = FakeDispatcher()
        tick(run, job_lookup=lambda _jid: None, dispatcher=dispatcher)

        a_job_id = next(s for s in run.steps if s.step_id == "a").simulation_job_id
        a_job = FakeJob(id=a_job_id, status="FAILED", error_message="engine exploded")
        result = tick(
            run,
            job_lookup=lambda jid: a_job if jid == a_job_id else None,
            dispatcher=dispatcher,
        )
        assert "a" in result.failed
        assert set(result.skipped) == {"b", "c"}
        assert run.status == "FAILED"


class TestTickForeachFanout:
    def test_fanout_dispatches_all_children(self):
        from backend.common.workflows import expand_foreach, tick, WorkflowSpec

        spec = WorkflowSpec.model_validate(
            {
                "name": "sweep",
                "steps": [
                    {
                        "id": "sweep",
                        "kind": "mock_static",
                        "foreach": {"T": list(range(8))},
                    }
                ],
            }
        )
        expanded = expand_foreach(spec)
        run = _build_run(expanded.model_dump())
        dispatcher = FakeDispatcher()
        tick(run, job_lookup=lambda _jid: None, dispatcher=dispatcher)
        assert len(dispatcher.calls) == 8
        assert {c["inputs"]["T"] for c in dispatcher.calls} == set(range(8))


class TestTickCancelledRun:
    def test_cancelled_run_is_a_noop(self):
        from backend.common.workflows import tick

        run = _build_run(
            {
                "name": "cancel-me",
                "steps": [{"id": "a", "kind": "mock_static"}],
            }
        )
        run.status = "CANCELLED"
        for s in run.steps:
            s.status = "CANCELLED"
        dispatcher = FakeDispatcher()
        result = tick(run, job_lookup=lambda _jid: None, dispatcher=dispatcher)
        assert dispatcher.calls == []
        assert result.advanced == []
        assert run.status == "CANCELLED"


class TestManifestAggregation:
    def test_build_manifest_shape(self):
        from backend.common.workflows import build_workflow_manifest

        records = [
            {
                "step_id": "a",
                "status": "COMPLETED",
                "job_id": "job-1",
                "outputs": {"energy_ev": -42.0},
                "artifact": {"bucket": "b", "key": "jobs/job-1/run.tgz"},
            },
            {
                "step_id": "b",
                "status": "COMPLETED",
                "job_id": "job-2",
                "outputs": {"energy_ev": -43.0},
                "artifact": None,
            },
        ]
        manifest = build_workflow_manifest(
            workflow_run_id="run-1", name="demo", step_records=records,
        )
        assert manifest["workflow_run_id"] == "run-1"
        assert manifest["manifest_schema"] == "workflow_run.v1"
        assert set(manifest["steps"]) == {"a", "b"}
        assert manifest["steps"]["a"]["outputs"] == {"energy_ev": -42.0}
        assert manifest["steps"]["b"]["artifact"] is None
