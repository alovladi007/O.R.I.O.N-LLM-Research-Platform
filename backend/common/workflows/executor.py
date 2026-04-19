"""Workflow tick executor — advance a WorkflowRun by one scheduler step.

Not a Celery task itself — this is the core logic, sync and testable.
The Celery entry points in ``src/worker/tasks.py`` wrap it with
session + commit.

One tick's job
--------------

For a given workflow_run that's not terminal:

1. For each step, map any completed SimulationJob's outputs back onto
   ``WorkflowRunStep.outputs`` + flip the step's status accordingly.
2. Propagate FAILED steps: every descendant becomes SKIPPED.
3. For each PENDING step, check if all predecessors are COMPLETED and
   all ``{"uses": ...}`` references resolve. If yes, dispatch it:
   create a SimulationJob + enqueue the corresponding Celery task,
   mark the step RUNNING with the new ``simulation_job_id``.
4. Update the run's aggregate status.

The dispatcher is passed in as a callable so unit tests can substitute
a no-op or a FakeDispatcher; production wiring uses
:class:`CeleryDispatcher` below.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from .resolver import ResolutionError, resolve_references
from .spec import StepSpec, WorkflowSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dispatcher protocol
# ---------------------------------------------------------------------------


class Dispatcher(Protocol):
    """How a tick actually starts a step's job.

    Implementations must be synchronous — the tick caller owns the DB
    session; spawning async work inside is fine, but the return has to
    be the created SimulationJob.id so the step row can be linked.
    """

    def dispatch(
        self,
        *,
        step_id: str,
        kind: str,
        structure_id: Optional[str],
        inputs: Dict[str, Any],
        owner_id: Any,
    ) -> Any:
        ...


# ---------------------------------------------------------------------------
# Tick result
# ---------------------------------------------------------------------------


@dataclass
class TickResult:
    """Observability payload from one tick."""

    advanced: List[str]          # step_ids that transitioned to RUNNING this tick
    completed: List[str]         # step_ids that transitioned to COMPLETED
    failed: List[str]
    skipped: List[str]           # newly-SKIPPED (upstream failure)
    aggregate_status: str        # new run status after this tick

    def as_dict(self) -> Dict[str, Any]:
        return {
            "advanced": self.advanced,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "aggregate_status": self.aggregate_status,
        }


# ---------------------------------------------------------------------------
# Core tick — pure on a minimal "run state" shape
# ---------------------------------------------------------------------------


# Keep dependency on the DB models behind a Protocol so tests don't
# need SQLAlchemy instances. Executor reads + writes these attrs.


class _RunStateProtocol(Protocol):
    status: str
    spec: Dict[str, Any]
    owner_id: Any
    steps: List[Any]


def _step_by_id(run, sid: str):
    for s in run.steps:
        if s.step_id == sid:
            return s
    return None


def _step_spec_map(spec_dict: Dict[str, Any]) -> Dict[str, StepSpec]:
    ws = WorkflowSpec.model_validate(spec_dict)
    return {s.id: s for s in ws.steps}


def _is_step_ready(
    step_spec: StepSpec,
    run,
    outputs_by_step: Dict[str, Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """All predecessors COMPLETED AND every `{"uses": ...}` resolves."""
    # Predecessor status check first
    deps = set(step_spec.depends_on)
    # Add implicit deps from `uses` references
    from .spec import _collect_uses_targets  # local import avoids cycle

    for ref in _collect_uses_targets(step_spec.inputs):
        deps.add(ref.split(".", 1)[0])

    for dep in deps:
        pred = _step_by_id(run, dep)
        if pred is None:
            return False, f"predecessor {dep!r} missing"
        if pred.status != "COMPLETED":
            return False, f"predecessor {dep!r} status={pred.status}"

    # Now try to resolve references — they should succeed if upstream
    # outputs are populated.
    try:
        resolve_references(step_spec.inputs, outputs_by_step)
    except ResolutionError as exc:
        return False, f"unresolved reference: {exc}"
    return True, None


def _propagate_failure(run) -> List[str]:
    """After a FAILED step, mark descendants SKIPPED.

    Uses the spec to find descendants; idempotent (already-SKIPPED
    steps are ignored).
    """
    spec_map = _step_spec_map(run.spec)
    reverse: Dict[str, List[str]] = {sid: [] for sid in spec_map}
    for sid, sspec in spec_map.items():
        deps = set(sspec.depends_on)
        from .spec import _collect_uses_targets

        for ref in _collect_uses_targets(sspec.inputs):
            deps.add(ref.split(".", 1)[0])
        for d in deps:
            if d in reverse:
                reverse[d].append(sid)

    failed_steps = {s.step_id for s in run.steps if s.status == "FAILED"}
    newly_skipped: List[str] = []
    queue = list(failed_steps)
    while queue:
        cur = queue.pop()
        for succ in reverse.get(cur, []):
            step = _step_by_id(run, succ)
            if step is None or step.status in ("SKIPPED", "FAILED", "CANCELLED", "COMPLETED"):
                continue
            step.status = "SKIPPED"
            newly_skipped.append(succ)
            queue.append(succ)
    return newly_skipped


def _recompute_aggregate(run) -> str:
    """Roll up step statuses to the run-level status."""
    statuses = {s.status for s in run.steps}
    if "CANCELLED" in statuses and all(
        s in ("CANCELLED", "COMPLETED", "FAILED", "SKIPPED") for s in statuses
    ):
        return "CANCELLED"
    if "FAILED" in statuses and all(
        s in ("COMPLETED", "FAILED", "SKIPPED", "CANCELLED") for s in statuses
    ):
        return "FAILED"
    if all(s == "COMPLETED" for s in statuses):
        return "COMPLETED"
    if any(s == "RUNNING" for s in statuses) or any(
        s in ("COMPLETED", "FAILED", "SKIPPED", "CANCELLED") for s in statuses
    ):
        return "RUNNING"
    return "PENDING"


def tick(
    run,
    *,
    job_lookup: Callable[[Any], Any],
    dispatcher: Dispatcher,
    now: Optional[datetime] = None,
) -> TickResult:
    """Advance *run* by one scheduler step.

    Parameters
    ----------
    run
        A :class:`WorkflowRun` row (SQLAlchemy) or an object with the
        same attribute shape. Mutated in-place.
    job_lookup
        Given a ``simulation_job_id``, return a job-like object exposing
        ``status`` (string enum name or value) and a way to reach
        ``parameters['_outputs']``. Usually
        ``session.get(SimulationJob, ...)``.
    dispatcher
        How to submit a new step. See :class:`Dispatcher`.
    now
        Injected for deterministic testing of ``finished_at`` writes.
    """
    now = now or datetime.utcnow()

    if run.status == "CANCELLED":
        return TickResult([], [], [], [], run.status)

    spec_map = _step_spec_map(run.spec)

    # 1) Poll running children — pull outputs + update status.
    newly_completed: List[str] = []
    newly_failed: List[str] = []
    for step in run.steps:
        if step.status != "RUNNING":
            continue
        if step.simulation_job_id is None:
            continue
        job = job_lookup(step.simulation_job_id)
        if job is None:
            continue
        job_status = _job_status_str(job)
        if job_status == "COMPLETED":
            outs = _extract_outputs(job)
            step.outputs = outs
            step.status = "COMPLETED"
            newly_completed.append(step.step_id)
        elif job_status in ("FAILED", "TIMEOUT"):
            step.status = "FAILED"
            step.error_message = getattr(job, "error_message", None)
            newly_failed.append(step.step_id)
        elif job_status == "CANCELLED":
            step.status = "CANCELLED"

    # 2) Cascade failures → SKIPPED descendants.
    newly_skipped = _propagate_failure(run)

    # 3) Find dispatchable steps and fire them.
    outputs_by_step = {
        s.step_id: (s.outputs or {}) for s in run.steps if s.status == "COMPLETED"
    }
    newly_advanced: List[str] = []
    for step in run.steps:
        if step.status != "PENDING":
            continue
        step_spec = spec_map.get(step.step_id)
        if step_spec is None:
            step.status = "FAILED"
            step.error_message = f"step {step.step_id!r} not found in spec"
            newly_failed.append(step.step_id)
            continue
        ready, reason = _is_step_ready(step_spec, run, outputs_by_step)
        if not ready:
            continue
        # Resolve references for real now (we know they resolve).
        resolved_inputs = resolve_references(step_spec.inputs, outputs_by_step)
        structure_id = step_spec.structure_id or (
            run.spec.get("default_structure_id")
        )
        try:
            job_id = dispatcher.dispatch(
                step_id=step.step_id,
                kind=step_spec.kind,
                structure_id=structure_id,
                inputs=resolved_inputs,
                owner_id=run.owner_id,
            )
        except Exception as exc:  # noqa: BLE001
            step.status = "FAILED"
            step.error_message = f"dispatch_failed: {exc}"
            newly_failed.append(step.step_id)
            continue
        step.simulation_job_id = job_id
        step.status = "RUNNING"
        newly_advanced.append(step.step_id)

    # 4) Aggregate status
    aggregate = _recompute_aggregate(run)
    run.status = aggregate
    if aggregate in ("COMPLETED", "FAILED", "CANCELLED") and run.finished_at is None:
        run.finished_at = now
    if aggregate == "RUNNING" and run.started_at is None:
        run.started_at = now

    return TickResult(
        advanced=newly_advanced,
        completed=newly_completed,
        failed=newly_failed,
        skipped=newly_skipped,
        aggregate_status=aggregate,
    )


# ---------------------------------------------------------------------------
# Helpers — interpret job objects defensively
# ---------------------------------------------------------------------------


def _job_status_str(job) -> str:
    raw = getattr(job, "status", None)
    if raw is None:
        return "UNKNOWN"
    return getattr(raw, "value", str(raw)).upper()


def _extract_outputs(job) -> Optional[Dict[str, Any]]:
    params = getattr(job, "parameters", None) or {}
    outs = params.get("_outputs") if isinstance(params, dict) else None
    if outs is not None:
        return outs
    # Fallback: extra_metadata.outputs if a later session moves the slot.
    extra = getattr(job, "extra_metadata", None) or {}
    return extra.get("outputs") if isinstance(extra, dict) else None
