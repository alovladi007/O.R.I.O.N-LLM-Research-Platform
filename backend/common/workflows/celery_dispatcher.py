"""Celery-backed Dispatcher for workflow steps.

Creates a :class:`SimulationJob` row in QUEUED, enqueues the matching
Celery task, and returns the job id. The mapping from ``kind`` →
Celery task name is shared with the API router's dispatch endpoint
(Session 2.2) via the same table:

    kind ∈ {"mock_static", ...}   ─►   orion.<kind>.static / .relax / ...

Session 2.4 only lists ``mock_static``; Phase 3+ fill in real engines.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Shared with src.api.routers.jobs._DISPATCH_TASKS — kept separate to
# avoid cyclic imports. When Phase 3 adds DFT kinds, extend both.
_KIND_TO_TASK: Dict[str, str] = {
    "mock_static": "orion.mock.static",
}


class CeleryDispatcher:
    """Sync :class:`Dispatcher` that creates a SimulationJob + enqueues a task.

    Takes a SQLAlchemy session (sync mode — matches the tick entrypoint
    in the Celery worker).
    """

    def __init__(self, session: Any) -> None:
        self.session = session

    def dispatch(
        self,
        *,
        step_id: str,
        kind: str,
        structure_id: Optional[str],
        inputs: Dict[str, Any],
        owner_id: Any,
    ) -> uuid.UUID:
        if kind not in _KIND_TO_TASK:
            raise ValueError(f"workflow step kind {kind!r} has no Celery task")
        if structure_id is None:
            raise ValueError(
                f"workflow step {step_id!r}: structure_id required "
                "(set per-step or via default_structure_id)"
            )

        from src.api.models import JobStatus, SimulationJob, WorkflowTemplate
        from sqlalchemy import select

        # Ensure a built-in template exists (same lazy pattern as the
        # /api/v1/jobs/dispatch endpoint).
        tpl_name = _builtin_template_name_for_kind(kind)
        tpl = (
            self.session.execute(
                select(WorkflowTemplate).where(WorkflowTemplate.name == tpl_name)
            )
            .scalar_one_or_none()
        )
        if tpl is None:
            tpl = WorkflowTemplate(
                name=tpl_name,
                display_name=f"{kind} (workflow-dispatched)",
                description="Auto-created by workflow executor.",
                engine=_engine_for_kind(kind),
                category="workflow",
                default_parameters={},
                default_resources={"cores": 1, "memory_gb": 1, "walltime_minutes": 5},
                is_active=True,
                is_public=True,
            )
            self.session.add(tpl)
            self.session.flush()

        new_job = SimulationJob(
            owner_id=owner_id,
            structure_id=uuid.UUID(str(structure_id)),
            workflow_template_id=tpl.id,
            name=f"workflow-step:{step_id}",
            status=JobStatus.QUEUED,
            priority=5,
            engine=tpl.engine,
            kind=kind,
            parameters={**(tpl.default_parameters or {}), **inputs},
            resources=tpl.default_resources,
            extra_metadata={"workflow_step_id": step_id},
            submitted_at=datetime.utcnow(),
        )
        self.session.add(new_job)
        self.session.flush()

        # Enqueue via send_task so we don't import the task module here
        # (it would load src.worker.tasks, creating a circular import).
        from src.worker.celery_app import celery_app

        task_name = _KIND_TO_TASK[kind]
        async_result = celery_app.send_task(task_name, args=[str(new_job.id)])
        new_job.celery_task_id = async_result.id
        self.session.flush()

        logger.info(
            "workflow dispatched step=%s kind=%s job=%s task_id=%s",
            step_id, kind, new_job.id, async_result.id,
        )
        return new_job.id


def _builtin_template_name_for_kind(kind: str) -> str:
    return f"{kind}_default"


def _engine_for_kind(kind: str) -> str:
    if kind.startswith("mock"):
        return "mock"
    if kind.startswith("dft"):
        return "qe"
    if kind.startswith("md"):
        return "lammps"
    if kind.startswith("continuum"):
        return "continuum"
    if kind.startswith("mesoscale"):
        return "mesoscale"
    return "mock"
