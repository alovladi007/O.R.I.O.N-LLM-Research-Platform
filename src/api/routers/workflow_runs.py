"""Workflow DAG runs router (Session 2.4).

Distinct from :mod:`.workflows` (which manages single-step
``WorkflowTemplate`` rows). This router submits multi-step DAGs built
from ``backend.common.workflows.WorkflowSpec``.

Endpoints:

- POST /api/v1/workflow-runs            submit a spec
- GET  /api/v1/workflow-runs/{id}       get aggregate state
- POST /api/v1/workflow-runs/{id}/cancel cancel the run
- GET  /api/v1/workflow-runs/{id}/manifest  get workflow.json manifest
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query, status
from pydantic import ValidationError as PydValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..auth.security import get_current_active_user
from ..database import get_db
from ..exceptions import AuthorizationError, ConflictError, NotFoundError, ValidationError
from ..models import (
    SimulationJob,
    User,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowRunStep,
    WorkflowStepStatus,
)
from ..schemas.workflow_run import (
    WorkflowRunCreate,
    WorkflowRunResponse,
)

logger = logging.getLogger(__name__)


router = APIRouter(
    tags=["workflow-runs"],
    dependencies=[Depends(get_current_active_user)],
    responses={
        401: {"description": "Not authenticated"},
        404: {"description": "Workflow run not found"},
    },
)


@router.post(
    "",
    response_model=WorkflowRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a workflow DAG for execution",
)
async def submit_workflow_run(
    body: WorkflowRunCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> WorkflowRunResponse:
    if not current_user.can_run_simulations():
        raise AuthorizationError("You don't have permission to run workflows")

    from backend.common.workflows import (
        WorkflowSpec,
        WorkflowSpecError,
        expand_foreach,
        toposort_steps,
    )  # noqa: F401 — WorkflowSpecError is used in exception handler below

    # 1) Parse + validate spec (pydantic)
    try:
        spec = WorkflowSpec.model_validate(body.spec)
    except PydValidationError as exc:
        raise ValidationError("WorkflowSpec validation failed", details={"errors": exc.errors()}) from exc

    # 2) foreach expansion
    try:
        expanded = expand_foreach(spec)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"foreach expansion failed: {exc}") from exc

    # 3) Topological sort (also catches cycles + unknown deps)
    try:
        order = toposort_steps(expanded)
    except WorkflowSpecError as exc:
        raise ValidationError(str(exc)) from exc

    # 4) Persist run + steps.
    run = WorkflowRun(
        owner_id=current_user.id,
        name=expanded.name,
        description=expanded.description,
        status=WorkflowRunStatus.PENDING.value,
        spec=expanded.model_dump(),
    )
    db.add(run)
    await db.flush()

    topo_idx = {sid: i for i, sid in enumerate(order)}
    for step in expanded.steps:
        db.add(
            WorkflowRunStep(
                workflow_run_id=run.id,
                step_id=step.id,
                kind=step.kind,
                status=WorkflowStepStatus.PENDING.value,
                topo_index=topo_idx[step.id],
                spec=step.model_dump(),
            )
        )
    await db.commit()

    # Poke the tick task so the user sees progress without waiting for
    # the next beat cycle.
    try:
        from src.worker.celery_app import celery_app

        celery_app.send_task("orion.workflows.tick")
    except Exception as exc:  # noqa: BLE001
        logger.warning("submit: could not kick workflow tick: %s", exc)

    loaded = await _reload_run_with_steps(db, run.id)
    return WorkflowRunResponse.model_validate(loaded)


@router.get("/{run_id}", response_model=WorkflowRunResponse)
async def get_workflow_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> WorkflowRunResponse:
    run = await _reload_run_with_steps(db, run_id)
    if run is None:
        raise NotFoundError("WorkflowRun", run_id)
    return WorkflowRunResponse.model_validate(run)


@router.post("/{run_id}/cancel", response_model=WorkflowRunResponse)
async def cancel_workflow_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> WorkflowRunResponse:
    run = await _reload_run_with_steps(db, run_id)
    if run is None:
        raise NotFoundError("WorkflowRun", run_id)

    if run.status in (
        WorkflowRunStatus.COMPLETED.value,
        WorkflowRunStatus.FAILED.value,
        WorkflowRunStatus.CANCELLED.value,
    ):
        raise ConflictError(
            f"Workflow run already terminal: {run.status}",
            details={"run_id": str(run_id), "status": run.status},
        )

    # Mark pending steps CANCELLED; running steps stay alive — the
    # underlying SimulationJobs can be cancelled via the jobs router.
    for step in run.steps:
        if step.status in (
            WorkflowStepStatus.PENDING.value,
            WorkflowStepStatus.DISPATCHABLE.value,
        ):
            step.status = WorkflowStepStatus.CANCELLED.value
    run.status = WorkflowRunStatus.CANCELLED.value
    run.finished_at = datetime.utcnow()
    await db.commit()

    return WorkflowRunResponse.model_validate(run)


@router.get("/{run_id}/manifest")
async def get_workflow_manifest(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    run = await _reload_run_with_steps(db, run_id)
    if run is None:
        raise NotFoundError("WorkflowRun", run_id)
    if run.manifest is not None:
        return run.manifest
    # Build on-the-fly for non-terminal runs so clients can poll.
    from backend.common.workflows import build_workflow_manifest

    records: List[Dict[str, Any]] = []
    for s in run.steps:
        artifact = None
        if s.simulation_job_id is not None:
            job = await db.get(SimulationJob, s.simulation_job_id)
            if job is not None:
                extra = job.extra_metadata or {}
                artifact = extra.get("artifact") if isinstance(extra, dict) else None
        records.append(
            {
                "step_id": s.step_id,
                "status": s.status,
                "job_id": str(s.simulation_job_id) if s.simulation_job_id else None,
                "outputs": s.outputs,
                "artifact": artifact,
            }
        )
    return build_workflow_manifest(
        workflow_run_id=str(run.id), name=run.name, step_records=records,
    )


# ---------------------------------------------------------------------------
# Template endpoints — Session 3.3
# ---------------------------------------------------------------------------


@router.post(
    "/templates/qe/{template_name}",
    response_model=WorkflowRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a pre-built QE workflow (Session 3.3)",
    description=(
        "Short-circuits spec authoring: pass `structure_id` as a query "
        "param and the matching template (from "
        "`backend.common.workflows.templates.qe`) is constructed + "
        "submitted. Available templates: `relax_then_static`, "
        "`band_structure`, `dos`, `phonons_gamma`."
    ),
)
async def submit_qe_template(
    template_name: str,
    structure_id: uuid.UUID = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> WorkflowRunResponse:
    if not current_user.can_run_simulations():
        raise AuthorizationError("You don't have permission to run workflows")

    from backend.common.workflows.templates.qe import SPEC_BUILDERS

    builder = SPEC_BUILDERS.get(template_name)
    if builder is None:
        raise ValidationError(
            f"Unknown QE template {template_name!r}",
            details={"known": sorted(SPEC_BUILDERS)},
        )

    # Structure must exist.
    from ..models import Structure

    structure = await db.get(Structure, structure_id)
    if structure is None:
        raise NotFoundError("Structure", structure_id)

    spec = builder(str(structure_id))
    # Reuse the submit_workflow_run path so validation + tick kick is
    # shared.
    from backend.common.workflows import expand_foreach, toposort_steps

    expanded = expand_foreach(spec)
    toposort_steps(expanded)  # raises on cycle / unknown deps

    run = WorkflowRun(
        owner_id=current_user.id,
        name=expanded.name,
        description=expanded.description,
        status=WorkflowRunStatus.PENDING.value,
        spec=expanded.model_dump(),
    )
    db.add(run)
    await db.flush()

    topo_idx = {sid: i for i, sid in enumerate([s.id for s in expanded.steps])}
    for step in expanded.steps:
        db.add(
            WorkflowRunStep(
                workflow_run_id=run.id,
                step_id=step.id,
                kind=step.kind,
                status=WorkflowStepStatus.PENDING.value,
                topo_index=topo_idx[step.id],
                spec=step.model_dump(),
            )
        )
    await db.commit()

    try:
        from src.worker.celery_app import celery_app

        celery_app.send_task("orion.workflows.tick")
    except Exception as exc:  # noqa: BLE001
        logger.warning("template submit: could not kick workflow tick: %s", exc)

    loaded = await _reload_run_with_steps(db, run.id)
    return WorkflowRunResponse.model_validate(loaded)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _reload_run_with_steps(db: AsyncSession, run_id: uuid.UUID):
    q = (
        select(WorkflowRun)
        .where(WorkflowRun.id == run_id)
        .options(selectinload(WorkflowRun.steps))
    )
    res = await db.execute(q)
    return res.scalar_one_or_none()
