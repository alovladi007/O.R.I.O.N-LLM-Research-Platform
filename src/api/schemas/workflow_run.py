"""Pydantic schemas for /api/v1/workflow-runs (Session 2.4)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class WorkflowRunCreate(BaseModel):
    """Body for POST /api/v1/workflow-runs.

    The ``spec`` is the :class:`backend.common.workflows.WorkflowSpec`
    dict; validation happens in the router so we can return 422 with
    the pydantic-style error list.
    """

    model_config = ConfigDict(extra="forbid")

    spec: Dict[str, Any] = Field(..., description="WorkflowSpec as a dict.")


class WorkflowRunStepResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="ignore")

    step_id: str
    kind: str
    status: str
    topo_index: int
    simulation_job_id: Optional[uuid.UUID] = None
    outputs: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class WorkflowRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="ignore")

    id: uuid.UUID
    name: str
    description: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    steps: List[WorkflowRunStepResponse] = Field(default_factory=list)
    manifest: Optional[Dict[str, Any]] = None
