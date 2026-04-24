"""Phase 7 / Session 7.3 — agent-loop router.

Two endpoints today:

- ``POST /api/v1/agent/campaigns/{campaign_id}/stop`` — flip the
  kill switch for a running agent loop. The loop checks
  ``KillSwitch.is_tripped()`` at the top of every round and before
  every tool call, so trip latency is bounded by the longest tool
  call (sub-second for the stub handlers; up to a DFT walltime in
  production once 7.3b lands).

- ``GET /api/v1/agent/campaigns/{campaign_id}/log`` — read back the
  AgentLogEntry rows for a campaign. Used by the frontend to render
  the agent's decision history.

The ``POST /api/v1/agent/campaigns/{campaign_id}/run`` endpoint that
actually drives the loop lives behind a Celery dispatch in 7.3b —
synchronously executing an LLM-driven loop inside a request handler
risks tying up an HTTP worker for minutes. Session 7.3 ships the
engine + kill switch wired correctly so the 7.3b dispatcher is a
thin shim.

The kill-switch registry is process-local — the existing
:mod:`backend.common.agent_v2` ``KillSwitch`` keys live in a single
process's memory. Multi-worker deployments will need a Redis-backed
registry (Session 7.3b); for the in-process tests + the single-
worker uvicorn dev path used in CI, the in-memory registry is
correct.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..auth.security import get_current_active_user
from ..models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent",
    tags=["agent-loop"],
    dependencies=[Depends(get_current_active_user)],
    responses={
        401: {"description": "Not authenticated"},
        404: {"description": "Agent campaign not found"},
    },
)


# ---------------------------------------------------------------------------
# Process-local kill-switch registry
# ---------------------------------------------------------------------------


_REGISTRY_LOCK = threading.Lock()
_KILL_SWITCHES: Dict[str, "object"] = {}


def get_or_create_kill_switch(campaign_id: str):
    """Return (and cache) the :class:`KillSwitch` for a campaign id.

    Used by both the HTTP endpoint (to ``trip`` it) and the loop
    driver (to read it). Same in-memory dict, so they're guaranteed
    to refer to the same flag.
    """
    from backend.common.agent_v2 import KillSwitch

    with _REGISTRY_LOCK:
        ks = _KILL_SWITCHES.get(campaign_id)
        if ks is None:
            ks = KillSwitch()
            _KILL_SWITCHES[campaign_id] = ks
    return ks


def reset_registry_for_tests() -> None:  # pragma: no cover (test helper)
    """Clear the kill-switch registry — only call from test fixtures."""
    with _REGISTRY_LOCK:
        _KILL_SWITCHES.clear()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class StopResponse(BaseModel):
    campaign_id: uuid.UUID
    tripped: bool
    message: str


class AgentLogResponse(BaseModel):
    campaign_id: uuid.UUID
    entries: List[Dict]
    n_entries: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/campaigns/{campaign_id}/stop",
    response_model=StopResponse,
    status_code=status.HTTP_200_OK,
    summary="Halt a running agent loop",
)
async def stop_agent_campaign(
    campaign_id: uuid.UUID,
    current_user: User = Depends(get_current_active_user),
) -> StopResponse:
    """Flip the kill switch.

    Idempotent — re-stopping a tripped switch is a no-op (still
    returns 200 with ``tripped=True``). The loop driver must be
    polling :meth:`KillSwitch.is_tripped` for this to take effect;
    7.3b adds a Redis-backed registry that survives process
    restarts.
    """
    ks = get_or_create_kill_switch(str(campaign_id))
    already = ks.is_tripped()
    ks.trip()
    msg = "kill switch already tripped" if already else "kill switch tripped"
    logger.info(
        "agent campaign %s: %s by user %s", campaign_id, msg, current_user.id,
    )
    return StopResponse(
        campaign_id=campaign_id, tripped=True, message=msg,
    )
