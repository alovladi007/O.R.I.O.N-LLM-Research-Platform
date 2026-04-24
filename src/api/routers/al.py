"""Active-learning campaign router (Phase 6 / Session 6.5).

Provides ``POST /al/campaigns`` to start a campaign and ``GET
/al/campaigns/{id}`` to read its cycle log + cumulative best.

Storage is **in-memory** for Session 6.5 — a process-local dict keyed
by campaign UUID. The DB-backed equivalent (``al_campaigns`` /
``al_cycles`` tables, Alembic migration, Celery dispatch) lands in
Session 6.5b, mirroring the Session 6.2 → 6.2b and 6.3 → 6.3b pattern
from earlier in the phase. Restarting the API server clears the
in-memory store; the cycles themselves are deterministic given the
input snapshot + seed, so re-running gets you the same trajectory.

The cheating-oracle path (oracle reads ``y_pool[i]`` directly, the
default for the engine's :meth:`ALCampaign.run`) is the only path
exposed today — it's enough to drive the engine end-to-end against the
Session 6.3 snapshot. Real DFT-as-oracle is wired in 6.5b once the
``ml`` Celery queue learns to block on a child ``dft_static`` job.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..auth.security import get_current_active_user
from ..models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/al",
    tags=["active-learning"],
    dependencies=[Depends(get_current_active_user)],
    responses={
        401: {"description": "Not authenticated"},
        404: {"description": "Campaign not found"},
    },
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ALCampaignCreate(BaseModel):
    """Inline-corpus campaign request.

    Session 6.5 ships an inline-corpus shape (caller posts a feature
    matrix + targets directly) so the API is testable without the
    DB-backed dataset registry. Session 6.5b extends with
    ``dataset_name`` / ``dataset_version`` to look the corpus up from
    the registry, plus a ``model_id`` to drive a real CGCNN ensemble
    via Celery.
    """

    name: str = Field(..., min_length=1, max_length=128)
    X_pool: List[List[float]] = Field(..., description="2-D feature matrix")
    y_pool: List[float] = Field(..., description="Targets, 1-D")
    initial_train_indices: List[int] = Field(
        ..., min_length=1, description="Seed labelled set (pool indices)",
    )
    acquisition: str = Field(
        "max_sigma",
        description="One of: max_sigma, ucb, ei, bald",
    )
    query_size: int = Field(10, gt=0, le=200)
    n_cycles: int = Field(10, gt=0, le=50)
    maximize: bool = True
    beta: float = Field(2.0, ge=0.0)
    xi: float = Field(0.0, ge=0.0)
    X_val: Optional[List[List[float]]] = None
    y_val: Optional[List[float]] = None
    seed: int = 0
    model_kind: str = Field(
        "random_forest",
        description="One of: mean, random_forest. XGBoost-quantile is "
                    "supported by the engine but kept off the API surface "
                    "for Session 6.5 because its uncertainty calibration "
                    "needs the larger 6.3 corpus to stabilise.",
    )


class ALCycleResponse(BaseModel):
    cycle_index: int
    n_train_before: int
    n_train_after: int
    n_pool_before: int
    n_pool_after: int
    queried_indices: List[int]
    queried_targets: List[float]
    val_mae: Optional[float]
    val_rmse: Optional[float]
    val_r2: Optional[float]
    val_spearman: Optional[float]
    cumulative_best: float
    acquisition: str


class ALCampaignResponse(BaseModel):
    id: uuid.UUID
    name: str
    owner_id: uuid.UUID
    created_at: datetime
    n_cycles_completed: int
    cycles: List[ALCycleResponse]
    cumulative_best_history: List[float]
    final_val_mae: Optional[float]


# ---------------------------------------------------------------------------
# In-memory store (Session 6.5b promotes this to a DB table)
# ---------------------------------------------------------------------------


_STORE: Dict[uuid.UUID, ALCampaignResponse] = {}


def _model_for(kind: str):
    from backend.common.ml.models.baselines import (
        MeanRegressor,
        RandomForestRegressor,
    )

    if kind == "mean":
        return MeanRegressor()
    if kind == "random_forest":
        return RandomForestRegressor(
            n_estimators=200, n_jobs=1, random_state=0,
        )
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"unknown model_kind {kind!r}; choose: mean, random_forest",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/campaigns",
    response_model=ALCampaignResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start an active-learning campaign",
)
async def create_al_campaign(
    body: ALCampaignCreate,
    current_user: User = Depends(get_current_active_user),
) -> ALCampaignResponse:
    """Run a synchronous AL campaign and return the cycle log.

    The engine refits the chosen baseline model each cycle, scores the
    unlabelled pool with the requested acquisition function, takes the
    top-``query_size`` candidates, and pulls their ground-truth labels
    from the supplied ``y_pool`` (the cheating-oracle path).

    Synchronous on purpose for Session 6.5 — the engine's runtime on
    a few-thousand-row pool with the RandomForest baseline is < 30 s,
    well inside a typical HTTP timeout. Session 6.5b moves the
    long-running CGCNN-ensemble path to a Celery ``ml`` task and
    introduces ``GET /al/campaigns/{id}/status`` for polling.
    """
    from backend.common.ml.active_learning_v2 import ALCampaign

    X_pool = np.asarray(body.X_pool, dtype=np.float64)
    y_pool = np.asarray(body.y_pool, dtype=np.float64)
    if X_pool.ndim != 2 or X_pool.shape[0] != y_pool.shape[0]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"X_pool must be 2-D matching y_pool length; got "
                f"X_pool shape={X_pool.shape}, y_pool length={y_pool.shape[0]}"
            ),
        )
    X_val = (
        np.asarray(body.X_val, dtype=np.float64)
        if body.X_val is not None else None
    )
    y_val = (
        np.asarray(body.y_val, dtype=np.float64)
        if body.y_val is not None else None
    )

    try:
        camp = ALCampaign(
            X_pool=X_pool, y_pool=y_pool,
            initial_train_indices=body.initial_train_indices,
            model=_model_for(body.model_kind),
            acquisition=body.acquisition,
            query_size=body.query_size,
            n_cycles=body.n_cycles,
            maximize=body.maximize,
            beta=body.beta, xi=body.xi,
            X_val=X_val, y_val=y_val, seed=body.seed,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc),
        ) from exc

    cycles = camp.run()
    campaign_id = uuid.uuid4()
    response = ALCampaignResponse(
        id=campaign_id, name=body.name,
        owner_id=current_user.id, created_at=datetime.utcnow(),
        n_cycles_completed=len(cycles),
        cycles=[
            ALCycleResponse(
                cycle_index=c.cycle_index,
                n_train_before=c.n_train_before,
                n_train_after=c.n_train_after,
                n_pool_before=c.n_pool_before,
                n_pool_after=c.n_pool_after,
                queried_indices=c.queried_indices,
                queried_targets=c.queried_targets,
                val_mae=c.val_mae, val_rmse=c.val_rmse,
                val_r2=c.val_r2, val_spearman=c.val_spearman,
                cumulative_best=c.cumulative_best,
                acquisition=c.acquisition,
            )
            for c in cycles
        ],
        cumulative_best_history=[c.cumulative_best for c in cycles],
        final_val_mae=cycles[-1].val_mae if cycles else None,
    )
    _STORE[campaign_id] = response
    logger.info(
        "AL campaign %s (%s) completed: %d cycles, final_val_mae=%s",
        campaign_id, body.name, len(cycles), response.final_val_mae,
    )
    return response


@router.get(
    "/campaigns/{campaign_id}",
    response_model=ALCampaignResponse,
    summary="Read an active-learning campaign log",
)
async def get_al_campaign(
    campaign_id: uuid.UUID,
    current_user: User = Depends(get_current_active_user),
) -> ALCampaignResponse:
    camp = _STORE.get(campaign_id)
    if camp is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"AL campaign {campaign_id} not found",
        )
    if camp.owner_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="not authorized to view this AL campaign",
        )
    return camp


@router.get(
    "/campaigns",
    response_model=List[ALCampaignResponse],
    summary="List active-learning campaigns",
)
async def list_al_campaigns(
    current_user: User = Depends(get_current_active_user),
) -> List[ALCampaignResponse]:
    """List campaigns owned by the current user (admins see all)."""
    if current_user.is_admin:
        return list(_STORE.values())
    return [c for c in _STORE.values() if c.owner_id == current_user.id]
