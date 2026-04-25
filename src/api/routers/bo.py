"""Phase 7 / Session 7.1 — Bayesian-optimization router.

Single endpoint: ``POST /api/v1/bo/suggest`` accepts a JSON
``{space, objectives, history, q, ...}`` body, runs the
:func:`backend.common.ml.bo_v2.suggest` engine, and returns the next
``q`` decoded points.

Stateless on purpose for Session 7.1 — the user supplies the full
history with each call, and the router never persists it. Session 7.2
introduces the campaign orchestrator, which owns the history + budget
+ halting logic and calls this endpoint as a sub-step.

Constraints in the request body:

- ``inequalities`` → linear inequality constraints over encoded dims
  (``coeffs · x <= rhs``).
- Charge-neutrality / prototype filters → caller-side ``feasibility_fn``
  isn't supported over JSON; for those, callers either drop down to
  the Python engine directly (recommended for materials work) or wait
  for Session 7.2's campaign-level feasibility hook.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..auth.security import get_current_active_user
from ..models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/bo",
    tags=["bayesian-optimization"],
    dependencies=[Depends(get_current_active_user)],
    responses={
        401: {"description": "Not authenticated"},
        400: {"description": "Invalid space / history / objectives"},
    },
)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class _ContinuousDimSchema(BaseModel):
    kind: str = Field("continuous", pattern="^continuous$")
    name: str
    low: float
    high: float


class _IntegerDimSchema(BaseModel):
    kind: str = Field("integer", pattern="^integer$")
    name: str
    low: int
    high: int


class _CategoricalDimSchema(BaseModel):
    kind: str = Field("categorical", pattern="^categorical$")
    name: str
    choices: List[Any] = Field(..., min_length=2)


class _SimplexDimSchema(BaseModel):
    kind: str = Field("simplex", pattern="^simplex$")
    name: str
    components: List[str] = Field(..., min_length=2)
    minimum: float = 0.0


# Discriminated union with manual dispatch — Pydantic v2 supports
# ``Discriminator`` but FastAPI generates cleaner OpenAPI for plain
# ``Union[...]`` here. We dispatch in :func:`_build_space`.
DimSchema = Any  # one of the four above


class _ObjectiveSchema(BaseModel):
    name: str
    minimize: bool = False


class _HistoryPointSchema(BaseModel):
    point: Dict[str, Any]
    y: List[float] = Field(..., min_length=1)


class _LinearInequalitySchema(BaseModel):
    coeffs: Dict[str, float]
    rhs: float


class BOSuggestRequest(BaseModel):
    """Single-shot BO suggest request.

    The ``history`` field uses **decoded** points (human-readable
    dicts), which the router encodes via :meth:`Space.encode` before
    handing off to the engine. This keeps the API caller's payload
    aligned with the response format and avoids leaking the encoded
    layout (which is an internal detail of how simplex/categorical
    dims are flattened for the GP).
    """

    space: List[Dict[str, Any]] = Field(
        ..., min_length=1, description="List of dim specs",
    )
    objectives: List[_ObjectiveSchema] = Field(..., min_length=1)
    history: List[_HistoryPointSchema] = Field(
        ..., min_length=2,
        description="Past observations (≥ 2 required to fit the GP).",
    )
    q: int = Field(1, ge=1, le=20)
    inequalities: List[_LinearInequalitySchema] = Field(default_factory=list)
    ref_point: Optional[List[float]] = None
    seed: Optional[int] = None


class BOSuggestResponse(BaseModel):
    candidates: List[Dict[str, Any]]
    n_candidates: int


class ParetoFrontRequest(BaseModel):
    """Request body for the Pareto-front helper.

    Carries the raw N×M objective matrix ``Y`` (Python list of lists)
    and a ``minimize`` flag per objective. Matches the
    :func:`backend.common.ml.bo_v2.pareto_front` signature exactly.
    """

    Y: List[List[float]] = Field(
        ..., min_length=1,
        description="N×M matrix of objective values.",
    )
    minimize: List[bool] = Field(
        ..., min_length=1,
        description="Per-objective minimize flag (length must match Y[0]).",
    )


class ParetoFrontResponse(BaseModel):
    """Boolean mask + the indices of Pareto-optimal rows.

    The frontend's MO campaign-detail page consumes ``optimal_indices``
    to highlight the front in a recharts scatter; the boolean mask is
    handy for callers that prefer indexed-into-original-order access.
    """

    mask: List[bool]
    optimal_indices: List[int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_space(specs: List[Dict[str, Any]]):
    """Translate JSON dim specs to :class:`Space`."""
    from backend.common.ml.bo_v2 import (
        CategoricalDim, ContinuousDim, IntegerDim, SimplexSpace, Space,
    )

    dims = []
    for spec in specs:
        kind = spec.get("kind")
        if kind == "continuous":
            dims.append(ContinuousDim(
                name=spec["name"], low=spec["low"], high=spec["high"],
            ))
        elif kind == "integer":
            dims.append(IntegerDim(
                name=spec["name"], low=spec["low"], high=spec["high"],
            ))
        elif kind == "categorical":
            dims.append(CategoricalDim(
                name=spec["name"], choices=spec["choices"],
            ))
        elif kind == "simplex":
            dims.append(SimplexSpace(
                name=spec["name"], components=spec["components"],
                minimum=spec.get("minimum", 0.0),
            ))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"unknown dim kind {kind!r}; expected one of "
                    "continuous, integer, categorical, simplex"
                ),
            )
    return Space(dims=dims)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/suggest",
    response_model=BOSuggestResponse,
    status_code=status.HTTP_200_OK,
    summary="Suggest next BO candidates",
    description=(
        "Fit a GP on the supplied history and return ``q`` next-best "
        "candidates from a `qLogExpectedImprovement` (single-objective) "
        "or `qLogExpectedHypervolumeImprovement` (multi-objective, "
        "qEHVI) acquisition. Stateless — caller owns the history."
    ),
)
async def suggest_bo(
    body: BOSuggestRequest,
    current_user: User = Depends(get_current_active_user),
) -> BOSuggestResponse:
    from backend.common.ml.bo_v2 import (
        History, LinearInequality, Objective, suggest,
    )

    space = _build_space(body.space)
    objectives = [
        Objective(name=o.name, minimize=o.minimize) for o in body.objectives
    ]
    hist = History(objectives=objectives)
    try:
        for hp in body.history:
            x = space.encode(hp.point)
            hist.add(x, hp.y)
        ineqs = [
            LinearInequality(coeffs=ie.coeffs, rhs=ie.rhs)
            for ie in body.inequalities
        ]
        cands = suggest(
            space=space,
            objectives=objectives,
            history=hist,
            q=body.q,
            inequalities=ineqs,
            ref_point=body.ref_point,
            seed=body.seed,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        # Engine raises RuntimeError when feasibility fails after
        # retries — surface as 422 (semantic but unprocessable).
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc),
        ) from exc

    logger.info(
        "BO suggest: user=%s q=%d objectives=%d history=%d → %d candidates",
        current_user.id, body.q, len(objectives), len(body.history), len(cands),
    )
    return BOSuggestResponse(candidates=cands, n_candidates=len(cands))


@router.post(
    "/pareto-front",
    response_model=ParetoFrontResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute the Pareto-optimal mask of an objective matrix",
    description=(
        "Cheap pure-numpy helper for the frontend's multi-objective "
        "campaign-detail page (Phase 9 / Session 9.4). Wraps "
        "``backend.common.ml.bo_v2.pareto_front``."
    ),
)
async def pareto_front_endpoint(
    body: ParetoFrontRequest,
    current_user: User = Depends(get_current_active_user),
) -> ParetoFrontResponse:
    import numpy as np

    from backend.common.ml.bo_v2 import pareto_front

    Y = np.asarray(body.Y, dtype=float)
    if Y.ndim != 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Y must be 2-D; got shape {Y.shape}",
        )
    if len(body.minimize) != Y.shape[1]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"minimize length {len(body.minimize)} doesn't match "
                f"Y's column count {Y.shape[1]}"
            ),
        )
    try:
        mask = pareto_front(Y, minimize=body.minimize)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc),
        ) from exc
    return ParetoFrontResponse(
        mask=mask.tolist(),
        optimal_indices=[i for i, b in enumerate(mask) if b],
    )
