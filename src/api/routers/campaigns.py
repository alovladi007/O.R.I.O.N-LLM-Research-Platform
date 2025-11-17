"""
Design Campaign API Router
===========================

Provides comprehensive endpoints for managing AI-driven design campaigns:

- Campaign lifecycle: Create, start, pause, resume, cancel
- Iteration management: Run campaign steps, view iterations
- Progress tracking: View campaign details, statistics, summaries
- Access control: Owner-based permissions with admin override

Each campaign represents an autonomous design loop that:
1. Generates candidate structures using AI strategies
2. Evaluates candidates using ML predictions
3. Iteratively improves designs toward target properties
4. Tracks best discoveries and optimization progress
"""

from fastapi import APIRouter, Depends, Query, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from sqlalchemy.orm import selectinload
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from ..database import get_db
from ..models.user import User
from ..models.campaign import DesignCampaign, DesignIteration, CampaignStatus
from ..schemas.campaign import (
    DesignCampaignCreate,
    DesignCampaignUpdate,
    DesignCampaignResponse,
    DesignIterationResponse,
    CampaignStepRequest,
    CampaignStepResponse,
    CampaignListResponse,
    CampaignSummaryResponse,
)
from ..auth.security import get_current_active_user
from backend.common.campaigns import DesignLoopService

logger = logging.getLogger(__name__)

router = APIRouter(
    responses={
        401: {"description": "Not authenticated"},
        403: {"description": "Not authorized"},
        404: {"description": "Campaign not found"}
    }
)


# Helper functions

def check_campaign_ownership(campaign: DesignCampaign, user: User) -> None:
    """
    Check if user owns campaign or is admin.

    Args:
        campaign: Campaign to check
        user: Current user

    Raises:
        HTTPException: 403 if user doesn't have access
    """
    if campaign.owner_id != user.id and not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this campaign"
        )


async def get_campaign_or_404(
    db: AsyncSession,
    campaign_id: uuid.UUID,
    user: User
) -> DesignCampaign:
    """
    Get campaign by ID with ownership check.

    Args:
        db: Database session
        campaign_id: Campaign UUID
        user: Current user

    Returns:
        DesignCampaign object

    Raises:
        HTTPException: 404 if not found, 403 if no access
    """
    campaign = await db.get(DesignCampaign, campaign_id)

    if not campaign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign {campaign_id} not found"
        )

    check_campaign_ownership(campaign, user)

    return campaign


# Endpoints

@router.post(
    "",
    response_model=DesignCampaignResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new design campaign",
    description="""
    Create a new AI-driven design campaign for materials discovery.

    A campaign defines:
    - **Target properties**: What you want to optimize (e.g., bandgap=2.0 eV)
    - **Constraints**: Structural limitations (elements, size, dimensionality)
    - **Strategy**: How to generate candidates (random, Bayesian, genetic, RL)
    - **Budget**: Max iterations and candidates per iteration

    The campaign starts in CREATED status. Use `/campaigns/{id}/start` to begin.

    Example:
    ```json
    {
        "name": "TMD Bandgap Optimization",
        "description": "Find 2D materials with 2eV bandgap",
        "config": {
            "target_properties": {
                "bandgap": {"value": 2.0, "tolerance": 0.2, "weight": 1.0}
            },
            "constraints": {
                "elements": ["Mo", "W", "S", "Se"],
                "dimensionality": 2,
                "max_atoms": 20
            },
            "max_iterations": 20,
            "candidates_per_iteration": 10,
            "generation_strategy": "bayesian"
        }
    }
    ```
    """,
    responses={
        201: {
            "description": "Campaign created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "owner_id": "660e8400-e29b-41d4-a716-446655440000",
                        "name": "TMD Bandgap Optimization",
                        "status": "CREATED",
                        "config": {
                            "target_properties": {
                                "bandgap": {"value": 2.0, "tolerance": 0.2}
                            },
                            "max_iterations": 20
                        },
                        "max_iterations": 20,
                        "current_iteration": 0,
                        "created_at": "2025-01-15T10:00:00Z"
                    }
                }
            }
        },
        400: {"description": "Invalid campaign configuration"}
    }
)
async def create_campaign(
    campaign_data: DesignCampaignCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> DesignCampaignResponse:
    """Create a new design campaign."""

    logger.info(f"Creating campaign '{campaign_data.name}' for user {current_user.id}")

    # Extract max_iterations from config
    max_iterations = campaign_data.config.max_iterations

    # Create campaign
    campaign = DesignCampaign(
        owner_id=current_user.id,
        name=campaign_data.name,
        description=campaign_data.description,
        status=CampaignStatus.CREATED,
        config=campaign_data.config.model_dump(),
        max_iterations=max_iterations,
        current_iteration=0,
        metadata=campaign_data.metadata or {}
    )

    db.add(campaign)
    await db.commit()
    await db.refresh(campaign)

    logger.info(f"Campaign created: {campaign.id}")

    return DesignCampaignResponse.model_validate(campaign)


@router.get(
    "",
    response_model=CampaignListResponse,
    summary="List campaigns",
    description="""
    List design campaigns with filtering and pagination.

    **Filtering:**
    - By status: `?status=RUNNING`
    - By owner: `?owner_id=<uuid>` (admin only)

    **Pagination:**
    - `skip`: Number of records to skip (default: 0)
    - `limit`: Max records to return (default: 20, max: 100)

    **Permissions:**
    - Regular users see only their own campaigns
    - Admins can see all campaigns and filter by owner

    **Sorting:**
    - Results sorted by created_at descending (newest first)
    """,
    responses={
        200: {
            "description": "List of campaigns",
            "content": {
                "application/json": {
                    "example": {
                        "campaigns": [
                            {
                                "id": "550e8400-e29b-41d4-a716-446655440000",
                                "name": "TMD Optimization",
                                "status": "RUNNING",
                                "current_iteration": 5,
                                "best_score": 0.87
                            }
                        ],
                        "total": 1,
                        "page": 1,
                        "page_size": 20
                    }
                }
            }
        }
    }
)
async def list_campaigns(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=100, description="Max records to return"),
    status: Optional[CampaignStatus] = Query(None, description="Filter by status"),
    owner_id: Optional[uuid.UUID] = Query(None, description="Filter by owner (admin only)"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> CampaignListResponse:
    """List user's campaigns with filtering and pagination."""

    # Build query
    query = select(DesignCampaign)

    # Filter by owner
    if owner_id:
        # Only admins can filter by other users
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can filter campaigns by owner"
            )
        query = query.where(DesignCampaign.owner_id == owner_id)
    else:
        # Non-admins see only their campaigns
        if not current_user.is_admin:
            query = query.where(DesignCampaign.owner_id == current_user.id)

    # Filter by status
    if status:
        query = query.where(DesignCampaign.status == status)

    # Sort by created_at descending
    query = query.order_by(DesignCampaign.created_at.desc())

    # Get total count
    count_query = select(func.count()).select_from(query.alias())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Apply pagination
    query = query.offset(skip).limit(limit)

    # Execute query
    result = await db.execute(query)
    campaigns = result.scalars().all()

    logger.info(f"Listed {len(campaigns)} campaigns (total: {total})")

    return CampaignListResponse(
        campaigns=[DesignCampaignResponse.model_validate(c) for c in campaigns],
        total=total,
        page=(skip // limit) + 1,
        page_size=limit
    )


@router.get(
    "/{campaign_id}",
    response_model=DesignCampaignResponse,
    summary="Get campaign details",
    description="""
    Get detailed information about a specific campaign.

    Returns:
    - Campaign configuration and status
    - Current progress (iteration count, best score)
    - Timestamps (created, started, completed)
    - Owner information

    **Permissions:**
    - Owner can view their campaigns
    - Admins can view any campaign
    """,
    responses={
        200: {
            "description": "Campaign details",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "name": "TMD Optimization",
                        "status": "RUNNING",
                        "current_iteration": 5,
                        "max_iterations": 20,
                        "best_score": 0.87,
                        "best_structure_id": "770e8400-e29b-41d4-a716-446655440000",
                        "created_at": "2025-01-15T10:00:00Z",
                        "started_at": "2025-01-15T10:05:00Z"
                    }
                }
            }
        },
        404: {"description": "Campaign not found"},
        403: {"description": "Not authorized to view this campaign"}
    }
)
async def get_campaign(
    campaign_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> DesignCampaignResponse:
    """Get campaign details with ownership check."""

    campaign = await get_campaign_or_404(db, campaign_id, current_user)

    return DesignCampaignResponse.model_validate(campaign)


@router.patch(
    "/{campaign_id}",
    response_model=DesignCampaignResponse,
    summary="Update campaign",
    description="""
    Update campaign properties.

    **Updatable fields:**
    - name: Campaign name
    - description: Campaign description
    - config: Campaign configuration

    **Restrictions:**
    - Can only update campaigns in CREATED or PAUSED status
    - Cannot update RUNNING, COMPLETED, FAILED, or CANCELLED campaigns
    - Must be owner or admin

    **Note:** Updating config will affect future iterations but not completed ones.
    """,
    responses={
        200: {
            "description": "Campaign updated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "name": "Updated Campaign Name",
                        "status": "CREATED",
                        "updated_at": "2025-01-15T10:30:00Z"
                    }
                }
            }
        },
        400: {"description": "Invalid state for update"},
        403: {"description": "Not authorized to update this campaign"},
        404: {"description": "Campaign not found"}
    }
)
async def update_campaign(
    campaign_id: uuid.UUID,
    update_data: DesignCampaignUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> DesignCampaignResponse:
    """Update campaign with state validation."""

    campaign = await get_campaign_or_404(db, campaign_id, current_user)

    # Check if campaign can be updated
    if campaign.status not in [CampaignStatus.CREATED, CampaignStatus.PAUSED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot update campaign in {campaign.status.value} status. "
                   f"Only CREATED or PAUSED campaigns can be updated."
        )

    # Update fields
    if update_data.name is not None:
        campaign.name = update_data.name

    if update_data.description is not None:
        campaign.description = update_data.description

    if update_data.config is not None:
        campaign.config = update_data.config.model_dump()
        campaign.max_iterations = update_data.config.max_iterations

    if update_data.metadata is not None:
        campaign.metadata = update_data.metadata

    campaign.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(campaign)

    logger.info(f"Campaign {campaign_id} updated")

    return DesignCampaignResponse.model_validate(campaign)


@router.post(
    "/{campaign_id}/start",
    response_model=DesignCampaignResponse,
    summary="Start campaign",
    description="""
    Start a campaign that is in CREATED status.

    This transitions the campaign from CREATED → RUNNING and records the start time.

    **Prerequisites:**
    - Campaign must be in CREATED status
    - Must be owner or admin

    **After starting:**
    - Use `/campaigns/{id}/step` to run iterations
    - Campaign status becomes RUNNING
    - started_at timestamp is recorded
    """,
    responses={
        200: {
            "description": "Campaign started",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "status": "RUNNING",
                        "started_at": "2025-01-15T10:05:00Z"
                    }
                }
            }
        },
        400: {"description": "Campaign not in CREATED status"},
        403: {"description": "Not authorized"},
        404: {"description": "Campaign not found"}
    }
)
async def start_campaign(
    campaign_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> DesignCampaignResponse:
    """Start a created campaign."""

    campaign = await get_campaign_or_404(db, campaign_id, current_user)

    # Validate status
    if campaign.status != CampaignStatus.CREATED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot start campaign in {campaign.status.value} status. "
                   f"Only CREATED campaigns can be started."
        )

    # Update status
    campaign.status = CampaignStatus.RUNNING
    campaign.started_at = datetime.utcnow()
    campaign.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(campaign)

    logger.info(f"Campaign {campaign_id} started")

    return DesignCampaignResponse.model_validate(campaign)


@router.post(
    "/{campaign_id}/pause",
    response_model=DesignCampaignResponse,
    summary="Pause running campaign",
    description="""
    Pause a campaign that is currently RUNNING.

    This transitions the campaign from RUNNING → PAUSED.

    **Use cases:**
    - Review intermediate results before continuing
    - Adjust configuration based on progress
    - Temporarily halt resource usage

    **After pausing:**
    - Use `/campaigns/{id}/update` to modify config
    - Use `/campaigns/{id}/resume` to continue
    """,
    responses={
        200: {
            "description": "Campaign paused",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "status": "PAUSED",
                        "current_iteration": 5
                    }
                }
            }
        },
        400: {"description": "Campaign not in RUNNING status"},
        403: {"description": "Not authorized"},
        404: {"description": "Campaign not found"}
    }
)
async def pause_campaign(
    campaign_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> DesignCampaignResponse:
    """Pause a running campaign."""

    campaign = await get_campaign_or_404(db, campaign_id, current_user)

    # Validate status
    if campaign.status != CampaignStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot pause campaign in {campaign.status.value} status. "
                   f"Only RUNNING campaigns can be paused."
        )

    # Update status
    campaign.status = CampaignStatus.PAUSED
    campaign.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(campaign)

    logger.info(f"Campaign {campaign_id} paused")

    return DesignCampaignResponse.model_validate(campaign)


@router.post(
    "/{campaign_id}/resume",
    response_model=DesignCampaignResponse,
    summary="Resume paused campaign",
    description="""
    Resume a campaign that is PAUSED.

    This transitions the campaign from PAUSED → RUNNING.

    **After resuming:**
    - Use `/campaigns/{id}/step` to continue running iterations
    - Campaign continues from where it left off
    """,
    responses={
        200: {
            "description": "Campaign resumed",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "status": "RUNNING",
                        "current_iteration": 5
                    }
                }
            }
        },
        400: {"description": "Campaign not in PAUSED status"},
        403: {"description": "Not authorized"},
        404: {"description": "Campaign not found"}
    }
)
async def resume_campaign(
    campaign_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> DesignCampaignResponse:
    """Resume a paused campaign."""

    campaign = await get_campaign_or_404(db, campaign_id, current_user)

    # Validate status
    if campaign.status != CampaignStatus.PAUSED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot resume campaign in {campaign.status.value} status. "
                   f"Only PAUSED campaigns can be resumed."
        )

    # Update status
    campaign.status = CampaignStatus.RUNNING
    campaign.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(campaign)

    logger.info(f"Campaign {campaign_id} resumed")

    return DesignCampaignResponse.model_validate(campaign)


@router.post(
    "/{campaign_id}/cancel",
    response_model=DesignCampaignResponse,
    summary="Cancel campaign",
    description="""
    Cancel a campaign permanently.

    This transitions the campaign to CANCELLED status (terminal state).

    **Warning:**
    - This action cannot be undone
    - Campaign cannot be resumed after cancellation
    - Results and iterations are preserved for analysis

    **Use cases:**
    - Campaign is not converging
    - Changed requirements
    - Resource constraints
    """,
    responses={
        200: {
            "description": "Campaign cancelled",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "status": "CANCELLED",
                        "completed_at": "2025-01-15T11:00:00Z"
                    }
                }
            }
        },
        400: {"description": "Campaign already in terminal state"},
        403: {"description": "Not authorized"},
        404: {"description": "Campaign not found"}
    }
)
async def cancel_campaign(
    campaign_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> DesignCampaignResponse:
    """Cancel a campaign."""

    campaign = await get_campaign_or_404(db, campaign_id, current_user)

    # Check if already in terminal state
    if campaign.is_terminal:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Campaign is already in terminal state {campaign.status.value}"
        )

    # Update status
    campaign.status = CampaignStatus.CANCELLED
    campaign.completed_at = datetime.utcnow()
    campaign.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(campaign)

    logger.info(f"Campaign {campaign_id} cancelled")

    return DesignCampaignResponse.model_validate(campaign)


@router.post(
    "/{campaign_id}/step",
    response_model=CampaignStepResponse,
    summary="Run campaign iterations",
    description="""
    Execute one or more design loop iterations.

    **What happens during an iteration:**
    1. Generate candidate structures using AI strategy
    2. Evaluate candidates with ML property predictions
    3. Score candidates against target properties
    4. Update campaign best score and structure
    5. Record iteration metrics for analysis

    **Requirements:**
    - Campaign must be in RUNNING status
    - Campaign must not have reached max_iterations
    - Must be owner or admin

    **Parameters:**
    - `num_iterations`: Number of iterations to run (default: 1, max: 100)

    **Response:**
    - Updated campaign with new progress
    - List of iteration results
    - Status message

    **Auto-completion:**
    - Campaign automatically transitions to COMPLETED when max_iterations reached
    """,
    responses={
        200: {
            "description": "Iterations completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "campaign": {
                            "id": "550e8400-e29b-41d4-a716-446655440000",
                            "status": "RUNNING",
                            "current_iteration": 6,
                            "best_score": 0.87
                        },
                        "iterations": [
                            {
                                "iteration_index": 5,
                                "best_score_this_iter": 0.87,
                                "metrics": {
                                    "mean_score": 0.81,
                                    "improvement_from_previous": 0.05
                                }
                            }
                        ],
                        "message": "Completed 1 iteration. Best score: 0.87"
                    }
                }
            }
        },
        400: {
            "description": "Campaign not in RUNNING status or reached max iterations"
        },
        403: {"description": "Not authorized"},
        404: {"description": "Campaign not found"},
        500: {"description": "Iteration execution failed"}
    }
)
async def step_campaign(
    campaign_id: uuid.UUID,
    step_request: CampaignStepRequest = CampaignStepRequest(),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> CampaignStepResponse:
    """Run one or more campaign iterations."""

    campaign = await get_campaign_or_404(db, campaign_id, current_user)

    # Validate status
    if campaign.status != CampaignStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot run iterations on campaign in {campaign.status.value} status. "
                   f"Campaign must be RUNNING."
        )

    # Check iteration limit
    if campaign.current_iteration >= campaign.max_iterations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Campaign has reached max iterations ({campaign.max_iterations})"
        )

    # Run iterations
    iterations_to_run = min(
        step_request.num_iterations,
        campaign.max_iterations - campaign.current_iteration
    )

    logger.info(
        f"Running {iterations_to_run} iteration(s) for campaign {campaign_id}"
    )

    completed_iterations = []

    try:
        for i in range(iterations_to_run):
            logger.info(
                f"Running iteration {campaign.current_iteration + 1}/"
                f"{campaign.max_iterations}"
            )

            # Run iteration using DesignLoopService
            iteration = await DesignLoopService.run_iteration(db, campaign_id)

            completed_iterations.append(iteration)

            # Refresh campaign to get updated state
            await db.refresh(campaign)

            logger.info(
                f"Iteration {iteration.iteration_index} completed. "
                f"Best score this iter: {iteration.best_score_this_iter}"
            )

    except Exception as e:
        logger.error(f"Error running iteration: {str(e)}", exc_info=True)

        # Mark campaign as failed
        campaign.status = CampaignStatus.FAILED
        campaign.completed_at = datetime.utcnow()
        campaign.metadata = campaign.metadata or {}
        campaign.metadata["error"] = str(e)
        await db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Iteration failed: {str(e)}"
        )

    # Generate response message
    best_score_msg = (
        f"Best score: {campaign.best_score:.3f}"
        if campaign.best_score is not None
        else "No scores yet"
    )

    message = (
        f"Completed {len(completed_iterations)} iteration(s). {best_score_msg}. "
        f"Campaign status: {campaign.status.value}"
    )

    return CampaignStepResponse(
        campaign=DesignCampaignResponse.model_validate(campaign),
        iterations=[
            DesignIterationResponse.model_validate(it)
            for it in completed_iterations
        ],
        message=message
    )


@router.get(
    "/{campaign_id}/iterations",
    response_model=List[DesignIterationResponse],
    summary="List campaign iterations",
    description="""
    Get all iterations for a campaign.

    Returns iterations in order (iteration_index ascending).

    Each iteration contains:
    - Structures created and evaluated
    - Best score and structure for that iteration
    - Metrics (scores, diversity, improvement)
    - Strategy used
    - Timing information

    **Use cases:**
    - Analyze campaign progress over time
    - Identify which iterations found best results
    - Compare different generation strategies
    - Debug convergence issues
    """,
    responses={
        200: {
            "description": "List of iterations",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "iteration_index": 0,
                            "best_score_this_iter": 0.75,
                            "metrics": {"mean_score": 0.70}
                        },
                        {
                            "iteration_index": 1,
                            "best_score_this_iter": 0.82,
                            "metrics": {
                                "mean_score": 0.78,
                                "improvement_from_previous": 0.07
                            }
                        }
                    ]
                }
            }
        },
        403: {"description": "Not authorized"},
        404: {"description": "Campaign not found"}
    }
)
async def list_iterations(
    campaign_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> List[DesignIterationResponse]:
    """List all iterations for a campaign."""

    campaign = await get_campaign_or_404(db, campaign_id, current_user)

    # Query iterations
    query = (
        select(DesignIteration)
        .where(DesignIteration.campaign_id == campaign_id)
        .order_by(DesignIteration.iteration_index)
    )

    result = await db.execute(query)
    iterations = result.scalars().all()

    logger.info(f"Found {len(iterations)} iterations for campaign {campaign_id}")

    return [DesignIterationResponse.model_validate(it) for it in iterations]


@router.get(
    "/{campaign_id}/iterations/{iteration_index}",
    response_model=DesignIterationResponse,
    summary="Get specific iteration",
    description="""
    Get detailed information about a specific iteration.

    **Parameters:**
    - `iteration_index`: 0-based iteration number

    Returns complete iteration data including:
    - All structures created and evaluated
    - Detailed metrics and scores
    - Generation strategy used
    - Timing and performance data
    """,
    responses={
        200: {
            "description": "Iteration details",
            "content": {
                "application/json": {
                    "example": {
                        "id": "880e8400-e29b-41d4-a716-446655440000",
                        "campaign_id": "550e8400-e29b-41d4-a716-446655440000",
                        "iteration_index": 5,
                        "best_score_this_iter": 0.87,
                        "metrics": {
                            "scores": [0.87, 0.75, 0.82],
                            "mean_score": 0.81,
                            "improvement_from_previous": 0.05
                        },
                        "strategy_used": "bayesian"
                    }
                }
            }
        },
        403: {"description": "Not authorized"},
        404: {"description": "Campaign or iteration not found"}
    }
)
async def get_iteration(
    campaign_id: uuid.UUID,
    iteration_index: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> DesignIterationResponse:
    """Get specific iteration details."""

    campaign = await get_campaign_or_404(db, campaign_id, current_user)

    # Query iteration
    query = (
        select(DesignIteration)
        .where(
            and_(
                DesignIteration.campaign_id == campaign_id,
                DesignIteration.iteration_index == iteration_index
            )
        )
    )

    result = await db.execute(query)
    iteration = result.scalar_one_or_none()

    if not iteration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Iteration {iteration_index} not found for campaign {campaign_id}"
        )

    return DesignIterationResponse.model_validate(iteration)


@router.get(
    "/{campaign_id}/summary",
    response_model=CampaignSummaryResponse,
    summary="Get campaign statistics",
    description="""
    Get comprehensive statistics and analytics for a campaign.

    **Includes:**
    - Total iterations completed
    - Total structures created and evaluated
    - Best score and structure overall
    - Score history (best score at each iteration)
    - Mean scores over time
    - Improvements between iterations

    **Use cases:**
    - Visualize optimization progress
    - Analyze convergence behavior
    - Compare campaign performance
    - Generate reports and dashboards
    """,
    responses={
        200: {
            "description": "Campaign statistics",
            "content": {
                "application/json": {
                    "example": {
                        "campaign_id": "550e8400-e29b-41d4-a716-446655440000",
                        "total_iterations": 10,
                        "total_structures_created": 100,
                        "total_structures_evaluated": 100,
                        "best_score_overall": 0.92,
                        "best_structure_id": "770e8400-e29b-41d4-a716-446655440000",
                        "score_history": [0.65, 0.70, 0.75, 0.82, 0.87, 0.89, 0.90, 0.91, 0.92, 0.92],
                        "mean_scores": [0.60, 0.65, 0.70, 0.75, 0.80, 0.83, 0.85, 0.87, 0.88, 0.89],
                        "improvements": [None, 0.05, 0.05, 0.07, 0.05, 0.02, 0.01, 0.01, 0.01, 0.0]
                    }
                }
            }
        },
        403: {"description": "Not authorized"},
        404: {"description": "Campaign not found"}
    }
)
async def get_campaign_summary(
    campaign_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> CampaignSummaryResponse:
    """Get campaign statistics and summary."""

    campaign = await get_campaign_or_404(db, campaign_id, current_user)

    # Query all iterations
    query = (
        select(DesignIteration)
        .where(DesignIteration.campaign_id == campaign_id)
        .order_by(DesignIteration.iteration_index)
    )

    result = await db.execute(query)
    iterations = result.scalars().all()

    # Calculate statistics
    total_iterations = len(iterations)

    total_structures_created = sum(
        len(it.created_structures or [])
        for it in iterations
    )

    total_structures_evaluated = sum(
        len(it.evaluated_structures or [])
        for it in iterations
    )

    # Build score history and improvements
    score_history = []
    mean_scores = []
    improvements = []

    prev_best = None

    for iteration in iterations:
        # Best score at this iteration
        best_score = iteration.best_score_this_iter
        score_history.append(best_score)

        # Mean score from metrics
        mean_score = iteration.metrics.get("mean_score") if iteration.metrics else None
        mean_scores.append(mean_score)

        # Improvement from previous
        if prev_best is not None and best_score is not None:
            improvement = best_score - prev_best
            improvements.append(improvement)
        else:
            improvements.append(None)

        # Update running best
        if best_score is not None:
            if prev_best is None or best_score > prev_best:
                prev_best = best_score

    logger.info(f"Generated summary for campaign {campaign_id}: {total_iterations} iterations")

    return CampaignSummaryResponse(
        campaign_id=campaign_id,
        total_iterations=total_iterations,
        total_structures_created=total_structures_created,
        total_structures_evaluated=total_structures_evaluated,
        best_score_overall=campaign.best_score,
        best_structure_id=campaign.best_structure_id,
        score_history=score_history,
        mean_scores=mean_scores,
        improvements=improvements
    )
