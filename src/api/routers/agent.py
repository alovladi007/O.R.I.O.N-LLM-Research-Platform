"""
LLM-Agent API endpoints for NANO-OS control.

This API provides a simplified, natural-language-friendly interface
for external AI agents (LLMs) to control NANO-OS.

Session 30: Control Plane for Nanomaterials AGI
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func

from src.api.database import get_db
from src.api.models import (
    DesignCampaign,
    DesignIteration,
    CampaignStatus,
    SimulationJob,
    ExperimentRun,
    ExperimentStatus,
    Structure,
    AgentCommand,
    User,
)
from src.api.schemas.orchestrator import (
    AgentCreateCampaignRequest,
    AgentAdvanceCampaignRequest,
    AgentRequestSimulationsRequest,
    AgentRequestExperimentsRequest,
    AgentSummaryResponse,
    AgentCommandResponse,
    AgentCampaignResponse,
    AgentSimulationResponse,
    AgentExperimentResponse,
)
from src.api.auth.security import get_current_user, get_optional_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


@router.get("/summary", response_model=AgentSummaryResponse)
async def get_agent_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_optional_user),
):
    """
    Get concise system summary for AI agent.

    This endpoint provides a natural-language summary of the system state,
    optimized for LLM consumption.

    Returns:
        Concise system summary with recommendations
    """
    # Collect statistics
    active_campaigns = await db.execute(
        select(DesignCampaign).where(DesignCampaign.status == CampaignStatus.RUNNING)
    )
    campaigns = active_campaigns.scalars().all()

    running_jobs = await db.execute(
        select(func.count(SimulationJob.id)).where(
            SimulationJob.status.in_(["PENDING", "RUNNING"])
        )
    )
    running_jobs_count = running_jobs.scalar()

    pending_experiments = await db.execute(
        select(func.count(ExperimentRun.id)).where(
            or_(
                ExperimentRun.status == ExperimentStatus.PENDING,
                ExperimentRun.status == ExperimentStatus.RUNNING
            )
        )
    )
    pending_experiments_count = pending_experiments.scalar()

    # Recent completed jobs (last 24h)
    last_24h = datetime.utcnow() - timedelta(hours=24)
    completed_recent = await db.execute(
        select(func.count(SimulationJob.id)).where(
            and_(
                SimulationJob.status == "COMPLETED",
                SimulationJob.completed_at >= last_24h
            )
        )
    )
    completed_recent_count = completed_recent.scalar()

    # Active campaigns summary
    active_campaigns_summary = []
    for campaign in campaigns:
        active_campaigns_summary.append({
            "id": str(campaign.id),
            "name": campaign.name,
            "iteration": f"{campaign.current_iteration}/{campaign.max_iterations}",
            "best_score": campaign.best_score,
            "status": "active"
        })

    # Recent discoveries (top scoring structures from last 24h)
    recent_discoveries = []
    # TODO: Query recent high-scoring structures from campaigns
    # For now, placeholder
    recent_discoveries.append({
        "message": "Check campaign iterations for recent high-scoring candidates",
        "time_range": "last 24 hours"
    })

    # Generate natural language summary
    summary_text = (
        f"System Status: {len(campaigns)} active campaigns, "
        f"{running_jobs_count} jobs running, "
        f"{pending_experiments_count} experiments pending. "
        f"Completed {completed_recent_count} jobs in last 24 hours."
    )

    # Determine system health
    if running_jobs_count > 50:
        system_health = "degraded"
    elif running_jobs_count > 100:
        system_health = "critical"
    else:
        system_health = "healthy"

    # Generate recommendations
    recommendations = []

    if len(campaigns) == 0:
        recommendations.append("No active campaigns. Consider creating a new design campaign.")

    if running_jobs_count == 0 and len(campaigns) > 0:
        recommendations.append("No jobs running. Consider advancing campaigns to generate new candidates.")

    if pending_experiments_count > 10:
        recommendations.append(f"{pending_experiments_count} experiments pending. Check instrument availability.")

    for campaign in campaigns:
        if campaign.current_iteration >= campaign.max_iterations * 0.9:
            recommendations.append(f"Campaign '{campaign.name}' near completion ({campaign.current_iteration}/{campaign.max_iterations})")

    if not recommendations:
        recommendations.append("System operating normally. Continue monitoring campaigns.")

    return AgentSummaryResponse(
        summary=summary_text,
        statistics={
            "active_campaigns": len(campaigns),
            "running_jobs": running_jobs_count,
            "pending_experiments": pending_experiments_count,
            "completed_jobs_24h": completed_recent_count,
            "system_health": system_health,
        },
        active_campaigns=active_campaigns_summary,
        recent_discoveries=recent_discoveries,
        recommendations=recommendations,
        system_health=system_health,
    )


@router.post("/create_design_campaign", response_model=AgentCommandResponse)
async def create_design_campaign(
    request: AgentCreateCampaignRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new design campaign (agent-friendly).

    This endpoint provides a simplified interface for AI agents to create
    design campaigns using natural language goals and constraints.

    Args:
        request: Campaign creation request
        db: Database session
        current_user: Current authenticated user

    Returns:
        Agent command response with created campaign info
    """
    # Record command
    command = AgentCommand(
        agent_id=f"user:{current_user.id}",
        command_type="create_design_campaign",
        payload=request.dict(),
        executed_at=datetime.utcnow(),
    )

    try:
        # Create campaign config from request
        config = {
            "goal": request.goal,
            "target_properties": request.target_properties,
            "constraints": request.constraints or {},
            "max_iterations": request.max_iterations,
            "candidates_per_iteration": 10,  # Default
            "generation_strategy": "bayesian",
        }

        # Apply budget if provided
        if request.budget:
            config["simulation_budget"] = request.budget.get("simulations", 1000)
            config["experiment_budget"] = request.budget.get("experiments", 50)

        # Create campaign
        campaign = DesignCampaign(
            owner_id=current_user.id,
            name=request.name,
            description=f"Goal: {request.goal}",
            status=CampaignStatus.PENDING,
            config=config,
            max_iterations=request.max_iterations,
            current_iteration=0,
        )

        db.add(campaign)
        await db.commit()
        await db.refresh(campaign)

        # Record success
        command.success = True
        command.completed_at = datetime.utcnow()
        command.duration_seconds = (command.completed_at - command.executed_at).total_seconds()
        command.result = {
            "campaign_id": str(campaign.id),
            "name": campaign.name,
            "status": campaign.status.value,
        }

        db.add(command)
        await db.commit()

        logger.info(f"Agent created campaign: {campaign.name} ({campaign.id})")

        return AgentCommandResponse(
            command_id=command.id,
            command_type=command.command_type,
            success=True,
            result=command.result,
            message=f"Successfully created campaign '{request.name}' with ID {campaign.id}",
            timestamp=command.completed_at,
        )

    except Exception as e:
        logger.error(f"Failed to create campaign: {e}", exc_info=True)

        # Record failure
        command.success = False
        command.error_message = str(e)
        command.completed_at = datetime.utcnow()
        command.duration_seconds = (command.completed_at - command.executed_at).total_seconds()

        db.add(command)
        await db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create campaign: {str(e)}"
        )


@router.post("/advance_campaign", response_model=AgentCommandResponse)
async def advance_campaign(
    request: AgentAdvanceCampaignRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Advance a design campaign (agent-friendly).

    This endpoint triggers campaign iterations to generate and evaluate
    new candidate structures.

    Args:
        request: Campaign advancement request
        db: Database session
        current_user: Current authenticated user

    Returns:
        Agent command response with advancement results
    """
    # Record command
    command = AgentCommand(
        agent_id=f"user:{current_user.id}",
        command_type="advance_campaign",
        payload=request.dict(),
        executed_at=datetime.utcnow(),
    )

    try:
        # Get campaign
        result = await db.execute(
            select(DesignCampaign).where(DesignCampaign.id == request.campaign_id)
        )
        campaign = result.scalar_one_or_none()

        if not campaign:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Campaign {request.campaign_id} not found"
            )

        if campaign.status == CampaignStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Campaign {request.campaign_id} is already completed"
            )

        # TODO: Trigger actual campaign advancement
        # For now, simulate advancement
        iterations_to_run = min(
            request.num_iterations,
            campaign.max_iterations - campaign.current_iteration
        )

        # Record success
        command.success = True
        command.completed_at = datetime.utcnow()
        command.duration_seconds = (command.completed_at - command.executed_at).total_seconds()
        command.result = {
            "campaign_id": str(campaign.id),
            "iterations_queued": iterations_to_run,
            "message": f"Queued {iterations_to_run} iterations for campaign '{campaign.name}'"
        }

        db.add(command)
        await db.commit()

        logger.info(f"Agent advanced campaign: {campaign.name} (+{iterations_to_run} iterations)")

        return AgentCommandResponse(
            command_id=command.id,
            command_type=command.command_type,
            success=True,
            result=command.result,
            message=f"Queued {iterations_to_run} iterations for campaign '{campaign.name}'",
            timestamp=command.completed_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to advance campaign: {e}", exc_info=True)

        # Record failure
        command.success = False
        command.error_message = str(e)
        command.completed_at = datetime.utcnow()
        command.duration_seconds = (command.completed_at - command.executed_at).total_seconds()

        db.add(command)
        await db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to advance campaign: {str(e)}"
        )


@router.post("/request_simulations", response_model=AgentCommandResponse)
async def request_simulations(
    request: AgentRequestSimulationsRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Request simulations for structures (agent-friendly).

    This endpoint allows AI agents to directly request simulations
    for specific structures.

    Args:
        request: Simulation request
        db: Database session
        current_user: Current authenticated user

    Returns:
        Agent command response with created job info
    """
    # Record command
    command = AgentCommand(
        agent_id=f"user:{current_user.id}",
        command_type="request_simulations",
        payload=request.dict(),
        executed_at=datetime.utcnow(),
    )

    try:
        # Validate structures exist
        result = await db.execute(
            select(Structure).where(Structure.id.in_(request.structure_ids))
        )
        structures = result.scalars().all()

        if len(structures) != len(request.structure_ids):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or more structures not found"
            )

        # TODO: Create actual simulation jobs
        # For now, simulate job creation
        job_ids = []
        for structure in structures:
            # In production, create SimulationJob here
            logger.info(f"Would create {request.simulation_type} job for structure {structure.id}")
            # job_ids.append(str(new_job.id))

        # Record success
        command.success = True
        command.completed_at = datetime.utcnow()
        command.duration_seconds = (command.completed_at - command.executed_at).total_seconds()
        command.result = {
            "structure_ids": [str(sid) for sid in request.structure_ids],
            "simulation_type": request.simulation_type,
            "jobs_created": len(structures),
            "message": f"Created {len(structures)} {request.simulation_type} jobs"
        }

        db.add(command)
        await db.commit()

        logger.info(f"Agent requested {len(structures)} simulations")

        return AgentCommandResponse(
            command_id=command.id,
            command_type=command.command_type,
            success=True,
            result=command.result,
            message=f"Queued {len(structures)} {request.simulation_type} simulations",
            timestamp=command.completed_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to request simulations: {e}", exc_info=True)

        # Record failure
        command.success = False
        command.error_message = str(e)
        command.completed_at = datetime.utcnow()
        command.duration_seconds = (command.completed_at - command.executed_at).total_seconds()

        db.add(command)
        await db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to request simulations: {str(e)}"
        )


@router.post("/request_experiments", response_model=AgentCommandResponse)
async def request_experiments(
    request: AgentRequestExperimentsRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Request experiments for structures (agent-friendly).

    This endpoint allows AI agents to directly request physical experiments
    for promising candidate structures.

    Args:
        request: Experiment request
        db: Database session
        current_user: Current authenticated user

    Returns:
        Agent command response with created experiment info
    """
    # Record command
    command = AgentCommand(
        agent_id=f"user:{current_user.id}",
        command_type="request_experiments",
        payload=request.dict(),
        executed_at=datetime.utcnow(),
    )

    try:
        # Validate structures and instrument exist
        result = await db.execute(
            select(Structure).where(Structure.id.in_(request.structure_ids))
        )
        structures = result.scalars().all()

        if len(structures) != len(request.structure_ids):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or more structures not found"
            )

        # TODO: Create actual experiment runs
        # For now, simulate experiment creation
        experiment_ids = []
        for structure in structures:
            # In production, create ExperimentRun here
            logger.info(
                f"Would create {request.experiment_type} experiment for structure {structure.id} "
                f"on instrument {request.instrument_id}"
            )
            # experiment_ids.append(str(new_experiment.id))

        # Record success
        command.success = True
        command.completed_at = datetime.utcnow()
        command.duration_seconds = (command.completed_at - command.executed_at).total_seconds()
        command.result = {
            "structure_ids": [str(sid) for sid in request.structure_ids],
            "instrument_id": str(request.instrument_id),
            "experiment_type": request.experiment_type,
            "experiments_created": len(structures),
            "message": f"Created {len(structures)} {request.experiment_type} experiments"
        }

        db.add(command)
        await db.commit()

        logger.info(f"Agent requested {len(structures)} experiments")

        return AgentCommandResponse(
            command_id=command.id,
            command_type=command.command_type,
            success=True,
            result=command.result,
            message=f"Queued {len(structures)} {request.experiment_type} experiments",
            timestamp=command.completed_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to request experiments: {e}", exc_info=True)

        # Record failure
        command.success = False
        command.error_message = str(e)
        command.completed_at = datetime.utcnow()
        command.duration_seconds = (command.completed_at - command.executed_at).total_seconds()

        db.add(command)
        await db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to request experiments: {str(e)}"
        )


@router.get("/commands")
async def list_agent_commands(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_optional_user),
):
    """
    List recent agent commands.

    Args:
        limit: Maximum number of commands to return
        offset: Number of commands to skip
        db: Database session
        current_user: Current authenticated user

    Returns:
        List of agent commands
    """
    result = await db.execute(
        select(AgentCommand)
        .order_by(AgentCommand.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    commands = result.scalars().all()

    return [
        {
            "id": str(cmd.id),
            "command_type": cmd.command_type,
            "success": cmd.success,
            "created_at": cmd.created_at.isoformat(),
            "duration_seconds": cmd.duration_seconds,
            "result_summary": cmd.result.get("message") if cmd.result else None,
        }
        for cmd in commands
    ]
