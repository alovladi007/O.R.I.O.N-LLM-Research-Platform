"""
Orchestrator core logic for NANO-OS AGI control plane.

The orchestrator is the central brain that:
- Manages design campaigns
- Decides when to retrain models
- Schedules simulations and experiments
- Provides control interface for external LLM/agents

Session 30: Control Plane for Nanomaterials AGI
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from src.api.models import (
    OrchestratorState,
    OrchestratorRun,
    DesignCampaign,
    DesignIteration,
    CampaignStatus,
    SimulationJob,
    ExperimentRun,
    ExperimentStatus,
    MLModelRegistry,
    Structure,
)

logger = logging.getLogger(__name__)


def get_default_config() -> Dict[str, Any]:
    """
    Get default orchestrator configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        "max_simultaneous_simulations": 10,
        "max_simultaneous_experiments": 5,
        "training_frequency_hours": 24,
        "min_new_samples_for_retrain": 100,
        "experiment_budget_per_campaign": 50,
        "simulation_budget_per_campaign": 1000,
        "active_learning_threshold": 0.8,
        "bo_acquisition_function": "ei",
        "max_iterations_per_run": 3,
        "experiment_trigger_score_threshold": 0.9,
        "campaign_priorities": {},
    }


def collect_orchestrator_stats(db: Session) -> Dict[str, Any]:
    """
    Collect current system statistics.

    Args:
        db: Database session

    Returns:
        Statistics dictionary
    """
    # Count active campaigns
    active_campaigns = db.query(DesignCampaign).filter(
        DesignCampaign.status == CampaignStatus.RUNNING
    ).count()

    # Count pending and running jobs
    pending_jobs = db.query(SimulationJob).filter(
        SimulationJob.status == "PENDING"
    ).count()

    running_jobs = db.query(SimulationJob).filter(
        SimulationJob.status == "RUNNING"
    ).count()

    # Count completed jobs in last 24h
    last_24h = datetime.utcnow() - timedelta(hours=24)
    completed_jobs_24h = db.query(SimulationJob).filter(
        and_(
            SimulationJob.status == "COMPLETED",
            SimulationJob.completed_at >= last_24h
        )
    ).count()

    # Count pending experiments
    pending_experiments = db.query(ExperimentRun).filter(
        or_(
            ExperimentRun.status == ExperimentStatus.PENDING,
            ExperimentRun.status == ExperimentStatus.RUNNING
        )
    ).count()

    # Total structures
    total_structures = db.query(Structure).count()

    # Labeled samples (structures with simulation results)
    labeled_samples = db.query(Structure).join(
        SimulationJob,
        SimulationJob.structure_id == Structure.id
    ).filter(
        SimulationJob.status == "COMPLETED"
    ).distinct().count()

    # Models info
    models = db.query(MLModelRegistry).filter(
        MLModelRegistry.is_active == True
    ).all()

    models_ready_for_retrain = []
    for model in models:
        # Check if model has enough new samples since last training
        # (simplified logic - in production, track this properly)
        if model.training_samples_count and labeled_samples > model.training_samples_count * 1.2:
            models_ready_for_retrain.append(model.name)

    # Campaigns needing attention (stalled or low performance)
    campaigns_needing_attention = []
    campaigns = db.query(DesignCampaign).filter(
        DesignCampaign.status == CampaignStatus.RUNNING
    ).all()

    for campaign in campaigns:
        # Check if campaign hasn't advanced recently
        latest_iteration = db.query(DesignIteration).filter(
            DesignIteration.campaign_id == campaign.id
        ).order_by(DesignIteration.created_at.desc()).first()

        if latest_iteration:
            hours_since_last = (datetime.utcnow() - latest_iteration.created_at).total_seconds() / 3600
            if hours_since_last > 24:
                campaigns_needing_attention.append(str(campaign.id))

    return {
        "active_campaigns": active_campaigns,
        "pending_jobs": pending_jobs,
        "running_jobs": running_jobs,
        "completed_jobs_last_24h": completed_jobs_24h,
        "pending_experiments": pending_experiments,
        "total_structures": total_structures,
        "total_labeled_samples": labeled_samples,
        "models_ready_for_retrain": models_ready_for_retrain,
        "campaigns_needing_attention": campaigns_needing_attention,
        "timestamp": datetime.utcnow().isoformat(),
    }


def run_orchestrator_step(
    db: Session,
    orchestrator_id: UUID,
    triggered_by: str = "manual",
    trigger_context: Optional[Dict[str, Any]] = None
) -> OrchestratorRun:
    """
    Execute one orchestrator step.

    This is the main orchestration logic that:
    1. Inspects current system state
    2. Makes decisions about what to do next
    3. Triggers appropriate actions

    Args:
        db: Database session
        orchestrator_id: Orchestrator state ID
        triggered_by: Who/what triggered this run
        trigger_context: Additional context about the trigger

    Returns:
        OrchestratorRun record
    """
    logger.info(f"Starting orchestrator step (triggered by: {triggered_by})")

    # Get orchestrator state
    orchestrator = db.query(OrchestratorState).filter(
        OrchestratorState.id == orchestrator_id
    ).first()

    if not orchestrator:
        raise ValueError(f"Orchestrator {orchestrator_id} not found")

    if not orchestrator.is_active:
        raise ValueError(f"Orchestrator {orchestrator_id} is not active")

    # Create run record
    run = OrchestratorRun(
        orchestrator_id=orchestrator_id,
        triggered_by=triggered_by,
        trigger_context=trigger_context or {},
        started_at=datetime.utcnow(),
    )

    # Collect stats before
    stats_before = collect_orchestrator_stats(db)
    run.stats_before = stats_before

    # Initialize actions
    actions = {
        "campaigns_advanced": [],
        "simulations_launched": 0,
        "experiments_launched": 0,
        "models_retrained": [],
        "decisions": [],
    }

    try:
        config = orchestrator.config or get_default_config()

        # Step 1: Check if we should retrain models
        actions["decisions"].extend(
            _check_model_retraining(db, orchestrator, config, actions)
        )

        # Step 2: Advance design campaigns
        actions["decisions"].extend(
            _advance_campaigns(db, orchestrator, config, actions)
        )

        # Step 3: Schedule experiments for promising candidates
        actions["decisions"].extend(
            _schedule_experiments(db, orchestrator, config, actions)
        )

        # Mark run as successful
        run.success = True
        run.actions = actions

    except Exception as e:
        logger.error(f"Orchestrator step failed: {e}", exc_info=True)
        run.success = False
        run.error_message = str(e)
        actions["decisions"].append({
            "type": "error",
            "reason": str(e),
            "result": "failed"
        })
        run.actions = actions

    # Collect stats after
    stats_after = collect_orchestrator_stats(db)
    run.stats_after = stats_after

    # Complete run
    run.completed_at = datetime.utcnow()
    run.duration_seconds = (run.completed_at - run.started_at).total_seconds()

    # Update orchestrator state
    orchestrator.last_run_at = datetime.utcnow()
    orchestrator.run_count += 1
    orchestrator.total_simulations_launched += actions["simulations_launched"]
    orchestrator.total_experiments_launched += actions["experiments_launched"]
    orchestrator.total_trainings_launched += len(actions["models_retrained"])
    orchestrator.stats = stats_after

    # Save to database
    db.add(run)
    db.commit()
    db.refresh(run)

    logger.info(
        f"Orchestrator step completed: "
        f"{len(actions['campaigns_advanced'])} campaigns advanced, "
        f"{actions['simulations_launched']} simulations launched, "
        f"{actions['experiments_launched']} experiments launched, "
        f"{len(actions['models_retrained'])} models retrained"
    )

    return run


def _check_model_retraining(
    db: Session,
    orchestrator: OrchestratorState,
    config: Dict[str, Any],
    actions: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Check if any models need retraining.

    Args:
        db: Database session
        orchestrator: Orchestrator state
        config: Configuration
        actions: Actions dict to update

    Returns:
        List of decision records
    """
    decisions = []

    # Check if enough time has passed since last training
    training_frequency_hours = config.get("training_frequency_hours", 24)
    min_new_samples = config.get("min_new_samples_for_retrain", 100)

    if orchestrator.last_training_at:
        hours_since_training = (
            datetime.utcnow() - orchestrator.last_training_at
        ).total_seconds() / 3600

        if hours_since_training < training_frequency_hours:
            decisions.append({
                "type": "model_retraining_check",
                "reason": f"too soon (last training {hours_since_training:.1f}h ago)",
                "result": "skipped"
            })
            return decisions

    # Get active models
    models = db.query(MLModelRegistry).filter(
        MLModelRegistry.is_active == True
    ).all()

    for model in models:
        # Count new labeled samples
        total_labeled = db.query(Structure).join(
            SimulationJob,
            SimulationJob.structure_id == Structure.id
        ).filter(
            SimulationJob.status == "COMPLETED"
        ).distinct().count()

        new_samples = total_labeled - (model.training_samples_count or 0)

        if new_samples >= min_new_samples:
            # Trigger retraining (in production, create a training job)
            logger.info(
                f"Model {model.name} ready for retraining "
                f"({new_samples} new samples)"
            )

            decisions.append({
                "type": "model_retrain_triggered",
                "model": model.name,
                "reason": f"{new_samples} new samples available",
                "result": "queued"
            })

            actions["models_retrained"].append(model.name)

            # Update last training time
            orchestrator.last_training_at = datetime.utcnow()

        else:
            decisions.append({
                "type": "model_retraining_check",
                "model": model.name,
                "reason": f"insufficient new samples ({new_samples}/{min_new_samples})",
                "result": "skipped"
            })

    return decisions


def _advance_campaigns(
    db: Session,
    orchestrator: OrchestratorState,
    config: Dict[str, Any],
    actions: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Advance active design campaigns.

    Args:
        db: Database session
        orchestrator: Orchestrator state
        config: Configuration
        actions: Actions dict to update

    Returns:
        List of decision records
    """
    decisions = []

    # Get active campaigns
    campaigns = db.query(DesignCampaign).filter(
        DesignCampaign.status == CampaignStatus.RUNNING
    ).all()

    if not campaigns:
        decisions.append({
            "type": "campaign_advancement",
            "reason": "no active campaigns",
            "result": "skipped"
        })
        return decisions

    # Get resource limits
    max_sim_jobs = config.get("max_simultaneous_simulations", 10)
    max_iterations_per_run = config.get("max_iterations_per_run", 3)

    # Count current running jobs
    running_jobs = db.query(SimulationJob).filter(
        SimulationJob.status.in_(["PENDING", "RUNNING"])
    ).count()

    campaigns_advanced_this_run = 0

    for campaign in campaigns:
        # Check if we've hit iteration limit for this run
        if campaigns_advanced_this_run >= max_iterations_per_run:
            decisions.append({
                "type": "campaign_advancement",
                "campaign_id": str(campaign.id),
                "reason": f"max iterations per run reached ({max_iterations_per_run})",
                "result": "deferred"
            })
            continue

        # Check campaign budget
        sim_budget = config.get("simulation_budget_per_campaign", 1000)

        # Count simulations for this campaign
        campaign_iterations = db.query(DesignIteration).filter(
            DesignIteration.campaign_id == campaign.id
        ).count()

        # Estimate simulations per iteration (simplified)
        sims_per_iteration = campaign.config.get("candidates_per_iteration", 10)
        total_campaign_sims = campaign_iterations * sims_per_iteration

        if total_campaign_sims >= sim_budget:
            decisions.append({
                "type": "campaign_advancement",
                "campaign_id": str(campaign.id),
                "reason": f"budget exhausted ({total_campaign_sims}/{sim_budget} simulations)",
                "result": "budget_exceeded"
            })
            continue

        # Check if we have capacity for more jobs
        if running_jobs + sims_per_iteration > max_sim_jobs:
            decisions.append({
                "type": "campaign_advancement",
                "campaign_id": str(campaign.id),
                "reason": f"insufficient capacity (would exceed {max_sim_jobs} max jobs)",
                "result": "capacity_exceeded"
            })
            continue

        # Check if campaign has reached max iterations
        if campaign.current_iteration >= campaign.max_iterations:
            decisions.append({
                "type": "campaign_advancement",
                "campaign_id": str(campaign.id),
                "reason": f"max iterations reached ({campaign.max_iterations})",
                "result": "completed"
            })

            # Mark campaign as completed
            campaign.status = CampaignStatus.COMPLETED
            campaign.completed_at = datetime.utcnow()
            continue

        # All checks passed - advance campaign
        logger.info(f"Advancing campaign {campaign.id} (iteration {campaign.current_iteration + 1})")

        # In production, this would call the campaign advancement logic
        # For now, we just record the decision
        decisions.append({
            "type": "campaign_iteration",
            "campaign_id": str(campaign.id),
            "iteration": campaign.current_iteration + 1,
            "reason": "budget available and capacity sufficient",
            "result": "queued",
            "simulations_to_launch": sims_per_iteration
        })

        actions["campaigns_advanced"].append(str(campaign.id))
        actions["simulations_launched"] += sims_per_iteration
        running_jobs += sims_per_iteration
        campaigns_advanced_this_run += 1

        # Update campaign iteration count (in production, this happens in campaign step)
        # campaign.current_iteration += 1

    return decisions


def _schedule_experiments(
    db: Session,
    orchestrator: OrchestratorState,
    config: Dict[str, Any],
    actions: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Schedule experiments for promising candidates.

    Args:
        db: Database session
        orchestrator: Orchestrator state
        config: Configuration
        actions: Actions dict to update

    Returns:
        List of decision records
    """
    decisions = []

    # Get resource limits
    max_experiments = config.get("max_simultaneous_experiments", 5)
    experiment_score_threshold = config.get("experiment_trigger_score_threshold", 0.9)
    experiment_budget_per_campaign = config.get("experiment_budget_per_campaign", 50)

    # Count current pending/running experiments
    running_experiments = db.query(ExperimentRun).filter(
        or_(
            ExperimentRun.status == ExperimentStatus.PENDING,
            ExperimentRun.status == ExperimentStatus.RUNNING
        )
    ).count()

    if running_experiments >= max_experiments:
        decisions.append({
            "type": "experiment_scheduling",
            "reason": f"max experiments reached ({running_experiments}/{max_experiments})",
            "result": "capacity_exceeded"
        })
        return decisions

    # Get active campaigns
    campaigns = db.query(DesignCampaign).filter(
        DesignCampaign.status == CampaignStatus.RUNNING
    ).all()

    for campaign in campaigns:
        # Check experiment budget for this campaign
        campaign_experiments = db.query(ExperimentRun).filter(
            ExperimentRun.linked_structure_id.in_(
                db.query(Structure.id).join(
                    DesignIteration,
                    DesignIteration.id == Structure.id  # Simplified join
                ).filter(
                    DesignIteration.campaign_id == campaign.id
                )
            )
        ).count()

        if campaign_experiments >= experiment_budget_per_campaign:
            decisions.append({
                "type": "experiment_scheduling",
                "campaign_id": str(campaign.id),
                "reason": f"experiment budget exhausted ({campaign_experiments}/{experiment_budget_per_campaign})",
                "result": "budget_exceeded"
            })
            continue

        # Find high-scoring candidates from recent iterations
        # (In production, query predicted properties and scores)
        # For now, simplified logic

        decisions.append({
            "type": "experiment_scheduling",
            "campaign_id": str(campaign.id),
            "reason": "checking for high-scoring candidates",
            "result": "evaluated"
        })

    return decisions


def get_or_create_orchestrator(
    db: Session,
    name: str = "default",
    config: Optional[Dict[str, Any]] = None
) -> OrchestratorState:
    """
    Get or create orchestrator state.

    Args:
        db: Database session
        name: Orchestrator name
        config: Configuration (uses default if not provided)

    Returns:
        OrchestratorState instance
    """
    orchestrator = db.query(OrchestratorState).filter(
        OrchestratorState.name == name
    ).first()

    if not orchestrator:
        orchestrator = OrchestratorState(
            name=name,
            description="NANO-OS AGI Control Plane",
            config=config or get_default_config(),
            is_active=True,
        )
        db.add(orchestrator)
        db.commit()
        db.refresh(orchestrator)
        logger.info(f"Created new orchestrator: {name}")

    return orchestrator
