"""
Pydantic schemas for orchestrator and agent API endpoints.

Session 30: Control Plane for Nanomaterials AGI
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


# Orchestrator Schemas

class OrchestratorConfigUpdate(BaseModel):
    """Request to update orchestrator configuration."""
    max_simultaneous_simulations: Optional[int] = Field(None, ge=1, le=100)
    max_simultaneous_experiments: Optional[int] = Field(None, ge=1, le=50)
    training_frequency_hours: Optional[float] = Field(None, ge=1, le=168)
    min_new_samples_for_retrain: Optional[int] = Field(None, ge=10, le=10000)
    experiment_budget_per_campaign: Optional[int] = Field(None, ge=1, le=1000)
    simulation_budget_per_campaign: Optional[int] = Field(None, ge=1, le=100000)
    active_learning_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    bo_acquisition_function: Optional[str] = Field(None, pattern="^(ei|pi|ucb)$")
    max_iterations_per_run: Optional[int] = Field(None, ge=1, le=10)
    experiment_trigger_score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    campaign_priorities: Optional[Dict[str, float]] = None


class OrchestratorStateResponse(BaseModel):
    """Orchestrator state response."""
    id: UUID
    name: str
    description: Optional[str] = None
    mode: str
    config: Dict[str, Any]
    last_run_at: Optional[datetime] = None
    last_training_at: Optional[datetime] = None
    last_experiment_at: Optional[datetime] = None
    run_count: int
    total_simulations_launched: int
    total_experiments_launched: int
    total_trainings_launched: int
    is_active: bool
    error_message: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class OrchestratorRunResponse(BaseModel):
    """Orchestrator run response."""
    id: UUID
    orchestrator_id: UUID
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    actions: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    stats_before: Optional[Dict[str, Any]] = None
    stats_after: Optional[Dict[str, Any]] = None
    triggered_by: Optional[str] = None

    class Config:
        from_attributes = True


class OrchestratorRunRequest(BaseModel):
    """Request to run orchestrator step."""
    trigger_context: Optional[Dict[str, Any]] = None


# Agent API Schemas (LLM-friendly)

class AgentCreateCampaignRequest(BaseModel):
    """
    Agent request to create a design campaign.

    Simplified, LLM-friendly schema.
    """
    name: str = Field(..., description="Campaign name", min_length=1, max_length=200)
    goal: str = Field(..., description="Campaign goal in natural language", min_length=1)
    target_properties: Dict[str, Any] = Field(
        ...,
        description="Target properties, e.g., {'bandgap': {'value': 2.0, 'tolerance': 0.2}}"
    )
    constraints: Optional[Dict[str, Any]] = Field(
        None,
        description="Constraints, e.g., {'elements': ['Mo', 'S'], 'dimensionality': 2}"
    )
    max_iterations: int = Field(default=20, ge=1, le=100, description="Maximum iterations")
    budget: Optional[Dict[str, int]] = Field(
        None,
        description="Budget limits, e.g., {'simulations': 1000, 'experiments': 50}"
    )


class AgentAdvanceCampaignRequest(BaseModel):
    """Agent request to advance a campaign."""
    campaign_id: UUID = Field(..., description="Campaign ID to advance")
    num_iterations: int = Field(default=1, ge=1, le=10, description="Number of iterations to run")
    override_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Temporary configuration overrides"
    )


class AgentRequestSimulationsRequest(BaseModel):
    """Agent request to run simulations."""
    structure_ids: List[UUID] = Field(..., description="Structure IDs to simulate", min_items=1)
    simulation_type: str = Field(default="dft", description="Simulation type (dft, ml_prediction)")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Simulation parameters")
    priority: int = Field(default=5, ge=1, le=10, description="Job priority")


class AgentRequestExperimentsRequest(BaseModel):
    """Agent request to run experiments."""
    structure_ids: List[UUID] = Field(..., description="Structure IDs to experiment on", min_items=1)
    instrument_id: UUID = Field(..., description="Instrument ID to use")
    experiment_type: str = Field(..., description="Experiment type (synthesis, measurement, etc.)")
    parameters: Dict[str, Any] = Field(..., description="Experiment parameters")


class AgentSummaryResponse(BaseModel):
    """
    Agent summary response.

    Concise, LLM-friendly summary of system state.
    """
    summary: str = Field(..., description="Natural language summary")
    statistics: Dict[str, Any] = Field(..., description="Key statistics")
    active_campaigns: List[Dict[str, Any]] = Field(..., description="Active campaigns summary")
    recent_discoveries: List[Dict[str, Any]] = Field(..., description="Recent high-value discoveries")
    recommendations: List[str] = Field(..., description="Recommended next actions")
    system_health: str = Field(..., description="System health status (healthy, degraded, critical)")


class AgentCommandResponse(BaseModel):
    """Agent command response."""
    command_id: UUID
    command_type: str
    success: bool
    result: Dict[str, Any]
    message: str
    timestamp: datetime


class AgentCampaignResponse(BaseModel):
    """Simplified campaign response for agents."""
    campaign_id: UUID
    name: str
    status: str
    current_iteration: int
    max_iterations: int
    best_score: Optional[float] = None
    total_structures_created: int
    total_structures_evaluated: int
    recent_progress: str


class AgentSimulationResponse(BaseModel):
    """Simplified simulation response for agents."""
    job_ids: List[UUID]
    total_jobs_created: int
    estimated_completion_time_hours: Optional[float] = None
    message: str


class AgentExperimentResponse(BaseModel):
    """Simplified experiment response for agents."""
    experiment_ids: List[UUID]
    total_experiments_created: int
    estimated_completion_time_hours: Optional[float] = None
    message: str
