"""
Provenance tracking router for NANO-OS API.

Provides:
- Get provenance chain for entities (jobs, predictions, structures)
- Get timeline visualization data
- Get provenance summaries
- Query provenance records by type and time
"""

from fastapi import APIRouter, Depends, Query, status, Path
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from ..database import get_db
from ..models import User
from ..models.provenance import ProvenanceRecord, EntityType, EventType
from ..schemas.provenance import (
    ProvenanceRecordResponse,
    ProvenanceChainResponse,
    ProvenanceTimelineResponse,
    ProvenanceSummaryResponse,
    TimelineEvent
)
from ..auth.security import get_current_active_user
from ..exceptions import NotFoundError, ValidationError
from backend.common.provenance import (
    get_provenance_chain,
    get_provenance_summary,
    get_code_version
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/provenance",
    tags=["provenance"],
    dependencies=[Depends(get_current_active_user)],
    responses={
        401: {"description": "Not authenticated"},
        404: {"description": "Entity not found"}
    }
)


def _format_timeline_event(record: ProvenanceRecord, next_record: Optional[ProvenanceRecord] = None) -> TimelineEvent:
    """
    Format a provenance record as a timeline event for UI visualization.

    Args:
        record: The provenance record to format
        next_record: The next record in sequence (for duration calculation)

    Returns:
        TimelineEvent with formatted data for display
    """
    # Determine color based on event type
    color_map = {
        EventType.CREATED: "info",
        EventType.QUEUED: "info",
        EventType.STARTED: "info",
        EventType.COMPLETED: "success",
        EventType.FAILED: "error",
        EventType.CANCELLED: "warning",
        EventType.TIMEOUT: "error",
        EventType.PREDICTED: "success",
        EventType.UPLOADED: "info",
        EventType.MODIFIED: "warning",
        EventType.DELETED: "error",
    }

    # Determine title based on entity and event type
    entity_name = record.entity_type.value.lower().capitalize()
    event_name = record.event_type.value.replace("_", " ").title()
    title = f"{entity_name} {event_name}"

    # Create description from details
    description_parts = []
    if "engine_name" in record.details:
        description_parts.append(f"Engine: {record.details['engine_name']}")
    if "model_name" in record.details:
        description_parts.append(f"Model: {record.details['model_name']}")
    if "worker_id" in record.details:
        description_parts.append(f"Worker: {record.details['worker_id']}")
    if "error_message" in record.details:
        description_parts.append(f"Error: {record.details['error_message']}")

    description = " | ".join(description_parts) if description_parts else event_name

    # Calculate duration to next event
    duration_ms = None
    if next_record:
        duration_ms = (next_record.timestamp - record.timestamp).total_seconds() * 1000

    return TimelineEvent(
        type=record.event_type.value,
        timestamp=record.timestamp,
        title=title,
        description=description,
        color=color_map.get(record.event_type, "info"),
        details=record.details,
        duration_ms=duration_ms
    )


@router.get(
    "/{entity_type}/{entity_id}",
    response_model=ProvenanceChainResponse,
    summary="Get provenance chain",
    description="""
    Get the complete provenance chain for an entity.

    Returns all provenance records for the specified entity in chronological order.
    Includes summary statistics like total records, first/last event times, and duration.

    Entity types:
    - JOB: Simulation jobs
    - PREDICTION: ML predictions
    - STRUCTURE: Atomic structures
    - MATERIAL: Materials

    Example:
    GET /provenance/JOB/423e4567-e89b-12d3-a456-426614174000
    """,
    responses={
        200: {
            "description": "Provenance chain retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "entity_type": "JOB",
                        "entity_id": "423e4567-e89b-12d3-a456-426614174000",
                        "total_records": 5,
                        "first_event": "2025-11-17T10:00:00Z",
                        "last_event": "2025-11-17T10:05:30Z",
                        "duration_ms": 330000,
                        "records": []
                    }
                }
            }
        }
    }
)
async def get_entity_provenance(
    entity_type: str = Path(..., description="Entity type (JOB, PREDICTION, STRUCTURE, MATERIAL)"),
    entity_id: uuid.UUID = Path(..., description="Entity UUID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> ProvenanceChainResponse:
    """Get complete provenance chain for an entity."""
    # Validate entity type
    try:
        entity_type_enum = EntityType(entity_type.upper())
    except ValueError:
        raise ValidationError(
            f"Invalid entity type: {entity_type}. "
            f"Valid types: {', '.join(e.value for e in EntityType)}"
        )

    # Get provenance records
    records = await get_provenance_chain(db, entity_type_enum, entity_id)

    if not records:
        logger.info(f"No provenance records found for {entity_type}:{entity_id}")

    # Build response
    first_event = records[0].timestamp if records else None
    last_event = records[-1].timestamp if records else None
    duration_ms = None

    if first_event and last_event:
        duration_ms = (last_event - first_event).total_seconds() * 1000

    return ProvenanceChainResponse(
        entity_type=entity_type.upper(),
        entity_id=entity_id,
        records=[ProvenanceRecordResponse.model_validate(r) for r in records],
        total_records=len(records),
        first_event=first_event,
        last_event=last_event,
        duration_ms=duration_ms
    )


@router.get(
    "/{entity_type}/{entity_id}/timeline",
    response_model=ProvenanceTimelineResponse,
    summary="Get provenance timeline",
    description="""
    Get timeline data formatted for UI visualization.

    Returns provenance data formatted as timeline events with:
    - Human-readable titles and descriptions
    - Color codes for event types
    - Duration calculations between events
    - Code version information

    Optimized for rendering in timeline components.
    """
)
async def get_entity_timeline(
    entity_type: str = Path(..., description="Entity type"),
    entity_id: uuid.UUID = Path(..., description="Entity UUID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> ProvenanceTimelineResponse:
    """Get timeline visualization data for an entity."""
    # Validate entity type
    try:
        entity_type_enum = EntityType(entity_type.upper())
    except ValueError:
        raise ValidationError(
            f"Invalid entity type: {entity_type}. "
            f"Valid types: {', '.join(e.value for e in EntityType)}"
        )

    # Get provenance records
    records = await get_provenance_chain(db, entity_type_enum, entity_id)

    # Format as timeline events
    events = []
    for i, record in enumerate(records):
        next_record = records[i + 1] if i + 1 < len(records) else None
        event = _format_timeline_event(record, next_record)
        events.append(event)

    # Calculate total duration
    total_duration_ms = None
    if len(records) >= 2:
        total_duration_ms = (records[-1].timestamp - records[0].timestamp).total_seconds() * 1000

    # Get code version from first record if available
    code_version = None
    if records and "code_version" in records[0].details:
        code_version = records[0].details["code_version"]
    else:
        code_version = get_code_version()

    return ProvenanceTimelineResponse(
        entity_type=entity_type.upper(),
        entity_id=entity_id,
        events=events,
        total_duration_ms=total_duration_ms,
        code_version=code_version
    )


@router.get(
    "/{entity_type}/{entity_id}/summary",
    response_model=ProvenanceSummaryResponse,
    summary="Get provenance summary",
    description="""
    Get summary statistics for an entity's provenance.

    Returns high-level metrics:
    - Total number of records
    - First and last event timestamps
    - List of unique event types
    - Total duration

    Useful for quick overview without fetching all records.
    """
)
async def get_entity_summary(
    entity_type: str = Path(..., description="Entity type"),
    entity_id: uuid.UUID = Path(..., description="Entity UUID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> ProvenanceSummaryResponse:
    """Get provenance summary for an entity."""
    # Validate entity type
    try:
        entity_type_enum = EntityType(entity_type.upper())
    except ValueError:
        raise ValidationError(
            f"Invalid entity type: {entity_type}. "
            f"Valid types: {', '.join(e.value for e in EntityType)}"
        )

    # Get summary from utility function
    summary = await get_provenance_summary(db, entity_type_enum, entity_id)

    # Parse datetime strings back to datetime objects
    first_event = datetime.fromisoformat(summary["first_event"]) if summary["first_event"] else None
    last_event = datetime.fromisoformat(summary["last_event"]) if summary["last_event"] else None

    return ProvenanceSummaryResponse(
        entity_type=entity_type.upper(),
        entity_id=entity_id,
        total_records=summary["total_records"],
        first_event=first_event,
        last_event=last_event,
        event_types=summary["event_types"],
        duration_ms=summary["duration_ms"]
    )


@router.get(
    "/job/{job_id}",
    response_model=ProvenanceChainResponse,
    summary="Get job provenance",
    description="""
    Convenience endpoint for getting provenance of a simulation job.

    Equivalent to: GET /provenance/JOB/{job_id}
    """
)
async def get_job_provenance(
    job_id: uuid.UUID = Path(..., description="Job UUID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> ProvenanceChainResponse:
    """Get provenance chain for a simulation job."""
    return await get_entity_provenance("JOB", job_id, db, current_user)


@router.get(
    "/prediction/{prediction_id}",
    response_model=ProvenanceChainResponse,
    summary="Get prediction provenance",
    description="""
    Convenience endpoint for getting provenance of a prediction.

    Equivalent to: GET /provenance/PREDICTION/{prediction_id}
    """
)
async def get_prediction_provenance(
    prediction_id: uuid.UUID = Path(..., description="Prediction UUID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> ProvenanceChainResponse:
    """Get provenance chain for a prediction."""
    return await get_entity_provenance("PREDICTION", prediction_id, db, current_user)
