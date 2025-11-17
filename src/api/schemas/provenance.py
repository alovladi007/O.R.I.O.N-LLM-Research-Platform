"""Provenance schemas for API requests and responses."""

from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class ProvenanceRecordResponse(BaseModel):
    """Schema for provenance record in API responses."""
    id: uuid.UUID
    entity_type: str = Field(..., description="Type of entity (JOB, PREDICTION, STRUCTURE, etc.)")
    entity_id: uuid.UUID = Field(..., description="ID of the tracked entity")
    event_type: str = Field(..., description="Type of event (CREATED, STARTED, COMPLETED, etc.)")
    timestamp: datetime = Field(..., description="When the event occurred (UTC)")
    details: dict = Field(default_factory=dict, description="Detailed event information")
    created_at: datetime = Field(..., description="When this record was created (UTC)")

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "entity_type": "JOB",
                "entity_id": "223e4567-e89b-12d3-a456-426614174000",
                "event_type": "STARTED",
                "timestamp": "2025-11-17T10:00:00Z",
                "details": {
                    "engine_name": "vasp",
                    "worker_id": "worker-01",
                    "host_info": {
                        "hostname": "compute-node-01",
                        "cpu_count": 16
                    }
                },
                "created_at": "2025-11-17T10:00:00Z"
            }
        }


class ProvenanceChainResponse(BaseModel):
    """Schema for complete provenance chain of an entity."""
    entity_type: str = Field(..., description="Type of entity")
    entity_id: uuid.UUID = Field(..., description="ID of the entity")
    records: List[ProvenanceRecordResponse] = Field(default_factory=list, description="Provenance records in chronological order")
    total_records: int = Field(..., description="Total number of records")
    first_event: Optional[datetime] = Field(None, description="Timestamp of first event")
    last_event: Optional[datetime] = Field(None, description="Timestamp of last event")
    duration_ms: Optional[float] = Field(None, description="Duration between first and last event in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "entity_type": "JOB",
                "entity_id": "223e4567-e89b-12d3-a456-426614174000",
                "total_records": 4,
                "first_event": "2025-11-17T10:00:00Z",
                "last_event": "2025-11-17T10:05:30Z",
                "duration_ms": 330000,
                "records": [
                    {
                        "event_type": "CREATED",
                        "timestamp": "2025-11-17T10:00:00Z",
                        "details": {"user_id": "user-123"}
                    },
                    {
                        "event_type": "QUEUED",
                        "timestamp": "2025-11-17T10:00:05Z",
                        "details": {"priority": 10}
                    },
                    {
                        "event_type": "STARTED",
                        "timestamp": "2025-11-17T10:00:30Z",
                        "details": {"worker_id": "worker-01"}
                    },
                    {
                        "event_type": "COMPLETED",
                        "timestamp": "2025-11-17T10:05:30Z",
                        "details": {"duration_ms": 300000}
                    }
                ]
            }
        }


class TimelineEvent(BaseModel):
    """Schema for timeline visualization event."""
    type: str = Field(..., description="Event type")
    timestamp: datetime = Field(..., description="Event timestamp")
    title: str = Field(..., description="Human-readable title")
    description: str = Field(..., description="Event description")
    color: str = Field(..., description="Color code for UI (success, error, info, warning)")
    details: dict = Field(default_factory=dict, description="Event details")
    duration_ms: Optional[float] = Field(None, description="Duration to next event in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "STARTED",
                "timestamp": "2025-11-17T10:00:30Z",
                "title": "Job Started",
                "description": "Simulation started on worker-01",
                "color": "info",
                "details": {
                    "worker_id": "worker-01",
                    "engine_name": "vasp"
                },
                "duration_ms": 300000
            }
        }


class ProvenanceTimelineResponse(BaseModel):
    """Schema for timeline visualization data."""
    entity_type: str
    entity_id: uuid.UUID
    events: List[TimelineEvent] = Field(default_factory=list)
    total_duration_ms: Optional[float] = None
    code_version: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "entity_type": "JOB",
                "entity_id": "223e4567-e89b-12d3-a456-426614174000",
                "total_duration_ms": 330000,
                "code_version": "a1b2c3d",
                "events": [
                    {
                        "type": "CREATED",
                        "timestamp": "2025-11-17T10:00:00Z",
                        "title": "Job Created",
                        "description": "Job created by user",
                        "color": "info",
                        "duration_ms": 5000
                    },
                    {
                        "type": "COMPLETED",
                        "timestamp": "2025-11-17T10:05:30Z",
                        "title": "Job Completed",
                        "description": "Simulation completed successfully",
                        "color": "success",
                        "duration_ms": None
                    }
                ]
            }
        }


class ProvenanceSummaryResponse(BaseModel):
    """Schema for provenance summary statistics."""
    entity_type: str
    entity_id: uuid.UUID
    total_records: int
    first_event: Optional[datetime] = None
    last_event: Optional[datetime] = None
    event_types: List[str] = Field(default_factory=list)
    duration_ms: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "entity_type": "JOB",
                "entity_id": "223e4567-e89b-12d3-a456-426614174000",
                "total_records": 5,
                "first_event": "2025-11-17T10:00:00Z",
                "last_event": "2025-11-17T10:05:30Z",
                "event_types": ["CREATED", "QUEUED", "STARTED", "COMPLETED"],
                "duration_ms": 330000
            }
        }
