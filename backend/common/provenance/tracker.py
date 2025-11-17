"""
Provenance tracker implementation.

Provides functions for recording and querying provenance data.
All timestamps use UTC and ISO 8601 format for consistency.
"""

import os
import platform
import socket
import subprocess
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.provenance import ProvenanceRecord, EntityType, EventType


async def record_provenance(
    db: AsyncSession,
    entity_type: EntityType,
    entity_id: uuid.UUID,
    event_type: EventType,
    details: Optional[Dict[str, Any]] = None
) -> ProvenanceRecord:
    """
    Create a provenance record.

    Records an event for an entity with detailed context information.
    All timestamps are stored in UTC.

    Args:
        db: Database session
        entity_type: Type of entity (JOB, PREDICTION, etc.)
        entity_id: UUID of the entity
        event_type: Type of event (CREATED, STARTED, etc.)
        details: Optional dictionary with event details

    Returns:
        The created ProvenanceRecord

    Example:
        ```python
        await record_provenance(
            db,
            EntityType.JOB,
            job.id,
            EventType.STARTED,
            details={
                "engine_name": "vasp",
                "worker_id": "worker-01",
                "host_info": get_system_info(),
                "started_at": datetime.now(timezone.utc).isoformat()
            }
        )
        ```
    """
    if details is None:
        details = {}

    # Ensure we have a timestamp
    if "timestamp" not in details:
        details["timestamp"] = datetime.now(timezone.utc).isoformat()

    record = ProvenanceRecord(
        entity_type=entity_type,
        entity_id=entity_id,
        event_type=event_type,
        timestamp=datetime.now(timezone.utc),
        details=details
    )

    db.add(record)
    await db.commit()
    await db.refresh(record)

    return record


async def get_provenance_chain(
    db: AsyncSession,
    entity_type: EntityType,
    entity_id: uuid.UUID,
    event_type: Optional[EventType] = None,
    limit: Optional[int] = None
) -> List[ProvenanceRecord]:
    """
    Get all provenance records for an entity.

    Retrieves the complete event history for an entity, ordered by timestamp.
    Optionally filter by event type and limit results.

    Args:
        db: Database session
        entity_type: Type of entity
        entity_id: UUID of the entity
        event_type: Optional filter by event type
        limit: Optional limit on number of records

    Returns:
        List of ProvenanceRecord objects, ordered by timestamp ascending

    Example:
        ```python
        # Get all events for a job
        records = await get_provenance_chain(db, EntityType.JOB, job_id)

        # Get only FAILED events
        failures = await get_provenance_chain(
            db, EntityType.JOB, job_id, event_type=EventType.FAILED
        )
        ```
    """
    query = select(ProvenanceRecord).where(
        ProvenanceRecord.entity_type == entity_type,
        ProvenanceRecord.entity_id == entity_id
    )

    if event_type is not None:
        query = query.where(ProvenanceRecord.event_type == event_type)

    query = query.order_by(ProvenanceRecord.timestamp.asc())

    if limit is not None:
        query = query.limit(limit)

    result = await db.execute(query)
    return list(result.scalars().all())


def get_system_info() -> Dict[str, Any]:
    """
    Get host/system information for provenance.

    Collects information about the system where code is running
    for reproducibility and debugging.

    Returns:
        Dictionary with system information:
        - hostname: Machine hostname
        - platform: Platform string (OS and version)
        - system: OS name (Linux, Darwin, Windows)
        - release: OS release version
        - machine: Machine type (x86_64, arm64, etc.)
        - python_version: Python version string
        - cpu_count: Number of CPU cores

    Example:
        ```python
        info = get_system_info()
        # {
        #     "hostname": "worker-01",
        #     "platform": "Linux-5.4.0-42-generic-x86_64-with-glibc2.29",
        #     "system": "Linux",
        #     "release": "5.4.0-42-generic",
        #     "machine": "x86_64",
        #     "python_version": "3.11.5",
        #     "cpu_count": 8
        # }
        ```
    """
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"

    return {
        "hostname": hostname,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def get_code_version() -> str:
    """
    Get current git commit hash or version string.

    Attempts to get the current git commit hash for code version tracking.
    Falls back to "unknown" if not in a git repository or git is unavailable.

    Returns:
        Short git commit hash (7 characters) or "unknown"

    Example:
        ```python
        version = get_code_version()
        # "a1b2c3d" or "unknown"
        ```
    """
    try:
        # Try to get git commit hash
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        # Not in a git repo, git not installed, or command failed
        return "unknown"


async def get_provenance_summary(
    db: AsyncSession,
    entity_type: EntityType,
    entity_id: uuid.UUID
) -> Dict[str, Any]:
    """
    Get a summary of provenance data for an entity.

    Provides high-level statistics about an entity's provenance chain.

    Args:
        db: Database session
        entity_type: Type of entity
        entity_id: UUID of the entity

    Returns:
        Dictionary with summary information:
        - total_records: Total number of provenance records
        - first_event: Timestamp of first event
        - last_event: Timestamp of last event
        - event_types: List of unique event types
        - duration_ms: Time between first and last event (if both exist)

    Example:
        ```python
        summary = await get_provenance_summary(db, EntityType.JOB, job_id)
        # {
        #     "total_records": 5,
        #     "first_event": "2025-11-17T10:00:00Z",
        #     "last_event": "2025-11-17T10:05:30Z",
        #     "event_types": ["CREATED", "QUEUED", "STARTED", "COMPLETED"],
        #     "duration_ms": 330000
        # }
        ```
    """
    records = await get_provenance_chain(db, entity_type, entity_id)

    if not records:
        return {
            "total_records": 0,
            "first_event": None,
            "last_event": None,
            "event_types": [],
            "duration_ms": None
        }

    first_event = records[0].timestamp
    last_event = records[-1].timestamp
    event_types = list(set(r.event_type.value for r in records))

    duration_ms = None
    if first_event and last_event:
        duration_ms = (last_event - first_event).total_seconds() * 1000

    return {
        "total_records": len(records),
        "first_event": first_event.isoformat(),
        "last_event": last_event.isoformat(),
        "event_types": sorted(event_types),
        "duration_ms": duration_ms
    }
