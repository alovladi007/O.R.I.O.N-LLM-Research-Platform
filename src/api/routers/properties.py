"""
Property ingestion router (Session 1.3).

Two endpoints:

- ``POST /properties/bulk`` — synchronous MVP. Accepts CSV inline plus a
  mapping spec, validates every row against
  :data:`backend.common.units.PROPERTY_REGISTRY`, persists the accepted
  rows as :class:`PredictedProperties` (with ``value_unit`` / SI-
  normalized value in ``properties['<name>_si']`` and the raw value in
  ``properties['<name>']``). Returns a :class:`BulkImportJob` snapshot.
- ``GET /jobs/bulk-import/{job_id}`` — progress + error log.
- ``GET /jobs/bulk-import/{job_id}/errors.csv`` — download the rejected
  rows, formatted as CSV.

Session 2.1 will move the work itself into a Celery task; the synchronous
path stays for testing and small uploads.
"""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth.security import get_current_active_user
from ..database import get_db
from ..exceptions import NotFoundError, ValidationError
from ..models import (
    BulkImportJob,
    BulkImportKind,
    BulkImportStatus,
    PredictedProperties,
    Structure,
    User,
)
from ..schemas.bulk_import import (
    BulkImportJobResponse,
    BulkPropertyImportRequest,
)
from backend.common.properties.bulk_importer import (
    AcceptedRow,
    BulkImportResult,
    import_property_rows,
    parse_csv,
    render_errors_csv,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["properties"],
    dependencies=[Depends(get_current_active_user)],
    responses={401: {"description": "Not authenticated"}},
)


# ---------------------------------------------------------------------------
# POST /api/v1/properties/bulk
# ---------------------------------------------------------------------------


@router.post(
    "/properties/bulk",
    response_model=BulkImportJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Bulk-import property rows (synchronous MVP)",
    description=(
        "Validates every CSV row against the property registry, converts "
        "to the canonical SI unit, and persists the accepted rows. Returns "
        "a BulkImportJob snapshot — poll GET /jobs/bulk-import/{id} for "
        "progress (instant for the synchronous MVP)."
    ),
)
async def bulk_import_properties(
    payload: BulkPropertyImportRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> BulkImportJobResponse:
    # Create the job row first so we have an ID to echo back and a
    # durable record even if something blows up mid-way.
    job = BulkImportJob(
        owner_id=current_user.id,
        kind=BulkImportKind.PROPERTY.value,
        status=BulkImportStatus.PENDING.value,
        mapping=payload.mapping,
        source_kind="inline",
        source_payload=payload.csv_text,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    job.status = BulkImportStatus.RUNNING.value
    job.started_at = datetime.utcnow()
    await db.commit()

    # Parse + validate.
    try:
        rows = parse_csv(payload.csv_text)
    except Exception as exc:  # noqa: BLE001
        job.status = BulkImportStatus.FAILED.value
        job.error_message = f"CSV parse failed: {exc}"
        job.completed_at = datetime.utcnow()
        await db.commit()
        raise ValidationError(f"CSV parse failed: {exc}") from exc

    try:
        result: BulkImportResult = import_property_rows(
            rows=rows,
            mapping=payload.mapping,
            error_cap=BulkImportJob.ERROR_LOG_CAP,
        )
    except ValueError as exc:
        # mapping-level error (unknown property, missing unit, etc.)
        job.status = BulkImportStatus.FAILED.value
        job.error_message = str(exc)
        job.rows_total = len(rows)
        job.completed_at = datetime.utcnow()
        await db.commit()
        raise ValidationError(str(exc)) from exc

    # Persist accepted rows. Each accepted row becomes one PredictedProperties
    # row keyed by structure_ref.id, with the raw and SI-normalized values
    # both carried in `properties`.
    for accepted in result.accepted:
        pred = await _persist_accepted_row(db, current_user, accepted)
        if pred is None:
            # Structure reference couldn't be resolved (MP lookup or
            # missing Structure). Downgrade to row-level error.
            result.rows_ok -= 1
            result.rows_failed += 1
            if len(result.errors) < BulkImportJob.ERROR_LOG_CAP:
                from backend.common.properties.bulk_importer import RowError

                result.errors.append(
                    RowError(
                        row=accepted.row,
                        reason=(
                            f"structure_ref {accepted.structure_ref!r} "
                            "could not be resolved to an existing Structure."
                        ),
                        raw={},
                    )
                )

    # Finalize the job row.
    job.rows_total = result.rows_total
    job.rows_ok = result.rows_ok
    job.rows_failed = result.rows_failed
    job.errors = [e.as_dict() for e in result.errors]
    job.status = BulkImportStatus.COMPLETED.value
    job.completed_at = datetime.utcnow()
    await db.commit()
    await db.refresh(job)

    logger.info(
        "bulk_import_properties job=%s ok=%d/%d failed=%d",
        job.id, job.rows_ok, job.rows_total, job.rows_failed,
    )
    return BulkImportJobResponse.model_validate(job)


async def _persist_accepted_row(
    db: AsyncSession, user: User, accepted: AcceptedRow,
) -> PredictedProperties | None:
    """
    Resolve the structure ref and create one PredictedProperties row.

    Returns ``None`` if the ref can't be resolved; caller records an error.
    For Session 1.3 only the ``kind=uuid`` path resolves; external_id
    resolution lands alongside Session 1.5 Materials Project seed work.
    """
    ref = accepted.structure_ref
    if ref.get("kind") != "uuid":
        return None  # external_id — not yet supported

    try:
        structure_id = UUID(ref["id"])
    except (ValueError, KeyError):
        return None

    structure = await db.get(Structure, structure_id)
    if structure is None:
        return None

    pred = PredictedProperties(
        owner_id=user.id,
        structure_id=structure.id,
        model_name="BULK_IMPORT",
        model_version=accepted.method.get("functional")
        or accepted.method.get("kind")
        or "unspecified",
        properties={
            accepted.property_name: accepted.value_raw,
            f"{accepted.property_name}_si": accepted.value_si,
        },
        confidence_scores={},
        value_unit=accepted.input_unit,
        extra_metadata={
            "source": "bulk_import",
            "method": accepted.method,
            "conditions": accepted.conditions,
            "canonical_unit": accepted.value_canonical_unit,
        },
    )
    db.add(pred)
    return pred


# ---------------------------------------------------------------------------
# GET /api/v1/jobs/bulk-import/{job_id}
# GET /api/v1/jobs/bulk-import/{job_id}/errors.csv
# ---------------------------------------------------------------------------


@router.get(
    "/jobs/bulk-import/{job_id}",
    response_model=BulkImportJobResponse,
    summary="Get bulk-import job status",
)
async def get_bulk_import_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> BulkImportJobResponse:
    result = await db.execute(
        select(BulkImportJob).where(BulkImportJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise NotFoundError("BulkImportJob", job_id)
    return BulkImportJobResponse.model_validate(job)


@router.get(
    "/jobs/bulk-import/{job_id}/errors.csv",
    summary="Download the rejected rows as CSV",
)
async def download_bulk_import_errors(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Response:
    result = await db.execute(
        select(BulkImportJob).where(BulkImportJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise NotFoundError("BulkImportJob", job_id)

    csv_body = render_errors_csv(job.errors or [])
    return Response(
        content=csv_body,
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f'attachment; filename="bulk_import_{job_id}_errors.csv"'
            ),
        },
    )
