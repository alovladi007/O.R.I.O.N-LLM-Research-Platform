"""Artifact storage helpers — MinIO client factory and presigned URLs.

These wrappers exist so that:

1. Celery tasks, the API, and tests share a single place to build a
   MinIO client from ``src.api.config.settings``.
2. The artifact router can produce presigned download URLs without the
   SQLAlchemy code needing to know anything about the MinIO SDK.
3. Environments without MinIO reachable (CI, docs builds, some unit
   test runs) can import the module without exploding — failures are
   reported per-call, not at import.

The MinIO Python SDK (``minio``) is a required dependency in
requirements.txt (it's used by bulk_import), so importing it here is
safe. We still guard against connection-time errors so the helpers
can be called from tests that don't have MinIO running.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


DEFAULT_ARTIFACTS_BUCKET = "orion-artifacts"
DEFAULT_PRESIGN_EXPIRY = timedelta(hours=1)


def build_minio_client(
    *,
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    secure: Optional[bool] = None,
) -> Any:
    """Return a ``minio.Minio`` instance bound to ORION's settings.

    All parameters default to ``src.api.config.settings``. Overriding
    keeps the function usable from tests that want a fake endpoint.
    """
    from minio import Minio  # lazy
    from src.api.config import settings

    if endpoint is None:
        endpoint = settings.minio_endpoint
    if access_key is None:
        access_key = settings.minio_access_key
    if secret_key is None:
        secret_key = settings.minio_secret_key.get_secret_value()
    if secure is None:
        secure = settings.minio_secure

    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )


def ensure_bucket(client: Any, bucket: str = DEFAULT_ARTIFACTS_BUCKET) -> None:
    """Create *bucket* on demand; idempotent."""
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensure_bucket(%s) failed: %s", bucket, exc)


def presign_artifact(
    client: Any,
    *,
    bucket: str,
    key: str,
    expires: timedelta = DEFAULT_PRESIGN_EXPIRY,
) -> str:
    """Return a presigned GET URL for *bucket*/*key*.

    Propagates any SDK exception up — callers (the artifact router)
    turn the failure into a 502 because an unreachable MinIO is an
    infra issue, not a user issue.
    """
    return client.presigned_get_object(bucket, key, expires=expires)
