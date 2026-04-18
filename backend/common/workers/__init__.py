"""ORION worker infrastructure (Phase 2).

This package is deliberately thin. The canonical Celery application lives
at ``src.worker.celery_app``; everything generic goes here so both
Celery tasks and any non-Celery job runners (e.g. inline execution in
tests) can share the same lifecycle contract.
"""

from .base import (
    JobLifecycle,
    JobLifecycleError,
    TransientEngineError,
    ArtifactBundle,
    build_run_dir,
    tar_and_upload_run_dir,
)
from .events import (
    EventEmitter,
    NullEventEmitter,
    RedisPubSubEmitter,
    event_channel,
)

__all__ = [
    "JobLifecycle",
    "JobLifecycleError",
    "TransientEngineError",
    "ArtifactBundle",
    "build_run_dir",
    "tar_and_upload_run_dir",
    "EventEmitter",
    "NullEventEmitter",
    "RedisPubSubEmitter",
    "event_channel",
]
