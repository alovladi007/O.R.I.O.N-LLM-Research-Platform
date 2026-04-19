"""Execution backends — local subprocess and SLURM.

Phase 2 / Session 2.3. The abstraction lets engine runners submit work
the same way whether the physics binary runs locally or via a job
scheduler. ``LocalBackend`` is the default; ``SlurmBackend`` kicks in
when the submitter has `SlurmResources` and matching config.

Design summary
--------------

- :class:`ExecutionBackend` is an async ``Protocol`` (see
  :mod:`.base`).
- :class:`Resources` is the dataclass the engines populate with
  cpus / gpus / memory / walltime. The ``queue`` and ``account`` fields
  are SLURM-specific but live on the shared type so engine code doesn't
  have to branch.
- :class:`SubmissionHandle` carries the backend-specific job id plus the
  run directory so ``poll``/``cancel``/``fetch_artifacts`` are
  stateless across worker restarts.
"""

from .base import (
    ExecutionBackend,
    JobState,
    Resources,
    SubmissionHandle,
    TimedOut,
)
from .local import LocalBackend, sync_execute
from .slurm import SlurmBackend, SlurmSubmitError

__all__ = [
    "ExecutionBackend",
    "JobState",
    "LocalBackend",
    "Resources",
    "SlurmBackend",
    "SlurmSubmitError",
    "SubmissionHandle",
    "TimedOut",
    "get_execution_backend",
    "sync_execute",
]


def get_execution_backend(kind: str = "local") -> ExecutionBackend:
    """Return a configured :class:`ExecutionBackend` for *kind*.

    ``kind`` is typically ``inputs.execution.kind`` from a job's
    parameters. Unknown kinds fall back to ``local`` with a warning.
    SLURM pulls host/user/key/partition from :mod:`src.api.config.settings`
    — when ``slurm_host`` is unset, SlurmBackend runs sbatch locally,
    which is the expected behavior on a cluster submit node.
    """
    import logging

    logger = logging.getLogger(__name__)

    normalized = (kind or "local").strip().lower()
    if normalized == "slurm":
        try:
            from src.api.config import settings

            return SlurmBackend(
                host=settings.slurm_host,
                user=settings.slurm_user,
                key_path=settings.slurm_key_path,
                partition=settings.slurm_partition,
            )
        except Exception as exc:  # noqa: BLE001 — fall back, don't crash the worker.
            logger.warning("SlurmBackend config failed (%s); falling back to local", exc)
            return LocalBackend()
    if normalized != "local":
        logger.warning("Unknown execution backend %r; using local", kind)
    return LocalBackend()
