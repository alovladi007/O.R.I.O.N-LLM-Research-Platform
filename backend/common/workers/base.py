"""
Worker-side lifecycle + artifact helpers (Phase 2 / Session 2.1).

What lives here
---------------

- :class:`JobLifecycle` — context-manager style wrapper that runs the
  roadmap's four hooks (``on_start``, ``on_success``, ``on_failure``,
  ``on_retry``) by cooperating with :class:`SimulationJob` and a
  :class:`backend.common.workers.events.EventEmitter`. Celery tasks
  just do ``with JobLifecycle(job_id, db, emitter) as lc: ... lc.outputs = {...}``.
- :class:`ArtifactBundle` — a tiny data class describing a run directory
  worth tar.gz'ing and uploading.
- :func:`build_run_dir` / :func:`tar_and_upload_run_dir` — pure helpers
  to avoid mocking the filesystem in every test.
- :class:`TransientEngineError` — the exception Celery tasks raise to
  get a retry. ``autoretry_for=(TransientEngineError,)`` is set on the
  DFT/MD task decorators.

Design boundary
---------------

This module must not import Celery or Redis at module top. It's
deliberately loadable in pure-Python tests without those installed —
``EventEmitter`` from :mod:`backend.common.workers.events` is also
defensive about Redis absence.

It *does* import SQLAlchemy (for :class:`SimulationJob` typing), which
is safe because the whole project runs under SQLAlchemy anyway.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class JobLifecycleError(Exception):
    """Any explicit lifecycle violation (missing job, illegal transition, etc.)."""


class TransientEngineError(Exception):
    """
    Raised by engines to signal a retry-worthy failure.

    Celery task decorators configure ``autoretry_for=(TransientEngineError,)``
    with exponential backoff. Permanent failures should raise a different
    exception class so the autoretry machinery doesn't kick in.
    """


# ---------------------------------------------------------------------------
# Artifact bundle
# ---------------------------------------------------------------------------


@dataclass
class ArtifactBundle:
    """Everything a task needs to describe its run-dir artifact."""

    run_dir: Path
    job_id: str
    # Optional extra files that live outside run_dir but should ship too.
    extra_files: list[Path] = field(default_factory=list)
    # Manifest dict dumped to ``manifest.json`` inside the tarball.
    manifest: Dict[str, Any] = field(default_factory=dict)

    def manifest_with_defaults(self) -> Dict[str, Any]:
        """Add standard manifest fields that every bundle should carry."""
        m = dict(self.manifest)
        m.setdefault("job_id", self.job_id)
        m.setdefault("bundled_at", datetime.utcnow().isoformat() + "Z")
        m.setdefault("orion_artifact_schema", "v1")
        return m


def build_run_dir(
    job_id: str,
    *,
    parent: Optional[Path] = None,
    prefix: str = "orion-run-",
) -> Path:
    """
    Create a fresh run directory for *job_id*.

    Uses :func:`tempfile.mkdtemp` under *parent* (or the system tmp).
    The directory name includes a short digest of *job_id* so that two
    simultaneous tasks for the same job id don't collide even if one is
    a retry.
    """
    parent = parent or Path(tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(f"{job_id}-{os.getpid()}-{uuid.uuid4()}".encode()).hexdigest()[:8]
    run_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}{digest}-", dir=parent))
    return run_dir


def _tar_dir_to(
    run_dir: Path,
    tarball_path: Path,
    manifest: Dict[str, Any],
    extra_files: Optional[list[Path]] = None,
) -> Path:
    """Tar *run_dir* + *extra_files* + manifest.json into *tarball_path*."""
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(run_dir, arcname=run_dir.name)
        for extra in extra_files or []:
            if not extra.exists():
                logger.warning("tar_and_upload: extra file missing: %s", extra)
                continue
            tar.add(extra, arcname=f"{run_dir.name}/extras/{extra.name}")
    return tarball_path


def tar_and_upload_run_dir(
    bundle: ArtifactBundle,
    minio_client: Optional[Any] = None,
    bucket: str = "orion-artifacts",
    object_key_prefix: str = "jobs",
) -> Dict[str, Any]:
    """
    Tar the bundle's run-dir and (optionally) upload to MinIO.

    If *minio_client* is ``None`` we only create the tarball locally and
    return its path — useful in tests and in environments without MinIO.
    Otherwise we call ``minio_client.fput_object(bucket, key, path)``
    against the object storage the service already has configured (the
    API's MinIO client, the Docker service container, etc.).

    Returns a dict suitable for ``SimulationJob.extra_metadata["artifact"]``:

        {"tarball": "/tmp/orion-run-…/run.tgz",
         "bucket":  "orion-artifacts",
         "key":     "jobs/<job-id>/run.tgz",
         "size_bytes": 12345,
         "uploaded": True,
         "manifest": {…}}
    """
    manifest = bundle.manifest_with_defaults()
    tarball_dir = bundle.run_dir.parent
    tarball_path = tarball_dir / f"{bundle.run_dir.name}.tgz"
    _tar_dir_to(bundle.run_dir, tarball_path, manifest, bundle.extra_files)

    size_bytes = tarball_path.stat().st_size
    key = f"{object_key_prefix}/{bundle.job_id}/run.tgz"

    uploaded = False
    if minio_client is not None:
        try:
            minio_client.fput_object(bucket, key, str(tarball_path))
            uploaded = True
        except Exception as exc:  # noqa: BLE001 — log and continue
            logger.error(
                "minio upload failed for job %s: %s", bundle.job_id, exc,
            )

    return {
        "tarball": str(tarball_path),
        "bucket": bucket,
        "key": key,
        "size_bytes": size_bytes,
        "uploaded": uploaded,
        "manifest": manifest,
    }


# ---------------------------------------------------------------------------
# SQLAlchemy Protocol — lightweight contract so JobLifecycle doesn't require
# an actual DB session in unit tests.
# ---------------------------------------------------------------------------


class _JobRowProtocol(Protocol):
    """Minimal interface :class:`JobLifecycle` expects from a job row."""

    id: Any
    status: Any
    progress: Optional[float]
    current_step: Optional[str]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    retry_count: int
    error_message: Optional[str]
    worker_id: Optional[str]
    worker_hostname: Optional[str]

    def transition_to(
        self,
        target: Any,
        *,
        error_message: Optional[str] = None,
        set_started: bool = False,
        set_finished: bool = False,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# JobLifecycle — the context manager around every task body
# ---------------------------------------------------------------------------


class JobLifecycle:
    """
    Lifecycle wrapper for a Celery task handling a :class:`SimulationJob`.

    Usage inside a Celery task::

        @celery_app.task(
            name="orion.dft.relax",
            base=DatabaseTask,
            bind=True,
            autoretry_for=(TransientEngineError,),
            retry_backoff=True,
            max_retries=2,
        )
        def dft_relax(self, job_id):
            with JobLifecycle(job_id, session_factory=self.session_factory) as lc:
                run_dir = build_run_dir(job_id)
                # ... call engine, populate outputs ...
                lc.outputs = {"energy_ev": -123.4}
                lc.bundle = ArtifactBundle(run_dir=run_dir, job_id=job_id)

    Side effects on enter (``on_start``):

    - Load the SimulationJob row.
    - Transition ``QUEUED → RUNNING`` (or ``PENDING → RUNNING`` if Celery
      picked it up before the transition-to-QUEUED could hit the DB).
    - Record ``worker_id`` / ``worker_hostname`` / ``started_at`` via the
      state machine.
    - Emit a ``job.status.running`` event.

    Side effects on exit:

    - If no exception: ``RUNNING → COMPLETED`` + emit ``job.status.completed``.
    - If :class:`TransientEngineError`: rethrow after incrementing
      ``retry_count`` and emitting ``job.status.retrying``. Celery's
      autoretry machinery picks it up from here.
    - Any other exception: ``RUNNING → FAILED``, store the traceback,
      emit ``job.status.failed``, and reraise so Celery logs it.

    The class is sync because ``SimulationJob.transition_to`` is sync; it
    accepts either a sync SQLAlchemy Session or an async session *plus* a
    provided ``commit`` callable. Session 2.2 moves tasks fully async;
    until then the Celery worker entrypoint owns the commit.
    """

    def __init__(
        self,
        job_id: str,
        *,
        session: Any = None,
        commit: Optional[Callable[[], None]] = None,
        emitter: Optional["EventEmitter"] = None,
        worker_id: Optional[str] = None,
    ) -> None:
        self.job_id = job_id
        self._session = session
        self._commit = commit or (lambda: session.commit() if session else None)
        # EventEmitter is importable without Redis installed — use Null fallback.
        if emitter is None:
            from .events import NullEventEmitter
            emitter = NullEventEmitter()
        self._emitter = emitter
        self._worker_id = worker_id or f"worker-{os.getpid()}"
        self._worker_hostname = os.getenv("HOSTNAME") or os.uname().nodename

        # Set inside the `with` block by the task body:
        self.outputs: Dict[str, Any] = {}
        self.bundle: Optional[ArtifactBundle] = None
        self.job: Optional[_JobRowProtocol] = None

    # -----------------------------------------------------------------
    # Context manager hooks
    # -----------------------------------------------------------------
    def __enter__(self) -> "JobLifecycle":
        self._on_start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # returns False to propagate
        if exc is None:
            self._on_success()
            return False

        if isinstance(exc, TransientEngineError):
            self._on_retry(exc)
            return False  # let the task body reraise for Celery's autoretry

        # Everything else is a terminal failure.
        self._on_failure(exc, tb)
        return False

    # -----------------------------------------------------------------
    # Lifecycle hooks — public for subclass/callers that want to call
    # them manually (non-context-manager execution paths).
    # -----------------------------------------------------------------
    def _on_start(self) -> None:
        job = self._load_job()
        self.job = job
        # Try QUEUED→RUNNING, fall back to PENDING→RUNNING (the job was
        # scheduled through a path that skipped QUEUED).
        from src.api.models.simulation import JobStatus, IllegalJobTransitionError

        for source in (JobStatus.QUEUED, JobStatus.PENDING):
            try:
                job.status = source  # type: ignore[assignment]
                job.transition_to(JobStatus.RUNNING, set_started=True)
                break
            except IllegalJobTransitionError:
                continue
        job.worker_id = self._worker_id
        job.worker_hostname = self._worker_hostname
        self._commit()
        self._emitter.emit(
            channel=event_channel(job.id),
            event="job.status.running",
            payload={"job_id": str(job.id), "worker_id": self._worker_id},
        )

    def _on_success(self) -> None:
        job = self.job
        assert job is not None
        from src.api.models.simulation import JobStatus

        # Persist outputs if the task populated them.
        if self.outputs:
            existing = getattr(job, "parameters", None) or {}
            job.parameters = {**existing, "_outputs": self.outputs}  # type: ignore[assignment]

        job.transition_to(JobStatus.COMPLETED, set_finished=True)
        self._commit()
        self._emitter.emit(
            channel=event_channel(job.id),
            event="job.status.completed",
            payload={"job_id": str(job.id), "outputs": self.outputs},
        )

    def _on_retry(self, exc: TransientEngineError) -> None:
        job = self.job
        if job is None:
            return
        job.retry_count = (job.retry_count or 0) + 1
        self._commit()
        self._emitter.emit(
            channel=event_channel(job.id),
            event="job.status.retrying",
            payload={
                "job_id": str(job.id),
                "retry_count": job.retry_count,
                "reason": str(exc),
            },
        )

    def _on_failure(self, exc: BaseException, tb: Any) -> None:
        job = self.job
        if job is None:
            logger.exception("task failure before job load", exc_info=exc)
            return
        from src.api.models.simulation import JobStatus

        try:
            job.transition_to(
                JobStatus.FAILED,
                set_finished=True,
                error_message=f"{type(exc).__name__}: {exc}"[:2000],
            )
        except Exception as transition_exc:  # noqa: BLE001
            # State machine said no. Set error_message anyway so the
            # row carries a clue.
            job.error_message = (
                f"failure-during-failure: {transition_exc!r} while handling {exc!r}"
            )[:2000]
        self._commit()
        self._emitter.emit(
            channel=event_channel(job.id),
            event="job.status.failed",
            payload={"job_id": str(job.id), "reason": str(exc)},
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _load_job(self) -> _JobRowProtocol:
        if self._session is None:
            raise JobLifecycleError(
                "JobLifecycle requires a DB session — pass `session=...`."
            )
        from src.api.models import SimulationJob

        job = self._session.get(SimulationJob, self.job_id)
        if job is None:
            raise JobLifecycleError(f"SimulationJob {self.job_id} not found")
        return job


# ---------------------------------------------------------------------------
# Utilities re-exported from the events module so callers don't need to
# know the submodule layout.
# ---------------------------------------------------------------------------


def event_channel(job_id: Any) -> str:
    """Redis pubsub channel name for a job's event stream."""
    return f"orion:events:job:{job_id}"


# Type-only re-import so forward references resolve when this module is
# loaded before ``events``.
from .events import EventEmitter  # noqa: E402
