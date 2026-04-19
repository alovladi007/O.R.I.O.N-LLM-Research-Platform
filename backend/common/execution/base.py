"""Core types for the execution backends.

``ExecutionBackend`` is the async protocol that every backend implements.
Engine runners stay simple: they hand the backend a command + run dir,
poll until terminal, then fetch artifacts.

The old session-27 Resources dataclass had ``nodes`` and
``cores_per_node``; the Session 2.3 roadmap is cpus / gpus /
memory_gb / walltime_minutes / queue / account. We ship the new shape
— nothing imported the old one outside its own package.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class JobState(str, enum.Enum):
    """Terminal-or-not lifecycle state of a submitted job.

    Deliberately coarse: engines translate backend-specific enums into
    one of these so callers only branch on five outcomes.
    """

    PENDING = "pending"     # queued but not yet running (SLURM only)
    RUNNING = "running"
    COMPLETED = "completed"  # exit code 0
    FAILED = "failed"        # exit code != 0 or scheduler FAIL
    CANCELLED = "cancelled"  # caller invoked cancel()

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED)


class TimedOut(Exception):
    """Raised when a ``poll`` call exceeded the backend's walltime budget."""


@dataclass
class Resources:
    """Resource request per the roadmap spec.

    Fields
    ------
    cpus
        Logical CPUs (LocalBackend uses this only for logging; SLURM
        maps it to ``--cpus-per-task``).
    gpus
        GPU count; SLURM emits ``--gres=gpu:N`` when non-zero.
    memory_gb
        Memory ask in GB; SLURM emits ``--mem=NG``.
    walltime_minutes
        Hard wall-clock limit. LocalBackend raises :class:`TimedOut`
        from ``poll`` once the deadline passes; SLURM emits ``--time``.
    queue
        SLURM partition name (``--partition``). Unused by LocalBackend.
    account
        SLURM billing account (``--account``). Unused by LocalBackend.
    env
        Extra environment variables merged into the child process env.
    """

    cpus: int = 1
    gpus: int = 0
    memory_gb: Optional[int] = None
    walltime_minutes: Optional[int] = None
    queue: Optional[str] = None
    account: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)

    def walltime_seconds(self) -> Optional[int]:
        return self.walltime_minutes * 60 if self.walltime_minutes is not None else None


@dataclass
class SubmissionHandle:
    """Opaque-ish handle returned by ``submit``.

    ``backend_kind`` identifies the producing backend so a caller that
    persists the handle (e.g. in ``job.extra_metadata``) can route a
    later ``poll`` to the right ExecutionBackend. ``external_id`` is
    the backend-specific id (PID for Local, SLURM jobid for SLURM).

    ``run_dir`` is where stdout/stderr files land. Keeping it on the
    handle means ``fetch_artifacts`` is stateless (no membership lookup
    in an in-memory dict).
    """

    backend_kind: str            # "local" | "slurm"
    external_id: str
    run_dir: Path
    submitted_at_epoch: float
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def stdout_path(self) -> Path:
        return self.run_dir / "stdout.txt"

    @property
    def stderr_path(self) -> Path:
        return self.run_dir / "stderr.txt"


@runtime_checkable
class ExecutionBackend(Protocol):
    """Async protocol implemented by LocalBackend and SlurmBackend.

    All methods are async so SSH-backed backends (SLURM remote) can
    share the interface without blocking the worker thread. Sync
    callers wrap with ``asyncio.run`` or use the ``sync_execute``
    convenience.
    """

    kind: str  # "local" | "slurm"

    async def submit(
        self,
        cmd: List[str],
        run_dir: Path,
        resources: Resources,
    ) -> SubmissionHandle:
        """Submit *cmd* for execution in *run_dir* under *resources*.

        Preconditions:
        - ``run_dir`` exists and is writable.
        - ``cmd[0]`` is either a binary name on $PATH or an absolute path.

        Postconditions:
        - Returns a :class:`SubmissionHandle`. Does not wait for the
          process to finish; poll in a loop until
          ``state.is_terminal``.
        - On local backend, the child is running by the time this
          returns (PID in handle.external_id).
        - On SLURM, the batch script has been queued; state is PENDING
          until the scheduler picks it up.
        """
        ...

    async def poll(self, handle: SubmissionHandle) -> JobState:
        """Return the current lifecycle state.

        Raises :class:`TimedOut` if ``resources.walltime_minutes`` was
        set and has already been exceeded for this handle.
        """
        ...

    async def cancel(self, handle: SubmissionHandle) -> None:
        """Best-effort cancel.

        Must flip the job to :attr:`JobState.CANCELLED` in the next
        ``poll``. Local: SIGTERM then SIGKILL after a 3 s grace; SLURM:
        ``scancel``.

        Idempotent on already-terminal handles.
        """
        ...

    async def fetch_artifacts(
        self,
        handle: SubmissionHandle,
        dest_dir: Path,
    ) -> Path:
        """Stage run-dir artifacts into *dest_dir* for bundling.

        Local backend: ``run_dir`` is already local, this is typically
        a no-op that just returns ``handle.run_dir``. SLURM remote:
        ``scp``/``rsync`` the files down.
        """
        ...
