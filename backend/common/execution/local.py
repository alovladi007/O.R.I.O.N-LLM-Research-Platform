"""LocalBackend — run the command in a subprocess on this host.

Design notes
------------

- Uses ``asyncio.create_subprocess_exec`` so ``submit`` and ``cancel``
  don't block an event loop. The child process's pid is stored on the
  :class:`SubmissionHandle` so ``poll``/``cancel`` are stateless: a
  process restart can re-poll an existing PID by building a new
  backend and calling ``poll(handle)``.
- ``cancel`` uses :mod:`psutil` to reach the full process tree. A
  bare ``os.kill(pid, SIGTERM)`` won't stop child shells spawned by
  the engine script.
- Resource accounting lives in ``poll`` via ``psutil.Process(...)``.
  When the process has already exited, we fall back to reading the
  exit code from ``/proc/<pid>`` via asyncio's returncode.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover — unit tests import the module without psutil.
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

from .base import (
    ExecutionBackend,
    JobState,
    Resources,
    SubmissionHandle,
    TimedOut,
)

logger = logging.getLogger(__name__)


# In-process cache: PID → (asyncio.subprocess.Process, start_epoch).
# Lets poll() see the still-running handle we spawned; once a process
# exits, psutil handles the corpse lookup.
_SPAWNED: Dict[int, Dict[str, Any]] = {}


class LocalBackend(ExecutionBackend):
    """Run the submitted command in a subprocess on this host."""

    kind = "local"

    def __init__(self, *, cancel_grace_seconds: float = 3.0) -> None:
        self.cancel_grace_seconds = cancel_grace_seconds

    # ------------------------------------------------------------------
    # submit
    # ------------------------------------------------------------------
    async def submit(
        self,
        cmd: List[str],
        run_dir: Path,
        resources: Resources,
    ) -> SubmissionHandle:
        if not cmd:
            raise ValueError("LocalBackend.submit: cmd is empty")
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        env = {**os.environ, **(resources.env or {})}

        stdout_f = stdout_path.open("wb")
        stderr_f = stderr_path.open("wb")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(run_dir),
            stdout=stdout_f,
            stderr=stderr_f,
            env=env,
        )

        _SPAWNED[proc.pid] = {
            "process": proc,
            "stdout_f": stdout_f,
            "stderr_f": stderr_f,
            "started_at": time.time(),
        }

        handle = SubmissionHandle(
            backend_kind=self.kind,
            external_id=str(proc.pid),
            run_dir=run_dir,
            submitted_at_epoch=time.time(),
            meta={
                "cmd": list(cmd),
                "cpus": resources.cpus,
                "gpus": resources.gpus,
                "walltime_seconds": resources.walltime_seconds(),
            },
        )
        logger.info("LocalBackend submitted pid=%s cmd=%s", proc.pid, cmd)
        return handle

    # ------------------------------------------------------------------
    # poll
    # ------------------------------------------------------------------
    async def poll(self, handle: SubmissionHandle) -> JobState:
        pid = int(handle.external_id)

        # Walltime check first — even a finished process should report
        # the timeout if the caller set a deadline.
        walltime_sec = _resolve_walltime(handle)
        if walltime_sec is not None:
            if time.time() - handle.submitted_at_epoch > walltime_sec:
                # Reclaim the resources and surface the timeout.
                await self.cancel(handle)
                raise TimedOut(
                    f"local pid={pid} exceeded walltime {walltime_sec}s"
                )

        cached = _SPAWNED.get(pid)
        if cached is not None:
            proc = cached["process"]
            rc = proc.returncode
            if rc is None:
                return JobState.RUNNING
            # Close file handles once the child has exited.
            _close_if_open(cached.pop("stdout_f", None))
            _close_if_open(cached.pop("stderr_f", None))
            cached["exit_code"] = rc
            cached["finished_at"] = time.time()
            if cached.get("cancelled"):
                return JobState.CANCELLED
            return JobState.COMPLETED if rc == 0 else JobState.FAILED

        # Not in-cache: either another process submitted it or we got
        # restarted. Use psutil to check if the PID is still alive.
        if psutil is not None:
            try:
                proc_ps = psutil.Process(pid)
                if proc_ps.is_running() and proc_ps.status() != psutil.STATUS_ZOMBIE:
                    return JobState.RUNNING
            except psutil.NoSuchProcess:
                pass
        # No asyncio handle + no psutil signal → we can't tell
        # completion from cancellation; report FAILED so callers surface
        # the ambiguity instead of silently succeeding.
        return JobState.FAILED

    # ------------------------------------------------------------------
    # cancel
    # ------------------------------------------------------------------
    async def cancel(self, handle: SubmissionHandle) -> None:
        pid = int(handle.external_id)
        cached = _SPAWNED.get(pid)
        if cached is not None:
            cached["cancelled"] = True

        # Build the process tree using psutil so we reach the engine's
        # children (the shell script → pw.x case).
        victims: List[int] = [pid]
        if psutil is not None:
            try:
                proc = psutil.Process(pid)
                victims += [p.pid for p in proc.children(recursive=True)]
            except psutil.NoSuchProcess:
                return  # already dead

        # SIGTERM first
        for victim in reversed(victims):  # children before parent
            _send_signal(victim, signal.SIGTERM)

        # Grace window
        await asyncio.sleep(self.cancel_grace_seconds)

        # SIGKILL any survivors
        for victim in reversed(victims):
            if _pid_alive(victim):
                _send_signal(victim, signal.SIGKILL)

        # If we launched the process ourselves, drain asyncio's
        # returncode to avoid zombies.
        if cached is not None:
            proc = cached.get("process")
            if proc is not None and proc.returncode is None:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    logger.warning("LocalBackend.cancel: pid=%s did not reap", pid)

    # ------------------------------------------------------------------
    # fetch_artifacts
    # ------------------------------------------------------------------
    async def fetch_artifacts(
        self,
        handle: SubmissionHandle,
        dest_dir: Path,
    ) -> Path:
        # Local: run_dir is already on disk. If dest_dir differs, we
        # still hand back handle.run_dir — the caller tar's from there.
        return handle.run_dir


# ---------------------------------------------------------------------------
# Sync convenience — used by engine runners that aren't yet async.
# ---------------------------------------------------------------------------


def sync_execute(
    backend: ExecutionBackend,
    cmd: List[str],
    run_dir: Path,
    resources: Resources,
    *,
    poll_interval_seconds: float = 0.5,
) -> JobState:
    """Run submit → poll loop → return terminal state, all under asyncio.run.

    Handy for the QE/LAMMPS runners that are still synchronous. Raises
    :class:`TimedOut` identically to :meth:`ExecutionBackend.poll`.
    """

    async def _run() -> JobState:
        handle = await backend.submit(cmd, run_dir, resources)
        while True:
            state = await backend.poll(handle)
            if state.is_terminal:
                return state
            await asyncio.sleep(poll_interval_seconds)

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_walltime(handle: SubmissionHandle) -> Optional[int]:
    meta = handle.meta or {}
    value = meta.get("walltime_seconds")
    return int(value) if value else None


def _close_if_open(fh: Any) -> None:
    if fh is None:
        return
    try:
        fh.close()
    except Exception:  # noqa: BLE001
        pass


def _send_signal(pid: int, sig: signal.Signals) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def _pid_alive(pid: int) -> bool:
    if psutil is not None:
        try:
            return psutil.Process(pid).is_running()
        except psutil.NoSuchProcess:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
