"""SlurmBackend — submit via sbatch/squeue/scancel.

Two transport modes:

1. **Local submit** (default): run the SLURM commands on this host via
   ``asyncio.create_subprocess_exec``. The worker is assumed to be on
   the cluster's login node.
2. **Remote submit**: when ``host`` is set, tunnel through SSH using
   ``asyncssh``. ``asyncssh`` is an optional dep; we import lazily so
   the module loads in environments without it.

The Session 2.3 roadmap ships the local-submit path end-to-end and
keeps the remote path narrowly useful: remote tests require
``ORION_SLURM_HOST`` + a keyfile (``@pytest.mark.requires_slurm``).
"""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
import time
from pathlib import Path
from typing import Any, List, Optional

from .base import (
    ExecutionBackend,
    JobState,
    Resources,
    SubmissionHandle,
    TimedOut,
)

logger = logging.getLogger(__name__)


_SUBMITTED_RE = re.compile(r"Submitted batch job (\d+)")


class SlurmSubmitError(RuntimeError):
    """``sbatch`` rejected the submission (malformed script, bad partition, etc.)."""


class SlurmBackend(ExecutionBackend):
    """sbatch-based execution backend.

    Parameters
    ----------
    host
        Remote SLURM submit host. ``None`` = run ``sbatch`` locally.
    user, key_path
        SSH credentials used when *host* is set. Unused in local mode.
    partition
        Default ``--partition`` when a :class:`Resources` doesn't set one.
    """

    kind = "slurm"

    def __init__(
        self,
        *,
        host: Optional[str] = None,
        user: Optional[str] = None,
        key_path: Optional[str] = None,
        partition: Optional[str] = None,
    ) -> None:
        self.host = host
        self.user = user
        self.key_path = key_path
        self.partition = partition

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
            raise ValueError("SlurmBackend.submit: cmd is empty")
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Generate batch script with SBATCH directives.
        script = self._render_batch_script(cmd, run_dir, resources)
        script_path = run_dir / "orion-submit.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)

        rc, stdout, stderr = await self._run_slurm_cmd(["sbatch", str(script_path)])
        if rc != 0:
            raise SlurmSubmitError(
                f"sbatch rc={rc} stderr={stderr.strip()!r} stdout={stdout.strip()!r}"
            )
        m = _SUBMITTED_RE.search(stdout)
        if not m:
            raise SlurmSubmitError(f"could not parse sbatch output: {stdout!r}")
        slurm_jobid = m.group(1)

        return SubmissionHandle(
            backend_kind=self.kind,
            external_id=slurm_jobid,
            run_dir=run_dir,
            submitted_at_epoch=time.time(),
            meta={
                "cmd": list(cmd),
                "partition": resources.queue or self.partition,
                "walltime_seconds": resources.walltime_seconds(),
            },
        )

    # ------------------------------------------------------------------
    # poll
    # ------------------------------------------------------------------
    async def poll(self, handle: SubmissionHandle) -> JobState:
        jobid = handle.external_id

        walltime_sec = (handle.meta or {}).get("walltime_seconds")
        if walltime_sec and time.time() - handle.submitted_at_epoch > walltime_sec:
            await self.cancel(handle)
            raise TimedOut(f"slurm jobid={jobid} exceeded walltime {walltime_sec}s")

        # Ask squeue first; empty stdout means the job is no longer
        # tracked — go to sacct for a terminal verdict.
        rc, stdout, _ = await self._run_slurm_cmd(
            ["squeue", "-j", jobid, "-h", "-o", "%T"]
        )
        if rc == 0 and stdout.strip():
            return _translate_squeue_state(stdout.strip())

        # Fallthrough to sacct for completed jobs.
        rc_s, stdout_s, _ = await self._run_slurm_cmd(
            ["sacct", "-j", jobid, "--format=State,ExitCode", "-n", "-P"]
        )
        if rc_s != 0 or not stdout_s.strip():
            return JobState.FAILED  # can't find it anywhere → surface
        first_line = stdout_s.strip().splitlines()[0]
        parts = first_line.split("|", 1)
        state = parts[0].strip().split()[0]
        return _translate_sacct_state(state)

    # ------------------------------------------------------------------
    # cancel
    # ------------------------------------------------------------------
    async def cancel(self, handle: SubmissionHandle) -> None:
        await self._run_slurm_cmd(["scancel", handle.external_id])

    # ------------------------------------------------------------------
    # fetch_artifacts
    # ------------------------------------------------------------------
    async def fetch_artifacts(
        self,
        handle: SubmissionHandle,
        dest_dir: Path,
    ) -> Path:
        # Local SLURM: files land in run_dir and are already visible.
        # Remote SLURM: rsync over SSH. We ship the no-op path and
        # leave the rsync branch for the first remote deployment.
        if self.host is None:
            return handle.run_dir
        await self._rsync_down(handle.run_dir, dest_dir)
        return dest_dir

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _render_batch_script(
        self,
        cmd: List[str],
        run_dir: Path,
        resources: Resources,
    ) -> str:
        lines: List[str] = ["#!/bin/bash"]
        lines.append(f"#SBATCH --job-name=orion-{run_dir.name}")
        lines.append(f"#SBATCH --output={run_dir}/stdout.txt")
        lines.append(f"#SBATCH --error={run_dir}/stderr.txt")
        lines.append(f"#SBATCH --cpus-per-task={max(1, resources.cpus)}")
        if resources.gpus:
            lines.append(f"#SBATCH --gres=gpu:{resources.gpus}")
        if resources.memory_gb:
            lines.append(f"#SBATCH --mem={resources.memory_gb}G")
        if resources.walltime_minutes:
            h = resources.walltime_minutes // 60
            m = resources.walltime_minutes % 60
            lines.append(f"#SBATCH --time={h:02d}:{m:02d}:00")
        partition = resources.queue or self.partition
        if partition:
            lines.append(f"#SBATCH --partition={partition}")
        if resources.account:
            lines.append(f"#SBATCH --account={resources.account}")
        lines.append("")
        # Engine env
        for k, v in (resources.env or {}).items():
            lines.append(f"export {shlex.quote(k)}={shlex.quote(v)}")
        lines.append(f"cd {shlex.quote(str(run_dir))}")
        lines.append("")
        lines.append(" ".join(shlex.quote(x) for x in cmd))
        return "\n".join(lines) + "\n"

    async def _run_slurm_cmd(self, argv: List[str]) -> tuple[int, str, str]:
        """Run a SLURM command locally or via SSH."""
        if self.host is None:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await proc.communicate()
            return proc.returncode or 0, stdout_b.decode(), stderr_b.decode()
        return await self._run_ssh_cmd(argv)

    async def _run_ssh_cmd(self, argv: List[str]) -> tuple[int, str, str]:
        try:
            import asyncssh  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "SlurmBackend remote mode requires `asyncssh` — "
                "install with `pip install asyncssh`."
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.user:
            kwargs["username"] = self.user
        if self.key_path:
            kwargs["client_keys"] = [self.key_path]
        kwargs["known_hosts"] = None  # cluster SSH is typically trusted

        command = " ".join(shlex.quote(x) for x in argv)
        async with asyncssh.connect(self.host, **kwargs) as conn:
            result = await conn.run(command, check=False)
            return int(result.exit_status or 0), str(result.stdout or ""), str(result.stderr or "")

    async def _rsync_down(self, remote_dir: Path, dest_dir: Path) -> None:
        try:
            import asyncssh  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "remote SLURM fetch_artifacts requires `asyncssh`."
            ) from exc

        dest_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {}
        if self.user:
            kwargs["username"] = self.user
        if self.key_path:
            kwargs["client_keys"] = [self.key_path]
        kwargs["known_hosts"] = None

        async with asyncssh.connect(self.host, **kwargs) as conn:
            async with conn.start_sftp_client() as sftp:
                await sftp.get(str(remote_dir), str(dest_dir), recurse=True)


# ---------------------------------------------------------------------------
# State translators
# ---------------------------------------------------------------------------


_SQUEUE_MAP = {
    "PENDING": JobState.PENDING,
    "CONFIGURING": JobState.PENDING,
    "RUNNING": JobState.RUNNING,
    "SUSPENDED": JobState.RUNNING,
    "COMPLETING": JobState.RUNNING,
    "COMPLETED": JobState.COMPLETED,
    "CANCELLED": JobState.CANCELLED,
    "FAILED": JobState.FAILED,
    "TIMEOUT": JobState.FAILED,
    "NODE_FAIL": JobState.FAILED,
    "OUT_OF_MEMORY": JobState.FAILED,
    "PREEMPTED": JobState.FAILED,
}


def _translate_squeue_state(raw: str) -> JobState:
    return _SQUEUE_MAP.get(raw.upper(), JobState.RUNNING)


def _translate_sacct_state(raw: str) -> JobState:
    if raw.upper().startswith("CANCELLED"):
        return JobState.CANCELLED
    if raw.upper() == "COMPLETED":
        return JobState.COMPLETED
    return JobState.FAILED
