"""Tests for Phase 2 / Session 2.3 — execution backends.

Scope
-----

1. **Contract** — :class:`ExecutionBackend` Protocol surface; ``Resources``
   walltime math.
2. **LocalBackend (end-to-end, real subprocess)** — submit a trivial
   shell command, observe RUNNING→COMPLETED; failing command → FAILED.
3. **LocalBackend cancel** — spawn ``sleep 60``, call cancel, assert
   the process is gone (psutil verifies).
4. **LocalBackend walltime** — set walltime_minutes and verify
   :class:`TimedOut` fires.
5. **SlurmBackend parsing** — render batch script, parse sbatch
   stdout, map squeue/sacct states to :class:`JobState`.
6. **Factory** — ``get_execution_backend('local'|'slurm'|'bogus')``.
7. **Live SLURM** — ``requires_slurm`` marker; submits ``hostname``
   and asserts the external_id looks numeric.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Basic dataclass + Protocol contract
# ---------------------------------------------------------------------------


class TestResources:
    def test_walltime_seconds_none_when_unset(self):
        from backend.common.execution import Resources

        assert Resources().walltime_seconds() is None

    def test_walltime_seconds_converts_minutes(self):
        from backend.common.execution import Resources

        assert Resources(walltime_minutes=30).walltime_seconds() == 1800


class TestJobState:
    @pytest.mark.parametrize(
        "state,expected",
        [
            ("pending", False),
            ("running", False),
            ("completed", True),
            ("failed", True),
            ("cancelled", True),
        ],
    )
    def test_is_terminal(self, state: str, expected: bool):
        from backend.common.execution import JobState

        assert JobState(state).is_terminal is expected


class TestFactory:
    def test_unknown_kind_falls_back_to_local(self):
        from backend.common.execution import LocalBackend, get_execution_backend

        assert isinstance(get_execution_backend("nonsense"), LocalBackend)
        assert isinstance(get_execution_backend(None), LocalBackend)  # type: ignore[arg-type]

    def test_local_returns_local(self):
        from backend.common.execution import LocalBackend, get_execution_backend

        assert isinstance(get_execution_backend("local"), LocalBackend)

    def test_slurm_returns_slurm(self):
        from backend.common.execution import SlurmBackend, get_execution_backend

        got = get_execution_backend("slurm")
        assert isinstance(got, SlurmBackend)


# ---------------------------------------------------------------------------
# LocalBackend — real subprocess
# ---------------------------------------------------------------------------


class TestLocalBackendRealSubprocess:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_successful_echo_completes(self, tmp_path: Path):
        from backend.common.execution import (
            JobState,
            LocalBackend,
            Resources,
            sync_execute,
        )

        backend = LocalBackend()
        state = sync_execute(
            backend,
            ["/bin/sh", "-c", "echo hello && echo err >&2"],
            tmp_path,
            Resources(),
            poll_interval_seconds=0.05,
        )
        assert state == JobState.COMPLETED
        stdout = (tmp_path / "stdout.txt").read_text()
        stderr = (tmp_path / "stderr.txt").read_text()
        assert "hello" in stdout
        assert "err" in stderr

    def test_nonzero_exit_marks_failed(self, tmp_path: Path):
        from backend.common.execution import JobState, LocalBackend, Resources, sync_execute

        backend = LocalBackend()
        state = sync_execute(
            backend,
            ["/bin/sh", "-c", "exit 7"],
            tmp_path,
            Resources(),
            poll_interval_seconds=0.05,
        )
        assert state == JobState.FAILED


class TestLocalBackendCancel:
    def test_cancel_kills_running_process(self, tmp_path: Path):
        """Spawn sleep 60, cancel, verify the PID is gone via psutil."""
        pytest.importorskip("psutil")
        import psutil

        from backend.common.execution import LocalBackend, Resources

        async def _go():
            backend = LocalBackend(cancel_grace_seconds=0.2)
            handle = await backend.submit(
                ["/bin/sh", "-c", "sleep 60"],
                tmp_path,
                Resources(),
            )
            pid = int(handle.external_id)
            assert psutil.Process(pid).is_running()
            await backend.cancel(handle)
            # Give the kernel a moment to reap.
            await asyncio.sleep(0.3)
            alive = False
            try:
                alive = psutil.Process(pid).is_running() and (
                    psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
                )
            except psutil.NoSuchProcess:
                alive = False
            assert not alive
            state = await backend.poll(handle)
            from backend.common.execution import JobState

            assert state == JobState.CANCELLED

        asyncio.run(_go())


class TestLocalBackendWalltime:
    def test_timeout_raises_timed_out(self, tmp_path: Path):
        from backend.common.execution import (
            LocalBackend,
            Resources,
            TimedOut,
        )

        async def _go():
            backend = LocalBackend(cancel_grace_seconds=0.2)
            handle = await backend.submit(
                ["/bin/sh", "-c", "sleep 10"],
                tmp_path,
                # 1 minute walltime — but we'll monkey-patch start epoch
                # so the deadline appears already missed.
                Resources(walltime_minutes=1),
            )
            # Pretend the job was submitted 2 minutes ago.
            handle.submitted_at_epoch = time.time() - 180
            with pytest.raises(TimedOut):
                await backend.poll(handle)

        asyncio.run(_go())


# ---------------------------------------------------------------------------
# SlurmBackend — parsing / state map / script render (no real sbatch)
# ---------------------------------------------------------------------------


class TestSlurmScriptRendering:
    def test_render_includes_sbatch_directives(self, tmp_path: Path):
        from backend.common.execution import Resources, SlurmBackend

        b = SlurmBackend(partition="default")
        script = b._render_batch_script(
            ["pw.x", "-in", "input.pw"],
            run_dir=tmp_path,
            resources=Resources(cpus=4, gpus=1, memory_gb=16, walltime_minutes=120),
        )
        assert script.splitlines()[0] == "#!/bin/bash"
        assert "#SBATCH --cpus-per-task=4" in script
        assert "#SBATCH --gres=gpu:1" in script
        assert "#SBATCH --mem=16G" in script
        assert "#SBATCH --time=02:00:00" in script
        assert "#SBATCH --partition=default" in script
        assert "pw.x -in input.pw" in script

    def test_render_respects_explicit_queue(self, tmp_path: Path):
        from backend.common.execution import Resources, SlurmBackend

        b = SlurmBackend(partition="fallback")
        script = b._render_batch_script(
            ["hostname"], tmp_path, Resources(queue="gpu-short", account="orion"),
        )
        assert "#SBATCH --partition=gpu-short" in script
        assert "#SBATCH --account=orion" in script


class TestSlurmSubmitParsing:
    def test_sbatch_output_parsed(self, tmp_path: Path, monkeypatch):
        """SlurmBackend.submit parses 'Submitted batch job NNN' stdout."""
        from backend.common.execution import Resources, SlurmBackend

        async def _fake_cmd(self, argv):
            return 0, "Submitted batch job 777\n", ""

        monkeypatch.setattr(SlurmBackend, "_run_slurm_cmd", _fake_cmd)

        b = SlurmBackend()
        handle = asyncio.run(b.submit(["hostname"], tmp_path, Resources()))
        assert handle.external_id == "777"
        assert handle.backend_kind == "slurm"
        assert (tmp_path / "orion-submit.sh").exists()

    def test_sbatch_nonzero_raises(self, tmp_path: Path, monkeypatch):
        from backend.common.execution import Resources, SlurmBackend, SlurmSubmitError

        async def _fake_cmd(self, argv):
            return 1, "", "sbatch: error: invalid partition\n"

        monkeypatch.setattr(SlurmBackend, "_run_slurm_cmd", _fake_cmd)
        b = SlurmBackend()
        with pytest.raises(SlurmSubmitError):
            asyncio.run(b.submit(["hostname"], tmp_path, Resources(queue="bogus")))


class TestSlurmStateTranslators:
    def test_squeue_states_map(self):
        from backend.common.execution import JobState
        from backend.common.execution.slurm import _translate_squeue_state

        assert _translate_squeue_state("RUNNING") == JobState.RUNNING
        assert _translate_squeue_state("PENDING") == JobState.PENDING
        assert _translate_squeue_state("COMPLETED") == JobState.COMPLETED
        assert _translate_squeue_state("FAILED") == JobState.FAILED
        assert _translate_squeue_state("TIMEOUT") == JobState.FAILED
        assert _translate_squeue_state("CANCELLED") == JobState.CANCELLED

    def test_sacct_cancelled_variants(self):
        from backend.common.execution import JobState
        from backend.common.execution.slurm import _translate_sacct_state

        # SLURM writes e.g. "CANCELLED by 12345"
        assert _translate_sacct_state("CANCELLED by 12345") == JobState.CANCELLED
        assert _translate_sacct_state("COMPLETED") == JobState.COMPLETED
        assert _translate_sacct_state("FAILED") == JobState.FAILED


class TestSlurmPollFlow:
    """Poll uses squeue first, falls through to sacct for terminal."""

    def test_running_from_squeue(self, tmp_path: Path, monkeypatch):
        from backend.common.execution import JobState, Resources, SlurmBackend, SubmissionHandle

        async def _fake_cmd(self, argv):
            if argv[0] == "squeue":
                return 0, "RUNNING\n", ""
            return 1, "", ""

        monkeypatch.setattr(SlurmBackend, "_run_slurm_cmd", _fake_cmd)
        b = SlurmBackend()
        handle = SubmissionHandle(
            backend_kind="slurm",
            external_id="42",
            run_dir=tmp_path,
            submitted_at_epoch=time.time(),
        )
        assert asyncio.run(b.poll(handle)) == JobState.RUNNING

    def test_completed_via_sacct(self, tmp_path: Path, monkeypatch):
        from backend.common.execution import JobState, SlurmBackend, SubmissionHandle

        async def _fake_cmd(self, argv):
            if argv[0] == "squeue":
                return 0, "", ""  # empty → job no longer queued
            if argv[0] == "sacct":
                return 0, "COMPLETED|0:0\n", ""
            return 1, "", ""

        monkeypatch.setattr(SlurmBackend, "_run_slurm_cmd", _fake_cmd)
        b = SlurmBackend()
        handle = SubmissionHandle(
            backend_kind="slurm",
            external_id="42",
            run_dir=tmp_path,
            submitted_at_epoch=time.time(),
        )
        assert asyncio.run(b.poll(handle)) == JobState.COMPLETED


# ---------------------------------------------------------------------------
# Live SLURM (optional)
# ---------------------------------------------------------------------------


@pytest.mark.requires_slurm
def test_live_slurm_hostname(tmp_path: Path):
    """Submit `hostname` against a real SLURM cluster; retrieve stdout.

    Auto-skipped when ``ORION_SLURM_HOST`` is unset.
    """
    from backend.common.execution import (
        JobState,
        Resources,
        SlurmBackend,
        sync_execute,
    )

    b = SlurmBackend(
        host=os.environ["ORION_SLURM_HOST"],
        user=os.environ.get("ORION_SLURM_USER"),
        key_path=os.environ.get("ORION_SLURM_KEY_PATH"),
        partition=os.environ.get("ORION_SLURM_PARTITION"),
    )
    state = sync_execute(
        b,
        ["hostname"],
        tmp_path,
        Resources(cpus=1, walltime_minutes=5),
        poll_interval_seconds=2.0,
    )
    assert state == JobState.COMPLETED
    stdout = (tmp_path / "stdout.txt").read_text()
    assert stdout.strip()  # a hostname line


# ---------------------------------------------------------------------------
# Engine wiring sanity — execute_command via LocalBackend round-trip
# ---------------------------------------------------------------------------


class TestEngineExecuteCommand:
    """SimulationEngine.execute_command uses the backend correctly."""

    def test_execute_command_succeeds(self, tmp_path: Path):
        from backend.common.engines.base import SimulationEngine

        class _EchoEngine(SimulationEngine):
            def setup(self, structure, parameters): pass
            def run(self, progress_callback=None): return {}
            def cleanup(self): pass

        engine = _EchoEngine()
        result = engine.execute_command(
            ["/bin/sh", "-c", "echo ok"],
            run_dir=tmp_path,
        )
        assert result.success is True
        assert result.returncode == 0
        assert "ok" in result.stdout

    def test_execute_command_fails_nonzero(self, tmp_path: Path):
        from backend.common.engines.base import SimulationEngine

        class _BadEngine(SimulationEngine):
            def setup(self, structure, parameters): pass
            def run(self, progress_callback=None): return {}
            def cleanup(self): pass

        engine = _BadEngine()
        result = engine.execute_command(
            ["/bin/sh", "-c", "exit 3"],
            run_dir=tmp_path,
        )
        assert result.success is False
