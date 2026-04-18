"""
Tests for the Session 2.1 Celery spine.

What's covered here
-------------------

- **Queue topology** — the 5 roadmap queues exist on the app and the
  name-prefix routes go where they should.
- **Beat schedule** — reaper is on; orchestrator tick is off unless
  ``ORION_ENABLE_ORCHESTRATOR_BEAT`` is set.
- **Artifact helpers** — ``build_run_dir`` + ``tar_and_upload_run_dir``
  with a fake MinIO client.
- **Event emitter** — NullEventEmitter captures; RedisPubSubEmitter
  publishes via an injected fake client; envelope format is stable.
- **JobLifecycle** — end-to-end context-manager flow against a
  fake job + fake session, covering success / transient retry /
  permanent failure paths.

None of these tests require Redis, a real MinIO, or SQLAlchemy-backed
DB — they use narrowly scoped fakes so CI (and dev) stays fast.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Dict

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes — shared across test classes
# ---------------------------------------------------------------------------


class _FakeJob:
    """Minimal stand-in for SimulationJob with the attributes the lifecycle touches."""

    def __init__(self, status):
        from src.api.models.simulation import SimulationJob

        self.id = "00000000-0000-0000-0000-000000000001"
        self.status = status
        self.progress = None
        self.current_step = None
        self.started_at = None
        self.finished_at = None
        self.updated_at = None
        self.retry_count = 0
        self.error_message = None
        self.worker_id = None
        self.worker_hostname = None
        self.parameters: Dict[str, Any] = {}
        # Bind the real state-machine method.
        self._transition_to = SimulationJob.transition_to.__get__(self)

    def transition_to(self, target, *, error_message=None, set_started=False, set_finished=False):
        return self._transition_to(
            target,
            error_message=error_message,
            set_started=set_started,
            set_finished=set_finished,
        )


class _FakeSession:
    """Session stand-in: stores a single job and returns it from .get()."""

    def __init__(self, job: _FakeJob):
        self._job = job
        self.commits = 0

    def get(self, model, job_id):
        return self._job if job_id == self._job.id else None

    def commit(self):
        self.commits += 1


class _FakeMinio:
    """Captures fput_object calls so the artifact test doesn't need MinIO."""

    def __init__(self, *, raise_on: str = ""):
        self.calls: list[tuple[str, str, str]] = []
        self.raise_on = raise_on

    def fput_object(self, bucket: str, key: str, file_path: str) -> None:
        if self.raise_on and self.raise_on in key:
            raise RuntimeError(f"simulated minio failure on {key}")
        self.calls.append((bucket, key, file_path))


class _FakeRedis:
    """Captures .publish() calls; .ping() returns True."""

    def __init__(self):
        self.published: list[tuple[str, str]] = []

    def ping(self) -> bool:
        return True

    def publish(self, channel: str, message: str) -> int:
        self.published.append((channel, message))
        return 1


# ---------------------------------------------------------------------------
# Queue topology
# ---------------------------------------------------------------------------


class TestCeleryApp:
    def test_canonical_queues_present(self):
        from src.worker.celery_app import celery_app

        got = {q.name for q in celery_app.conf.task_queues}
        for q in ("default", "io", "dft", "md", "ml"):
            assert q in got, f"queue {q!r} missing: {got}"

    def test_name_prefix_routes_are_registered(self):
        from src.worker.celery_app import celery_app

        routes = celery_app.conf.task_routes
        assert routes["orion.dft.*"]["queue"] == "dft"
        assert routes["orion.md.*"]["queue"] == "md"
        assert routes["orion.ml.*"]["queue"] == "ml"
        assert routes["orion.io.*"]["queue"] == "io"

    def test_reaper_registered_in_beat_schedule(self):
        from src.worker.celery_app import celery_app

        sched = celery_app.conf.beat_schedule
        assert "orion.reaper.stalled-jobs" in sched
        assert sched["orion.reaper.stalled-jobs"]["task"] == "orion.io.reap_stalled_jobs"

    def test_orchestrator_beat_gated_by_env(self, monkeypatch):
        """
        Without the env flag, the orchestrator beat entry must not be in
        the schedule. The module uses ``_build_beat_schedule`` at import;
        re-invoke it here with the flag forced.
        """
        from src.worker.celery_app import _build_beat_schedule

        monkeypatch.delenv("ORION_ENABLE_ORCHESTRATOR_BEAT", raising=False)
        sched_off = _build_beat_schedule()
        assert "orion.orchestrator.tick" not in sched_off

        monkeypatch.setenv("ORION_ENABLE_ORCHESTRATOR_BEAT", "true")
        sched_on = _build_beat_schedule()
        assert "orion.orchestrator.tick" in sched_on

    def test_reap_task_callable_exists(self):
        from src.worker.tasks import _reap_stalled_sync  # noqa: F401

        # The sync helper is a function; we don't invoke it here (it
        # needs a live DB). The registered Celery task is covered above.


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


class TestArtifactHelpers:
    def test_build_run_dir_creates_unique_paths(self, tmp_path):
        from backend.common.workers import build_run_dir

        a = build_run_dir("job-1", parent=tmp_path)
        b = build_run_dir("job-1", parent=tmp_path)
        assert a.is_dir() and b.is_dir()
        assert a != b  # same job id, two calls → two dirs

    def test_tar_and_upload_without_minio_writes_local_tarball(self, tmp_path):
        from backend.common.workers import ArtifactBundle, build_run_dir, tar_and_upload_run_dir

        run_dir = build_run_dir("job-42", parent=tmp_path)
        (run_dir / "stdout.txt").write_text("all good\n")
        (run_dir / "energies.json").write_text('{"total": -123.4}')

        bundle = ArtifactBundle(
            run_dir=run_dir,
            job_id="job-42",
            manifest={"engine": "mock", "kind": "mock_static"},
        )
        result = tar_and_upload_run_dir(bundle, minio_client=None)

        assert result["uploaded"] is False
        assert result["size_bytes"] > 0
        assert result["bucket"] == "orion-artifacts"
        assert result["key"] == "jobs/job-42/run.tgz"
        assert Path(result["tarball"]).exists()

        # Verify the manifest landed inside the tarball.
        with tarfile.open(result["tarball"], "r:gz") as tar:
            names = tar.getnames()
            manifest_member = next((n for n in names if n.endswith("manifest.json")), None)
            assert manifest_member is not None
            manifest_bytes = tar.extractfile(manifest_member).read()
            manifest = json.loads(manifest_bytes)
            assert manifest["engine"] == "mock"
            assert manifest["job_id"] == "job-42"
            assert "bundled_at" in manifest
            assert manifest["orion_artifact_schema"] == "v1"

    def test_tar_and_upload_with_minio_calls_fput(self, tmp_path):
        from backend.common.workers import ArtifactBundle, build_run_dir, tar_and_upload_run_dir

        run_dir = build_run_dir("job-99", parent=tmp_path)
        (run_dir / "out.txt").write_text("ok\n")
        minio = _FakeMinio()

        result = tar_and_upload_run_dir(
            ArtifactBundle(run_dir=run_dir, job_id="job-99"),
            minio_client=minio,
        )
        assert result["uploaded"] is True
        assert len(minio.calls) == 1
        bucket, key, path = minio.calls[0]
        assert bucket == "orion-artifacts"
        assert key == "jobs/job-99/run.tgz"
        assert Path(path).exists()

    def test_tar_and_upload_failed_upload_is_non_fatal(self, tmp_path):
        """An exception from minio.fput_object must not kill the caller."""
        from backend.common.workers import ArtifactBundle, build_run_dir, tar_and_upload_run_dir

        run_dir = build_run_dir("job-err", parent=tmp_path)
        minio = _FakeMinio(raise_on="job-err")
        result = tar_and_upload_run_dir(
            ArtifactBundle(run_dir=run_dir, job_id="job-err"),
            minio_client=minio,
        )
        assert result["uploaded"] is False
        # Tarball still got written locally.
        assert Path(result["tarball"]).exists()


# ---------------------------------------------------------------------------
# Event emitters
# ---------------------------------------------------------------------------


class TestEventEmitters:
    def test_null_emitter_captures_events(self):
        from backend.common.workers.events import NullEventEmitter

        e = NullEventEmitter()
        e.emit(channel="ch", event="job.status.running", payload={"x": 1})
        e.emit(channel="ch", event="job.status.completed", payload={"outputs": {}})
        assert [x["event"] for x in e.events] == [
            "job.status.running", "job.status.completed",
        ]

    def test_redis_emitter_publishes_valid_envelope(self):
        from backend.common.workers.events import EMITTER_VERSION, RedisPubSubEmitter

        fake = _FakeRedis()
        e = RedisPubSubEmitter(client=fake)
        rc = e.emit(channel="orion:events:job:123", event="job.status.running",
                    payload={"worker_id": "w-1"})
        assert rc == 1
        channel, body = fake.published[0]
        assert channel == "orion:events:job:123"
        envelope = json.loads(body)
        assert envelope["event"] == "job.status.running"
        assert envelope["payload"] == {"worker_id": "w-1"}
        assert envelope["emitter_version"] == EMITTER_VERSION
        assert "ts" in envelope

    def test_redis_emitter_dropped_emit_returns_zero(self):
        """Redis failure must never propagate — emitters are fire-and-forget."""
        from backend.common.workers.events import RedisPubSubEmitter

        class _Broken:
            def publish(self, *_a, **_k):
                raise RuntimeError("redis down")

        e = RedisPubSubEmitter(client=_Broken())
        rc = e.emit(channel="x", event="y", payload={})
        assert rc == 0


# ---------------------------------------------------------------------------
# JobLifecycle — the context manager
# ---------------------------------------------------------------------------


class TestJobLifecycleSuccess:
    def test_queued_to_running_to_completed(self):
        from backend.common.workers import JobLifecycle
        from backend.common.workers.events import NullEventEmitter
        from src.api.models.simulation import JobStatus

        job = _FakeJob(JobStatus.QUEUED)
        session = _FakeSession(job)
        emitter = NullEventEmitter()

        with JobLifecycle(
            job.id, session=session, emitter=emitter, worker_id="w-1",
        ) as lc:
            lc.outputs = {"energy_ev": -42.0}

        assert job.status == JobStatus.COMPLETED
        assert job.started_at is not None
        assert job.finished_at is not None
        assert job.parameters.get("_outputs") == {"energy_ev": -42.0}
        assert job.worker_id == "w-1"
        # on_start commit + on_success commit = 2
        assert session.commits >= 2
        events = [e["event"] for e in emitter.events]
        assert events[0] == "job.status.running"
        assert events[-1] == "job.status.completed"

    def test_pending_directly_to_running_allowed(self):
        """If Celery pulls the job before the QUEUED transition landed, still OK."""
        from backend.common.workers import JobLifecycle
        from src.api.models.simulation import JobStatus

        job = _FakeJob(JobStatus.PENDING)
        session = _FakeSession(job)
        with JobLifecycle(job.id, session=session) as lc:
            lc.outputs = {}
        assert job.status == JobStatus.COMPLETED


class TestJobLifecycleTransientRetry:
    def test_transient_error_increments_retry_count_and_reraises(self):
        from backend.common.workers import JobLifecycle, TransientEngineError
        from backend.common.workers.events import NullEventEmitter
        from src.api.models.simulation import JobStatus

        job = _FakeJob(JobStatus.QUEUED)
        session = _FakeSession(job)
        emitter = NullEventEmitter()

        with pytest.raises(TransientEngineError):
            with JobLifecycle(job.id, session=session, emitter=emitter):
                raise TransientEngineError("SCF didn't converge; try again")

        assert job.retry_count == 1
        # Not a terminal transition — job stays RUNNING for the next attempt.
        assert job.status == JobStatus.RUNNING
        events = [e["event"] for e in emitter.events]
        assert "job.status.retrying" in events


class TestJobLifecyclePermanentFailure:
    def test_arbitrary_exception_marks_failed(self):
        from backend.common.workers import JobLifecycle
        from backend.common.workers.events import NullEventEmitter
        from src.api.models.simulation import JobStatus

        job = _FakeJob(JobStatus.QUEUED)
        session = _FakeSession(job)
        emitter = NullEventEmitter()

        with pytest.raises(ValueError):
            with JobLifecycle(job.id, session=session, emitter=emitter):
                raise ValueError("something fundamentally wrong")

        assert job.status == JobStatus.FAILED
        assert job.finished_at is not None
        assert "ValueError" in (job.error_message or "")
        events = [e["event"] for e in emitter.events]
        assert events[-1] == "job.status.failed"

    def test_missing_job_row_raises_lifecycle_error(self):
        from backend.common.workers import JobLifecycle, JobLifecycleError

        class _EmptySession:
            def get(self, *_a, **_k):
                return None

            def commit(self):
                pass

        with pytest.raises(JobLifecycleError):
            with JobLifecycle("ghost", session=_EmptySession()):
                pass


# ---------------------------------------------------------------------------
# ExecutionResult — added in Session 2.1 (unblocks engine imports)
# ---------------------------------------------------------------------------


class TestExecutionResult:
    def test_shape(self):
        from backend.common.engines.base import ExecutionResult

        r = ExecutionResult(success=True, returncode=0, stdout="hi", stderr="")
        assert r.success is True
        assert r.timed_out is False
        as_dict = r.as_dict()
        assert as_dict["returncode"] == 0
        assert as_dict["stdout"] == "hi"
