"""Tests for Session 2.2 — mock_static engine end-to-end.

Scope
-----

1. :class:`backend.common.jobs.MockStaticOutput` — schema validation.
2. :func:`backend.common.jobs.run_mock_static` — physics-shaped math,
   determinism, forces within spec, 10-frame trajectory.
3. :func:`backend.common.jobs.write_trajectory_xyz` — correct shape.
4. The Celery task orchestration (``orion.mock.static``) with fake
   session, fake MinIO, and fake Redis. The task is invoked directly
   (as a plain Python callable) so we don't need a running broker.
5. Router wiring — dispatch knows about ``mock_static``, artifacts
   endpoint returns the expected shape given a job row with an
   ``artifact`` blob on ``extra_metadata``.

No Redis / Postgres / MinIO required.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. MockStaticOutput schema
# ---------------------------------------------------------------------------


class TestMockStaticOutputSchema:
    def test_schema_version_tag(self):
        from backend.common.jobs import MOCK_STATIC_OUTPUT_SCHEMA_VERSION

        assert MOCK_STATIC_OUTPUT_SCHEMA_VERSION == "mock_static.v1"

    def test_validated_construction(self):
        from backend.common.jobs import MockForceEntry, MockStaticOutput, MockTrajectoryFrame

        out = MockStaticOutput(
            n_atoms=1,
            formula="Si",
            energy_ev=-21.0,
            energy_per_atom_ev=-21.0,
            max_abs_force_ev_per_ang=0.01,
            forces=[MockForceEntry(species="Si", fx=0.0, fy=0.0, fz=0.01)],
            trajectory_n_frames=1,
            trajectory=[MockTrajectoryFrame(index=0, positions=[[0.0, 0.0, 0.0]])],
            seed=42,
        )
        assert out.schema_version == "mock_static.v1"
        assert out.n_atoms == 1

    def test_extra_fields_rejected(self):
        from pydantic import ValidationError

        from backend.common.jobs import MockStaticOutput

        with pytest.raises(ValidationError):
            MockStaticOutput(
                n_atoms=1,
                formula="Si",
                energy_ev=-21,
                energy_per_atom_ev=-21,
                max_abs_force_ev_per_ang=0.01,
                forces=[],
                trajectory_n_frames=1,
                trajectory=[],
                seed=1,
                garbage="no thanks",
            )


# ---------------------------------------------------------------------------
# 2. run_mock_static — physics, determinism, shape
# ---------------------------------------------------------------------------


_SI_ATOMS = [
    {"species": "Si", "position": [0.0, 0.0, 0.0]},
    {"species": "Si", "position": [0.25, 0.25, 0.25]},
]

_NACL_ATOMS = [
    {"species": "Na", "position": [0.0, 0.0, 0.0]},
    {"species": "Cl", "position": [0.5, 0.5, 0.5]},
]


class TestRunMockStatic:
    def test_energy_magnitudes_are_physics_shaped(self):
        from backend.common.jobs import run_mock_static
        from backend.common.jobs.mock_static import COEFFICIENT_EV_PER_Z

        out = run_mock_static(
            structure_id="si-fixture",
            atoms=_SI_ATOMS,
            formula="Si2",
        )
        # Si Z=14; two atoms → expected ≈ -42 eV baseline.
        expected = COEFFICIENT_EV_PER_Z * (14 + 14)
        assert abs(out.energy_ev - expected) < 0.5, out.energy_ev
        assert out.n_atoms == 2
        assert out.formula == "Si2"
        assert out.energy_per_atom_ev == pytest.approx(out.energy_ev / 2)
        assert out.converged is True

    def test_is_deterministic_by_structure_id(self):
        from backend.common.jobs import run_mock_static

        a = run_mock_static(structure_id="x", atoms=_SI_ATOMS)
        b = run_mock_static(structure_id="x", atoms=_SI_ATOMS)
        c = run_mock_static(structure_id="y", atoms=_SI_ATOMS)

        assert a.energy_ev == b.energy_ev
        assert a.seed == b.seed
        assert a.energy_ev != c.energy_ev  # different structure → different noise

    def test_forces_are_small_and_one_per_atom(self):
        from backend.common.jobs import FORCE_NOISE_SIGMA_EV_PER_ANG, run_mock_static

        out = run_mock_static(structure_id="si", atoms=_SI_ATOMS)
        assert len(out.forces) == 2
        # At σ≈0.05, ≥3σ bound gives 0.15 eV/Å — accept generous
        # envelope: the max abs component is well under 0.5 in practice.
        assert out.max_abs_force_ev_per_ang < 0.5
        for f in out.forces:
            for comp in (f.fx, f.fy, f.fz):
                assert abs(comp) < 10 * FORCE_NOISE_SIGMA_EV_PER_ANG

    def test_trajectory_has_ten_frames(self):
        from backend.common.jobs import TRAJECTORY_N_FRAMES, run_mock_static

        out = run_mock_static(structure_id="x", atoms=_NACL_ATOMS)
        assert out.trajectory_n_frames == TRAJECTORY_N_FRAMES == 10
        assert len(out.trajectory) == 10
        # Frame 0 must equal the input positions exactly.
        base = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        assert out.trajectory[0].positions == base
        # Later frames are perturbed — positions change but stay nearby.
        for frame in out.trajectory[1:]:
            for pos_new, pos_base in zip(frame.positions, base):
                dx = pos_new[0] - pos_base[0]
                dy = pos_new[1] - pos_base[1]
                dz = pos_new[2] - pos_base[2]
                assert abs(dx) < 0.2
                assert abs(dy) < 0.2
                assert abs(dz) < 0.2

    def test_rejects_empty_atoms(self):
        from backend.common.jobs import run_mock_static

        with pytest.raises(ValueError):
            run_mock_static(structure_id="x", atoms=[])


class TestWriteTrajectoryXYZ:
    def test_xyz_frame_layout(self):
        from backend.common.jobs import run_mock_static, write_trajectory_xyz

        out = run_mock_static(structure_id="si", atoms=_SI_ATOMS)
        species = [f.species for f in out.forces]
        body = write_trajectory_xyz(out.trajectory, species)
        lines = body.splitlines()
        # Each of 10 frames: n_atoms line + comment line + 2 atom lines = 4 lines.
        assert len(lines) == 10 * (2 + 2)
        assert lines[0] == "2"
        assert lines[1].startswith("frame=0")
        assert lines[2].startswith("Si ")
        # Last frame header sanity.
        assert lines[-4] == "2"
        assert lines[-3].startswith("frame=9")

    def test_size_mismatch_raises(self):
        from backend.common.jobs import MockTrajectoryFrame, write_trajectory_xyz

        with pytest.raises(ValueError):
            write_trajectory_xyz(
                [MockTrajectoryFrame(index=0, positions=[[0, 0, 0], [1, 1, 1]])],
                species=["Si"],  # one species, two positions → mismatch
            )


# ---------------------------------------------------------------------------
# 3. Celery task orchestration (no broker, no DB, no MinIO)
# ---------------------------------------------------------------------------


class _FakeStructure:
    def __init__(self, structure_id: str):
        self.id = structure_id
        self.atoms = [
            {"species": "Na", "position": [0.0, 0.0, 0.0]},
            {"species": "Cl", "position": [0.5, 0.5, 0.5]},
        ]
        self.formula = "NaCl"
        self.num_atoms = 2
        self.name = "NaCl test"


class _FakeJob:
    def __init__(self, job_id: str, structure_id: str):
        from src.api.models.simulation import JobStatus, SimulationJob

        self.id = job_id
        self.structure_id = structure_id
        self.status = JobStatus.QUEUED
        self.progress = None
        self.current_step = None
        self.started_at = None
        self.finished_at = None
        self.updated_at = None
        self.retry_count = 0
        self.error_message = None
        self.worker_id = None
        self.worker_hostname = None
        self.parameters: dict = {}
        self.extra_metadata: dict = {}
        self._transition_to = SimulationJob.transition_to.__get__(self)

    def transition_to(self, target, *, error_message=None, set_started=False, set_finished=False):
        return self._transition_to(
            target,
            error_message=error_message,
            set_started=set_started,
            set_finished=set_finished,
        )


class _FakeSession:
    def __init__(self, job, structure):
        self._job = job
        self._structure = structure
        self.commits = 0

    def get(self, model, id_):
        # Match by class name so we don't need to import the real classes.
        cls = model.__name__
        if cls == "SimulationJob" and id_ == self._job.id:
            return self._job
        if cls == "Structure" and id_ == self._structure.id:
            return self._structure
        return None

    def commit(self):
        self.commits += 1


class _FakeMinio:
    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []

    def fput_object(self, bucket, key, path):
        self.calls.append((bucket, key, path))

    def bucket_exists(self, bucket):
        return True

    def make_bucket(self, bucket):
        pass


def _install_task_fakes(monkeypatch, session):
    """Patch the task's infra: sync session, MinIO, event emitter."""
    from backend.common.workers.events import NullEventEmitter

    import src.worker.tasks as worker_tasks

    class _SessFactory:
        def __call__(self):
            return self

        def __enter__(self):
            return session

        def __exit__(self, *_a):
            return False

    fake_engine = type("FakeEngine", (), {"dispose": lambda self: None})()
    monkeypatch.setattr(
        worker_tasks,
        "_sync_session_for_worker",
        lambda: (fake_engine, _SessFactory()),
    )

    fake_minio = _FakeMinio()
    monkeypatch.setattr(
        "backend.common.jobs.build_minio_client", lambda **_kw: fake_minio,
    )
    monkeypatch.setattr(
        "backend.common.jobs.artifacts.build_minio_client",
        lambda **_kw: fake_minio,
    )
    monkeypatch.setattr(
        "backend.common.workers.events.RedisPubSubEmitter",
        lambda *a, **kw: NullEventEmitter(),
    )
    return fake_minio


def _invoke_task(task, job_id: str):
    """Invoke a bound Celery task in-process without a broker."""
    result = task.apply(args=[job_id])
    return result.get(propagate=True)


class TestMockStaticTaskOrchestration:
    """Exercise the task via Celery's eager apply(), without a broker."""

    def test_task_populates_outputs_and_bundle(self, monkeypatch):
        from src.api.models.simulation import JobStatus
        from src.worker.tasks import run_mock_static_job

        job = _FakeJob("00000000-0000-0000-0000-000000000001", "struct-123")
        structure = _FakeStructure("struct-123")
        session = _FakeSession(job, structure)

        _install_task_fakes(monkeypatch, session)

        result = _invoke_task(run_mock_static_job, str(job.id))

        assert result["status"] == "success"
        assert result["n_atoms"] == 2
        assert job.status == JobStatus.COMPLETED
        outs = job.parameters.get("_outputs")
        assert outs is not None
        assert outs["schema_version"] == "mock_static.v1"
        assert outs["n_atoms"] == 2
        art = job.extra_metadata["artifact"]
        assert art["bucket"] == "orion-artifacts"
        assert art["key"] == f"jobs/{job.id}/run.tgz"
        assert art["uploaded"] is True
        assert Path(art["tarball"]).exists()

    def test_task_fails_cleanly_when_structure_missing(self, monkeypatch):
        from src.api.models.simulation import JobStatus
        from src.worker.tasks import run_mock_static_job

        job = _FakeJob("00000000-0000-0000-0000-000000000002", "missing-struct")
        # SimulationJob.get returns the job; Structure.get returns None.
        session = _FakeSession(job, _FakeStructure("other-struct"))

        _install_task_fakes(monkeypatch, session)

        with pytest.raises(ValueError):
            _invoke_task(run_mock_static_job, str(job.id))

        assert job.status == JobStatus.FAILED
        assert "ValueError" in (job.error_message or "")


# ---------------------------------------------------------------------------
# 4. Router wiring — dispatch catalog + artifacts shape
# ---------------------------------------------------------------------------


class TestDispatchCatalog:
    def test_mock_static_is_dispatchable(self):
        from src.api.routers.jobs import _BUILTIN_TEMPLATES, _DISPATCH_TASKS

        assert "mock_static" in _DISPATCH_TASKS
        assert _DISPATCH_TASKS["mock_static"] == "orion.mock.static"
        assert "mock_static" in _BUILTIN_TEMPLATES

    def test_dispatch_schema_accepts_minimal_body(self):
        from src.api.schemas.simulation import SimulationJobDispatch

        body = SimulationJobDispatch(
            kind="mock_static",
            structure_id="11111111-1111-1111-1111-111111111111",
        )
        assert body.kind == "mock_static"
        assert body.priority == 5
