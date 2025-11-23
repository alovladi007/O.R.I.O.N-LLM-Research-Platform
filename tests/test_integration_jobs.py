"""
Integration Tests for Job Lifecycle
====================================

Tests the complete job lifecycle from creation to completion:
- Job creation
- Job submission to Celery
- Status transitions
- Job cancellation
- Job completion and results
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.simulation import SimulationJob, SimulationResult, JobStatus
from src.api.models.user import User
from src.api.models.structure import Structure
from src.api.models.material import Material
from src.worker.tasks import run_simulation_job, cancel_job, run_dummy_job


class TestJobLifecycle:
    """Test complete job lifecycle"""

    @pytest.mark.asyncio
    async def test_job_creation_and_status(self, db_session: AsyncSession, test_user: User):
        """Test job creation with proper initial status"""
        # Create material and structure
        material = Material(
            owner_id=test_user.id,
            formula="MoS2",
            name="Molybdenum Disulfide",
            description="2D material"
        )
        db_session.add(material)
        await db_session.commit()
        await db_session.refresh(material)

        structure = Structure(
            material_id=material.id,
            name="MoS2 1H",
            format="CIF",
            raw_text="data_MoS2\\n...",
            formula="MoS2",
            num_atoms=3
        )
        db_session.add(structure)
        await db_session.commit()
        await db_session.refresh(structure)

        # Create job
        job = SimulationJob(
            owner_id=test_user.id,
            structure_id=structure.id,
            engine="MOCK",
            status=JobStatus.PENDING,
            parameters={"cutoff": 500},
            priority=5
        )
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

        # Verify
        assert job.id is not None
        assert job.status == JobStatus.PENDING
        assert job.progress == 0.0
        assert job.created_at is not None
        assert job.started_at is None
        assert job.finished_at is None

    @pytest.mark.asyncio
    async def test_job_status_transitions(self, db_session: AsyncSession, test_user: User):
        """Test valid job status transitions"""
        job = SimulationJob(
            owner_id=test_user.id,
            engine="MOCK",
            status=JobStatus.PENDING,
            parameters={}
        )
        db_session.add(job)
        await db_session.commit()

        # PENDING → QUEUED
        job.status = JobStatus.QUEUED
        await db_session.commit()
        assert job.status == JobStatus.QUEUED

        # QUEUED → RUNNING
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        await db_session.commit()
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None

        # RUNNING → COMPLETED
        job.status = JobStatus.COMPLETED
        job.finished_at = datetime.utcnow()
        job.progress = 1.0
        await db_session.commit()
        assert job.status == JobStatus.COMPLETED
        assert job.finished_at is not None
        assert job.is_terminal

    @pytest.mark.asyncio
    async def test_job_cancellation_flow(self, db_session: AsyncSession, test_user: User):
        """Test job cancellation from different states"""
        # Test cancelling PENDING job
        job_pending = SimulationJob(
            owner_id=test_user.id,
            engine="MOCK",
            status=JobStatus.PENDING,
            parameters={}
        )
        db_session.add(job_pending)
        await db_session.commit()

        job_pending.status = JobStatus.CANCELLED
        job_pending.finished_at = datetime.utcnow()
        await db_session.commit()

        assert job_pending.status == JobStatus.CANCELLED
        assert job_pending.is_terminal

        # Test cancelling RUNNING job
        job_running = SimulationJob(
            owner_id=test_user.id,
            engine="MOCK",
            status=JobStatus.RUNNING,
            started_at=datetime.utcnow(),
            task_id="test-task-123",
            parameters={}
        )
        db_session.add(job_running)
        await db_session.commit()

        job_running.status = JobStatus.CANCELLED
        job_running.finished_at = datetime.utcnow()
        await db_session.commit()

        assert job_running.status == JobStatus.CANCELLED
        assert job_running.is_terminal

    @pytest.mark.asyncio
    async def test_job_with_results(self, db_session: AsyncSession, test_user: User):
        """Test job completion with results"""
        job = SimulationJob(
            owner_id=test_user.id,
            engine="MOCK",
            status=JobStatus.COMPLETED,
            started_at=datetime.utcnow() - timedelta(minutes=5),
            finished_at=datetime.utcnow(),
            progress=1.0,
            parameters={}
        )
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

        # Add results
        result = SimulationResult(
            simulation_job_id=job.id,
            summary={
                "total_energy": -42.5,
                "band_gap": 1.8,
                "formula": "MoS2"
            },
            convergence_reached=True,
            quality_score=0.95,
            metadata={"notes": "Test simulation"}
        )
        db_session.add(result)
        await db_session.commit()

        # Verify
        await db_session.refresh(job)
        assert len(job.results) == 1
        assert job.results[0].summary["total_energy"] == -42.5
        assert job.results[0].convergence_reached is True

    def test_job_priority_levels(self):
        """Test job priority handling"""
        low_priority = SimulationJob(
            owner_id=uuid.uuid4(),
            engine="MOCK",
            status=JobStatus.PENDING,
            priority=1,
            parameters={}
        )
        high_priority = SimulationJob(
            owner_id=uuid.uuid4(),
            engine="MOCK",
            status=JobStatus.PENDING,
            priority=10,
            parameters={}
        )

        assert low_priority.priority == 1
        assert high_priority.priority == 10
        assert high_priority.priority > low_priority.priority

    @pytest.mark.asyncio
    async def test_job_error_handling(self, db_session: AsyncSession, test_user: User):
        """Test job failure with error message"""
        job = SimulationJob(
            owner_id=test_user.id,
            engine="MOCK",
            status=JobStatus.RUNNING,
            started_at=datetime.utcnow(),
            parameters={}
        )
        db_session.add(job)
        await db_session.commit()

        # Simulate failure
        job.status = JobStatus.FAILED
        job.finished_at = datetime.utcnow()
        job.error_message = "Simulation diverged: energy > 1e10"
        await db_session.commit()

        # Verify
        await db_session.refresh(job)
        assert job.status == JobStatus.FAILED
        assert job.is_terminal
        assert "diverged" in job.error_message

    @pytest.mark.asyncio
    async def test_job_retry_count(self, db_session: AsyncSession, test_user: User):
        """Test job retry tracking"""
        job = SimulationJob(
            owner_id=test_user.id,
            engine="MOCK",
            status=JobStatus.QUEUED,
            retry_count=0,
            parameters={}
        )
        db_session.add(job)
        await db_session.commit()

        # Simulate retries
        for i in range(1, 4):
            job.retry_count = i
            await db_session.commit()
            await db_session.refresh(job)
            assert job.retry_count == i

    @pytest.mark.asyncio
    async def test_job_progress_tracking(self, db_session: AsyncSession, test_user: User):
        """Test job progress updates"""
        job = SimulationJob(
            owner_id=test_user.id,
            engine="MOCK",
            status=JobStatus.RUNNING,
            started_at=datetime.utcnow(),
            progress=0.0,
            current_step="Initializing",
            parameters={}
        )
        db_session.add(job)
        await db_session.commit()

        # Update progress
        progress_steps = [
            (0.25, "Processing structure"),
            (0.50, "Running calculation"),
            (0.75, "Analyzing results"),
            (1.00, "Completed")
        ]

        for progress, step in progress_steps:
            job.progress = progress
            job.current_step = step
            await db_session.commit()
            await db_session.refresh(job)
            assert job.progress == progress
            assert job.current_step == step


class TestCeleryTaskIntegration:
    """Test Celery task integration"""

    def test_dummy_job_task_signature(self):
        """Test dummy job task has correct signature"""
        from src.worker.tasks import run_dummy_job

        assert callable(run_dummy_job)
        assert run_dummy_job.name == "run_dummy_job"

    def test_simulation_job_task_signature(self):
        """Test simulation job task has correct signature"""
        from src.worker.tasks import run_simulation_job

        assert callable(run_simulation_job)
        assert run_simulation_job.name == "run_simulation_job"

    def test_cancel_job_task_signature(self):
        """Test cancel job task has correct signature"""
        from src.worker.tasks import cancel_job

        assert callable(cancel_job)
        assert run_cancel_job.name == "cancel_job"

    def test_continuum_task_signature(self):
        """Test continuum simulation task"""
        from src.worker.tasks import run_continuum_simulation

        assert callable(run_continuum_simulation)
        assert run_continuum_simulation.name == "run_continuum_simulation"

    def test_mesoscale_task_signature(self):
        """Test mesoscale simulation task"""
        from src.worker.tasks import run_mesoscale_simulation

        assert callable(run_mesoscale_simulation)
        assert run_mesoscale_simulation.name == "run_mesoscale_simulation"


class TestJobValidation:
    """Test job validation rules"""

    @pytest.mark.asyncio
    async def test_job_requires_owner(self, db_session: AsyncSession):
        """Test that jobs require an owner"""
        with pytest.raises(Exception):  # Should raise integrity error
            job = SimulationJob(
                engine="MOCK",
                status=JobStatus.PENDING,
                parameters={}
                # Missing owner_id
            )
            db_session.add(job)
            await db_session.commit()

    def test_job_status_is_terminal(self):
        """Test is_terminal property"""
        completed_job = SimulationJob(
            owner_id=uuid.uuid4(),
            engine="MOCK",
            status=JobStatus.COMPLETED,
            parameters={}
        )
        failed_job = SimulationJob(
            owner_id=uuid.uuid4(),
            engine="MOCK",
            status=JobStatus.FAILED,
            parameters={}
        )
        cancelled_job = SimulationJob(
            owner_id=uuid.uuid4(),
            engine="MOCK",
            status=JobStatus.CANCELLED,
            parameters={}
        )
        running_job = SimulationJob(
            owner_id=uuid.uuid4(),
            engine="MOCK",
            status=JobStatus.RUNNING,
            parameters={}
        )

        assert completed_job.is_terminal
        assert failed_job.is_terminal
        assert cancelled_job.is_terminal
        assert not running_job.is_terminal

    def test_job_parameters_validation(self):
        """Test job parameters are stored correctly"""
        params = {
            "cutoff_energy": 500,
            "k_points": [4, 4, 4],
            "convergence_threshold": 1e-6,
            "max_iterations": 100
        }

        job = SimulationJob(
            owner_id=uuid.uuid4(),
            engine="VASP",
            status=JobStatus.PENDING,
            parameters=params
        )

        assert job.parameters == params
        assert job.parameters["cutoff_energy"] == 500
        assert job.parameters["k_points"] == [4, 4, 4]


# Run integration tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
