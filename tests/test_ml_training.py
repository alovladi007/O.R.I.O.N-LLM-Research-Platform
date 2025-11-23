"""
Integration Tests for ML Training Pipeline
===========================================

Tests the ML model training pipeline:
- Training job creation and submission
- Celery worker task execution
- Model registry integration
- Progress tracking
- Error handling
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.user import User
from src.api.models.ml_training import MLTrainingJob, TrainingStatus
from src.api.models.ml_model_registry import MLModelRegistry


class TestMLTrainingJobModel:
    """Test ML training job model"""

    @pytest.mark.asyncio
    async def test_create_training_job(self, db_session: AsyncSession, test_user: User):
        """Test creating an ML training job"""
        training_job = MLTrainingJob(
            owner_id=test_user.id,
            name="test_cgcnn_bandgap",
            description="Test CGCNN model for bandgap prediction",
            model_type="CGCNN",
            target_property="bandgap",
            training_config={
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
            },
            dataset_config={
                "min_samples": 100,
                "train_fraction": 0.8,
                "target_property": "bandgap",
            },
            status=TrainingStatus.PENDING,
            total_epochs=100,
        )

        db_session.add(training_job)
        await db_session.commit()
        await db_session.refresh(training_job)

        # Verify job created
        assert training_job.id is not None
        assert training_job.name == "test_cgcnn_bandgap"
        assert training_job.model_type == "CGCNN"
        assert training_job.target_property == "bandgap"
        assert training_job.status == TrainingStatus.PENDING
        assert training_job.current_epoch is None
        assert training_job.progress is None

    @pytest.mark.asyncio
    async def test_training_job_status_transitions(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test training job status transitions"""
        job = MLTrainingJob(
            owner_id=test_user.id,
            name="test_model",
            model_type="CGCNN",
            target_property="formation_energy",
            training_config={},
            dataset_config={},
            status=TrainingStatus.PENDING,
        )

        db_session.add(job)
        await db_session.commit()

        # PENDING → QUEUED
        job.status = TrainingStatus.QUEUED
        job.celery_task_id = "test-task-123"
        await db_session.commit()
        assert job.status == TrainingStatus.QUEUED

        # QUEUED → RUNNING
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.utcnow()
        job.current_epoch = 1
        job.progress = 0.01
        await db_session.commit()
        assert job.status == TrainingStatus.RUNNING
        assert job.started_at is not None

        # RUNNING → COMPLETED
        job.status = TrainingStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.current_epoch = 100
        job.progress = 1.0
        job.best_val_loss = 0.05
        job.final_metrics = {
            "train_loss": 0.04,
            "val_loss": 0.05,
        }
        await db_session.commit()
        assert job.status == TrainingStatus.COMPLETED
        assert job.completed_at is not None
        assert job.progress == 1.0

    @pytest.mark.asyncio
    async def test_training_job_to_dict(self, db_session: AsyncSession, test_user: User):
        """Test training job to_dict conversion"""
        job = MLTrainingJob(
            owner_id=test_user.id,
            name="test_model",
            model_type="CGCNN",
            target_property="bandgap",
            training_config={"epochs": 50},
            dataset_config={"train_fraction": 0.8},
            status=TrainingStatus.COMPLETED,
            current_epoch=50,
            total_epochs=50,
            progress=1.0,
            best_val_loss=0.123,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )

        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

        job_dict = job.to_dict()

        assert job_dict["id"] == str(job.id)
        assert job_dict["name"] == "test_model"
        assert job_dict["model_type"] == "CGCNN"
        assert job_dict["status"] == "COMPLETED"
        assert job_dict["current_epoch"] == 50
        assert job_dict["best_val_loss"] == 0.123
        assert "duration_seconds" in job_dict


class TestMLTrainingEndpoint:
    """Test ML training API endpoint"""

    @pytest.mark.asyncio
    async def test_submit_training_job(self, db_session: AsyncSession, test_user: User):
        """Test submitting a training job via API"""
        from src.api.schemas.ml import TrainingRequest

        request = TrainingRequest(
            target_property="bandgap",
            model_type="CGCNN",
            model_name="cgcnn_bandgap_test",
            training_config={
                "epochs": 50,
                "batch_size": 16,
                "learning_rate": 0.001,
            },
            min_samples=50,
            train_fraction=0.8,
        )

        # Mock Celery task
        with patch("src.api.routers.ml.run_ml_training") as mock_task:
            mock_result = Mock()
            mock_result.id = "celery-task-123"
            mock_task.apply_async.return_value = mock_result

            from src.api.routers.ml import start_training_job

            response = await start_training_job(
                request=request, db=db_session, current_user=test_user
            )

        # Verify response
        assert response.status in ["PENDING", "QUEUED"]
        assert response.model_name == "cgcnn_bandgap_test"
        assert "submitted successfully" in response.message.lower()
        assert response.estimated_time_minutes is not None

        # Verify job in database
        job_id = uuid.UUID(response.job_id)
        job = await db_session.get(MLTrainingJob, job_id)

        assert job is not None
        assert job.name == "cgcnn_bandgap_test"
        assert job.model_type == "CGCNN"
        assert job.target_property == "bandgap"
        assert job.status == TrainingStatus.QUEUED

    @pytest.mark.asyncio
    async def test_submit_training_duplicate_name_error(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test that duplicate model names are rejected"""
        from src.api.schemas.ml import TrainingRequest
        from fastapi import HTTPException

        # Create existing model in registry
        existing_model = MLModelRegistry(
            name="existing_model",
            version="1.0.0",
            target="bandgap",
            model_type="CGCNN",
            checkpoint_path="/models/existing.pth",
            training_config={},
            metrics={},
            is_system_provided=False,
        )
        db_session.add(existing_model)
        await db_session.commit()

        # Try to create training job with same name
        request = TrainingRequest(
            target_property="bandgap",
            model_type="CGCNN",
            model_name="existing_model",
            training_config={"epochs": 50},
            min_samples=50,
            train_fraction=0.8,
        )

        with patch("src.api.routers.ml.run_ml_training"):
            from src.api.routers.ml import start_training_job

            # Should raise authorization error for non-admin
            with pytest.raises(Exception):  # AuthorizationError
                await start_training_job(
                    request=request, db=db_session, current_user=test_user
                )


class TestMLTrainingTask:
    """Test ML training Celery task"""

    def test_ml_training_task_exists(self):
        """Test that run_ml_training task exists"""
        from src.worker.tasks import run_ml_training

        assert callable(run_ml_training)
        assert run_ml_training.name == "run_ml_training"

    def test_ml_training_task_signature(self):
        """Test training task has correct signature"""
        from src.worker.tasks import run_ml_training

        # Verify task configuration
        assert run_ml_training.max_retries == 2
        assert run_ml_training.default_retry_delay == 300  # 5 minutes


class TestTrainingStatusEnum:
    """Test TrainingStatus enum"""

    def test_training_status_values(self):
        """Test all training status values"""
        from src.api.models.ml_training import TrainingStatus

        expected_statuses = {
            "PENDING",
            "QUEUED",
            "RUNNING",
            "COMPLETED",
            "FAILED",
            "CANCELLED",
        }

        actual_statuses = {status.value for status in TrainingStatus}

        assert actual_statuses == expected_statuses

    def test_training_status_progression(self):
        """Test typical status progression"""
        from src.api.models.ml_training import TrainingStatus

        # Typical progression
        progression = [
            TrainingStatus.PENDING,
            TrainingStatus.QUEUED,
            TrainingStatus.RUNNING,
            TrainingStatus.COMPLETED,
        ]

        for i, status in enumerate(progression):
            assert isinstance(status, TrainingStatus)
            assert status.value in [
                "PENDING",
                "QUEUED",
                "RUNNING",
                "COMPLETED",
                "FAILED",
                "CANCELLED",
            ]


class TestModelRegistryIntegration:
    """Test integration with model registry"""

    @pytest.mark.asyncio
    async def test_training_job_links_to_registry(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test that completed training job links to model registry"""
        # Create model in registry
        model = MLModelRegistry(
            name="trained_model",
            version="1.0.0",
            target="bandgap",
            model_type="CGCNN",
            checkpoint_path="/models/trained.pth",
            training_config={},
            metrics={"val_loss": 0.05},
            is_system_provided=False,
        )
        db_session.add(model)
        await db_session.commit()
        await db_session.refresh(model)

        # Create training job that references it
        job = MLTrainingJob(
            owner_id=test_user.id,
            name="trained_model",
            model_type="CGCNN",
            target_property="bandgap",
            training_config={},
            dataset_config={},
            status=TrainingStatus.COMPLETED,
            model_registry_id=model.id,
        )
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

        # Verify relationship
        assert job.model_registry_id == model.id
        assert job.registered_model is not None
        assert job.registered_model.name == "trained_model"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
