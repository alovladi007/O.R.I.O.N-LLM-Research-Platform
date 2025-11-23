"""
Smoke Tests for Critical Fixes
================================

Tests for the critical fixes implemented:
1. Import fixes (orchestrator.py, agent.py)
2. get_optional_user() function
3. Orchestrator exports
4. Celery async/await fixes
5. Database sync session support

These tests ensure the application can start and basic functionality works.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio


class TestImports:
    """Test that critical modules can be imported"""

    def test_import_security_module(self):
        """Test security module imports"""
        from src.api.auth.security import get_current_user, get_optional_user, SecurityService

        assert callable(get_current_user)
        assert callable(get_optional_user)
        assert SecurityService is not None

    def test_import_orchestrator_routers(self):
        """Test orchestrator router imports"""
        from src.api.routers.orchestrator import router

        assert router is not None
        assert router.prefix == "/orchestrator"

    def test_import_agent_routers(self):
        """Test agent router imports"""
        from src.api.routers.agent import router

        assert router is not None
        assert router.prefix == "/agent"

    def test_import_orchestrator_backend(self):
        """Test orchestrator backend imports"""
        from backend.orchestrator import (
            run_orchestrator_step,
            collect_orchestrator_stats,
            get_default_config,
            get_or_create_orchestrator
        )

        assert callable(run_orchestrator_step)
        assert callable(collect_orchestrator_stats)
        assert callable(get_default_config)
        assert callable(get_or_create_orchestrator)

    def test_import_database_module(self):
        """Test database module imports including sync support"""
        from src.api.database import (
            get_db,
            get_db_context,
            get_sync_db,
            async_session_factory,
            sync_session_factory
        )

        assert callable(get_db)
        assert callable(get_db_context)
        assert callable(get_sync_db)


class TestSecurity:
    """Test security functions"""

    @pytest.mark.asyncio
    async def test_get_optional_user_returns_none_without_token(self):
        """Test that get_optional_user returns None when no token provided"""
        from src.api.auth.security import get_optional_user

        # Mock dependencies
        mock_db = MagicMock()

        # Call with no token
        result = await get_optional_user(token=None, bearer=None, db=mock_db)

        assert result is None

    def test_security_service_hash_password(self):
        """Test password hashing"""
        from src.api.auth.security import SecurityService

        password = "TestPassword123!"
        hashed = SecurityService.hash_password(password)

        assert hashed != password
        assert SecurityService.verify_password(password, hashed)
        assert not SecurityService.verify_password("WrongPassword", hashed)

    def test_security_service_create_access_token(self):
        """Test access token creation"""
        from src.api.auth.security import SecurityService

        data = {"sub": "user123", "role": "user"}
        token = SecurityService.create_access_token(data)

        assert isinstance(token, str)
        assert len(token) > 0


class TestOrchestratorConfig:
    """Test orchestrator configuration"""

    def test_get_default_config(self):
        """Test orchestrator default config"""
        from backend.orchestrator import get_default_config

        config = get_default_config()

        assert isinstance(config, dict)
        assert "max_simultaneous_simulations" in config
        assert "max_simultaneous_experiments" in config
        assert "training_frequency_hours" in config
        assert config["max_simultaneous_simulations"] == 10
        assert config["max_simultaneous_experiments"] == 5


class TestDatabaseSessions:
    """Test database session management"""

    def test_sync_session_factory_exists(self):
        """Test that sync session factory is defined"""
        from src.api.database import sync_session_factory, sync_engine

        # These will be None until init_db() is called, but they should exist
        assert 'sync_session_factory' in dir()
        assert 'sync_engine' in dir()

    def test_create_sync_engine_function(self):
        """Test that sync engine creation function exists"""
        from src.api.database import create_sync_engine_with_pool

        assert callable(create_sync_engine_with_pool)


class TestCeleryTasks:
    """Test Celery task definitions"""

    def test_celery_tasks_import(self):
        """Test that Celery tasks can be imported"""
        from src.worker.tasks import (
            run_dummy_job,
            run_simulation_job,
            update_job_status,
            cancel_job,
            run_orchestrator_step_task
        )

        assert callable(run_dummy_job)
        assert callable(run_simulation_job)
        assert callable(update_job_status)
        assert callable(cancel_job)
        assert callable(run_orchestrator_step_task)

    def test_database_task_base_class(self):
        """Test DatabaseTask base class"""
        from src.worker.tasks import DatabaseTask

        assert DatabaseTask is not None
        assert hasattr(DatabaseTask, 'get_db_session')


class TestSchemas:
    """Test schema definitions"""

    def test_user_response_schema(self):
        """Test UserResponse schema exists"""
        from src.api.schemas.auth import UserResponse

        assert UserResponse is not None

    def test_model_registry_response_schema(self):
        """Test ModelRegistryResponse schema exists"""
        from src.api.schemas.ml import ModelRegistryResponse

        assert ModelRegistryResponse is not None


class TestRouterEndpoints:
    """Test that router endpoints are properly defined"""

    def test_orchestrator_endpoints(self):
        """Test orchestrator endpoints are defined"""
        from src.api.routers.orchestrator import router

        # Get all routes
        routes = [route.path for route in router.routes]

        assert "/state" in routes
        assert "/config" in routes
        assert "/run_once" in routes
        assert "/runs" in routes
        assert "/stats" in routes
        assert "/activate" in routes
        assert "/deactivate" in routes

    def test_agent_endpoints(self):
        """Test agent endpoints are defined"""
        from src.api.routers.agent import router

        # Get all routes
        routes = [route.path for route in router.routes]

        assert "/summary" in routes
        assert "/create_design_campaign" in routes
        assert "/advance_campaign" in routes
        assert "/request_simulations" in routes
        assert "/request_experiments" in routes
        assert "/commands" in routes


class TestAsyncAwaitPatterns:
    """Test async/await patterns are correct"""

    @pytest.mark.asyncio
    async def test_async_session_context_manager(self):
        """Test async session context manager structure"""
        from src.api.database import get_db_context

        # Just verify it's a context manager
        assert hasattr(get_db_context, '__aenter__')
        assert hasattr(get_db_context, '__aexit__')

    def test_sync_session_context_manager(self):
        """Test sync session context manager structure"""
        from src.api.database import get_sync_db

        # Just verify it's a context manager
        import inspect
        assert inspect.isgeneratorfunction(get_sync_db)


# Run smoke tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
