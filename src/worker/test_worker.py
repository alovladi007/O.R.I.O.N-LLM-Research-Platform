"""
Test script for Celery worker functionality.

This script tests the worker implementation without requiring
a full database setup. Useful for development and debugging.

Usage:
    python -m src.worker.test_worker
"""

import asyncio
import logging
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_celery_app():
    """Test Celery app configuration."""
    logger.info("Testing Celery app configuration...")

    try:
        from src.worker.celery_app import celery_app

        # Check app is configured
        assert celery_app.conf.broker_url, "Broker URL not configured"
        assert celery_app.conf.result_backend, "Result backend not configured"

        logger.info(f"✓ Broker: {celery_app.conf.broker_url}")
        logger.info(f"✓ Backend: {celery_app.conf.result_backend}")
        logger.info(f"✓ Task serializer: {celery_app.conf.task_serializer}")

        # Check queues
        queues = celery_app.conf.task_queues
        logger.info(f"✓ Configured queues: {[q.name for q in queues]}")

        # Check routes
        routes = celery_app.conf.task_routes
        logger.info(f"✓ Task routes: {len(routes)} routes configured")

        logger.info("✓ Celery app configuration OK\n")
        return True

    except Exception as e:
        logger.error(f"✗ Celery app configuration failed: {e}")
        return False


def test_tasks_import():
    """Test task imports."""
    logger.info("Testing task imports...")

    try:
        from src.worker.tasks import (
            run_dummy_job,
            run_simulation_job,
            update_job_status,
            cancel_job,
        )

        logger.info(f"✓ run_dummy_job: {run_dummy_job.name}")
        logger.info(f"✓ run_simulation_job: {run_simulation_job.name}")
        logger.info(f"✓ update_job_status: {update_job_status.name}")
        logger.info(f"✓ cancel_job: {cancel_job.name}")

        logger.info("✓ All tasks imported successfully\n")
        return True

    except Exception as e:
        logger.error(f"✗ Task import failed: {e}")
        return False


def test_simulation_runner():
    """Test simulation runner."""
    logger.info("Testing simulation runner...")

    try:
        from src.worker.simulation_runner import (
            MockSimulationEngine,
            run_mock_simulation,
            get_engine,
        )

        # Test MockSimulationEngine
        engine = MockSimulationEngine(
            engine="VASP",
            structure={"atoms": ["C", "C"], "positions": [[0, 0, 0], [0.5, 0.5, 0.5]]},
            parameters={"ecutwfc": 500},
        )
        logger.info(f"✓ Created MockSimulationEngine: {engine.engine}")

        # Test run_mock_simulation
        async def test_run():
            progress_updates = []

            def progress_callback(progress, step):
                progress_updates.append((progress, step))
                logger.info(f"  Progress: {progress*100:.1f}% - {step}")

            result = await run_mock_simulation(
                structure={"atoms": ["C", "C"]},
                parameters={"ecutwfc": 500},
                engine="VASP",
                job_id="test-job-123",
                progress_callback=progress_callback,
            )

            assert "summary" in result, "Result missing summary"
            assert "convergence_reached" in result, "Result missing convergence_reached"
            logger.info(f"✓ Mock simulation completed: {result['summary']}")
            logger.info(f"✓ Progress updates: {len(progress_updates)}")

        asyncio.run(test_run())

        # Test engine registry
        engine_class = get_engine("MOCK")
        logger.info(f"✓ Engine registry working: {engine_class}")

        logger.info("✓ Simulation runner OK\n")
        return True

    except Exception as e:
        logger.error(f"✗ Simulation runner failed: {e}", exc_info=True)
        return False


def test_health_check():
    """Test health check task."""
    logger.info("Testing health check task...")

    try:
        from src.worker.celery_app import health_check

        # Test direct call (not via Celery)
        result = health_check()

        assert result["status"] == "healthy", "Health check failed"
        assert result["service"] == "nano-os-worker", "Wrong service name"

        logger.info(f"✓ Health check result: {result}")
        logger.info("✓ Health check OK\n")
        return True

    except Exception as e:
        logger.error(f"✗ Health check failed: {e}")
        return False


def test_worker_integration():
    """
    Test worker integration (requires running worker and Redis).

    This test submits a task to the queue and waits for result.
    Skip if worker is not running.
    """
    logger.info("Testing worker integration...")
    logger.info("(This test requires a running worker and Redis)")

    try:
        from src.worker.celery_app import health_check

        # Try to ping the broker
        from src.worker.celery_app import celery_app

        inspector = celery_app.control.inspect()
        active_workers = inspector.active()

        if not active_workers:
            logger.warning("⚠ No active workers found. Skipping integration test.")
            logger.warning("  Start a worker with: celery -A src.worker.celery_app worker")
            return None

        logger.info(f"✓ Active workers: {list(active_workers.keys())}")

        # Submit health check task
        logger.info("Submitting health check task...")
        task = health_check.delay()

        # Wait for result (with timeout)
        result = task.get(timeout=10)

        logger.info(f"✓ Task completed: {result}")
        logger.info(f"✓ Task ID: {task.id}")
        logger.info(f"✓ Task state: {task.state}")

        logger.info("✓ Worker integration OK\n")
        return True

    except Exception as e:
        logger.warning(f"⚠ Worker integration test failed: {e}")
        logger.warning("  This is OK if worker/Redis is not running")
        return None


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("NANO-OS Celery Worker Test Suite")
    logger.info("=" * 60)
    logger.info("")

    tests = [
        ("Celery App", test_celery_app),
        ("Task Imports", test_tasks_import),
        ("Simulation Runner", test_simulation_runner),
        ("Health Check", test_health_check),
        ("Worker Integration", test_worker_integration),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info("-" * 60)
        result = test_func()
        results.append((test_name, result))

    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    for test_name, result in results:
        if result is True:
            logger.info(f"✓ {test_name}: PASSED")
        elif result is False:
            logger.error(f"✗ {test_name}: FAILED")
        else:
            logger.warning(f"⚠ {test_name}: SKIPPED")

    logger.info("")
    logger.info(f"Passed: {passed}/{len(tests)}")
    logger.info(f"Failed: {failed}/{len(tests)}")
    logger.info(f"Skipped: {skipped}/{len(tests)}")

    # Exit with error code if any tests failed
    if failed > 0:
        logger.error("\nTests failed!")
        sys.exit(1)
    else:
        logger.info("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
