"""
Celery Application Configuration
=================================

Production-ready Celery setup with:
- Redis as broker and result backend
- Task routing and queues
- JSON serialization
- Comprehensive logging
- Error handling and retries
"""

import logging
from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure
from kombu import Queue, Exchange

from src.api.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Create Celery application
celery_app = Celery(
    "nano_os_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

# Celery configuration
celery_app.conf.update(
    # Task execution
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task results
    result_expires=3600,  # Results expire after 1 hour
    result_backend_transport_options={
        "master_name": "mymaster",
        "retry_on_timeout": True,
    },

    # Task routing
    task_default_queue="default",
    task_default_exchange="tasks",
    task_default_routing_key="task.default",

    # Task queues
    task_queues=(
        Queue(
            "default",
            Exchange("tasks"),
            routing_key="task.default",
            queue_arguments={"x-max-priority": 10},
        ),
        Queue(
            "simulations",
            Exchange("simulations"),
            routing_key="simulation.#",
            queue_arguments={"x-max-priority": 20},
        ),
        Queue(
            "high_priority",
            Exchange("tasks"),
            routing_key="task.high",
            queue_arguments={"x-max-priority": 20},
        ),
    ),

    # Task routes
    task_routes={
        "src.worker.tasks.run_simulation_job": {
            "queue": "simulations",
            "routing_key": "simulation.run",
        },
        "src.worker.tasks.run_dummy_job": {
            "queue": "simulations",
            "routing_key": "simulation.dummy",
        },
        "src.worker.tasks.update_job_status": {
            "queue": "default",
            "routing_key": "task.update",
        },
        "run_orchestrator_step_task": {
            "queue": "default",
            "routing_key": "task.orchestrator",
        },
    },

    # Worker configuration
    worker_prefetch_multiplier=1,  # Fetch one task at a time for long-running tasks
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks (prevent memory leaks)
    worker_disable_rate_limits=False,

    # Task execution limits
    task_time_limit=3600,  # Hard limit: 1 hour
    task_soft_time_limit=3300,  # Soft limit: 55 minutes (raise exception)
    task_acks_late=True,  # Acknowledge task after execution (not before)
    task_reject_on_worker_lost=True,  # Re-queue task if worker dies

    # Retry configuration
    task_default_retry_delay=60,  # Wait 60s before retry
    task_max_retries=3,  # Maximum 3 retries

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,

    # Broker settings
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,

    # Result backend settings
    result_backend_always_retry=True,
    result_backend_max_retries=10,

    # Beat schedule (for periodic tasks)
    beat_schedule={
        # Run orchestrator every hour
        "run-orchestrator-hourly": {
            "task": "run_orchestrator_step_task",
            "schedule": 3600.0,  # Every hour (in seconds)
            "args": ("default", "scheduler"),
        },
        # Example: Clean up old results every day
        # "cleanup-old-results": {
        #     "task": "src.worker.tasks.cleanup_old_results",
        #     "schedule": crontab(hour=3, minute=0),
        # },
    },
)

# Auto-discover tasks from tasks.py
celery_app.autodiscover_tasks(["src.worker"])


# Signal handlers for monitoring and logging
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra_kwargs):
    """Log when a task starts."""
    logger.info(
        f"Task started: {task.name} [ID: {task_id}]",
        extra={
            "task_id": task_id,
            "task_name": task.name,
            "args": args,
            "kwargs": kwargs,
        },
    )


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, retval=None, state=None, **extra_kwargs):
    """Log when a task completes."""
    logger.info(
        f"Task completed: {task.name} [ID: {task_id}] - State: {state}",
        extra={
            "task_id": task_id,
            "task_name": task.name,
            "state": state,
        },
    )


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, **extra_kwargs):
    """Log when a task fails."""
    logger.error(
        f"Task failed: {sender.name} [ID: {task_id}] - {exception}",
        extra={
            "task_id": task_id,
            "task_name": sender.name,
            "exception": str(exception),
            "traceback": traceback,
        },
        exc_info=True,
    )


# Health check task
@celery_app.task(name="health_check")
def health_check():
    """Simple health check task for monitoring."""
    return {"status": "healthy", "service": "nano-os-worker"}


if __name__ == "__main__":
    celery_app.start()
