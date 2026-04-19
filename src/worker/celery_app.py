"""
Celery application configuration (Phase 2 / Session 2.1).

Queue layout, per roadmap:

- ``default``  — catch-all, short-running work (orchestrator ticks, IO)
- ``io``       — bulk-import chunks, artifact uploads, notifications
- ``dft``      — long-running DFT runs; 1 concurrency/worker recommended
- ``md``       — molecular dynamics (LAMMPS); CPU or GPU
- ``ml``       — ML training / inference; GPU when available

Routing is by task name: tasks registered under ``orion.dft.*`` go to
the ``dft`` queue, ``orion.md.*`` → ``md``, ``orion.ml.*`` → ``ml``,
``orion.io.*`` → ``io``. The legacy ``src.worker.tasks.run_*`` names
keep working via explicit entries in ``task_routes``.
"""

import logging

from celery import Celery
from celery.signals import task_failure, task_postrun, task_prerun
from kombu import Exchange, Queue

from src.api.config import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "orion_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
)


def _build_beat_schedule() -> dict:
    """Assemble the Celery beat schedule based on feature flags."""
    import os

    schedule: dict = {
        # Reaper: find jobs stuck in RUNNING past their stall timeout and
        # transition them to FAILED(reason="worker_lost"). Every minute.
        "orion.reaper.stalled-jobs": {
            "task": "orion.io.reap_stalled_jobs",
            "schedule": 60.0,
        },
        # Workflow DAG tick: advance every non-terminal WorkflowRun.
        "orion.workflows.tick": {
            "task": "orion.workflows.tick",
            "schedule": 5.0,
        },
    }
    if os.getenv("ORION_ENABLE_ORCHESTRATOR_BEAT", "").lower() == "true":
        schedule["orion.orchestrator.tick"] = {
            "task": "run_orchestrator_step_task",
            "schedule": 3600.0,
            "args": ("default", "scheduler"),
        }
    return schedule


celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Results
    result_expires=3600,
    result_backend_transport_options={
        "master_name": "mymaster",
        "retry_on_timeout": True,
    },

    # Queue layout (roadmap 2.1).
    task_default_queue="default",
    task_default_exchange="tasks",
    task_default_routing_key="task.default",
    task_queues=(
        Queue("default", Exchange("tasks"), routing_key="task.default",
              queue_arguments={"x-max-priority": 10}),
        Queue("io", Exchange("tasks"), routing_key="task.io",
              queue_arguments={"x-max-priority": 10}),
        Queue("dft", Exchange("tasks"), routing_key="task.dft",
              queue_arguments={"x-max-priority": 20}),
        Queue("md", Exchange("tasks"), routing_key="task.md",
              queue_arguments={"x-max-priority": 20}),
        Queue("ml", Exchange("tasks"), routing_key="task.ml",
              queue_arguments={"x-max-priority": 20}),
        # Back-compat — retained until Session 2.2 migrates callers.
        Queue("simulations", Exchange("simulations"), routing_key="simulation.#",
              queue_arguments={"x-max-priority": 20}),
    ),

    # Name-prefix routing. ``orion.dft.*`` → ``dft`` queue, etc.
    task_routes={
        "orion.dft.*": {"queue": "dft", "routing_key": "task.dft"},
        "orion.md.*": {"queue": "md", "routing_key": "task.md"},
        "orion.ml.*": {"queue": "ml", "routing_key": "task.ml"},
        "orion.io.*": {"queue": "io", "routing_key": "task.io"},
        # Mock tasks go to `default` — they're cheap and don't belong on
        # the DFT/MD queues where slots are expensive.
        "orion.mock.*": {"queue": "default", "routing_key": "task.default"},
        # Workflow ticks are DB reads + Celery send_task — always default.
        "orion.workflows.*": {"queue": "default", "routing_key": "task.default"},
        # Back-compat explicit routes.
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

    # Beat schedule (periodic tasks). Off by default — Session 7 (agent
    # loop) opts in via ORION_ENABLE_ORCHESTRATOR_BEAT=true. The reaper
    # for stalled jobs is on; it's pure-API work and doesn't need engines.
    beat_schedule=_build_beat_schedule(),
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
