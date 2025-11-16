"""
NANO-OS Celery Worker Module
============================

This module provides distributed task processing for NANO-OS simulations.

Components:
- celery_app: Main Celery application with Redis broker
- tasks: Task definitions for simulation jobs
- simulation_runner: Simulation execution logic

Usage:
    # Start worker
    celery -A src.worker.celery_app worker --loglevel=info -Q simulations,default

    # Start worker with concurrency
    celery -A src.worker.celery_app worker --loglevel=info -c 4

    # Monitor tasks
    celery -A src.worker.celery_app flower
"""

from .celery_app import celery_app
from .tasks import (
    run_dummy_job,
    run_simulation_job,
    update_job_status,
)

__all__ = [
    "celery_app",
    "run_dummy_job",
    "run_simulation_job",
    "update_job_status",
]

__version__ = "1.0.0"
