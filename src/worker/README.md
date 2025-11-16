# NANO-OS Celery Worker

Production-ready distributed task processing for NANO-OS simulations.

## Overview

The Celery worker handles asynchronous execution of computational chemistry and materials science simulations. It provides:

- **Distributed Processing**: Multiple workers can process jobs in parallel
- **Task Queues**: Priority-based job queuing with dedicated simulation queue
- **Progress Tracking**: Real-time progress updates stored in database
- **Error Handling**: Automatic retries with exponential backoff
- **Database Integration**: Full async database integration for job/result management
- **Monitoring**: Built-in monitoring with Flower and Prometheus metrics

## Architecture

```
┌─────────────────┐
│   FastAPI App   │
│   (Job Submit)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│  Redis Broker   │────▶│ Celery Worker│
│  (Task Queue)   │     │  (Process)   │
└─────────────────┘     └──────┬───────┘
         │                      │
         │                      ▼
         │              ┌──────────────┐
         │              │  Simulation  │
         │              │   Engines    │
         │              └──────┬───────┘
         │                      │
         ▼                      ▼
┌─────────────────┐     ┌──────────────┐
│   PostgreSQL    │◀────│   Results    │
│   (Jobs/Results)│     │              │
└─────────────────┘     └──────────────┘
```

## Components

### 1. `celery_app.py` - Celery Configuration

Main Celery application with:
- Redis broker and result backend
- Task routing and priority queues
- JSON serialization
- Comprehensive error handling
- Signal handlers for monitoring

### 2. `tasks.py` - Task Definitions

**Available Tasks:**

#### `run_dummy_job(job_id: str)`
Dummy simulation for testing Session 1 functionality.
- Updates job status to RUNNING
- Simulates work with progress updates
- Creates mock results
- Completes successfully

**Queue:** `simulations`
**Retry:** 3 attempts with 60s delay

#### `run_simulation_job(job_id: str)`
Main simulation task handler.
- Fetches job from database
- Validates parameters
- Executes simulation (mock or real engine)
- Stores results in database
- Updates job status throughout

**Queue:** `simulations`
**Retry:** 3 attempts with 60s delay
**Time Limit:** 1 hour (soft: 55min)

#### `update_job_status(job_id: str, status: str, **kwargs)`
Update job status in database.
- Utility task for status updates
- Updates progress, current_step, error messages
- Can be called from other services

**Queue:** `default`
**Retry:** 5 attempts with 10s delay

#### `cancel_job(job_id: str)`
Cancel a running simulation.
- Checks if job is cancellable
- Updates status to CANCELLED
- Future: Will terminate actual simulation process

**Queue:** `default`

### 3. `simulation_runner.py` - Simulation Engines

**Current Implementation:**

#### `MockSimulationEngine`
Mock engine for testing with realistic behavior:
- Input validation
- Multi-step execution
- Progress callbacks
- Random but realistic results
- Convergence simulation

#### `run_mock_simulation()`
Async function to run mock simulations.

**Future Implementations:**
- `VASPEngine` - DFT calculations with VASP
- `QuantumEspressoEngine` - DFT with Quantum Espresso
- `LAMMPSEngine` - Molecular dynamics with LAMMPS
- Additional engines as needed

## Installation

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Redis is running
docker run -d -p 6379:6379 redis:7-alpine

# Ensure PostgreSQL is running with asyncpg driver
# (See main API documentation)
```

### Install Celery Dependencies

Already included in `requirements.txt`:
```
celery>=5.3.0
kombu>=5.3.0
flower>=2.0.0
```

## Usage

### Starting the Worker

#### Basic Worker

```bash
# From project root
celery -A src.worker.celery_app worker --loglevel=info
```

#### Production Worker (Multiple Queues)

```bash
celery -A src.worker.celery_app worker \
    --loglevel=info \
    -Q simulations,default \
    -c 4 \
    --max-tasks-per-child=100
```

Options:
- `-Q`: Queue names to consume from
- `-c`: Concurrency (number of worker processes)
- `--max-tasks-per-child`: Restart worker after N tasks (prevents memory leaks)
- `--autoscale`: Dynamic scaling (e.g., `--autoscale=10,3` for 3-10 workers)

#### Worker Pool Types

```bash
# Prefork (default, good for CPU-bound)
celery -A src.worker.celery_app worker --pool=prefork -c 4

# Gevent (good for I/O-bound, many concurrent tasks)
celery -A src.worker.celery_app worker --pool=gevent -c 100

# Solo (single process, good for debugging)
celery -A src.worker.celery_app worker --pool=solo
```

### Monitoring

#### Flower Web UI

```bash
celery -A src.worker.celery_app flower --port=5555
```

Access at: `http://localhost:5555`

Features:
- Real-time task monitoring
- Worker statistics
- Task history
- Success/failure rates
- Broker monitoring

#### Command-line Monitoring

```bash
# List active tasks
celery -A src.worker.celery_app inspect active

# List registered tasks
celery -A src.worker.celery_app inspect registered

# Worker statistics
celery -A src.worker.celery_app inspect stats

# Reserved tasks
celery -A src.worker.celery_app inspect reserved

# Scheduled tasks (ETA)
celery -A src.worker.celery_app inspect scheduled
```

#### Controlling Workers

```bash
# Shutdown worker gracefully
celery -A src.worker.celery_app control shutdown

# Cancel task by ID
celery -A src.worker.celery_app control revoke <task_id>

# Enable/disable event monitoring
celery -A src.worker.celery_app control enable_events
celery -A src.worker.celery_app control disable_events

# Set worker concurrency
celery -A src.worker.celery_app control pool_grow 2  # Add 2 workers
celery -A src.worker.celery_app control pool_shrink 1  # Remove 1 worker
```

### Submitting Tasks

#### From Python Code

```python
from src.worker.tasks import run_simulation_job, run_dummy_job

# Submit dummy job (for testing)
task = run_dummy_job.delay("123e4567-e89b-12d3-a456-426614174000")
print(f"Task ID: {task.id}")

# Submit real simulation job
task = run_simulation_job.delay("223e4567-e89b-12d3-a456-426614174001")

# Check task status
result = task.get(timeout=10)  # Blocks until task completes
print(result)

# Or check without blocking
if task.ready():
    print(f"Result: {task.result}")
else:
    print(f"Task state: {task.state}")
```

#### From FastAPI Endpoint

```python
from fastapi import APIRouter, Depends
from src.worker.tasks import run_simulation_job

router = APIRouter()

@router.post("/jobs/{job_id}/submit")
async def submit_job(job_id: str):
    # Queue the job
    task = run_simulation_job.delay(job_id)

    # Store celery_task_id in database
    # (handled automatically in job creation endpoint)

    return {
        "job_id": job_id,
        "celery_task_id": task.id,
        "status": "queued"
    }
```

## Configuration

### Environment Variables

All configuration comes from `src.api.config.Settings`:

```bash
# Redis (broker and backend)
REDIS_URL=redis://:password@localhost:6379/0

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/orion_db

# Logging
LOG_LEVEL=INFO

# Worker settings (optional)
CELERYD_CONCURRENCY=4
CELERYD_MAX_TASKS_PER_CHILD=100
```

### Queue Configuration

Defined in `celery_app.py`:

- **default** (priority: 0-10): General tasks, status updates
- **simulations** (priority: 0-20): Simulation jobs
- **high_priority** (priority: 0-20): Urgent tasks

Priority levels:
- 0 = Lowest
- 5 = Normal (default)
- 10 = High
- 20 = Urgent

### Task Time Limits

Configured in `celery_app.py`:

- **Hard limit**: 1 hour (task killed)
- **Soft limit**: 55 minutes (exception raised, can be caught)

Override per task:
```python
@celery_app.task(time_limit=7200, soft_time_limit=7000)
def long_running_task():
    pass
```

## Development

### Running Tests

```python
# Test dummy job
from src.worker.tasks import run_dummy_job

task = run_dummy_job.delay("test-job-id")
result = task.get(timeout=30)
print(result)
```

### Adding New Tasks

1. Define task in `tasks.py`:

```python
@celery_app.task(
    name="my_custom_task",
    bind=True,
    max_retries=3,
)
def my_custom_task(self, arg1, arg2):
    try:
        # Task logic
        result = do_something(arg1, arg2)
        return result
    except Exception as e:
        # Retry on failure
        raise self.retry(exc=e, countdown=60)
```

2. Add route in `celery_app.py` (if needed):

```python
task_routes={
    "src.worker.tasks.my_custom_task": {
        "queue": "custom_queue",
        "routing_key": "custom.task",
    },
}
```

3. Export in `__init__.py`:

```python
from .tasks import my_custom_task

__all__ = [..., "my_custom_task"]
```

### Adding New Simulation Engines

1. Create engine class in `simulation_runner.py`:

```python
class MyEngine:
    def __init__(self, structure, parameters):
        self.structure = structure
        self.parameters = parameters

    async def run(self, progress_callback=None):
        # Implementation
        pass
```

2. Register in `ENGINE_REGISTRY`:

```python
ENGINE_REGISTRY = {
    "MY_ENGINE": MyEngine,
    # ...
}
```

3. Update `run_simulation_job` to use registry:

```python
# In tasks.py
engine_class = get_engine(job.engine)
engine = engine_class(structure_data, parameters)
result = await engine.run(progress_callback=...)
```

## Troubleshooting

### Worker Won't Start

**Problem:** ImportError or module not found

**Solution:**
```bash
# Ensure you're in project root
cd /home/user/O.R.I.O.N-LLM-Research-Platform

# Install dependencies
pip install -r requirements.txt

# Set PYTHONPATH if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Tasks Not Being Consumed

**Problem:** Tasks stuck in queue

**Solution:**
```bash
# Check worker is running
celery -A src.worker.celery_app inspect active

# Check worker is listening to correct queue
celery -A src.worker.celery_app inspect active_queues

# Purge queue if needed (CAUTION: deletes all tasks)
celery -A src.worker.celery_app purge
```

### Database Connection Issues

**Problem:** "Database not initialized" error

**Solution:**
- Ensure database is initialized in worker startup
- Check `DATABASE_URL` environment variable
- Verify PostgreSQL is running and accessible

### Memory Leaks

**Problem:** Worker memory grows over time

**Solution:**
```bash
# Use max-tasks-per-child to restart workers periodically
celery -A src.worker.celery_app worker --max-tasks-per-child=100

# Monitor memory usage
celery -A src.worker.celery_app inspect stats
```

## Production Deployment

### Systemd Service

Create `/etc/systemd/system/celery-worker.service`:

```ini
[Unit]
Description=Celery Worker for NANO-OS
After=network.target redis.target postgresql.target

[Service]
Type=forking
User=celery
Group=celery
WorkingDirectory=/opt/orion
Environment="PYTHONPATH=/opt/orion"
ExecStart=/opt/orion/venv/bin/celery -A src.worker.celery_app worker \
    --loglevel=info \
    -Q simulations,default \
    -c 4 \
    --pidfile=/var/run/celery/%n.pid \
    --logfile=/var/log/celery/%n.log \
    --max-tasks-per-child=100

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable celery-worker
sudo systemctl start celery-worker
sudo systemctl status celery-worker
```

### Docker Deployment

See `docker-compose.yml` in project root for containerized deployment.

### Monitoring in Production

1. **Flower**: Web-based monitoring
2. **Prometheus**: Metrics collection
3. **Grafana**: Dashboard visualization
4. **Sentry**: Error tracking
5. **CloudWatch/Stackdriver**: Cloud-native monitoring

## Performance Tuning

### Concurrency

```bash
# CPU-bound tasks (simulations)
celery -A src.worker.celery_app worker -c <num_cpu_cores>

# I/O-bound tasks (database updates)
celery -A src.worker.celery_app worker --pool=gevent -c 100
```

### Prefetch Multiplier

```python
# In celery_app.py
worker_prefetch_multiplier=1  # For long-running tasks
worker_prefetch_multiplier=4  # For short tasks (default)
```

### Task Routing

Route different task types to different workers:

```bash
# Worker 1: High-priority simulations
celery -A src.worker.celery_app worker -Q high_priority -c 2

# Worker 2: Normal simulations
celery -A src.worker.celery_app worker -Q simulations -c 4

# Worker 3: Background tasks
celery -A src.worker.celery_app worker -Q default -c 2
```

## Future Enhancements

- [ ] Real simulation engine integrations (VASP, QE, LAMMPS)
- [ ] Task cancellation with process termination
- [ ] Job dependencies (run job B after job A)
- [ ] Automatic resource allocation
- [ ] Result caching and deduplication
- [ ] Distributed file storage integration (MinIO/S3)
- [ ] Advanced scheduling (time-based, resource-based)
- [ ] Multi-tenant isolation
- [ ] GPU job support
- [ ] Workflow orchestration (Airflow integration)

## References

- [Celery Documentation](https://docs.celeryq.dev/)
- [Flower Documentation](https://flower.readthedocs.io/)
- [Redis Documentation](https://redis.io/documentation)
- [SQLAlchemy Async](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
