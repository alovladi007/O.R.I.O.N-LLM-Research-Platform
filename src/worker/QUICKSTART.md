# NANO-OS Worker Quick Start

Get up and running with the Celery worker in 5 minutes.

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New dependencies added:
- `celery>=5.3.0` - Task queue
- `kombu>=5.3.0` - Messaging library
- `flower>=2.0.0` - Monitoring UI

## 2. Start Redis (Broker)

```bash
# Using Docker
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Or using local Redis
redis-server
```

## 3. Start the Worker

```bash
# From project root
celery -A src.worker.celery_app worker --loglevel=info
```

## 4. Test the Worker

```python
# In Python shell or script
from src.worker.tasks import run_dummy_job

# Submit a test job
task = run_dummy_job.delay("test-job-123")

# Check status
print(f"Task ID: {task.id}")
print(f"State: {task.state}")

# Wait for result (blocks)
result = task.get(timeout=30)
print(f"Result: {result}")
```

## 5. Monitor with Flower (Optional)

```bash
# Start Flower web UI
celery -A src.worker.celery_app flower

# Open browser to http://localhost:5555
```

## Common Commands

```bash
# Start worker with specific queue
celery -A src.worker.celery_app worker -Q simulations

# Start with multiple workers
celery -A src.worker.celery_app worker -c 4

# List active tasks
celery -A src.worker.celery_app inspect active

# Purge all tasks (CAUTION!)
celery -A src.worker.celery_app purge
```

## Integration Example

```python
# In your FastAPI router
from fastapi import APIRouter
from src.worker.tasks import run_simulation_job

router = APIRouter()

@router.post("/simulations/{job_id}/run")
async def run_simulation(job_id: str):
    # Submit to Celery
    task = run_simulation_job.delay(job_id)

    return {
        "job_id": job_id,
        "celery_task_id": task.id,
        "status": "queued"
    }
```

## Troubleshooting

**Worker won't start?**
```bash
# Check imports
python -c "from src.worker.celery_app import celery_app; print('OK')"

# Check Redis connection
redis-cli ping
```

**Tasks not running?**
```bash
# Check worker is listening to correct queue
celery -A src.worker.celery_app inspect active_queues
```

**Database errors?**
- Ensure `DATABASE_URL` is set in environment or `.env`
- Check PostgreSQL is running
- Verify database is initialized

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Run the test suite: `python -m src.worker.test_worker`
3. Explore [Celery documentation](https://docs.celeryq.dev/)
4. Set up production deployment with systemd/Docker
