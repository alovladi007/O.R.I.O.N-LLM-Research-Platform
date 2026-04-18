# Orchestrator Deployment Guide

**Production Deployment Guide for NANO-OS Orchestrator AGI Control Plane**

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Database Setup](#database-setup)
4. [Celery Beat Configuration](#celery-beat-configuration)
5. [Environment Variables](#environment-variables)
6. [Initialization](#initialization)
7. [Monitoring](#monitoring)
8. [Scaling](#scaling)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The orchestrator is the central control plane for NANO-OS that enables:
- Autonomous campaign management
- Intelligent model retraining
- Simulation and experiment orchestration
- External LLM/agent control

---

## Prerequisites

**System Requirements**:
- PostgreSQL 14+ (with JSONB support)
- Redis 6+ (for Celery broker/backend)
- Python 3.10+
- Celery 5+

**Services Required**:
- FastAPI application server
- Celery worker
- Celery beat scheduler

---

## Database Setup

### 1. Run Database Migration

```bash
# Ensure you're in the project root
cd /path/to/O.R.I.O.N-LLM-Research-Platform

# Activate virtual environment
source venv/bin/activate

# Run Alembic migration
alembic upgrade head
```

This will create the following tables:
- `orchestrator_state` - Orchestrator configuration and state
- `orchestrator_runs` - Audit trail of orchestrator executions
- `agent_commands` - External agent command history

### 2. Initialize Default Orchestrator

```python
from sqlalchemy.orm import Session
from backend.orchestrator import get_or_create_orchestrator, get_default_config

# In your application startup or admin script
orchestrator = get_or_create_orchestrator(
    db=db,
    name="default",
    config=get_default_config()
)
```

### 3. Customize Configuration

Edit the orchestrator configuration via API or directly in database:

```python
config = {
    "max_simultaneous_simulations": 20,      # Production capacity
    "max_simultaneous_experiments": 10,      # Lab capacity
    "training_frequency_hours": 12,          # Train every 12 hours
    "min_new_samples_for_retrain": 200,      # Threshold for retraining
    "experiment_budget_per_campaign": 100,   # Experiments per campaign
    "simulation_budget_per_campaign": 5000,  # Simulations per campaign
    "active_learning_threshold": 0.85,       # Uncertainty threshold
    "bo_acquisition_function": "ei",         # Expected Improvement
    "max_iterations_per_run": 5,             # Campaigns per orchestrator run
    "experiment_trigger_score_threshold": 0.92  # Quality threshold for experiments
}

# Update via API
import requests
requests.post(
    "https://api.nano-os.ai/orchestrator/config?name=default",
    json=config,
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
```

---

## Celery Beat Configuration

### 1. Verify Beat Schedule

The beat schedule is configured in `src/worker/celery_app.py`:

```python
beat_schedule={
    "run-orchestrator-hourly": {
        "task": "run_orchestrator_step_task",
        "schedule": 3600.0,  # Every hour
        "args": ("default", "scheduler"),
    },
}
```

### 2. Start Celery Beat

```bash
# Start Celery beat scheduler
celery -A src.worker.celery_app beat \
    --loglevel=info \
    --logfile=logs/celery-beat.log \
    --pidfile=var/run/celery-beat.pid
```

**Production (systemd service)**:

Create `/etc/systemd/system/nano-os-beat.service`:

```ini
[Unit]
Description=NANO-OS Celery Beat Scheduler
After=network.target redis.service postgresql.service

[Service]
Type=simple
User=nano-os
Group=nano-os
WorkingDirectory=/opt/nano-os
Environment="PATH=/opt/nano-os/venv/bin"
ExecStart=/opt/nano-os/venv/bin/celery -A src.worker.celery_app beat \
    --loglevel=info \
    --logfile=/var/log/nano-os/celery-beat.log \
    --pidfile=/var/run/nano-os/celery-beat.pid

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable nano-os-beat
sudo systemctl start nano-os-beat
sudo systemctl status nano-os-beat
```

### 3. Adjust Schedule

To change the orchestrator run frequency:

```python
# Edit src/worker/celery_app.py
beat_schedule={
    "run-orchestrator-hourly": {
        "task": "run_orchestrator_step_task",
        "schedule": 1800.0,  # Every 30 minutes
        "args": ("default", "scheduler"),
    },
}
```

Or use crontab for specific times:

```python
from celery.schedules import crontab

beat_schedule={
    "run-orchestrator-hourly": {
        "task": "run_orchestrator_step_task",
        "schedule": crontab(minute=0, hour='*/2'),  # Every 2 hours
        "args": ("default", "scheduler"),
    },
}
```

---

## Environment Variables

### Required Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/nano_os

# Redis (for Celery)
REDIS_URL=redis://localhost:6379/0

# API Settings
API_PREFIX=/api
SECRET_KEY=your-secret-key-here

# Orchestrator Settings (optional)
ORCHESTRATOR_MODE=SCHEDULED  # MANUAL, SCHEDULED, CONTINUOUS, PAUSED
ORCHESTRATOR_DEFAULT_NAME=default
```

### Optional Variables

```bash
# Monitoring
ENABLE_METRICS=true
ENABLE_TRACING=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Rate Limiting
RATE_LIMIT_ENABLED=true
```

---

## Initialization

### 1. First-Time Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run database migrations
alembic upgrade head

# 3. Initialize orchestrator
python scripts/init_orchestrator.py

# 4. Start services
systemctl start nano-os-api
systemctl start nano-os-worker
systemctl start nano-os-beat
```

### 2. Create Initialization Script

Create `scripts/init_orchestrator.py`:

```python
#!/usr/bin/env python3
"""
Initialize orchestrator for production deployment.
"""

import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.api.config import settings
from backend.orchestrator import get_or_create_orchestrator

async def main():
    # Create async engine
    engine = create_async_engine(settings.database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Create default orchestrator
        orchestrator = get_or_create_orchestrator(
            db=session,
            name="default",
            config={
                "max_simultaneous_simulations": 20,
                "max_simultaneous_experiments": 10,
                "training_frequency_hours": 12,
                "min_new_samples_for_retrain": 200,
                "experiment_budget_per_campaign": 100,
                "simulation_budget_per_campaign": 5000,
                "active_learning_threshold": 0.85,
                "bo_acquisition_function": "ei",
                "max_iterations_per_run": 5,
                "experiment_trigger_score_threshold": 0.92
            }
        )

        print(f"✓ Orchestrator '{orchestrator.name}' initialized")
        print(f"  ID: {orchestrator.id}")
        print(f"  Mode: {orchestrator.mode}")
        print(f"  Active: {orchestrator.is_active}")

if __name__ == "__main__":
    asyncio.run(main())
```

Make executable:

```bash
chmod +x scripts/init_orchestrator.py
```

---

## Monitoring

### 1. Orchestrator Dashboard

Access the web dashboard:
```
https://your-domain.com/orchestrator
```

Features:
- Real-time status monitoring
- Manual orchestrator triggering
- Configuration management
- Run history

### 2. API Health Check

```bash
# Check orchestrator status
curl https://api.nano-os.ai/orchestrator/state?name=default

# Check recent runs
curl https://api.nano-os.ai/orchestrator/runs?limit=10

# Check system stats
curl https://api.nano-os.ai/orchestrator/stats
```

### 3. Celery Monitoring (Flower)

```bash
# Install Flower
pip install flower

# Start Flower dashboard
celery -A src.worker.celery_app flower \
    --port=5555 \
    --broker=redis://localhost:6379/0
```

Access at: `http://localhost:5555`

### 4. Prometheus Metrics

If metrics are enabled, orchestrator metrics are available at:
```
https://api.nano-os.ai/metrics
```

Key metrics:
- `orchestrator_runs_total` - Total orchestrator runs
- `orchestrator_campaigns_advanced_total` - Total campaigns advanced
- `orchestrator_simulations_launched_total` - Total simulations launched
- `orchestrator_run_duration_seconds` - Run duration histogram

### 5. Logging

**Orchestrator Logs**:
```bash
# Application logs
tail -f /var/log/nano-os/api.log | grep orchestrator

# Celery beat logs
tail -f /var/log/nano-os/celery-beat.log

# Worker logs
tail -f /var/log/nano-os/celery-worker.log
```

**Log Aggregation**:

Use LogStash, Fluentd, or similar:

```yaml
# filebeat.yml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/nano-os/*.log
    fields:
      service: nano-os
      component: orchestrator
```

---

## Scaling

### 1. Horizontal Scaling

**Multiple Workers**:

```bash
# Start multiple workers
celery -A src.worker.celery_app worker \
    --concurrency=10 \
    --loglevel=info \
    --hostname=worker1@%h

celery -A src.worker.celery_app worker \
    --concurrency=10 \
    --loglevel=info \
    --hostname=worker2@%h
```

**Only One Beat Instance**:
- Run Celery beat on only ONE server to avoid duplicate scheduling
- Use leader election (etcd, Consul) for HA beat

### 2. Resource Tuning

**High Throughput**:
```python
config = {
    "max_simultaneous_simulations": 100,
    "max_simultaneous_experiments": 50,
    "max_iterations_per_run": 10,
}
```

**Conservative (Limited Resources)**:
```python
config = {
    "max_simultaneous_simulations": 5,
    "max_simultaneous_experiments": 2,
    "max_iterations_per_run": 1,
}
```

### 3. Database Connection Pooling

```python
# In settings.py
SQLALCHEMY_POOL_SIZE = 20
SQLALCHEMY_MAX_OVERFLOW = 40
SQLALCHEMY_POOL_TIMEOUT = 30
```

---

## Troubleshooting

### Issue: Orchestrator Not Running

**Check orchestrator status**:
```bash
curl https://api.nano-os.ai/orchestrator/state?name=default
```

**Verify is_active = true**:
```bash
# Activate via API
curl -X POST https://api.nano-os.ai/orchestrator/activate?name=default
```

**Check Celery beat**:
```bash
# Verify beat is running
ps aux | grep celery.*beat

# Check beat logs
tail -f /var/log/nano-os/celery-beat.log
```

### Issue: Orchestrator Run Failures

**Check recent runs**:
```bash
curl https://api.nano-os.ai/orchestrator/runs?limit=10 | jq '.[] | {started_at, success, error_message}'
```

**Common causes**:
1. Database connection issues
2. Insufficient permissions
3. Missing campaigns or models
4. Resource exhaustion

**Check orchestrator error_message**:
```bash
curl https://api.nano-os.ai/orchestrator/state | jq '.error_message'
```

### Issue: Too Many Simulations Launched

**Check budget configuration**:
```bash
curl https://api.nano-os.ai/orchestrator/state | jq '.config.simulation_budget_per_campaign'
```

**Reduce limits**:
```bash
curl -X POST https://api.nano-os.ai/orchestrator/config \
    -H "Content-Type: application/json" \
    -d '{"max_simultaneous_simulations": 5, "simulation_budget_per_campaign": 100}'
```

### Issue: Orchestrator Not Triggering Experiments

**Check threshold**:
```bash
curl https://api.nano-os.ai/orchestrator/state | jq '.config.experiment_trigger_score_threshold'
```

**Lower threshold**:
```bash
curl -X POST https://api.nano-os.ai/orchestrator/config \
    -H "Content-Type: application/json" \
    -d '{"experiment_trigger_score_threshold": 0.8}'
```

### Issue: Beat Schedule Not Working

**Verify schedule**:
```python
# In Python REPL
from src.worker.celery_app import celery_app
print(celery_app.conf.beat_schedule)
```

**Restart beat**:
```bash
systemctl restart nano-os-beat
```

**Check beat database** (if using DB backend):
```bash
# Celery beat stores schedule in database
# Check celerybeat-schedule file or DB
```

---

## Production Checklist

- [ ] Database migrations applied
- [ ] Default orchestrator created
- [ ] Configuration customized for environment
- [ ] Celery beat running (only one instance!)
- [ ] Celery workers running (multiple OK)
- [ ] Orchestrator activated (`is_active=true`)
- [ ] Monitoring configured (Prometheus, logs)
- [ ] Dashboard accessible
- [ ] Agent API tested
- [ ] Backup strategy in place
- [ ] Alerts configured for failures

---

## Quick Reference

### Start Services

```bash
# API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Celery worker
celery -A src.worker.celery_app worker --loglevel=info

# Celery beat
celery -A src.worker.celery_app beat --loglevel=info
```

### Trigger Orchestrator Manually

```bash
# Via API
curl -X POST https://api.nano-os.ai/orchestrator/run_once

# Via Python
from src.worker.tasks import run_orchestrator_step_task
run_orchestrator_step_task.delay("default", "manual")
```

### Update Configuration

```bash
curl -X POST https://api.nano-os.ai/orchestrator/config \
    -H "Content-Type: application/json" \
    -d '{
        "max_simultaneous_simulations": 20,
        "training_frequency_hours": 6
    }'
```

### Pause/Resume

```bash
# Pause
curl -X POST https://api.nano-os.ai/orchestrator/deactivate

# Resume
curl -X POST https://api.nano-os.ai/orchestrator/activate
```

---

## Contact & Support

For issues or questions:
- GitHub Issues: https://github.com/alovladi007/O.R.I.O.N-LLM-Research-Platform/issues
- Documentation: `SESSION_30_IMPLEMENTATION.md`

---

**Last Updated**: 2025-01-17
