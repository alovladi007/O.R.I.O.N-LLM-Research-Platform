"""Phase 9 / Session 9.3 — workflow-run SSE event-stream tests.

Pin the route registration and the ``snapshot`` event format so the
frontend's ``lib/sse.ts`` consumer can rely on stable event names.
The deeper integration test (live polling against a real DB across
state transitions) lands with Session 9.3b's pub/sub rewrite, since
the polling implementation requires a real SQL connection that the
unit suite doesn't provide.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit


class TestRouteRegistration:
    def test_workflow_run_events_path_is_present(self):
        from src.api.app import app

        paths = {r.path for r in app.routes}
        assert "/api/v1/workflow-runs/{run_id}/events" in paths

    def test_workflow_run_events_lives_on_workflow_runs_router(self):
        """Catches a future regression that mounts the SSE on /jobs
        by mistake — the frontend distinguishes the two stream
        sources by URL prefix."""
        from src.api.app import app

        events_routes = [
            r for r in app.routes
            if hasattr(r, "path") and r.path.endswith("/events")
        ]
        urls = {r.path for r in events_routes}
        assert "/api/v1/workflow-runs/{run_id}/events" in urls
        assert "/api/v1/jobs/{job_id}/events" in urls


class TestSseSnapshotShape:
    """Drive the endpoint with a mocked DB; consume the first SSE
    event and verify it's a ``snapshot`` carrying the run + step list
    in the documented shape."""

    @pytest.mark.asyncio
    async def test_snapshot_event_shape(self):
        from datetime import datetime
        from enum import Enum

        from src.api.app import app
        from src.api.auth.security import (
            get_current_active_user, get_current_user,
        )
        from src.api.database import get_db
        from src.api.models import WorkflowRunStatus
        from src.api.routers import workflow_runs

        # Fake user wired through the auth deps.
        class _FakeRole(str, Enum):
            researcher = "researcher"

        fake_user = SimpleNamespace(
            id="00000000-0000-0000-0000-000000000001",
            email="t@orion.dev", username="t", full_name="T",
            role=_FakeRole.researcher,
            is_active=True, is_verified=True, is_superuser=False,
            permissions=[], last_login=None,
            created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 1),
        )

        async def _stub_user():
            return fake_user

        # Two steps; one COMPLETED, one PENDING.
        fake_run = SimpleNamespace(
            id="11111111-1111-1111-1111-111111111111",
            status=WorkflowRunStatus.COMPLETED.value,
            steps=[
                SimpleNamespace(
                    step_id="parse",
                    status="SUCCEEDED",
                    simulation_job_id=None,
                ),
                SimulationStep := SimpleNamespace(
                    step_id="static",
                    status="PENDING",
                    simulation_job_id=None,
                ),
            ],
        )

        async def _stub_reload(db, rid):
            return fake_run

        async def _stub_db():
            db = MagicMock()
            db.commit = AsyncMock()
            yield db

        original_reload = workflow_runs._reload_run_with_steps
        workflow_runs._reload_run_with_steps = _stub_reload  # type: ignore
        app.dependency_overrides[get_current_active_user] = _stub_user
        app.dependency_overrides[get_current_user] = _stub_user
        app.dependency_overrides[get_db] = _stub_db
        try:
            client = TestClient(app)
            with client.stream(
                "GET",
                "/api/v1/workflow-runs/11111111-1111-1111-1111-111111111111/events",
            ) as r:
                assert r.status_code == 200
                assert r.headers["content-type"].startswith("text/event-stream")
                # Snapshot is the first event; the run is already
                # terminal so the loop emits ``terminal`` next and
                # stops. We only need the snapshot.
                buffer = b""
                snapshot_event_lines = []
                for chunk in r.iter_bytes():
                    buffer += chunk
                    if b"\n\n" in buffer:
                        block, _, buffer = buffer.partition(b"\n\n")
                        snapshot_event_lines = block.decode().splitlines()
                        break
        finally:
            app.dependency_overrides.clear()
            workflow_runs._reload_run_with_steps = original_reload  # type: ignore

        # Parse the SSE block.
        evt_line = next(
            l for l in snapshot_event_lines if l.startswith("event: ")
        )
        data_line = next(
            l for l in snapshot_event_lines if l.startswith("data: ")
        )
        assert evt_line == "event: snapshot"
        payload = json.loads(data_line[len("data: "):])
        assert payload["run_id"] == "11111111-1111-1111-1111-111111111111"
        assert payload["status"] == WorkflowRunStatus.COMPLETED.value
        assert len(payload["steps"]) == 2
        assert {s["step_id"] for s in payload["steps"]} == {"parse", "static"}
