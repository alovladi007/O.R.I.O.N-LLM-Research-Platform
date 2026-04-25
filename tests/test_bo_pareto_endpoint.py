"""Phase 9 / Session 9.4 — BO pareto-front endpoint tests.

The frontend's MO campaign-detail page calls
``POST /api/v1/bo/pareto-front`` to highlight the front in a
recharts scatter. We pin the route + the response shape + the
validation invariants.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit


def _client():
    from src.api.app import app
    from src.api.auth.security import (
        get_current_active_user, get_current_user,
    )
    from src.api.database import get_db

    class _FakeRole(str, Enum):
        researcher = "researcher"

    user = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000001",
        email="t@orion.dev", username="t", full_name="T",
        role=_FakeRole.researcher,
        is_active=True, is_verified=True, is_superuser=False,
        permissions=[], last_login=None,
        created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 1),
    )

    async def _stub_user():
        return user

    async def _stub_db():
        yield MagicMock()

    app.dependency_overrides[get_current_active_user] = _stub_user
    app.dependency_overrides[get_current_user] = _stub_user
    app.dependency_overrides[get_db] = _stub_db
    return TestClient(app)


@pytest.fixture
def client():
    from src.api.app import app

    c = _client()
    try:
        yield c
    finally:
        app.dependency_overrides.clear()


class TestRouteRegistration:
    def test_pareto_route_present(self):
        from src.api.app import app

        paths = {r.path for r in app.routes}
        assert "/api/v1/bo/pareto-front" in paths


class TestParetoCorrectness:
    def test_textbook_2d_minimize_both(self, client):
        r = client.post(
            "/api/v1/bo/pareto-front",
            json={
                # 3 candidates: (0, 0) dominates the other two when
                # minimizing both. So mask = [True, False, False].
                "Y": [[0.0, 0.0], [1.0, 1.0], [0.5, 2.0]],
                "minimize": [True, True],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["mask"] == [True, False, False]
        assert body["optimal_indices"] == [0]

    def test_mixed_directions(self, client):
        r = client.post(
            "/api/v1/bo/pareto-front",
            json={
                # Maximize first, minimize second.
                # (10, 0) and (0, -1) on the front; (5, 1) dominated.
                "Y": [[10.0, 0.0], [5.0, 1.0], [0.0, -1.0]],
                "minimize": [False, True],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["mask"] == [True, False, True]
        assert body["optimal_indices"] == [0, 2]


class TestValidation:
    def test_minimize_length_mismatch_rejected(self, client):
        r = client.post(
            "/api/v1/bo/pareto-front",
            json={"Y": [[1.0, 2.0]], "minimize": [True]},
        )
        assert r.status_code == 400
        # The ORION exception handler wraps HTTPException.detail under
        # {"error": {"message": ...}}.
        body = r.json()
        msg = body.get("error", {}).get("message") or body.get("detail", "")
        assert "doesn't match" in msg

    def test_empty_y_rejected(self, client):
        r = client.post(
            "/api/v1/bo/pareto-front",
            json={"Y": [], "minimize": [True]},
        )
        # Pydantic min_length=1 fires.
        assert r.status_code == 422
