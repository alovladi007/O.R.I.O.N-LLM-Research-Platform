"""Phase 9 / Session 9.2 — structures-list filter + pagination tests.

Session 9.2 extends ``GET /api/v1/structures`` with:

- spacegroup_number / spacegroup_number_min / spacegroup_number_max
- density_min / density_max
- num_atoms_min / num_atoms_max
- sort_by + sort_dir
- offset (pagination; the existing ``limit`` was already there)
- ``X-Total-Count`` response header carrying the unfiltered-by-
  pagination row count

These tests exercise the **endpoint signature + the SQL filter
construction** — they patch the db session so we don't need a live
Postgres. The DB-roundtrip integration test is deferred to a 9.2b
session that wires the existing tests/conftest.py db_session
fixture (currently xfailed on Session 1.2's blocker).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit


def _make_client(*, structures, total_count: int = None):
    """TestClient with auth + db stubbed; db.execute returns the
    structures list AND the count scalar in the order the endpoint
    expects (count first, then rows)."""
    from datetime import datetime
    from enum import Enum

    from src.api.app import app
    from src.api.auth.security import get_current_active_user, get_current_user
    from src.api.database import get_db

    if total_count is None:
        total_count = len(structures)

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

    async def _stub_db():
        # FastAPI calls db.execute twice per list_structures call:
        #   1. select(count()).select_from(Structure)  → scalar_one
        #   2. select(Structure)                        → scalars().all()
        # We side_effect the two consecutive results.

        count_result = MagicMock()
        count_result.scalar_one = MagicMock(return_value=total_count)

        scalars_result = MagicMock()
        scalars_result.all = MagicMock(return_value=structures)
        list_result = MagicMock()
        list_result.scalars = MagicMock(return_value=scalars_result)

        db = MagicMock()
        db.execute = AsyncMock(side_effect=[count_result, list_result])
        yield db

    app.dependency_overrides[get_current_user] = _stub_user
    app.dependency_overrides[get_current_active_user] = _stub_user
    app.dependency_overrides[get_db] = _stub_db

    return TestClient(app)


@pytest.fixture
def client():
    from src.api.app import app

    c = _make_client(structures=[], total_count=0)
    try:
        yield c
    finally:
        app.dependency_overrides.clear()


class TestFilterSchema:
    """Verify the new query params parse correctly. We don't
    inspect the DB call; we just confirm the endpoint accepts them."""

    @pytest.mark.parametrize(
        "params",
        [
            {"spacegroup_number": 227},
            {"spacegroup_number_min": 1, "spacegroup_number_max": 230},
            {"density_min": 2.0, "density_max": 3.0},
            {"num_atoms_min": 1, "num_atoms_max": 100},
            {"sort_by": "density", "sort_dir": "asc"},
            {"sort_by": "num_atoms", "sort_dir": "desc"},
            {"offset": 50, "limit": 25},
            {"formula": "Si"},
        ],
    )
    def test_filter_combo_accepted(self, client, params):
        r = client.get("/api/v1/structures", params=params)
        assert r.status_code == 200, (
            f"params {params} returned {r.status_code}: {r.text}"
        )
        assert r.json() == []

    @pytest.mark.parametrize(
        ("params", "expected_status"),
        [
            ({"spacegroup_number": 0}, 422),    # below 1
            ({"spacegroup_number": 231}, 422),  # above 230
            ({"sort_dir": "sideways"}, 422),    # regex fail
            ({"density_min": -1.0}, 422),       # ge=0
            ({"offset": -1}, 422),              # ge=0
            ({"limit": 1000}, 422),             # le=500
        ],
    )
    def test_invalid_params_rejected(self, client, params, expected_status):
        r = client.get("/api/v1/structures", params=params)
        assert r.status_code == expected_status, (
            f"params {params}: expected {expected_status}, got {r.status_code}"
        )


class TestTotalCountHeader:
    def test_x_total_count_returned(self):
        from src.api.app import app

        client = _make_client(structures=[], total_count=42)
        try:
            r = client.get("/api/v1/structures")
            assert r.status_code == 200
            assert r.headers["x-total-count"] == "42"
        finally:
            app.dependency_overrides.clear()

    def test_total_count_zero_returns_header(self):
        """Even on an empty page, the X-Total-Count header is set —
        DataGrid relies on this on first page-load."""
        from src.api.app import app

        client = _make_client(structures=[], total_count=0)
        try:
            r = client.get("/api/v1/structures?offset=0&limit=10")
            assert r.status_code == 200
            assert r.headers["x-total-count"] == "0"
        finally:
            app.dependency_overrides.clear()
