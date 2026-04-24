"""Phase 7 / Session 7.1 — BO router registration smoke tests.

No DB / no auth-bound HTTP — just verify the BoTorch-backed
``POST /api/v1/bo/suggest`` is registered and the request schema
rejects the obvious bad inputs.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestRouterRegistration:
    def test_bo_route_in_app(self):
        from src.api.app import app

        paths = [r.path for r in app.routes if "/bo/" in r.path]
        assert "/api/v1/bo/suggest" in paths

    def test_router_imports_engine(self):
        """Hard failure here is preferable to a 500 on the first user
        request when the engine path breaks.
        """
        from backend.common.ml import bo_v2  # noqa: F401
        from src.api.routers import bo as bo_router_mod

        assert bo_router_mod.router.prefix == "/bo"
        paths = {r.path for r in bo_router_mod.router.routes}
        assert "/bo/suggest" in paths


class TestRequestSchema:
    def test_history_min_length_two(self):
        from src.api.routers.bo import BOSuggestRequest

        with pytest.raises(Exception):  # pydantic ValidationError
            BOSuggestRequest(
                space=[{"kind": "continuous", "name": "x", "low": 0, "high": 1}],
                objectives=[{"name": "y", "minimize": True}],
                history=[{"point": {"x": 0.5}, "y": [1.0]}],  # only 1
                q=1,
            )

    def test_q_upper_bound(self):
        from src.api.routers.bo import BOSuggestRequest

        with pytest.raises(Exception):
            BOSuggestRequest(
                space=[{"kind": "continuous", "name": "x", "low": 0, "high": 1}],
                objectives=[{"name": "y"}],
                history=[
                    {"point": {"x": 0.5}, "y": [1.0]},
                    {"point": {"x": 0.7}, "y": [0.8]},
                ],
                q=999,
            )

    def test_objectives_required(self):
        from src.api.routers.bo import BOSuggestRequest

        with pytest.raises(Exception):
            BOSuggestRequest(
                space=[{"kind": "continuous", "name": "x", "low": 0, "high": 1}],
                objectives=[],
                history=[
                    {"point": {"x": 0.5}, "y": [1.0]},
                    {"point": {"x": 0.7}, "y": [0.8]},
                ],
            )

    def test_valid_minimal_request_constructs(self):
        from src.api.routers.bo import BOSuggestRequest

        req = BOSuggestRequest(
            space=[{"kind": "continuous", "name": "x", "low": 0, "high": 1}],
            objectives=[{"name": "y", "minimize": True}],
            history=[
                {"point": {"x": 0.5}, "y": [1.0]},
                {"point": {"x": 0.7}, "y": [0.8]},
            ],
            q=2,
        )
        assert req.q == 2
        assert req.objectives[0].minimize is True
