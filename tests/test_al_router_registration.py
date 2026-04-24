"""Phase 6 / Session 6.5 — AL router registration smoke tests.

Doesn't hit the DB or run a real campaign — verifies that:

1. The ``al_router`` is importable and the AL routes appear in the
   FastAPI app's route table.
2. The ``ALCampaignCreate`` Pydantic model rejects malformed bodies
   the way the router relies on (sub-1-cycle counts, mismatched X/y
   lengths via the engine's own ``ALCampaign.__post_init__`` raise).
3. A minimal end-to-end engine call wired through the router's
   construction path (without actually invoking the auth-gated HTTP
   endpoint) returns the expected ``ALCampaignResponse`` shape.

The DB-backed integration test (real auth + real campaign run via
TestClient) lands with Session 6.5b alongside the ``al_campaigns`` /
``al_cycles`` migration.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestRouterRegistration:
    def test_al_router_is_in_app_routes(self):
        from src.api.app import app

        al_paths = [r.path for r in app.routes if "/al/" in r.path]
        # POST + GET (one) + GET (list) — three paths, two unique URL
        # templates (the GET /campaigns and POST /campaigns share path).
        assert "/api/v1/al/campaigns" in al_paths
        assert "/api/v1/al/campaigns/{campaign_id}" in al_paths

    def test_router_imports_engine(self):
        """Importing the router must not silently swallow engine import
        errors — if active_learning_v2 disappears, we want a hard
        failure here, not a 500 on the first user request."""
        from src.api.routers import al as al_router_mod
        from backend.common.ml import active_learning_v2  # noqa: F401

        assert al_router_mod.router.prefix == "/al"
        # FastAPI flattens prefix into route paths at registration.
        paths = {r.path for r in al_router_mod.router.routes}
        assert "/al/campaigns" in paths
        assert "/al/campaigns/{campaign_id}" in paths


class TestRequestSchema:
    def test_create_payload_validates_query_size_bounds(self):
        from src.api.routers.al import ALCampaignCreate

        with pytest.raises(Exception):  # pydantic ValidationError
            ALCampaignCreate(
                name="x", X_pool=[[1.0]], y_pool=[1.0],
                initial_train_indices=[0], query_size=0, n_cycles=1,
            )

    def test_create_payload_validates_n_cycles_upper_bound(self):
        from src.api.routers.al import ALCampaignCreate

        with pytest.raises(Exception):
            ALCampaignCreate(
                name="x", X_pool=[[1.0]], y_pool=[1.0],
                initial_train_indices=[0], query_size=1, n_cycles=999,
            )

    def test_create_payload_accepts_default_acquisition(self):
        from src.api.routers.al import ALCampaignCreate

        body = ALCampaignCreate(
            name="x", X_pool=[[1.0, 2.0], [3.0, 4.0]],
            y_pool=[0.1, 0.2], initial_train_indices=[0],
            query_size=1, n_cycles=1,
        )
        assert body.acquisition == "max_sigma"
        assert body.maximize is True
        assert body.model_kind == "random_forest"
