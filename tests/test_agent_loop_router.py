"""Phase 7 / Session 7.3 — agent-loop router smoke tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestRouterRegistration:
    def test_stop_route_in_app(self):
        from src.api.app import app

        paths = sorted({r.path for r in app.routes})
        assert "/api/v1/agent/campaigns/{campaign_id}/stop" in paths

    def test_kill_switch_registry_consistent(self):
        from src.api.routers.agent_loop import (
            get_or_create_kill_switch, reset_registry_for_tests,
        )

        reset_registry_for_tests()
        ks1 = get_or_create_kill_switch("camp-A")
        ks2 = get_or_create_kill_switch("camp-A")
        assert ks1 is ks2  # same instance — same campaign id
        ks_other = get_or_create_kill_switch("camp-B")
        assert ks_other is not ks1
        assert ks_other.is_tripped() is False

    def test_trip_through_registry_visible_to_loop(self):
        """Tripping the kill switch via the registry handle should be
        visible to a loop that resolved the same campaign id — that's
        the core HTTP-endpoint → loop contract."""
        from src.api.routers.agent_loop import (
            get_or_create_kill_switch, reset_registry_for_tests,
        )

        reset_registry_for_tests()
        ks_for_endpoint = get_or_create_kill_switch("camp-X")
        ks_for_loop = get_or_create_kill_switch("camp-X")
        ks_for_endpoint.trip()
        assert ks_for_loop.is_tripped() is True
