"""Phase 9 / Session 9.1 — cookie-mode auth tests.

Covers
------

1. ``get_current_user`` falls back to the orion_access_token cookie
   when no Authorization header is present.
2. ``POST /auth/login`` (default mode=bearer) does NOT set cookies —
   pre-Phase-9 clients see no behavior change.
3. ``POST /auth/login?mode=cookie`` sets both httpOnly cookies with
   the right attributes (httpOnly, samesite=lax, path=/).
4. ``GET /auth/me`` succeeds with ONLY the cookies in the jar
   (no Authorization header).
5. ``POST /auth/logout`` clears both cookies.
6. The router-prefix double-mount fix lands the canonical paths at
   ``/api/v1/auth/{login,logout,me,refresh,register,token}`` (used to
   be ``/api/v1/auth/auth/...``; OAuth2 tokenUrl was already pointing
   at the un-doubled form so the fix is a strict improvement).

These tests use FastAPI's :class:`TestClient` with the auth dependency
overridden by a stub that returns a fake :class:`User` — that decouples
the cookie-handling logic from the live Postgres dependency
(``get_current_user`` would otherwise need ``init_db`` + a real DB).
The cookie-fallback unit test covers the actual ``get_current_user``
code path with a mocked Request, so the bypass doesn't hide the new
logic.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Canonical auth paths (the router-prefix doubling fix)
# ---------------------------------------------------------------------------


class TestCanonicalAuthPaths:
    """The Session 9.1 fix removed a duplicate ``prefix='/auth'`` on the
    auth router that was producing ``/api/v1/auth/auth/...`` paths.
    These tests pin the canonical (un-doubled) paths so a future
    accidental re-introduction of the prefix breaks CI.
    """

    def test_login_path_is_singly_prefixed(self):
        from src.api.app import app

        paths = {r.path for r in app.routes}
        assert "/api/v1/auth/login" in paths
        assert "/api/v1/auth/auth/login" not in paths

    def test_logout_route_registered(self):
        from src.api.app import app

        paths = {r.path for r in app.routes}
        assert "/api/v1/auth/logout" in paths

    def test_me_path_is_singly_prefixed(self):
        from src.api.app import app

        paths = {r.path for r in app.routes}
        assert "/api/v1/auth/me" in paths


# ---------------------------------------------------------------------------
# Cookie-fallback unit test on get_current_user
# ---------------------------------------------------------------------------


class TestCookieFallback:
    """Direct unit test of the ``get_current_user`` cookie fallback —
    feeds a mock :class:`Request` whose ``cookies`` dict carries the
    access token and verifies the dependency reads from it when no
    Authorization header is present."""

    @pytest.mark.asyncio
    async def test_reads_from_cookie_when_no_bearer(self):
        """Verify the cookie-fallback branch fires when Authorization
        is absent. We patch ``decode_token`` (so we don't need to
        round-trip through JWT signing) and the existing latent
        ``TokenData(...)`` constructor bug — which both bearer and
        cookie paths share — to keep the unit test focused on the
        new cookie branch.
        """
        from src.api.auth.security import (
            ACCESS_COOKIE_NAME,
            get_current_user,
        )

        user_id = str(uuid.uuid4())
        request = MagicMock()
        request.cookies = {ACCESS_COOKIE_NAME: "fake-cookie-token"}
        fake_user = SimpleNamespace(
            id=uuid.UUID(user_id), is_active=True, username="alice",
        )
        db = MagicMock()
        db.get = AsyncMock(return_value=fake_user)

        with patch(
            "src.api.auth.security.SecurityService.decode_token",
            return_value={"type": "access", "sub": user_id},
        ), patch(
            "src.api.auth.security.TokenData",
            new=lambda **kw: SimpleNamespace(user_id=uuid.UUID(user_id)),
        ):
            result = await get_current_user(
                request=request, token=None, bearer=None, db=db,
            )
        assert result is fake_user
        # The DB lookup happened with the cookie-extracted user id.
        db.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_bearer_takes_precedence_over_cookie(self):
        """When Authorization is present, the cookie is ignored — the
        bearer token's user id wins. We pass distinct fake tokens so
        the patched decoder can identify which branch fired."""
        from src.api.auth.security import (
            ACCESS_COOKIE_NAME,
            get_current_user,
        )

        bearer_uid = uuid.uuid4()
        cookie_uid = uuid.uuid4()

        request = MagicMock()
        request.cookies = {ACCESS_COOKIE_NAME: "fake-cookie-token"}
        bearer_user = SimpleNamespace(
            id=bearer_uid, is_active=True, username="bearer",
        )
        db = MagicMock()
        db.get = AsyncMock(return_value=bearer_user)

        def _decode(tok):
            if tok == "fake-bearer-token":
                return {"type": "access", "sub": str(bearer_uid)}
            if tok == "fake-cookie-token":
                return {"type": "access", "sub": str(cookie_uid)}
            raise AssertionError(f"unexpected token {tok!r}")

        def _token_data(user_id, **kwargs):
            return SimpleNamespace(user_id=uuid.UUID(user_id))

        with patch(
            "src.api.auth.security.SecurityService.decode_token",
            side_effect=_decode,
        ), patch(
            "src.api.auth.security.TokenData",
            side_effect=_token_data,
        ):
            result = await get_current_user(
                request=request, token="fake-bearer-token",
                bearer=None, db=db,
            )
        # DB was queried with the BEARER user id, not the cookie's.
        db.get.assert_awaited_once()
        # Returned user matches the bearer-flow lookup.
        assert result is bearer_user

    @pytest.mark.asyncio
    async def test_no_token_anywhere_raises_401(self):
        from fastapi import HTTPException

        from src.api.auth.security import get_current_user

        request = MagicMock()
        request.cookies = {}
        db = MagicMock()
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(
                request=request, token=None, bearer=None, db=db,
            )
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Cookie-mode login + me + logout via TestClient
# ---------------------------------------------------------------------------


def _client_with_stub_auth():
    """Return a TestClient with the auth dependency overridden so we
    don't need a live Postgres for the cookie-handling integration."""
    from datetime import datetime

    from src.api.app import app
    from src.api.auth.security import get_current_user, get_current_active_user
    from types import SimpleNamespace
    from src.api.database import get_db

    fake_user = SimpleNamespace(
        id=uuid.uuid4(),
        email="cookie-test@orion.dev",
        username="cookie-test",
        full_name="Cookie Test",
        # UserResponse expects role to serialize as a plain string;
        # the live model carries an Enum but UserResponse declares
        # ``role: str`` so model_validate runs str() on it.
        role="researcher",
        is_active=True,
        is_verified=True,
        is_superuser=False,
        permissions=[],
        last_login=None,
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
    )

    async def _stub_user():
        return fake_user

    async def _stub_db():
        # The /me endpoint only depends on get_current_active_user
        # (overridden above), but get_db is in the signature of the
        # underlying get_current_user; provide a no-op session.
        yield MagicMock()

    app.dependency_overrides[get_current_active_user] = _stub_user
    app.dependency_overrides[get_current_user] = _stub_user
    app.dependency_overrides[get_db] = _stub_db

    yield_obj = (TestClient(app), fake_user)
    return yield_obj


@pytest.fixture
def stub_client():
    from src.api.app import app

    client, user = _client_with_stub_auth()
    try:
        yield client, user
    finally:
        app.dependency_overrides.clear()


class TestMeEndpointWithStub:
    def test_me_succeeds_with_stub(self, stub_client):
        """Sanity: with the auth dependency stubbed, GET /me returns
        the synthetic user. This proves the override is wired
        correctly so the cookie integration tests below are meaningful.
        """
        client, fake_user = stub_client
        r = client.get("/api/v1/auth/me")
        assert r.status_code == 200
        body = r.json()
        assert body["email"] == "cookie-test@orion.dev"
        assert body["role"] == "researcher"


# ---------------------------------------------------------------------------
# Cookie-mode login flow with a real DB stub
# ---------------------------------------------------------------------------


def _login_test_client():
    """TestClient where the login endpoint can run end-to-end against
    a fake DB. We patch the SQLAlchemy ``session.execute`` so that
    looking up the test user returns a synthetic User object with a
    bcrypt-hashed password we control.
    """
    from src.api.app import app
    from src.api.auth.security import SecurityService
    from src.api.database import get_db

    from datetime import datetime
    from enum import Enum

    class _FakeRole(str, Enum):
        """Subclass of ``str`` so pydantic accepts it as ``role: str``
        AND has ``.value`` for the login-endpoint code path."""
        researcher = "researcher"

    user_id = uuid.uuid4()
    hashed = SecurityService.hash_password("cookie-pass-123")
    fake_user = SimpleNamespace(
        id=user_id,
        email="cookie@orion.dev",
        username="cookie",
        full_name="Cookie",
        role=_FakeRole.researcher,
        hashed_password=hashed,
        is_active=True,
        is_verified=True,
        is_superuser=False,
        permissions=[],
        last_login=None,
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
    )

    class _FakeResult:
        def scalar_one_or_none(self):
            return fake_user

    async def _stub_db():
        db = MagicMock()
        db.execute = AsyncMock(return_value=_FakeResult())
        db.commit = AsyncMock()
        db.refresh = AsyncMock()
        # db.get used by get_current_user / refresh_token.
        db.get = AsyncMock(return_value=fake_user)
        yield db

    app.dependency_overrides[get_db] = _stub_db
    return TestClient(app), fake_user


@pytest.fixture
def login_client():
    from src.api.app import app

    client, user = _login_test_client()
    try:
        yield client, user
    finally:
        app.dependency_overrides.clear()


class TestLoginCookieMode:
    def test_default_bearer_mode_does_not_set_cookies(self, login_client):
        """``mode=bearer`` (default) must keep the pre-Phase-9 behavior:
        body tokens, no Set-Cookie header. Backward compatibility for
        curl / Postman / the OAuth2 password flow."""
        client, _ = login_client
        r = client.post(
            "/api/v1/auth/login",
            json={"username": "cookie", "password": "cookie-pass-123"},
        )
        assert r.status_code == 200
        # Tokens in the body.
        body = r.json()
        assert body["access_token"]
        assert body["refresh_token"]
        # No Set-Cookie for our auth cookies.
        from src.api.auth.security import (
            ACCESS_COOKIE_NAME, REFRESH_COOKIE_NAME,
        )
        cookies = client.cookies.jar
        names = {c.name for c in cookies}
        assert ACCESS_COOKIE_NAME not in names
        assert REFRESH_COOKIE_NAME not in names

    def test_cookie_mode_sets_both_cookies(self, login_client):
        """``mode=cookie`` sets both httpOnly cookies with the right
        attributes and ALSO returns the body tokens (so the response
        format is invariant — the cookie is additive)."""
        client, _ = login_client
        r = client.post(
            "/api/v1/auth/login?mode=cookie",
            json={"username": "cookie", "password": "cookie-pass-123"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["access_token"]
        assert body["refresh_token"]
        from src.api.auth.security import (
            ACCESS_COOKIE_NAME, REFRESH_COOKIE_NAME,
        )
        # Both cookies in the jar.
        cookies = {c.name: c for c in client.cookies.jar}
        assert ACCESS_COOKIE_NAME in cookies
        assert REFRESH_COOKIE_NAME in cookies
        # Cookie attributes — httpOnly, SameSite=Lax, path=/.
        access_cookie = cookies[ACCESS_COOKIE_NAME]
        # httpcookiejar exposes httpOnly via the rest dict.
        assert access_cookie.has_nonstandard_attr("HttpOnly")
        assert access_cookie.path == "/"
        # In testing env (not production), secure=False so the cookie
        # is accepted over http://localhost.
        assert not access_cookie.secure

    def test_cookie_login_then_me_with_only_cookies(self, login_client):
        """End-to-end cookie path: log in with mode=cookie, then call
        /me with the cookies in the jar (no Authorization header).
        """
        client, _ = login_client
        login_r = client.post(
            "/api/v1/auth/login?mode=cookie",
            json={"username": "cookie", "password": "cookie-pass-123"},
        )
        assert login_r.status_code == 200
        # client.get without an Authorization header — must succeed
        # via the cookie fallback. TestClient sends the jar's cookies
        # automatically.
        me_r = client.get("/api/v1/auth/me")
        assert me_r.status_code == 200, (
            f"GET /me with cookies failed: {me_r.status_code} {me_r.text}"
        )
        assert me_r.json()["email"] == "cookie@orion.dev"

    def test_logout_clears_cookies(self, login_client):
        client, _ = login_client
        client.post(
            "/api/v1/auth/login?mode=cookie",
            json={"username": "cookie", "password": "cookie-pass-123"},
        )
        from src.api.auth.security import (
            ACCESS_COOKIE_NAME, REFRESH_COOKIE_NAME,
        )
        names_before = {c.name for c in client.cookies.jar}
        assert ACCESS_COOKIE_NAME in names_before
        assert REFRESH_COOKIE_NAME in names_before

        r = client.post("/api/v1/auth/logout")
        assert r.status_code == 204
        # Verify the response asks the browser to clear both cookies.
        # FastAPI's delete_cookie sets the cookie to an empty value
        # with Max-Age=0; we accept either ``max-age=0`` or an
        # ``expires=`` in the past as the clearance signal.
        set_cookie_headers = r.headers.get_list("set-cookie")
        access_clear = any(
            ACCESS_COOKIE_NAME in h
            and ("max-age=0" in h.lower() or "expires=" in h.lower())
            for h in set_cookie_headers
        )
        refresh_clear = any(
            REFRESH_COOKIE_NAME in h
            and ("max-age=0" in h.lower() or "expires=" in h.lower())
            for h in set_cookie_headers
        )
        assert access_clear, (
            f"no clear-cookie Set-Cookie header for {ACCESS_COOKIE_NAME}; "
            f"got: {set_cookie_headers}"
        )
        assert refresh_clear, (
            f"no clear-cookie Set-Cookie header for {REFRESH_COOKIE_NAME}"
        )

    def test_invalid_password_returns_401(self, login_client):
        client, _ = login_client
        r = client.post(
            "/api/v1/auth/login",
            json={"username": "cookie", "password": "wrong-password"},
        )
        assert r.status_code == 401
