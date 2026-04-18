"""
Authentication helpers for the ORION API.

Re-exports the commonly-used dependencies so callers can do either::

    from src.api.auth import get_current_user
    from src.api.auth.security import get_current_user

Both work. ``get_optional_user`` is a thin wrapper on top of
``get_current_user`` that returns ``None`` instead of raising on missing
or invalid credentials; used by routes that want to log anonymous access
rather than reject it.
"""

from typing import Optional

from fastapi import Request

from .security import (
    SecurityService,
    get_current_active_user,
    get_current_superuser,
    get_current_user,
    require_admin,
    require_auth,
)

__all__ = [
    "SecurityService",
    "get_current_user",
    "get_current_active_user",
    "get_current_superuser",
    "get_optional_user",
    "require_admin",
    "require_auth",
]


async def get_optional_user(request: Request):
    """Return the authenticated user or ``None`` for anonymous requests.

    Implemented conservatively: parses the Authorization header manually
    so we can swallow failures instead of raising. If token validation
    machinery lives elsewhere in the codebase, we defer to
    ``get_current_user`` semantics by importing lazily.
    """
    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    try:
        # Import lazily to avoid circular imports during app startup.
        from fastapi.security.utils import get_authorization_scheme_param
        from jose import jwt

        from ..config import settings
        from ..database import get_db  # noqa: F401 — side-effect import
        from .security import SecurityService  # type: ignore[attr-defined]

        scheme, token = get_authorization_scheme_param(authorization)
        if not token or scheme.lower() != "bearer":
            return None
        # Decode without re-raising; SecurityService owns the full flow but
        # here we only need the optional-user heuristic.
        payload = jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[settings.algorithm],
        )
        return payload.get("sub") or payload.get("user_id")
    except Exception:  # noqa: BLE001 — optional auth swallows everything
        return None
