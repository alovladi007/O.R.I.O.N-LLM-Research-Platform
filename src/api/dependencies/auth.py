"""
Auth dependencies — thin re-export shim.

Routers historically import from two slightly different locations:

    from src.api.auth.security import get_current_user
    from src.api.dependencies.auth import get_current_user, get_optional_user

Rather than rewrite every import at this stage of the refactor, this shim
re-exports the canonical helpers. Phase 11 / 13 will settle on a single
blessed module (`src.api.auth`) and drop this file.
"""

from ..auth import (
    SecurityService,
    get_current_active_user,
    get_current_superuser,
    get_current_user,
    get_optional_user,
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
