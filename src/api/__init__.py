"""
ORION API package.

This ``__init__`` intentionally does NOT eagerly import ``create_app`` or
``Settings``. Eager imports would drag ``routers`` and the stale models
package in whenever anything in ``backend.common`` touches
``src.api.exceptions`` (which several modules legitimately do for shared
exception types).

Consumers wanting the ASGI app:
    from src.api.app import app, create_app

Consumers wanting settings:
    from src.api.config import Settings, settings
"""
