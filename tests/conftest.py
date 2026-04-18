"""
Shared test fixtures for the ORION test suite.

Philosophy for Phase 0 / Session 0.5:

- Fixtures are defensive. Several subsystems still can't be imported
  end-to-end (see Session 1.2 blocker for src.api.app). Fixtures that
  depend on those subsystems SKIP their consumers cleanly rather than
  crashing the whole run.
- Integration fixtures use live Postgres / Redis when available (via
  testcontainers when installed, else a live DATABASE_URL / REDIS_URL).
  Tests marked `requires_db` / `requires_redis` skip gracefully when
  neither is reachable.
- No fixtures spin up the canonical FastAPI app directly until Session
  1.2 / 0.5b unblocks the model layer.

New sessions should prefer adding narrowly-scoped fixtures here rather
than widening existing ones.
"""

from __future__ import annotations

import importlib
import os
import socket
from pathlib import Path
from typing import Iterator

import pytest

# ---------------------------------------------------------------------------
# Session-wide flags (sourced from env once at collection time).
# ---------------------------------------------------------------------------


def _port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False


def _db_reachable() -> bool:
    """Cheap Postgres reachability check (no auth handshake)."""
    url = os.getenv("DATABASE_URL", "")
    # Expect something like postgresql+asyncpg://user:pw@host:port/db
    try:
        after_at = url.split("@", 1)[1]
        host_port = after_at.split("/", 1)[0]
        host, port = host_port.split(":")
        return _port_open(host, int(port))
    except (IndexError, ValueError):
        return False


def _redis_reachable() -> bool:
    url = os.getenv("REDIS_URL", "")
    try:
        after_at = url.split("@", 1)[1]
        host_port = after_at.split("/", 1)[0]
        host, port = host_port.split(":")
        return _port_open(host, int(port))
    except (IndexError, ValueError):
        # No password form: redis://host:port/db
        try:
            after_scheme = url.split("://", 1)[1]
            host_port = after_scheme.split("/", 1)[0]
            host, port = host_port.split(":")
            return _port_open(host, int(port))
        except (IndexError, ValueError):
            return False


# Cache so we only probe each service once per session.
_DB_OK: bool | None = None
_REDIS_OK: bool | None = None


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests that need infra we can't reach."""
    global _DB_OK, _REDIS_OK
    if _DB_OK is None:
        _DB_OK = _db_reachable()
    if _REDIS_OK is None:
        _REDIS_OK = _redis_reachable()

    skip_db = pytest.mark.skip(reason="Postgres not reachable at DATABASE_URL")
    skip_redis = pytest.mark.skip(reason="Redis not reachable at REDIS_URL")
    skip_engines = pytest.mark.skip(
        reason="Simulation engines not configured (set QE_EXECUTABLE / LAMMPS_EXECUTABLE)"
    )
    skip_gpu = pytest.mark.skip(reason="No CUDA GPU available")
    skip_slurm = pytest.mark.skip(reason="No SLURM cluster configured")

    for item in items:
        if "requires_db" in item.keywords and not _DB_OK:
            item.add_marker(skip_db)
        if "requires_redis" in item.keywords and not _REDIS_OK:
            item.add_marker(skip_redis)
        if "requires_engines" in item.keywords:
            # Engines require explicit env configuration; default = skip.
            if not (
                os.getenv("QE_EXECUTABLE") or os.getenv("LAMMPS_EXECUTABLE")
            ):
                item.add_marker(skip_engines)
        if "requires_gpu" in item.keywords:
            try:
                import torch

                if not torch.cuda.is_available():
                    item.add_marker(skip_gpu)
            except ImportError:
                item.add_marker(skip_gpu)
        if "requires_slurm" in item.keywords and not os.getenv("ORION_SLURM_HOST"):
            item.add_marker(skip_slurm)


# ---------------------------------------------------------------------------
# Common fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Force anyio-backed tests to use asyncio."""
    return "asyncio"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def settings_module():
    """
    Load `src.api.config` as a standalone module to avoid triggering the
    package-level `__init__` that currently imports broken routers /
    models. Session 1.2 lifts this workaround; for now tests wanting the
    Settings class should use this fixture instead of `from src.api.config
    import Settings`.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "orion_config_test", Path(__file__).parent.parent / "src/api/config.py"
    )
    assert spec and spec.loader, "src/api/config.py not found"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Clear the lru_cache so different tests can override env vars.
    module.get_settings.cache_clear()
    return module


@pytest.fixture
def set_env(monkeypatch):
    """
    Helper fixture — returns a function that applies env-var overrides
    within the test's scope. Use for testing config invariants.

    Example::

        def test_rejects_wildcard_origin_in_prod(settings_module, set_env):
            set_env(ORION_ENV="production", CORS_ORIGINS="*")
            with pytest.raises(ValueError):
                settings_module.Settings()
    """

    def _apply(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)

    return _apply


# ---------------------------------------------------------------------------
# Fixture stubs to be filled in as the backend comes online.
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session():
    """
    Async SQLAlchemy session bound to a per-test transaction that rolls
    back on teardown. NOT IMPLEMENTED in Session 0.5 because the model
    layer (`src.api.models.__init__`) still has broken imports. Lands
    in Session 1.2 once the model package imports cleanly.

    Tests marked `requires_db` that use this fixture today will xfail
    with a clear reason.
    """
    pytest.xfail(
        "db_session fixture blocked by Session 1.2 (models/__init__ "
        "references nonexistent workflow.py / embedding.py / provenance.py)"
    )


@pytest.fixture
def api_client():
    """
    httpx.AsyncClient bound to the FastAPI app.

    NOT IMPLEMENTED — same blocker as db_session. Lands in Session 1.2.
    """
    pytest.xfail("api_client fixture blocked by Session 1.2 (canonical app import)")


@pytest.fixture
def authenticated_user(api_client):
    """JWT-authenticated test user. Waits on api_client / db_session."""
    pytest.xfail("authenticated_user fixture blocked by Session 1.2")


# ---------------------------------------------------------------------------
# Fixture registry (discoverable via `pytest --fixtures`).
# ---------------------------------------------------------------------------


def pytest_report_header(config) -> str:
    parts = ["orion test harness (Phase 0 / Session 0.5)"]
    global _DB_OK, _REDIS_OK
    if _DB_OK is None:
        _DB_OK = _db_reachable()
    if _REDIS_OK is None:
        _REDIS_OK = _redis_reachable()
    parts.append(f"postgres reachable: {_DB_OK}")
    parts.append(f"redis reachable:    {_REDIS_OK}")
    return " | ".join(parts)
