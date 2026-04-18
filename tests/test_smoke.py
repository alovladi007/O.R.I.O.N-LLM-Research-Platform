"""
Smoke tests — Phase 0 / Session 0.5.

These tests cover the parts of the codebase that are known to be importable
and working today. They exist so that subsequent sessions have a regression
net even while the canonical app remains partially broken (see Session 1.2).

Scope:
- Config module loads and enforces the Session 0.4 security invariants.
- `backend.common.structures` imports and round-trips a simple structure.
- Repository structural invariants that should not regress across sessions.

When Session 1.2 lifts the model-import blocker, this file should grow to
cover health / auth / materials / structures HTTP endpoints via the
``api_client`` fixture.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Config smoke tests (Session 0.1 + 0.4)
# ---------------------------------------------------------------------------


class TestConfigLoadsInDev:
    """Config should load cleanly with the pytest.ini env block."""

    def test_settings_instantiates(self, settings_module):
        s = settings_module.Settings()
        assert s.environment == "testing"
        assert s.cors_allow_credentials is True

    def test_cors_methods_default_is_explicit(self, settings_module):
        s = settings_module.Settings()
        assert "*" not in s.cors_allow_methods
        assert "GET" in s.cors_allow_methods
        assert "OPTIONS" in s.cors_allow_methods

    def test_cors_headers_default_is_explicit(self, settings_module):
        s = settings_module.Settings()
        assert "*" not in s.cors_allow_headers
        assert "Authorization" in s.cors_allow_headers

    def test_secret_key_meets_minimum_length(self, settings_module):
        s = settings_module.Settings()
        assert len(s.secret_key.get_secret_value()) >= 32


class TestConfigRejectsInsecureProd:
    """Model validator from Session 0.4 — these must fail for non-dev envs."""

    def test_wildcard_origin_rejected_in_prod(self, settings_module, set_env):
        set_env(
            ORION_ENV="production",
            CORS_ORIGINS="*",
            JWT_SECRET_KEY="a" * 48,
            DATABASE_URL="postgresql+asyncpg://u:real_pw@h:5432/d",
            REDIS_URL="redis://:real_pw@h:6379/0",
        )
        with pytest.raises(ValueError, match=r"CORS_ORIGINS='\*'"):
            settings_module.Settings()

    def test_short_secret_rejected_in_prod(self, settings_module, set_env):
        set_env(
            ORION_ENV="production",
            CORS_ORIGINS="https://example.com",
            JWT_SECRET_KEY="too-short",
            DATABASE_URL="postgresql+asyncpg://u:real_pw@h:5432/d",
            REDIS_URL="redis://:real_pw@h:6379/0",
        )
        with pytest.raises(ValueError, match=r"at least 32 characters"):
            settings_module.Settings()

    def test_demo_db_password_rejected_in_prod(self, settings_module, set_env):
        set_env(
            ORION_ENV="production",
            CORS_ORIGINS="https://example.com",
            JWT_SECRET_KEY="a" * 48,
            DATABASE_URL="postgresql+asyncpg://u:orion_secure_pwd@h:5432/d",
            REDIS_URL="redis://:real_pw@h:6379/0",
        )
        with pytest.raises(ValueError, match=r"DATABASE_URL.*demo password"):
            settings_module.Settings()

    def test_demo_redis_password_rejected_in_prod(self, settings_module, set_env):
        set_env(
            ORION_ENV="production",
            CORS_ORIGINS="https://example.com",
            JWT_SECRET_KEY="a" * 48,
            DATABASE_URL="postgresql+asyncpg://u:real_pw@h:5432/d",
            REDIS_URL="redis://:orion_redis_pwd@h:6379/0",
        )
        with pytest.raises(ValueError, match=r"REDIS_URL.*demo password"):
            settings_module.Settings()

    def test_wildcard_methods_rejected_always(self, settings_module, set_env):
        # Wildcard methods are rejected even in development.
        set_env(ORION_ENV="development", CORS_ALLOW_METHODS="*")
        with pytest.raises(ValueError, match=r"cors_allow_methods contains"):
            settings_module.Settings()

    def test_dev_mode_tolerates_wildcard_origin(self, settings_module, set_env):
        # Dev must continue to work with permissive CORS.
        set_env(ORION_ENV="development", CORS_ORIGINS="*")
        s = settings_module.Settings()
        assert s.cors_origins == ["*"]


# ---------------------------------------------------------------------------
# Structure parser smoke test (backend.common.structures — real pymatgen)
# ---------------------------------------------------------------------------


# Minimal CIF for rock-salt NaCl (space group Fm-3m).
NACL_CIF = """data_NaCl
_cell_length_a    5.64056
_cell_length_b    5.64056
_cell_length_c    5.64056
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
_space_group_name_H-M_alt   'F m -3 m'
_symmetry_space_group_name_H-M   'F m -3 m'
_symmetry_Int_Tables_number   225
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Na1 Na 0.0 0.0 0.0 1.0
Cl1 Cl 0.5 0.5 0.5 1.0
"""


class TestBackendCommonStructures:
    """Prove the real pymatgen-backed parsers are reachable and sane."""

    def test_module_import(self):
        from backend.common.structures import (  # noqa: F401
            InternalStructureModel,
            StructureFormat,
            parse_structure,
            to_cif,
            to_poscar,
            to_xyz,
        )

    def test_parse_nacl_cif_returns_internal_model(self):
        pymatgen = pytest.importorskip("pymatgen")  # noqa: F841
        from backend.common.structures import StructureFormat, parse_structure

        parsed = parse_structure(NACL_CIF, StructureFormat.CIF)
        # Session 0.5 only asserts the shape. Session 1.1 will add the
        # physics-level checks (spacegroup 225, a=5.64 Å, formula='NaCl').
        assert parsed is not None
        assert hasattr(parsed, "formula") or hasattr(parsed, "atoms")


# ---------------------------------------------------------------------------
# Repository structural invariants
# ---------------------------------------------------------------------------


class TestRepoLayout:
    """Invariants that Phase 0 established and later sessions must not break."""

    def test_canonical_entry_points_exist(self, repo_root: Path):
        assert (repo_root / "src" / "api" / "app.py").is_file()
        assert (repo_root / "src" / "api" / "config.py").is_file()

    def test_legacy_entry_points_are_gone(self, repo_root: Path):
        for dead in (
            "simple_api.py",
            "src/api/app_dev.py",
            "demo_app.py",
            "run_demo.py",
            "setup.py",
        ):
            assert not (repo_root / dead).exists(), (
                f"Legacy entry point {dead} reappeared — see Phase 0 reports"
            )

    def test_legacy_src_packages_are_gone(self, repo_root: Path):
        for dead in (
            "src/core",
            "src/candidate_generation",
            "src/knowledge_graph",
            "src/rag",
            "src/data_ingest",
            "src/simulation",
            "src/experimental_design",
            "src/evaluation",
            "src/feedback_loop",
            "src/ui",
        ):
            assert not (repo_root / dead).exists(), (
                f"Legacy package {dead} reappeared — see Session 0.2 report"
            )

    def test_no_neo4j_refs_in_tracked_config(self, repo_root: Path):
        # Session 0.2 removed Neo4j. A later change that reintroduces it
        # without roadmap-level decision should fail this test.
        probe_paths = [
            repo_root / "docker-compose.yml",
            repo_root / "requirements.txt",
            repo_root / "pyproject.toml",
            repo_root / ".env.example",
            repo_root / "src" / "api" / "config.py",
        ]
        for p in probe_paths:
            text = p.read_text(encoding="utf-8") if p.exists() else ""
            # Allow mentions in comments explaining that neo4j was removed.
            for line in text.splitlines():
                if "neo4j" in line.lower() or "NEO4J" in line:
                    stripped = line.strip()
                    assert stripped.startswith(("#", "//")) or "Removed" in stripped, (
                        f"Non-comment neo4j reference reappeared in {p.name}: {line!r}"
                    )

    def test_root_markdown_count_within_budget(self, repo_root: Path):
        # Session 0.3 target: ≤ 6 root-level .md files.
        md_files = sorted(p.name for p in repo_root.glob("*.md"))
        assert len(md_files) <= 6, f"Too many root-level md files: {md_files}"

    def test_docs_history_and_guides_exist(self, repo_root: Path):
        assert (repo_root / "docs" / "history").is_dir()
        assert (repo_root / "docs" / "guides").is_dir()
        assert (repo_root / "docs" / "SECURITY.md").is_file()

    def test_changelog_tracks_phase_0(self, repo_root: Path):
        text = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
        # At minimum, every merged session leaves a fingerprint here.
        for token in ("Session 0.1", "Session 0.2", "Session 0.3"):
            assert token in text, f"CHANGELOG missing {token}"


# ---------------------------------------------------------------------------
# Canonical app health check — enabled in Session 1.2 once the model
# package imports cleanly. Uses FastAPI's in-process TestClient so we
# don't depend on Postgres/Redis being reachable for the smoke test.
# ---------------------------------------------------------------------------


def test_canonical_app_imports():
    """The canonical FastAPI app must import with all routers registered."""
    from src.api.app import app

    assert app.title == "ORION Platform API"
    # At least the core routers from Session 0.1 audit.
    registered = {r.path for r in app.routes}
    # Spot-check a few canonical paths.
    assert any(p.startswith("/api/v1") for p in registered), registered
    assert any(p in ("/", "/health", "/healthz") for p in registered)


def test_health_endpoint_returns_200():
    """
    Hits the app in-process via FastAPI's TestClient. No DB or Redis
    required for /health itself — the route is synchronous by design.
    """
    from fastapi.testclient import TestClient

    from src.api.app import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get("status") in {"ok", "healthy", "up"} or "status" in body
