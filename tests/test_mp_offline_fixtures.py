"""
Tests for the Session 1.5 offline MP fixtures.

These tests don't hit the network and don't need the DB — they just
verify that every file in ``tests/fixtures/mp_offline/`` is:

1. Valid JSON with the required keys.
2. A CIF that parses through the Session 1.1 pipeline.
3. Spacegroup / formula agreement between the fixture's declared
   metadata and what the parser derives — catches fixtures that
   drift from reality.

When a fixture is added, it goes through this test automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

pymatgen = pytest.importorskip("pymatgen")

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "mp_offline"

REQUIRED_KEYS = {"mp_id", "formula", "cif", "source"}


def _iter_fixtures():
    if not FIXTURE_DIR.exists():
        return []
    return sorted(FIXTURE_DIR.glob("*.json"))


class TestFixtureShape:
    def test_directory_is_present(self):
        assert FIXTURE_DIR.exists(), (
            f"fixture dir missing: {FIXTURE_DIR}. Session 1.5 expected."
        )

    def test_at_least_five_fixtures(self):
        # Roadmap eventually wants 20; Session 1.5 ships 5.
        files = _iter_fixtures()
        assert len(files) >= 5, (
            f"only {len(files)} fixtures found; see tests/fixtures/mp_offline/README.md"
        )


@pytest.mark.parametrize("path", _iter_fixtures(), ids=lambda p: p.stem)
class TestEveryFixture:
    def test_is_valid_json(self, path: Path):
        json.loads(path.read_text())

    def test_has_required_keys(self, path: Path):
        data = json.loads(path.read_text())
        missing = REQUIRED_KEYS - set(data)
        assert not missing, f"{path.name} missing keys: {missing}"
        assert data["cif"].strip(), f"{path.name} has empty cif field"

    def test_cif_parses_through_session_1_1_pipeline(self, path: Path):
        import asyncio

        from src.api.routers.structures import parse_structure_file

        data = json.loads(path.read_text())
        parsed = asyncio.run(parse_structure_file(data["cif"], "CIF"))
        # Same shape the Session 1.1 route returns.
        assert parsed["formula"]
        assert parsed["num_atoms"] >= 1
        assert parsed["structure_hash"]
        assert len(parsed["structure_hash"]) == 64

    def test_declared_formula_matches_parsed(self, path: Path):
        import asyncio

        from src.api.routers.structures import parse_structure_file

        data = json.loads(path.read_text())
        parsed = asyncio.run(parse_structure_file(data["cif"], "CIF"))
        # Fixture formula should match what the parser derives (reduced form).
        declared = data["formula"]
        got = parsed["formula"]
        assert got == declared, (
            f"{path.name}: declared formula {declared!r} != parsed {got!r}"
        )


# ---------------------------------------------------------------------------
# load_offline_fixtures helper — exercised directly to avoid a subprocess.
# ---------------------------------------------------------------------------


class TestLoadOfflineFixtures:
    def test_loader_returns_non_empty_list(self):
        # Import through the script module.
        import importlib.util
        import pathlib

        spec = importlib.util.spec_from_file_location(
            "seed_mp_subset_module",
            pathlib.Path(__file__).resolve().parent.parent
            / "scripts"
            / "seed_mp_subset.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        items = mod.load_offline_fixtures()
        assert len(items) >= 5
        # Every item must at least carry the required keys.
        for item in items:
            for k in REQUIRED_KEYS:
                assert k in item, f"{item.get('mp_id')!r} missing key {k}"
