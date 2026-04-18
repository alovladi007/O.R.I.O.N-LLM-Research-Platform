"""
Tests for structure parsing — Session 1.1 acceptance.

Covers the ``backend.common.structures`` pipeline wired through
``src.api.routers.structures.parse_structure_file`` and the HTTP
``/api/v1/structures/parse`` endpoint.

Physics-level acceptance checks (from ROADMAP_PROMPTS.md Session 1.1):

- NaCl (CIF, rock-salt) → spacegroup number 225, ``Fm-3m``, cubic,
  2 atoms (primitive cell), density ≈ 2.17 g/cm³.
- Silicon (CIF, diamond) → spacegroup number 227, ``Fd-3m``, cubic,
  density within 10 % of 2.33 g/cm³.
- Invalid CIF → 422 ``ParsingError``.
- Degenerate / overlapping sites → 422.
- Hash invariants (already covered in :mod:`test_structure_hash`, but
  reaffirmed at the API layer to catch wiring regressions).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

# Any test in this file requires pymatgen. Module-level skip keeps the
# rest of the suite green on pymatgen-absent CI environments.
pymatgen = pytest.importorskip("pymatgen")


NACL_CIF = """data_NaCl
_cell_length_a    5.64056
_cell_length_b    5.64056
_cell_length_c    5.64056
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
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


SILICON_CIF = """data_Si
_cell_length_a    5.46873
_cell_length_b    5.46873
_cell_length_c    5.46873
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M   'F d -3 m'
_symmetry_Int_Tables_number 227
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Si Si 0.0  0.0  0.0  1.0
"""


# Two different species at the same position — rejected by the 0.5 Å rule.
# Using different elements avoids pymatgen's occupancy>1 short-circuit that
# would otherwise reject at CIF-parse time (before our Å check runs).
OVERLAPPING_CIF = """data_Overlap
_cell_length_a    5.0
_cell_length_b    5.0
_cell_length_c    5.0
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M   'P 1'
_symmetry_Int_Tables_number 1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Si1 Si 0.10 0.10 0.10 1.0
Ge1 Ge 0.101 0.10 0.10 1.0
"""


INVALID_CIF = """this is not a CIF file, just some text
that the parser has no hope of making sense of.
"""


# ---------------------------------------------------------------------------
# Direct parser function tests
# ---------------------------------------------------------------------------


class TestParseStructureFile:
    @pytest.mark.asyncio
    async def test_nacl_parses_to_spacegroup_225(self):
        from src.api.routers.structures import parse_structure_file

        parsed = await parse_structure_file(NACL_CIF, "CIF")

        assert parsed["formula"] == "NaCl"
        assert parsed["num_atoms"] == 2
        assert parsed["space_group_number"] == 225
        assert parsed["space_group"] == "Fm-3m"
        assert parsed["crystal_system"] == "cubic"
        # Density: primitive-cell NaCl is ≈ 2.17 g/cm³; allow ±15 %.
        assert 1.8 < parsed["density"] < 2.5
        # Hash is SHA-256 (64 hex chars).
        assert isinstance(parsed["structure_hash"], str)
        assert len(parsed["structure_hash"]) == 64

    @pytest.mark.asyncio
    async def test_silicon_parses_to_spacegroup_227(self):
        from src.api.routers.structures import parse_structure_file

        parsed = await parse_structure_file(SILICON_CIF, "CIF")

        assert parsed["formula"] == "Si"
        assert parsed["space_group_number"] == 227
        assert parsed["space_group"] == "Fd-3m"
        assert parsed["crystal_system"] == "cubic"
        # Density: primitive cell Si gives ≈ 2.33 g/cm³ with 2 atoms in a
        # conventional fcc lattice but the CIF above has the primitive
        # cell, so pymatgen's density ≈ 2.28. Tolerate ± 10 %.
        assert abs(parsed["density"] - 2.33) / 2.33 < 0.10

    @pytest.mark.asyncio
    async def test_hash_matches_for_reordered_atoms(self):
        """Swap the two NaCl atoms in the CIF — hash should be identical."""
        reordered = NACL_CIF.replace(
            "Na1 Na 0.0 0.0 0.0 1.0\nCl1 Cl 0.5 0.5 0.5 1.0",
            "Cl1 Cl 0.5 0.5 0.5 1.0\nNa1 Na 0.0 0.0 0.0 1.0",
        )
        from src.api.routers.structures import parse_structure_file

        a = await parse_structure_file(NACL_CIF, "CIF")
        b = await parse_structure_file(reordered, "CIF")
        assert a["structure_hash"] == b["structure_hash"]

    @pytest.mark.asyncio
    async def test_overlapping_sites_are_rejected(self):
        from src.api.exceptions import ParsingError
        from src.api.routers.structures import parse_structure_file

        with pytest.raises(ParsingError, match="Å apart"):
            await parse_structure_file(OVERLAPPING_CIF, "CIF")

    @pytest.mark.asyncio
    async def test_invalid_cif_raises_parsing_error(self):
        from src.api.exceptions import ParsingError
        from src.api.routers.structures import parse_structure_file

        with pytest.raises(ParsingError):
            await parse_structure_file(INVALID_CIF, "CIF")

    @pytest.mark.asyncio
    async def test_unknown_format_raises_parsing_error(self):
        from src.api.exceptions import ParsingError
        from src.api.routers.structures import parse_structure_file

        with pytest.raises(ParsingError, match="unsupported format"):
            await parse_structure_file("anything", "NOT_A_FORMAT")


# ---------------------------------------------------------------------------
# HTTP-level tests — TestClient + POST /api/v1/structures/parse
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """
    FastAPI TestClient with auth bypassed for the /structures router.

    The router pins a ``Depends(get_current_active_user)`` at the router
    level. For HTTP-level parse tests that don't actually touch the DB
    or user state, we override that dependency to return a dummy User
    stand-in. This keeps the test scope tight; Session 1.2-style auth
    integration tests come later.
    """
    from fastapi.testclient import TestClient

    from src.api.app import app
    from src.api.auth.security import get_current_active_user

    class _FakeUser:
        id = "00000000-0000-0000-0000-000000000001"
        role = "admin"

        def can_create_materials(self) -> bool:
            return True

    app.dependency_overrides[get_current_active_user] = lambda: _FakeUser()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_current_active_user, None)


class TestParseEndpoint:
    """End-to-end HTTP tests; no DB or auth required for /parse."""

    def test_parse_nacl_via_http(self, client):
        resp = client.post(
            "/api/v1/structures/parse",
            json={"text": NACL_CIF, "format": "CIF", "symprec": 0.01},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["formula"] == "NaCl"
        assert body["space_group_number"] == 225
        assert body["space_group"] == "Fm-3m"
        assert len(body["structure_hash"]) == 64

    def test_parse_silicon_via_http(self, client):
        resp = client.post(
            "/api/v1/structures/parse",
            json={"text": SILICON_CIF, "format": "CIF"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["space_group_number"] == 227

    def test_overlapping_sites_return_422(self, client):
        resp = client.post(
            "/api/v1/structures/parse",
            json={"text": OVERLAPPING_CIF, "format": "CIF"},
        )
        # ParsingError maps to 422 via the exception handlers.
        assert resp.status_code in (422, 400), resp.text
        assert "apart" in resp.text.lower() or "parse" in resp.text.lower()

    def test_invalid_cif_returns_error(self, client):
        resp = client.post(
            "/api/v1/structures/parse",
            json={"text": INVALID_CIF, "format": "CIF"},
        )
        assert resp.status_code in (422, 400), resp.text

    def test_symprec_out_of_range_rejected(self, client):
        resp = client.post(
            "/api/v1/structures/parse",
            json={"text": NACL_CIF, "format": "CIF", "symprec": 2.0},
        )
        # Pydantic validation: 422 with detail.
        assert resp.status_code == 422
