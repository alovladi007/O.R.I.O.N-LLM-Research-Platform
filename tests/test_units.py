"""
Tests for backend.common.units — the SI / pint conversion layer.

Unit conventions per ROADMAP_PROMPTS.md:
- Energies: 1 eV == 1.602176634e-19 J (CODATA 2019).
- Pressures: 1 GPa == 1e9 Pa.
- Lengths: 1 Å == 1e-10 m.
- Densities: 1 g/cm³ == 1000 kg/m³.

If one of these asserts fails, don't relax the tolerance — we've broken
physical correctness somewhere.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def units():
    """Import the units module once per test — mirrors real usage."""
    from backend.common import units as mod
    return mod


# ---------------------------------------------------------------------------
# Pure conversions (to_si).
# ---------------------------------------------------------------------------


class TestToSi:
    def test_ev_to_joule(self, units):
        assert abs(units.to_si(1.0, "eV") - 1.602176634e-19) < 1e-29

    def test_gpa_to_pascal(self, units):
        assert units.to_si(10.0, "GPa") == 1e10

    def test_angstrom_to_meter(self, units):
        assert abs(units.to_si(5.43, "angstrom") - 5.43e-10) < 1e-16

    def test_g_per_cm3_to_kg_per_m3(self, units):
        # 1 g/cm³ == 1000 kg/m³
        assert abs(units.to_si(1.0, "g/cm**3") - 1000.0) < 1e-9

    def test_none_value_raises(self, units):
        with pytest.raises(ValueError):
            units.to_si(None, "eV")


# ---------------------------------------------------------------------------
# Property registry + normalization.
# ---------------------------------------------------------------------------


class TestPropertyRegistry:
    def test_registry_has_expected_properties(self, units):
        for name in (
            "bandgap",
            "formation_energy_per_atom",
            "density",
            "bulk_modulus",
            "poisson_ratio",
            "lattice_parameter",
            "temperature",
        ):
            assert name in units.PROPERTY_REGISTRY, name

    def test_bandgap_stored_as_joule(self, units):
        spec = units.PROPERTY_REGISTRY["bandgap"]
        assert spec.canonical_unit == "J"
        assert spec.bounds_si[0] == 0.0  # no negative bandgaps

    def test_poisson_ratio_bounds(self, units):
        # ν ∈ [-1, 0.5] thermodynamic limits
        spec = units.PROPERTY_REGISTRY["poisson_ratio"]
        assert spec.bounds_si == (-1.0, 0.5)

    def test_normalize_bandgap_ev_to_si(self, units):
        v = units.normalize_to_canonical("bandgap", 2.0, "eV")
        assert abs(v - 2.0 * 1.602176634e-19) < 1e-29

    def test_normalize_density_g_cm3_to_kg_m3(self, units):
        v = units.normalize_to_canonical("density", 2.33, "g/cm**3")
        assert abs(v - 2330.0) < 1e-6

    def test_negative_bandgap_rejected(self, units):
        with pytest.raises(ValueError, match="below the physical lower bound"):
            units.normalize_to_canonical("bandgap", -1.0, "eV")

    def test_poisson_above_half_rejected(self, units):
        with pytest.raises(ValueError, match="above the physical upper bound"):
            units.validate_value("poisson_ratio", 0.9)

    def test_unknown_property_raises(self, units):
        with pytest.raises(KeyError, match="Unknown property"):
            units.lookup_property("this_is_not_a_thing")


# ---------------------------------------------------------------------------
# Sanity: the SI round-trip through pint's base units.
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_ev_is_idempotent(self, units):
        # eV → J → J (already SI)
        v = units.to_si(1.0, "eV")
        assert v == units.to_si(v, "J")

    def test_kelvin_is_identity(self, units):
        assert units.to_si(300.0, "K") == 300.0
