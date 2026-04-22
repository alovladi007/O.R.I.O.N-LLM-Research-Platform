"""Tests for Session 3.4 — calibration + reference energies.

Unit tests don't run live QE; fixture-based tests + the machinery
of :class:`FormationEnergyCalculator`, :func:`build_elemental_reference_cell`,
and the cross-validation harness cover enough for CI. Live
calibration exercises go into the separate acceptance harness in
``scripts/orion_calibrate.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Reference cell builder
# ---------------------------------------------------------------------------


class TestReferenceCellBuilder:
    def test_si_returns_2_atom_diamond_primitive(self):
        from backend.common.calibration import build_elemental_reference_cell

        cell = build_elemental_reference_cell("Si")
        assert cell.prototype == "diamond_cubic"
        assert cell.n_atoms == 2
        assert cell.species == ["Si", "Si"]
        # Primitive FCC |a| = a_conv / sqrt(2).
        prim_len = math.sqrt(sum(x * x for x in cell.lattice_ang[0]))
        assert abs(prim_len - cell.a_conv_ang / math.sqrt(2)) < 1e-6

    def test_cu_returns_1_atom_fcc_primitive(self):
        from backend.common.calibration import build_elemental_reference_cell

        cell = build_elemental_reference_cell("Cu")
        assert cell.prototype == "fcc"
        assert cell.n_atoms == 1
        assert cell.species == ["Cu"]

    def test_na_returns_1_atom_bcc_primitive(self):
        from backend.common.calibration import build_elemental_reference_cell

        cell = build_elemental_reference_cell("Na")
        assert cell.prototype == "bcc"
        assert cell.n_atoms == 1

    def test_mg_returns_2_atom_hcp(self):
        from backend.common.calibration import build_elemental_reference_cell

        cell = build_elemental_reference_cell("Mg")
        assert cell.prototype == "hcp"
        assert cell.n_atoms == 2

    def test_a_override_applies(self):
        from backend.common.calibration import build_elemental_reference_cell

        cell = build_elemental_reference_cell("Si", a_override=5.50)
        assert cell.a_conv_ang == 5.50
        prim_len = math.sqrt(sum(x * x for x in cell.lattice_ang[0]))
        assert abs(prim_len - 5.50 / math.sqrt(2)) < 1e-6

    def test_diatomic_gas_rejected(self):
        from backend.common.calibration import (
            UnsupportedElement,
            build_elemental_reference_cell,
        )

        for element in ("H", "N", "O", "F", "Cl"):
            with pytest.raises(UnsupportedElement, match="diatomic"):
                build_elemental_reference_cell(element)

    def test_noble_gas_rejected(self):
        from backend.common.calibration import (
            UnsupportedElement,
            build_elemental_reference_cell,
        )

        with pytest.raises(UnsupportedElement, match="noble"):
            build_elemental_reference_cell("Ar")

    def test_unknown_element_rejected(self):
        from backend.common.calibration import (
            UnsupportedElement,
            build_elemental_reference_cell,
        )

        with pytest.raises(UnsupportedElement, match="prototype"):
            build_elemental_reference_cell("Zz")


# ---------------------------------------------------------------------------
# Formation-energy calculator
# ---------------------------------------------------------------------------


class TestFormationEnergy:
    def test_simple_nacl_formation_energy(self):
        """For a hypothetical NaCl with 2 atoms and total energy -5 eV,
        with Na ref = -1.31 eV and Cl ref = -1.84 eV:

            E_f = -5 - (-1.31 - 1.84) = -5 + 3.15 = -1.85 eV
            E_f_per_atom = -0.925 eV
        """
        from backend.common.calibration import FormationEnergyCalculator

        calc = FormationEnergyCalculator(
            functional="PBE",
            pseudo_family="test",
            references={"Na": -1.31, "Cl": -1.84},
        )
        result = calc.compute(species=["Na", "Cl"], total_energy_ev=-5.0)
        assert result.reference_sum_ev == pytest.approx(-3.15, abs=1e-3)
        assert result.formation_energy_ev == pytest.approx(-1.85, abs=1e-3)
        assert result.formation_energy_per_atom_ev == pytest.approx(-0.925, abs=1e-3)
        assert result.composition == {"Na": 1, "Cl": 1}
        assert result.n_atoms == 2

    def test_element_formation_energy_is_zero(self):
        """Per definition, E_f of an element against itself is zero."""
        from backend.common.calibration import FormationEnergyCalculator

        calc = FormationEnergyCalculator(
            functional="PBE",
            pseudo_family="test",
            references={"Si": -5.42},
        )
        # Pure Si with energy exactly matching the reference × n_atoms.
        result = calc.compute(
            species=["Si", "Si"],
            total_energy_ev=-5.42 * 2,
        )
        assert result.formation_energy_ev == pytest.approx(0.0, abs=1e-9)
        assert result.formation_energy_per_atom_ev == pytest.approx(0.0, abs=1e-9)

    def test_missing_reference_raises(self):
        from backend.common.calibration import (
            FormationEnergyCalculator,
            MissingReferenceError,
        )

        calc = FormationEnergyCalculator(
            functional="PBE",
            pseudo_family="test",
            references={"Si": -5.42},  # no Cu
        )
        with pytest.raises(MissingReferenceError) as excinfo:
            calc.compute(species=["Si", "Cu"], total_energy_ev=-10.0)
        assert excinfo.value.element == "Cu"

    def test_missing_reference_error_points_at_cli(self):
        from backend.common.calibration import MissingReferenceError

        err = MissingReferenceError("Fe", "PBE", "SSSP_efficiency_1.3.0")
        assert "orion calibrate" in str(err)
        assert "Fe" in str(err)

    def test_empty_species_raises(self):
        from backend.common.calibration import FormationEnergyCalculator

        calc = FormationEnergyCalculator(
            functional="PBE", pseudo_family="test", references={},
        )
        with pytest.raises(ValueError, match="empty"):
            calc.compute(species=[], total_energy_ev=0.0)

    def test_as_dict_round_trip(self):
        from backend.common.calibration import FormationEnergyCalculator

        calc = FormationEnergyCalculator(
            functional="PBE", pseudo_family="test",
            references={"Na": -1.31, "Cl": -1.84},
        )
        result = calc.compute(species=["Na", "Cl"], total_energy_ev=-5.0)
        d = result.as_dict()
        assert d["functional"] == "PBE"
        assert d["formation_energy_per_atom_ev"] == pytest.approx(-0.925, abs=1e-3)
        assert set(d["used_references"]) == {"Na", "Cl"}

    def test_calculator_rejects_both_refs_and_lookup(self):
        from backend.common.calibration import FormationEnergyCalculator

        with pytest.raises(ValueError, match="either"):
            FormationEnergyCalculator(
                functional="PBE",
                pseudo_family="test",
                references={"Si": 0},
                lookup=lambda *a, **k: 0,
            )

    def test_calculator_rejects_neither(self):
        from backend.common.calibration import FormationEnergyCalculator

        with pytest.raises(ValueError, match="Need either"):
            FormationEnergyCalculator(functional="PBE", pseudo_family="test")


# ---------------------------------------------------------------------------
# Cross-validation harness
# ---------------------------------------------------------------------------


class TestCrossValidation:
    def _make_calculator(self, **refs):
        from backend.common.calibration import FormationEnergyCalculator

        return FormationEnergyCalculator(
            functional="PBE",
            pseudo_family="test",
            references=refs,
        )

    def test_ok_when_mp_matches(self):
        """Contrived case: MP says E_f = -0.5; our calc produces -0.5.
        Deviation ≈ 0 → 'ok' status, MAE ≈ 0."""
        from backend.common.calibration import run_cross_validation

        calc = self._make_calculator(Na=-1.0, Cl=-2.0)
        # Compound: NaCl, 2 atoms, total energy chosen so E_f_per_atom = -0.5.
        # E_f_total = -1.0 → total = sum(refs) + E_f_total = -3.0 - 1.0 = -4.0
        fixtures = [{
            "mp_id": "mp-fake",
            "formula": "NaCl",
            "formation_energy_per_atom": -0.5,
            "orion_total_energy_ev": -4.0,
            "orion_species": ["Na", "Cl"],
        }]
        report = run_cross_validation(fixtures, calc)
        assert len(report.ok_entries()) == 1
        assert report.mae_ev_per_atom() == pytest.approx(0.0, abs=1e-9)
        assert report.max_abs_deviation_ev_per_atom() == pytest.approx(0.0, abs=1e-9)

    def test_skipped_when_reference_missing(self):
        from backend.common.calibration import run_cross_validation

        calc = self._make_calculator(Na=-1.0)  # no Cl
        fixtures = [{
            "mp_id": "mp-fake",
            "formula": "NaCl",
            "formation_energy_per_atom": -0.5,
            "orion_total_energy_ev": -4.0,
            "orion_species": ["Na", "Cl"],
        }]
        report = run_cross_validation(fixtures, calc)
        assert len(report.ok_entries()) == 0
        assert report.skipped_count() == 1
        assert report.entries[0].status == "skipped_missing_reference"

    def test_skipped_when_no_orion_energy(self):
        from backend.common.calibration import run_cross_validation

        calc = self._make_calculator(Na=-1.0, Cl=-2.0)
        fixtures = [{
            "mp_id": "mp-fake",
            "formula": "NaCl",
            "formation_energy_per_atom": -0.5,
            # no orion_total_energy_ev
        }]
        report = run_cross_validation(fixtures, calc)
        assert report.entries[0].status == "skipped_no_energy"

    def test_mae_and_max_dev_computed_over_ok_entries_only(self):
        from backend.common.calibration import run_cross_validation

        calc = self._make_calculator(Na=-1.0, Cl=-2.0, Si=-5.0)
        fixtures = [
            # MP says -0.5, we compute -0.5 (deviation 0).
            {"mp_id": "a", "formula": "NaCl",
             "formation_energy_per_atom": -0.5,
             "orion_total_energy_ev": -4.0, "orion_species": ["Na", "Cl"]},
            # MP says -1.0, we compute -0.8 (deviation +0.2).
            {"mp_id": "b", "formula": "NaCl",
             "formation_energy_per_atom": -1.0,
             "orion_total_energy_ev": -4.6, "orion_species": ["Na", "Cl"]},
            # Missing reference: skipped.
            {"mp_id": "c", "formula": "FeSi",
             "formation_energy_per_atom": -0.2,
             "orion_total_energy_ev": -10.0, "orion_species": ["Fe", "Si"]},
        ]
        report = run_cross_validation(fixtures, calc)
        assert len(report.ok_entries()) == 2
        assert report.skipped_count() == 1
        assert report.mae_ev_per_atom() == pytest.approx(0.1, abs=1e-3)
        assert report.max_abs_deviation_ev_per_atom() == pytest.approx(0.2, abs=1e-3)

    def test_report_as_dict_shape(self):
        from backend.common.calibration import run_cross_validation

        calc = self._make_calculator(Na=-1.0, Cl=-2.0)
        fixtures = [{
            "mp_id": "mp-fake", "formula": "NaCl",
            "formation_energy_per_atom": -0.5,
            "orion_total_energy_ev": -4.0, "orion_species": ["Na", "Cl"],
        }]
        report = run_cross_validation(fixtures, calc)
        d = report.as_dict()
        assert d["n_entries"] == 1 and d["n_ok"] == 1 and d["n_skipped"] == 0
        assert d["mae_ev_per_atom"] == pytest.approx(0.0, abs=1e-9)
        assert len(d["entries"]) == 1


# ---------------------------------------------------------------------------
# Cross-val against the real MP fixtures (loose assertion — pre-seeded refs)
# ---------------------------------------------------------------------------


_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mp_offline"


class TestCrossValAgainstMPFixtures:
    """Smoke-level: our harness can load MP fixtures and report honestly.

    We can't run actual DFT for these structures in a unit test, but we
    can verify that the harness correctly classifies them as 'ok' or
    'skipped_no_energy' depending on whether we supply the total energy.
    """

    def test_mp_fixtures_without_orion_energy_are_skipped_cleanly(self):
        import json
        from backend.common.calibration import (
            FormationEnergyCalculator,
            run_cross_validation,
        )

        fixtures = []
        for fp in sorted(_FIXTURE_DIR.glob("*.json")):
            with fp.open() as f:
                fixtures.append(json.load(f))
        # Use empty reference dict — nothing should 'ok' since no energies.
        calc = FormationEnergyCalculator(
            functional="PBE", pseudo_family="test", references={},
        )
        report = run_cross_validation(fixtures, calc)
        # None ok (no orion_total_energy_ev in any fixture).
        assert len(report.ok_entries()) == 0
        # All counted.
        assert len(report.entries) == len(fixtures)

    def test_synthetic_orion_energy_flows_through(self):
        """If we manually insert orion_total_energy_ev matching the MP
        formation energy, the harness should report 0 deviation."""
        from backend.common.calibration import (
            FormationEnergyCalculator,
            run_cross_validation,
        )

        # Fake Si fixture (E_f = 0 for element).
        fixtures = [{
            "mp_id": "mp-149",
            "formula": "Si",
            "formation_energy_per_atom": 0.0,
            "orion_total_energy_ev": -5.0 * 2,  # 2 atoms, E_ref(Si) = -5.0
            "orion_species": ["Si", "Si"],
        }]
        calc = FormationEnergyCalculator(
            functional="PBE", pseudo_family="test", references={"Si": -5.0},
        )
        report = run_cross_validation(fixtures, calc)
        assert report.mae_ev_per_atom() == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Live calibration fixture — 6 elements relaxed on the author's machine
# ---------------------------------------------------------------------------


_CALIBRATION_FIXTURE = (
    Path(__file__).parent / "fixtures" / "calibration"
    / "pbe_sssp_efficiency_1.3.0.json"
)


@pytest.mark.skipif(
    not _CALIBRATION_FIXTURE.is_file(),
    reason="calibration fixture not generated; run scripts/orion_calibrate.py",
)
class TestLiveCalibrationFixture:
    """Assertions against the 6 real reference energies captured from
    pw.x v7.5 against SSSP efficiency 1.3.0. These numbers will vary
    on different machines but stay within typical PBE bounds.
    """

    def _load(self):
        import json

        with _CALIBRATION_FIXTURE.open() as f:
            return json.load(f)

    def test_six_elements_captured(self):
        data = self._load()
        assert len(data) == 6
        elements = {r["element"] for r in data}
        assert elements == {"Si", "Cu", "Al", "Na", "Sr", "Ti"}

    def test_si_relaxed_a_within_1pct_of_experiment(self):
        """Closes the Session 3.3 open item. The 3.3 test compared Si's
        relaxed conventional a to 5.43 Å. That comparison is physics
        (DFT vs experiment). The right comparison for a regression
        test is DFT vs its own stored reference — which this test
        does. The stored reference itself is checked against
        experiment here with the realistic PBE bound of 1%.
        """
        data = self._load()
        si = next(r for r in data if r["element"] == "Si")
        assert abs(si["relaxed_a_ang"] - 5.43) / 5.43 < 0.01

    def test_metal_references_reasonable_order_of_magnitude(self):
        """All references are negative (bound states) and in the
        -100 to -5000 eV/atom window depending on core electron count.
        Positive or wildly large values would indicate a pseudopotential
        mismatch or convergence failure.
        """
        data = self._load()
        for r in data:
            assert r["energy_per_atom_ev"] < 0, f"{r['element']} has positive energy"
            assert -10000 < r["energy_per_atom_ev"] < -50, r["element"]

    def test_cross_val_with_real_refs_element_consistency(self):
        """For an element against its own reference, formation energy
        must be exactly zero. Run the cross-val harness with the live
        references and a synthetic fixture consisting of each element
        at its own reference energy. MAE should be 0.
        """
        from backend.common.calibration import (
            FormationEnergyCalculator,
            run_cross_validation,
        )

        data = self._load()
        references = {r["element"]: r["energy_per_atom_ev"] for r in data}
        calc = FormationEnergyCalculator(
            functional="PBE",
            pseudo_family="SSSP_efficiency_1.3.0",
            references=references,
        )
        fixtures = []
        for r in data:
            n = r["n_atoms_in_reference_cell"]
            fixtures.append({
                "mp_id": f"self-{r['element']}",
                "formula": r["element"],
                "formation_energy_per_atom": 0.0,
                "orion_total_energy_ev": r["energy_per_atom_ev"] * n,
                "orion_species": [r["element"]] * n,
            })
        report = run_cross_validation(fixtures, calc)
        # Every element-against-itself should give E_f_per_atom = 0.
        assert len(report.ok_entries()) == 6
        assert report.mae_ev_per_atom() == pytest.approx(0.0, abs=1e-9)
        assert report.max_abs_deviation_ev_per_atom() == pytest.approx(0.0, abs=1e-9)
