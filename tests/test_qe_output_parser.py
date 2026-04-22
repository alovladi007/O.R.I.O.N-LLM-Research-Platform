"""Tests for Phase 3 / Session 3.2 — pw.x output parser.

Golden fixtures live in ``tests/fixtures/pw_outputs/``. They're
hand-crafted to be minimal but realistic (matching pw.x 7.x format).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


FIXTURES = Path(__file__).parent / "fixtures" / "pw_outputs"


# ---------------------------------------------------------------------------
# Successful SCF
# ---------------------------------------------------------------------------


class TestParseConvergedScf:
    def test_energy_converted_to_ev(self):
        from backend.common.engines.qe_run import parse_pw_output
        from backend.common.engines.qe_run.output import RY_TO_EV

        result = parse_pw_output(FIXTURES / "si_scf_converged.out")
        assert result.energy is not None
        assert result.energy.total_ry == pytest.approx(-15.83, abs=1e-4)
        assert result.energy.total_ev == pytest.approx(-15.83 * RY_TO_EV, abs=1e-3)

    def test_scf_history(self):
        from backend.common.engines.qe_run import parse_pw_output

        result = parse_pw_output(FIXTURES / "si_scf_converged.out")
        # Final + per-iteration energies: iteration 1, 2, 3 (3 lines) + final (!).
        # The regex matches both the "     total energy" iteration lines and
        # the "!    total energy" final marker.
        assert len(result.energy.scf_history_ry) >= 3
        # Monotonically approaching final (decreasing absolute change).
        hist = result.energy.scf_history_ry
        assert hist[-1] == pytest.approx(-15.83, abs=1e-4)

    def test_convergence_flagged_converged(self):
        from backend.common.engines.qe_run import (
            ConvergenceStatus,
            parse_pw_output,
        )

        result = parse_pw_output(FIXTURES / "si_scf_converged.out")
        assert result.convergence == ConvergenceStatus.CONVERGED
        assert result.n_scf_iterations == 3

    def test_forces_parsed_and_converted(self):
        from backend.common.engines.qe_run import parse_pw_output
        from backend.common.engines.qe_run.output import RY_PER_BOHR_TO_EV_PER_ANG

        result = parse_pw_output(FIXTURES / "si_scf_converged.out")
        assert len(result.forces) == 2
        f0, f1 = result.forces
        # Atom indices converted from 1-based (pw.x) to 0-based.
        assert f0.atom_index == 0 and f1.atom_index == 1
        # Conversion check: 0.001 Ry/bohr -> ~0.0257 eV/Å
        assert f0.fx == pytest.approx(0.001 * RY_PER_BOHR_TO_EV_PER_ANG, rel=1e-5)
        # Forces on Si2 are the negative of Si1 (symmetric fixture).
        assert f1.fx == pytest.approx(-f0.fx, rel=1e-5)

    def test_max_abs_force_derivation(self):
        from backend.common.engines.qe_run import parse_pw_output

        result = parse_pw_output(FIXTURES / "si_scf_converged.out")
        # Largest component in the fixture is 0.003 Ry/bohr.
        from backend.common.engines.qe_run.output import RY_PER_BOHR_TO_EV_PER_ANG

        assert result.max_abs_force() == pytest.approx(
            0.003 * RY_PER_BOHR_TO_EV_PER_ANG, rel=1e-5,
        )

    def test_stress_tensor_in_gpa(self):
        from backend.common.engines.qe_run import parse_pw_output

        result = parse_pw_output(FIXTURES / "si_scf_converged.out")
        assert result.stress is not None
        tensor = result.stress.tensor_gpa
        assert len(tensor) == 3 and all(len(row) == 3 for row in tensor)
        # Fixture has 1.81 kbar on the diagonal → 0.181 GPa.
        assert tensor[0][0] == pytest.approx(0.181, abs=1e-4)
        assert tensor[2][2] == pytest.approx(0.998, abs=1e-4)

    def test_pressure_derived_from_trace(self):
        from backend.common.engines.qe_run import parse_pw_output

        result = parse_pw_output(FIXTURES / "si_scf_converged.out")
        tensor = result.stress.tensor_gpa
        expected = -(tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0
        assert result.stress.pressure_gpa == pytest.approx(expected)

    def test_counts_parsed(self):
        from backend.common.engines.qe_run import parse_pw_output

        result = parse_pw_output(FIXTURES / "si_scf_converged.out")
        assert result.n_atoms == 2
        assert result.n_species == 1
        assert result.n_electrons == pytest.approx(8.0)

    def test_wall_time(self):
        from backend.common.engines.qe_run import parse_pw_output

        result = parse_pw_output(FIXTURES / "si_scf_converged.out")
        assert result.wall_time_seconds == pytest.approx(0.40)

    def test_as_dict_shape(self):
        from backend.common.engines.qe_run import parse_pw_output

        result = parse_pw_output(FIXTURES / "si_scf_converged.out")
        d = result.as_dict()
        # Required keys for the job outputs payload (consumed by the
        # DAG resolver via uses: step_id.outputs.<key>).
        for key in (
            "energy_ev", "energy_ry", "max_force_ev_per_ang",
            "pressure_gpa", "convergence", "n_scf_iterations",
        ):
            assert key in d, f"missing {key}"
        assert d["convergence"] == "converged"


# ---------------------------------------------------------------------------
# Unconverged SCF
# ---------------------------------------------------------------------------


class TestParseUnconvergedScf:
    def test_convergence_flagged_unconverged(self):
        from backend.common.engines.qe_run import (
            ConvergenceStatus,
            parse_pw_output,
        )

        result = parse_pw_output(FIXTURES / "si_scf_unconverged.out")
        assert result.convergence == ConvergenceStatus.UNCONVERGED

    def test_partial_energy_still_extracted(self):
        from backend.common.engines.qe_run import parse_pw_output

        result = parse_pw_output(FIXTURES / "si_scf_unconverged.out")
        # No "!  total energy" marker → final energy is None.
        assert result.energy is None

    def test_no_forces_when_unconverged(self):
        from backend.common.engines.qe_run import parse_pw_output

        result = parse_pw_output(FIXTURES / "si_scf_unconverged.out")
        assert result.forces == []


# ---------------------------------------------------------------------------
# Hard error
# ---------------------------------------------------------------------------


class TestParseBFGSUnconverged:
    """Session 3.4b addition. A vc-relax that converged its SCF but
    failed BFGS step limits should report ``bfgs_unconverged``, not
    fatal ``unconverged``. The energy is usable as an approximate
    minimum. Caught by a false positive during live O₂ calibration.
    """

    _BFGS_FAIL_FRAGMENT = """
     Program PWSCF v.7.5 starts
     number of atoms/cell      =            2
     bravais-lattice index     =            0
     kinetic-energy cutoff     =      46.6000  Ry

     Self-consistent Calculation
     convergence has been achieved in   9 iterations

     Self-consistent Calculation
     convergence has been achieved in   5 iterations

     bfgs failed after   6 scf cycles and   3 bfgs steps, convergence not achieved

!    total energy              =     -83.04485614 Ry

     PWSCF        :     5m59.49s CPU   6m29.09s WALL
"""

    def test_bfgs_unconverged_distinct_from_unconverged(self):
        from backend.common.engines.qe_run import (
            ConvergenceStatus,
            parse_pw_output,
        )

        result = parse_pw_output(self._BFGS_FAIL_FRAGMENT)
        assert result.convergence == ConvergenceStatus.BFGS_UNCONVERGED
        # Energy still parseable.
        assert result.energy is not None
        assert result.energy.total_ry == pytest.approx(-83.04485, abs=1e-4)

    def test_scf_unconverged_takes_priority_over_bfgs(self):
        """When SCF itself diverges, that's fatal. We must NOT
        misclassify as bfgs_unconverged."""
        from backend.common.engines.qe_run import (
            ConvergenceStatus,
            parse_pw_output,
        )

        text = (
            "convergence NOT achieved after 200 iterations: stopping\n"
            "bfgs failed after 6 scf cycles and 3 bfgs steps, convergence not achieved\n"
        )
        result = parse_pw_output(text)
        assert result.convergence == ConvergenceStatus.UNCONVERGED


class TestParseWallTime:
    """Wall-time regex handles pw.x's multiple timer formats.
    The failure during O₂ calibration surfaced the missing h/m support.
    """

    def test_short_run_plain_seconds(self):
        from backend.common.engines.qe_run import parse_pw_output

        text = (
            "     Self-consistent Calculation\n"
            "     convergence has been achieved in 3 iterations\n"
            "!    total energy              =     -15.83 Ry\n"
            "     PWSCF        :     0.30s CPU      0.40s WALL\n"
        )
        r = parse_pw_output(text)
        assert r.wall_time_seconds == pytest.approx(0.40, abs=1e-3)

    def test_medium_run_minutes(self):
        from backend.common.engines.qe_run import parse_pw_output

        text = (
            "     Self-consistent Calculation\n"
            "     convergence has been achieved in 9 iterations\n"
            "!    total energy              =     -83.04 Ry\n"
            "     PWSCF        :   5m59.49s CPU   6m21.02s WALL\n"
        )
        r = parse_pw_output(text)
        # 6*60 + 21.02 = 381.02
        assert r.wall_time_seconds == pytest.approx(381.02, abs=1e-3)

    def test_long_run_hours(self):
        from backend.common.engines.qe_run import parse_pw_output

        text = (
            "     Self-consistent Calculation\n"
            "     convergence has been achieved in 9 iterations\n"
            "!    total energy              =     -1000.0 Ry\n"
            "     PWSCF        :  1h10m 5.12s CPU  1h11m20.5s WALL\n"
        )
        r = parse_pw_output(text)
        # 1h * 3600 + 11m * 60 + 20.5 = 3600 + 660 + 20.5 = 4280.5
        assert r.wall_time_seconds == pytest.approx(4280.5, abs=1e-3)


class TestParsePositionOnlyRelax:
    """A plain ``calculation='relax'`` (not vc-relax) emits no
    CELL_PARAMETERS block — the cell is fixed. The parser must fall
    back to the preamble's ``crystal axes`` block. This was the second
    bug from the O₂ calibration: relaxed=None even though the relax
    finished.
    """

    _RELAX_FRAGMENT = """
     Program PWSCF v.7.5 starts
     lattice parameter (alat)  =      28.3459  a.u.
     number of atoms/cell      =            2

     crystal axes: (cart. coord. in units of alat)
               a(1) = (   1.000000   0.000000   0.000000 )
               a(2) = (   0.000000   1.000000   0.000000 )
               a(3) = (   0.000000   0.000000   1.000000 )

     Self-consistent Calculation
     convergence has been achieved in 9 iterations

     Begin final coordinates

     ATOMIC_POSITIONS (crystal)
     O                0.4592647035        0.5000000000        0.5000000000
     O                0.5407352965        0.5000000000        0.5000000000
     End final coordinates

!    total energy              =     -83.04485614 Ry
"""

    def test_positions_only_relax_extracts_lattice_from_preamble(self):
        from backend.common.engines.qe_run import parse_pw_output

        r = parse_pw_output(self._RELAX_FRAGMENT)
        assert r.relaxed is not None
        # 15 Å box: alat is 28.3459 a.u. ≈ 15 Å. Lattice should be ~15 on diagonal.
        assert r.relaxed.lattice_ang[0][0] == pytest.approx(15.0, abs=0.05)
        assert r.relaxed.lattice_ang[1][1] == pytest.approx(15.0, abs=0.05)
        assert r.relaxed.lattice_ang[2][2] == pytest.approx(15.0, abs=0.05)

    def test_positions_only_relax_computes_bond_length(self):
        """For the O₂ fragment, distance between the two atoms in Å."""
        import math
        from backend.common.engines.qe_run import parse_pw_output

        r = parse_pw_output(self._RELAX_FRAGMENT)
        c = r.relaxed.cart_coords_ang
        bond = math.sqrt(sum((c[1][i] - c[0][i]) ** 2 for i in range(3)))
        # 0.0815 * 15 Å ≈ 1.22 Å — PBE-typical O₂ bond.
        assert 1.15 < bond < 1.30


class TestParseErrored:
    def test_error_banner_marks_errored(self):
        from backend.common.engines.qe_run import (
            ConvergenceStatus,
            parse_pw_output,
        )

        result = parse_pw_output(FIXTURES / "error_wrong_pseudo.out")
        assert result.convergence == ConvergenceStatus.ERRORED

    def test_error_messages_collected(self):
        from backend.common.engines.qe_run import parse_pw_output

        result = parse_pw_output(FIXTURES / "error_wrong_pseudo.out")
        assert result.error_messages
        assert any("readpp" in msg for msg in result.error_messages)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestParseEdgeCases:
    def test_empty_input_raises(self):
        from backend.common.engines.qe_run import PWOutputParseError, parse_pw_output

        with pytest.raises(PWOutputParseError):
            parse_pw_output("")

    def test_parse_accepts_raw_text(self):
        """Can pass a text blob directly, not just a path."""
        from backend.common.engines.qe_run import (
            ConvergenceStatus,
            parse_pw_output,
        )

        text = (FIXTURES / "si_scf_converged.out").read_text()
        result = parse_pw_output(text)
        assert result.convergence == ConvergenceStatus.CONVERGED

    def test_missing_sections_are_none_not_raise(self):
        """A minimal output with only an energy marker should parse."""
        from backend.common.engines.qe_run import parse_pw_output

        minimal = """
     Program PWSCF starts
     convergence has been achieved in   1 iterations
!    total energy              =     -10.00000000 Ry
        """
        result = parse_pw_output(minimal)
        assert result.energy.total_ry == pytest.approx(-10.0)
        assert result.forces == []
        assert result.stress is None
        assert result.wall_time_seconds is None
