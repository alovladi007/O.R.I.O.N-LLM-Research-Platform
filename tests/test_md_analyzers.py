"""Tests for Phase 4 / Session 4.3b — MD aggregate analyzers.

Synthetic step_outputs so the unit tests don't need lmp. Live LAMMPS
acceptance runs live in :mod:`tests.test_lammps_live_acceptance` —
they're skipped unless ``ORION_LMP_PATH`` is set.

Coverage
--------

- :func:`detect_melting_point` — clean transitions, edge-bracket
  cases, missing fields, under-length sweeps.
- :func:`arrhenius_fit` — synthetic Arrhenius data with known E_a,
  fit-quality thresholds, bad-input paths (D≤0, missing T).
- :func:`fit_elastic_constants` — isotropic cubic synthetic sweep
  with a known C_11, axis-missing tolerance, shear-axis pass-through,
  sign-convention sanity.
- :func:`_lstsq` — degenerate-input error, slope/intercept/R² math.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _lstsq
# ---------------------------------------------------------------------------


class TestLstsq:
    def test_perfect_line(self):
        from backend.common.reports.md import _lstsq

        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [1.0, 3.0, 5.0, 7.0]  # y = 2x + 1
        slope, intercept, r2 = _lstsq(xs, ys)
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(1.0)
        assert r2 == pytest.approx(1.0)

    def test_noisy_line_r2_below_one(self):
        from backend.common.reports.md import _lstsq

        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [1.0, 3.1, 4.9, 7.2]
        _slope, _intercept, r2 = _lstsq(xs, ys)
        assert 0.9 < r2 < 1.0

    def test_zero_variance_x_raises(self):
        from backend.common.reports.md import AnalyzerInputError, _lstsq

        with pytest.raises(AnalyzerInputError, match="zero variance"):
            _lstsq([1.0, 1.0, 1.0], [2.0, 3.0, 4.0])

    def test_too_few_points_raises(self):
        from backend.common.reports.md import AnalyzerInputError, _lstsq

        with pytest.raises(AnalyzerInputError):
            _lstsq([1.0], [2.0])


# ---------------------------------------------------------------------------
# detect_melting_point
# ---------------------------------------------------------------------------


def _make_sweep(points):
    """Helper: build step_outputs dict from a list of (T, TotEng, MSD[, P, V]) tuples."""
    step_outputs = {}
    for i, pt in enumerate(points):
        if len(pt) == 3:
            t, e, msd = pt
            p = None
            v = None
        elif len(pt) == 5:
            t, e, msd, p, v = pt
        else:
            raise ValueError("need 3 or 5 entries per point")
        final = {}
        if e is not None:
            final["TotEng"] = e
        if p is not None:
            final["Press"] = p
        if v is not None:
            final["Volume"] = v
        step_outputs[f"sweep__{int(t)}"] = {
            "temperature_k": t,
            "msd_final_ang2": msd,
            "diffusion_coefficient_ang2_per_ps": msd / 10.0 if msd else None,
            "final_thermo": final,
        }
    return step_outputs


class TestDetectMeltingPoint:
    def test_clean_melting_high_confidence(self):
        from backend.common.reports import detect_melting_point

        # Solid: MSD ~0.1 Å² with slow E rise. Liquid at T >= 1400: MSD
        # jumps 15×, energy jumps sharply. Pre-melt E slope ~2 meV/K,
        # post-melt ~30 meV/K (ratio > _DEDT_JUMP_THRESHOLD=2).
        step_outputs = _make_sweep([
            (1000, -4.00, 0.10),
            (1100, -3.90, 0.12),
            (1200, -3.80, 0.15),
            (1300, -3.70, 0.18),
            (1400, -0.70, 3.0),
            (1500, 0.00, 6.0),
        ])
        r = detect_melting_point(step_outputs)
        assert r.detected_melting_point_k is not None
        # Bracket is [1300, 1400] → midpoint 1350.
        assert r.detected_melting_point_k == pytest.approx(1350.0)
        assert r.detection_confidence == "high"
        assert "msd_jump" in r.detection_method

    def test_medium_confidence_when_only_msd_fires(self):
        from backend.common.reports import detect_melting_point

        # MSD jumps at 1400 but total energy is smooth (no enthalpy
        # discontinuity flagged) — medium confidence.
        step_outputs = _make_sweep([
            (1000, -4.0, 0.10),
            (1100, -3.9, 0.12),
            (1200, -3.8, 0.15),
            (1300, -3.7, 0.18),
            (1400, -3.6, 3.0),   # MSD jumps 16×, E slope unchanged
            (1500, -3.5, 6.0),
        ])
        r = detect_melting_point(step_outputs)
        assert r.detected_melting_point_k == pytest.approx(1350.0)
        assert r.detection_confidence == "medium"

    def test_no_transition_returns_none(self):
        from backend.common.reports import detect_melting_point

        # Monotonic but no jump — a fully-solid sweep.
        step_outputs = _make_sweep([
            (300, -4.0, 0.01),
            (400, -3.95, 0.015),
            (500, -3.90, 0.02),
            (600, -3.85, 0.025),
        ])
        r = detect_melting_point(step_outputs)
        assert r.detected_melting_point_k is None
        assert r.detection_method == "no_transition_detected"

    def test_rejects_empty_outputs(self):
        from backend.common.reports import AnalyzerInputError, detect_melting_point

        with pytest.raises(AnalyzerInputError):
            detect_melting_point({})

    def test_rejects_under_length_sweep(self):
        from backend.common.reports import AnalyzerInputError, detect_melting_point

        # One step only — can't compute ratios.
        with pytest.raises(AnalyzerInputError, match="at least 2"):
            detect_melting_point({"only": {"temperature_k": 500, "msd_final_ang2": 0.1}})

    def test_missing_temperature_raises(self):
        from backend.common.reports import AnalyzerInputError, detect_melting_point

        step_outputs = {
            "a": {"msd_final_ang2": 0.1},
            "b": {"temperature_k": 500, "msd_final_ang2": 0.1},
        }
        with pytest.raises(AnalyzerInputError, match="temperature_k"):
            detect_melting_point(step_outputs)

    def test_enthalpy_from_pressure_volume(self):
        from backend.common.reports import detect_melting_point

        # Feed P + V so the analyzer can compute H = E + PV.
        # PV correction at 1 bar · 1000 Å³ ≈ 6e-4 eV — tiny, but the
        # enthalpies_ev list should differ from total_energies_ev.
        step_outputs = _make_sweep([
            (300, -4.00, 0.10, 1.0, 1000.0),
            (400, -3.95, 0.12, 1.0, 1010.0),
            (500, -3.90, 0.15, 1.0, 1020.0),
        ])
        r = detect_melting_point(step_outputs)
        for e, h in zip(r.total_energies_ev, r.enthalpies_ev):
            assert e is not None and h is not None
            assert h != e  # correction applied
            assert abs(h - e) < 0.01  # but small

    def test_unsorted_input_is_sorted(self):
        from backend.common.reports import detect_melting_point

        # Insertion order reversed; analyzer must sort by T.
        step_outputs = _make_sweep([
            (1500, -3.5, 6.0),
            (1000, -4.0, 0.1),
            (1200, -3.8, 0.15),
            (1400, -0.7, 3.0),
        ])
        r = detect_melting_point(step_outputs)
        assert r.temperatures_k == sorted(r.temperatures_k)


# ---------------------------------------------------------------------------
# arrhenius_fit
# ---------------------------------------------------------------------------


def _make_arrhenius(ts, d0, ea_ev, noise=None):
    """Synthesize step_outputs with D(T) = d0 * exp(-E_a / k_B T)."""
    k_b = 8.617333262e-5
    outputs = {}
    for i, t in enumerate(ts):
        d = d0 * math.exp(-ea_ev / (k_b * t))
        if noise:
            d *= noise[i]
        outputs[f"sweep__{int(t)}"] = {
            "temperature_k": t,
            "diffusion_coefficient_ang2_per_ps": d,
        }
    return outputs


class TestArrheniusFit:
    def test_recovers_known_activation_energy(self):
        from backend.common.reports import arrhenius_fit

        # Synthesize with E_a = 0.5 eV, D_0 = 100 Å²/ps, no noise.
        ts = [400, 500, 600, 700, 800, 900, 1000]
        outputs = _make_arrhenius(ts, d0=100.0, ea_ev=0.5)
        r = arrhenius_fit(outputs)
        assert r.activation_energy_ev == pytest.approx(0.5, abs=1e-6)
        assert r.prefactor_ang2_per_ps == pytest.approx(100.0, rel=1e-6)
        assert r.r_squared == pytest.approx(1.0, abs=1e-9)
        assert r.fit_quality == "good"

    def test_fit_quality_thresholds(self):
        from backend.common.reports import arrhenius_fit

        # Heavier noise to push R² below the "good" threshold. Multi-
        # order-of-magnitude swings on D in a short T range break the
        # linear ln D vs 1/T relationship.
        ts = [400, 500, 600, 700, 800]
        noise = [3.0, 0.3, 4.0, 0.25, 2.5]
        outputs = _make_arrhenius(ts, d0=100.0, ea_ev=0.5, noise=noise)
        r = arrhenius_fit(outputs, min_r_squared_good=0.99, min_r_squared_fair=0.50)
        assert 0.50 <= r.r_squared < 0.99
        assert r.fit_quality == "fair"

    def test_low_r_squared_labeled_poor(self):
        from backend.common.reports import arrhenius_fit

        # Deliberately non-Arrhenius data: near-constant D → slope ~ 0,
        # R² drops toward zero.
        outputs = {
            "a": {"temperature_k": 400, "diffusion_coefficient_ang2_per_ps": 1.05},
            "b": {"temperature_k": 500, "diffusion_coefficient_ang2_per_ps": 0.95},
            "c": {"temperature_k": 600, "diffusion_coefficient_ang2_per_ps": 1.02},
            "d": {"temperature_k": 700, "diffusion_coefficient_ang2_per_ps": 1.01},
            "e": {"temperature_k": 800, "diffusion_coefficient_ang2_per_ps": 0.98},
        }
        r = arrhenius_fit(outputs)
        assert r.fit_quality == "poor"

    def test_drops_zero_and_negative_D(self):
        from backend.common.reports import arrhenius_fit

        ts = [400, 500, 600, 700, 800]
        outputs = _make_arrhenius(ts, d0=100.0, ea_ev=0.5)
        # Clobber two points with invalid D values.
        outputs["sweep__400"]["diffusion_coefficient_ang2_per_ps"] = 0.0
        outputs["sweep__500"]["diffusion_coefficient_ang2_per_ps"] = -0.1
        r = arrhenius_fit(outputs)
        # Fit succeeds on the remaining 3 points.
        assert r.activation_energy_ev == pytest.approx(0.5, abs=1e-6)
        assert r.fit_quality == "good"

    def test_insufficient_valid_points_raises(self):
        from backend.common.reports import AnalyzerInputError, arrhenius_fit

        outputs = {
            "a": {"temperature_k": 400, "diffusion_coefficient_ang2_per_ps": 0.0},
            "b": {"temperature_k": 500, "diffusion_coefficient_ang2_per_ps": None},
            "c": {"temperature_k": 600, "diffusion_coefficient_ang2_per_ps": 0.1},
        }
        with pytest.raises(AnalyzerInputError, match="at least 2"):
            arrhenius_fit(outputs)

    def test_missing_temperature_raises(self):
        from backend.common.reports import AnalyzerInputError, arrhenius_fit

        outputs = {"a": {"diffusion_coefficient_ang2_per_ps": 0.1}}
        with pytest.raises(AnalyzerInputError, match="temperature_k"):
            arrhenius_fit(outputs)

    def test_rejects_empty(self):
        from backend.common.reports import AnalyzerInputError, arrhenius_fit

        with pytest.raises(AnalyzerInputError):
            arrhenius_fit({})

    def test_report_lists_are_sorted(self):
        from backend.common.reports import arrhenius_fit

        outputs = _make_arrhenius([800, 400, 600], d0=10.0, ea_ev=0.3)
        r = arrhenius_fit(outputs)
        assert r.temperatures_k == sorted(r.temperatures_k)


# ---------------------------------------------------------------------------
# fit_elastic_constants
# ---------------------------------------------------------------------------


def _make_strain_sweep(c_target_gpa: float, strains=(-0.005, -0.002, 0.0, 0.002, 0.005)):
    """Synthesize a ±ε sweep along all three diagonal axes with stress = C·ε.

    Stress is returned in the runner's convention: ``final_thermo.Pxx`` in
    bar with compressive-positive sign; the analyzer flips sign + rescales
    bar → GPa internally, so we emit ``P = -σ / bar_to_gpa``.
    """
    outputs = {}
    bar_to_gpa = 1.0e-4
    for vi in (0, 1, 2):
        p_key = ("Pxx", "Pyy", "Pzz")[vi]
        for s in strains:
            sigma_gpa = c_target_gpa * s
            p_bar = -sigma_gpa / bar_to_gpa  # back to LAMMPS' sign
            sid = f"strain_v{vi}_{s:+.4f}"
            outputs[sid] = {
                "strain_voigt": vi,
                "strain_value": s,
                "final_thermo": {p_key: p_bar},
            }
    return outputs


class TestFitElasticConstants:
    def test_recovers_cubic_c11(self):
        from backend.common.reports import fit_elastic_constants

        outputs = _make_strain_sweep(c_target_gpa=108.0)
        r = fit_elastic_constants(outputs)
        for c, r2 in zip(
            (r.c11_gpa, r.c22_gpa, r.c33_gpa),
            (r.c11_r_squared, r.c22_r_squared, r.c33_r_squared),
        ):
            assert c == pytest.approx(108.0, abs=1e-6)
            assert r2 == pytest.approx(1.0, abs=1e-9)

    def test_missing_axis_yields_none_for_that_C(self):
        from backend.common.reports import fit_elastic_constants

        outputs = _make_strain_sweep(c_target_gpa=150.0)
        # Remove all Pyy entries.
        for sid in list(outputs):
            if "_v1_" in sid:
                del outputs[sid]
        r = fit_elastic_constants(outputs)
        assert r.c11_gpa == pytest.approx(150.0, abs=1e-6)
        assert r.c22_gpa is None
        assert r.c22_r_squared is None
        assert r.c33_gpa == pytest.approx(150.0, abs=1e-6)

    def test_single_axis_enough_to_not_raise(self):
        from backend.common.reports import fit_elastic_constants

        outputs = _make_strain_sweep(c_target_gpa=200.0)
        # Keep only the c11 axis.
        for sid in list(outputs):
            if "_v0_" not in sid:
                del outputs[sid]
        r = fit_elastic_constants(outputs)
        assert r.c11_gpa == pytest.approx(200.0, abs=1e-6)
        assert r.c22_gpa is None

    def test_rejects_no_valid_axis(self):
        from backend.common.reports import AnalyzerInputError, fit_elastic_constants

        outputs = {
            "only_shear": {
                "strain_voigt": 3, "strain_value": 0.01,
                "final_thermo": {"Pxx": 100.0},
            },
        }
        with pytest.raises(AnalyzerInputError, match="no diagonal axis"):
            fit_elastic_constants(outputs)

    def test_missing_strain_marker_raises(self):
        from backend.common.reports import AnalyzerInputError, fit_elastic_constants

        outputs = {"bad": {"final_thermo": {"Pxx": 0.0}}}
        with pytest.raises(AnalyzerInputError, match="strain_voigt|strain_value"):
            fit_elastic_constants(outputs)

    def test_sign_convention_tension_positive(self):
        from backend.common.reports import fit_elastic_constants

        # Positive strain → tension, σ > 0 in report. Positive C.
        outputs = {
            "pos": {
                "strain_voigt": 0, "strain_value": +0.005,
                "final_thermo": {"Pxx": -1000.0},  # Pxx=-1000 bar → σ=+0.1 GPa
            },
            "neg": {
                "strain_voigt": 0, "strain_value": -0.005,
                "final_thermo": {"Pxx": +1000.0},  # Pxx=+1000 bar → σ=-0.1 GPa
            },
        }
        r = fit_elastic_constants(outputs)
        assert r.c11_gpa is not None
        assert r.c11_gpa > 0
        # Slope σ/ε = 0.2 GPa / 0.01 strain = 20 GPa
        assert r.c11_gpa == pytest.approx(20.0, abs=1e-6)

    def test_shear_strain_pass_through_but_no_fit(self):
        from backend.common.reports import fit_elastic_constants

        outputs = _make_strain_sweep(c_target_gpa=100.0)
        outputs["shear"] = {
            "strain_voigt": 3, "strain_value": 0.01,
            "final_thermo": {"Pxx": 0.0},
        }
        r = fit_elastic_constants(outputs)
        # Shear entry recorded in strains_voigt / voigt_indices but with
        # None stress and no C fitted for index 3.
        assert 3 in r.voigt_indices
        assert r.c11_gpa == pytest.approx(100.0, abs=1e-6)

    def test_rejects_empty(self):
        from backend.common.reports import AnalyzerInputError, fit_elastic_constants

        with pytest.raises(AnalyzerInputError):
            fit_elastic_constants({})
