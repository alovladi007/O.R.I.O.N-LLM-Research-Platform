"""Tests for Phase 8 / Session 8.1 — elastic-tensor workflow.

Covers
------

1. Voigt strain/stress conversions — the engineering-shear factor of
   2 lives in the strain side only; round-trip 3×3 ↔ Voigt 6.
2. Strain-set generation — 6 directions × 2 magnitudes × 2 signs =
   24 specs by default.
3. ``fit_elastic_tensor`` — recovers a known cubic ``C`` from
   noise-free synthetic stresses.
4. ``derive_moduli`` — closed-form check on an isotropic cubic
   (C11=C12+2 G, C44=G) reduces V/R/H to identical bulk + shear.
5. ``born_stability`` — eigenvalue check + cubic-criteria flagging.
6. **Acceptance**:
   - Al PBE-like ``C``: K_H within 10% of 78 GPa.
   - Si experimental ``C``: C11/C12/C44 within 10% of 165/64/79 GPa.
   - Fictitious unstable cubic flagged ``born_stable=False`` with
     a negative minimum eigenvalue.

The engine is driven by :class:`QEStressDriverStub` everywhere —
acceptance against a real ``pw.x`` is parked behind the ``live``
marker (Session 8.1b) so CI doesn't depend on a DFT executable.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cubic_C(c11: float, c12: float, c44: float) -> np.ndarray:
    """Build the 6×6 Voigt C of a cubic crystal."""
    C = np.zeros((6, 6), dtype=np.float64)
    for i in range(3):
        C[i, i] = c11
    C[0, 1] = C[1, 0] = c12
    C[0, 2] = C[2, 0] = c12
    C[1, 2] = C[2, 1] = c12
    for i in range(3, 6):
        C[i, i] = c44
    return C


# ---------------------------------------------------------------------------
# Voigt conventions
# ---------------------------------------------------------------------------


class TestVoigtConventions:
    def test_voigt_strain_to_3x3_factor_of_two(self):
        from backend.common.elastic import voigt_strain_to_3x3

        v = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        eps = voigt_strain_to_3x3(v)
        # Diagonal entries match.
        assert eps[0, 0] == pytest.approx(0.01)
        assert eps[1, 1] == pytest.approx(0.02)
        assert eps[2, 2] == pytest.approx(0.03)
        # Off-diagonals carry factor 1/2.
        assert eps[1, 2] == pytest.approx(0.04 / 2)
        assert eps[0, 2] == pytest.approx(0.05 / 2)
        assert eps[0, 1] == pytest.approx(0.06 / 2)

    def test_voigt_strain_wrong_length_raises(self):
        from backend.common.elastic import voigt_strain_to_3x3

        with pytest.raises(ValueError, match="6 entries"):
            voigt_strain_to_3x3([0.0, 0.0, 0.0])

    def test_voigt_stress_no_factor_of_two(self):
        from backend.common.elastic import voigt_stress_from_3x3

        s = np.array([
            [10.0, 4.0, 5.0],
            [4.0, 20.0, 6.0],
            [5.0, 6.0, 30.0],
        ])
        v = voigt_stress_from_3x3(s)
        # Stress packs in the symmetric off-diagonals directly.
        assert v.tolist() == [10.0, 20.0, 30.0, 6.0, 5.0, 4.0]


# ---------------------------------------------------------------------------
# Strain set
# ---------------------------------------------------------------------------


class TestStrainSet:
    def test_default_set_size(self):
        from backend.common.elastic import generate_strain_set

        specs = generate_strain_set()
        # 6 directions × 2 magnitudes × 2 signs = 24.
        assert len(specs) == 24
        # Each direction appears 4 times (2 mags × 2 signs).
        from collections import Counter
        counts = Counter(s.direction_index for s in specs)
        assert dict(counts) == {i: 4 for i in range(6)}

    def test_one_magnitude_one_sign(self):
        from backend.common.elastic import generate_strain_set

        specs = generate_strain_set(
            magnitudes=(0.01,), include_negative=False,
        )
        assert len(specs) == 6  # one per direction.

    def test_empty_magnitudes_raises(self):
        from backend.common.elastic import generate_strain_set

        with pytest.raises(ValueError, match="non-empty"):
            generate_strain_set(magnitudes=())

    def test_voigt_strain_only_perturbs_one_direction(self):
        from backend.common.elastic import generate_strain_set

        for s in generate_strain_set():
            v = s.voigt_strain
            # Every spec is a unit vector × magnitude — exactly one
            # nonzero entry equal to the magnitude.
            nonzero = np.flatnonzero(v)
            assert nonzero.tolist() == [s.direction_index]
            assert v[s.direction_index] == pytest.approx(s.magnitude)


# ---------------------------------------------------------------------------
# LSQ fit
# ---------------------------------------------------------------------------


class TestFit:
    def test_recovers_known_cubic_C_with_no_noise(self):
        from backend.common.elastic import (
            QEStressDriverStub, compute_elastic_tensor,
        )

        C_true = _cubic_C(200.0, 100.0, 50.0)
        driver = QEStressDriverStub(C_voigt=C_true, noise_std_gpa=0.0)
        result = compute_elastic_tensor(reference=None, driver=driver)
        # No noise → fit must be ~exact (modulo float64 LSQ floor).
        np.testing.assert_allclose(result.c_voigt, C_true, atol=1e-6)
        assert result.fit_residual_norm < 1e-6

    def test_fit_too_few_points_raises(self):
        from backend.common.elastic import fit_elastic_tensor

        with pytest.raises(ValueError, match="at least 6"):
            fit_elastic_tensor(
                [np.zeros(6)] * 3, [np.zeros(6)] * 3,
            )

    def test_fit_input_length_mismatch_raises(self):
        from backend.common.elastic import fit_elastic_tensor

        with pytest.raises(ValueError, match="length mismatch"):
            fit_elastic_tensor([np.zeros(6)] * 6, [np.zeros(6)] * 5)

    def test_fit_symmetric_output(self):
        from backend.common.elastic import (
            QEStressDriverStub, compute_elastic_tensor,
        )

        # Cubic ground-state has symmetric C trivially; check that
        # the symmetrization step kills any residual asymmetry from
        # noise.
        C_true = _cubic_C(150.0, 80.0, 40.0)
        driver = QEStressDriverStub(C_voigt=C_true, noise_std_gpa=0.5)
        result = compute_elastic_tensor(reference=None, driver=driver)
        np.testing.assert_allclose(
            result.c_voigt, result.c_voigt.T, atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Derived moduli
# ---------------------------------------------------------------------------


class TestModuli:
    def test_isotropic_VRH_collapse(self):
        """For a truly isotropic material (cubic with C11 - C12 = 2 C44),
        Voigt and Reuss averages coincide — so Hill = both. The
        universal anisotropy A_U should be 0.
        """
        from backend.common.elastic import derive_moduli

        # Pick C11 = 200, C12 = 100, then C44 = 50 → isotropy.
        c11, c12, c44 = 200.0, 100.0, 50.0
        # Isotropy condition: C11 - C12 = 2 C44.
        assert c11 - c12 == pytest.approx(2 * c44)
        m = derive_moduli(_cubic_C(c11, c12, c44))
        assert m["bulk_voigt"] == pytest.approx(m["bulk_reuss"])
        assert m["shear_voigt"] == pytest.approx(m["shear_reuss"])
        assert m["universal_anisotropy"] == pytest.approx(0.0, abs=1e-9)

    def test_bulk_voigt_formula_for_cubic(self):
        """K_V for cubic = (C11 + 2 C12) / 3."""
        from backend.common.elastic import derive_moduli

        c11, c12, c44 = 200.0, 100.0, 50.0
        m = derive_moduli(_cubic_C(c11, c12, c44))
        assert m["bulk_voigt"] == pytest.approx((c11 + 2 * c12) / 3.0)


# ---------------------------------------------------------------------------
# Born stability
# ---------------------------------------------------------------------------


class TestBornStability:
    def test_stable_cubic(self):
        from backend.common.elastic import born_stability

        stable, diag = born_stability(_cubic_C(165.0, 64.0, 79.0))
        assert stable is True
        assert diag["min_eigenvalue_gpa"] > 0
        assert diag["all_born_cubic_pass"] is True

    def test_unstable_when_c11_below_c12(self):
        from backend.common.elastic import born_stability

        # |C12| > C11 violates the Born cubic criterion.
        stable, diag = born_stability(_cubic_C(50.0, 80.0, 30.0))
        assert stable is False
        assert diag["born_cubic_criteria"]["c11_gt_abs_c12"] is False
        assert diag["min_eigenvalue_gpa"] < 0

    def test_unstable_when_c44_negative(self):
        from backend.common.elastic import born_stability

        # Mechanically unstable shear modulus.
        stable, diag = born_stability(_cubic_C(200.0, 100.0, -10.0))
        assert stable is False
        assert diag["born_cubic_criteria"]["c44_gt_0"] is False


# ---------------------------------------------------------------------------
# Roadmap acceptance — Al / Si / unstable
# ---------------------------------------------------------------------------


class TestAcceptance:
    """Acceptance against the known PBE-literature elastic constants.

    The acceptance is run against the :class:`QEStressDriverStub`
    seeded with the literature C — that's the standard way to check
    the **fit + analysis pipeline** without a live DFT engine. The
    "real DFT against a pw.x binary" check lives behind the ``live``
    marker (deferred to 8.1b once the QE driver is wired).
    """

    def test_al_bulk_modulus_within_10_percent_of_78_gpa(self):
        """Roadmap: Al (PBE) bulk modulus B_H within 10 % of 78 GPa.

        Al PBE elastic constants from the Materials Project /
        Wang et al. 2021 (doi:10.1038/s41524-021-00564-y):
        C11 ≈ 107, C12 ≈ 61, C44 ≈ 28 GPa. K_H ≈ 76 GPa, well within
        the 10 % bound around the experimental ~78 GPa.
        """
        from backend.common.elastic import (
            QEStressDriverStub, compute_elastic_tensor,
        )

        C_al = _cubic_C(107.0, 61.0, 28.0)
        driver = QEStressDriverStub(C_voigt=C_al, noise_std_gpa=0.05, seed=0)
        result = compute_elastic_tensor(reference=None, driver=driver)
        target = 78.0
        rel_err = abs(result.bulk_hill - target) / target
        assert rel_err <= 0.10, (
            f"Al K_H = {result.bulk_hill:.2f} GPa, target = {target} GPa, "
            f"relative error = {rel_err:.1%} (expected ≤ 10 %)"
        )
        assert result.born_stable is True

    def test_si_c11_c12_c44_within_10_percent(self):
        """Roadmap: Si C11/C12/C44 within 10 % of 165/64/79 GPa."""
        from backend.common.elastic import (
            QEStressDriverStub, compute_elastic_tensor,
        )

        C_si = _cubic_C(165.0, 64.0, 79.0)
        driver = QEStressDriverStub(C_voigt=C_si, noise_std_gpa=0.05, seed=0)
        result = compute_elastic_tensor(reference=None, driver=driver)

        c11_fit = result.c_voigt[0, 0]
        c12_fit = result.c_voigt[0, 1]
        c44_fit = result.c_voigt[3, 3]
        for name, fit, target in [
            ("C11", c11_fit, 165.0),
            ("C12", c12_fit, 64.0),
            ("C44", c44_fit, 79.0),
        ]:
            rel_err = abs(fit - target) / target
            assert rel_err <= 0.10, (
                f"Si {name} = {fit:.2f} GPa, target = {target} GPa, "
                f"relative error = {rel_err:.1%} (expected ≤ 10 %)"
            )
        assert result.born_stable is True

    def test_unstable_polymorph_flagged(self):
        """Roadmap: a known unstable structure (e.g., fictitious
        high-symmetry polymorph) flagged unstable.

        Construct a clearly-unstable cubic with negative C44 (shear
        instability) — the classic "fictitious high-symmetry parent"
        of a Jahn-Teller-distorted structure that wants to break
        symmetry along a shear direction.
        """
        from backend.common.elastic import (
            QEStressDriverStub, compute_elastic_tensor,
        )

        C_bad = _cubic_C(200.0, 80.0, -25.0)  # negative C44
        driver = QEStressDriverStub(C_voigt=C_bad, noise_std_gpa=0.05, seed=0)
        result = compute_elastic_tensor(reference=None, driver=driver)
        assert result.born_stable is False, (
            "expected born_stable=False for negative-C44 structure; "
            f"got C44_fit = {result.c_voigt[3, 3]:.2f}, "
            f"min_eig = {result.stability_diagnostics['min_eigenvalue_gpa']:.2f}"
        )
        assert result.stability_diagnostics["min_eigenvalue_gpa"] < 0
        assert (
            result.stability_diagnostics["born_cubic_criteria"]["c44_gt_0"]
            is False
        )
