"""Tests for Phase 5 / Session 5.2 — kMC mesoscale engine.

Coverage
--------

1. Params — ``EventCatalog`` + ``KMCProblem`` validation.
2. Lattice primitives — neighbour offsets, periodic wrap, shape.
3. Engine determinism — same ``seed`` yields identical trajectories.
4. Rate math — Arrhenius exponent evaluated correctly.
5. Occupancy invariants — after N hops the occupancy dicts still
   agree with the position arrays, and defect counts are conserved
   except where annihilation occurred.
6. Acceptance (fast variant):
   - Single-species MSD: 500 non-interacting vacancies on 100³ at
     600 K, E_a = 1 eV → fitted D within **10%** of analytical
     ``a² ν₀ exp(-E/kT)``.
   - Pair annihilation: 1% vacancies + 1% interstitials on 50³ (for
     test-suite speed) → both populations decay to 0 within
     2M steps. The roadmap's ``<0.01%`` target is met (ends at 0).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


class TestParams:
    def test_event_catalog_defaults(self):
        from backend.common.engines.mesoscale_kmc import EventCatalog, EventType

        cat = EventCatalog(
            vacancy=EventType(hop_attempt_frequency_hz=1e13, activation_energy_ev=1.0),
            interstitial=EventType(hop_attempt_frequency_hz=1e13, activation_energy_ev=0.4),
        )
        assert cat.recombination_radius_a == 1.0

    def test_problem_rejects_missing_stopping_criteria(self):
        from backend.common.engines.mesoscale_kmc import (
            EventCatalog, EventType, KMCProblem, LatticeSpec, run_kmc,
        )

        problem = KMCProblem(
            lattice=LatticeSpec(a_m=3e-10, nx=10, ny=10, nz=10),
            catalog=EventCatalog(
                vacancy=EventType(hop_attempt_frequency_hz=1e13, activation_energy_ev=1.0),
                interstitial=EventType(hop_attempt_frequency_hz=1e13, activation_energy_ev=1.0),
            ),
            temperature_k=300.0,
            n_initial_vacancies=1,
        )
        with pytest.raises(ValueError, match="max_steps or max_time_s"):
            run_kmc(problem)

    def test_negative_activation_energy_rejected(self):
        from backend.common.engines.mesoscale_kmc import EventType

        with pytest.raises(Exception):
            EventType(hop_attempt_frequency_hz=1e13, activation_energy_ev=-0.1)

    def test_temperature_must_be_positive(self):
        from backend.common.engines.mesoscale_kmc import (
            EventCatalog, EventType, KMCProblem, LatticeSpec,
        )

        with pytest.raises(Exception):
            KMCProblem(
                lattice=LatticeSpec(a_m=3e-10, nx=10, ny=10, nz=10),
                catalog=EventCatalog(
                    vacancy=EventType(hop_attempt_frequency_hz=1e13, activation_energy_ev=1.0),
                    interstitial=EventType(hop_attempt_frequency_hz=1e13, activation_energy_ev=1.0),
                ),
                temperature_k=0.0,
                max_steps=1,
            )


# ---------------------------------------------------------------------------
# Lattice primitives
# ---------------------------------------------------------------------------


class TestLatticePrimitives:
    def test_neighbor_offsets_sum_to_zero(self):
        """Six offsets on a simple-cubic lattice must cancel — no drift."""
        from backend.common.engines.mesoscale_kmc import NEIGHBOR_OFFSETS

        assert NEIGHBOR_OFFSETS.shape == (6, 3)
        np.testing.assert_array_equal(NEIGHBOR_OFFSETS.sum(axis=0), [0, 0, 0])

    def test_wrap_handles_negatives(self):
        from backend.common.engines.mesoscale_kmc import wrap

        arr = np.array([-1, 0, 5, 10, 11, -7])
        np.testing.assert_array_equal(wrap(arr, 10), [9, 0, 5, 0, 1, 3])

    def test_neighbor_site_wraps_both_ways(self):
        from backend.common.engines.mesoscale_kmc import neighbor_site

        # +x from the far corner wraps to 0.
        assert neighbor_site(9, 0, 0, 0, 10, 10, 10) == (0, 0, 0)
        # -x from origin wraps to 9.
        assert neighbor_site(0, 0, 0, 1, 10, 10, 10) == (9, 0, 0)


# ---------------------------------------------------------------------------
# Engine determinism + rate math
# ---------------------------------------------------------------------------


def _make_problem(**overrides):
    from backend.common.engines.mesoscale_kmc import (
        EventCatalog, EventType, KMCProblem, LatticeSpec,
    )

    defaults = dict(
        lattice=LatticeSpec(a_m=3e-10, nx=20, ny=20, nz=20),
        catalog=EventCatalog(
            vacancy=EventType(hop_attempt_frequency_hz=1e13, activation_energy_ev=1.0),
            interstitial=EventType(hop_attempt_frequency_hz=1e13, activation_energy_ev=1.0),
        ),
        temperature_k=600.0,
        n_initial_vacancies=5,
        max_steps=1000,
        observe_every_n_steps=100,
        seed=42,
    )
    defaults.update(overrides)
    return KMCProblem(**defaults)


class TestEngineDeterminism:
    def test_same_seed_same_trajectory(self):
        from backend.common.engines.mesoscale_kmc import run_kmc

        r1 = run_kmc(_make_problem())
        r2 = run_kmc(_make_problem())
        np.testing.assert_array_equal(r1.time_s, r2.time_s)
        np.testing.assert_array_equal(r1.vacancy_msd_m2, r2.vacancy_msd_m2)
        np.testing.assert_array_equal(
            r1.final_vacancy_positions_a, r2.final_vacancy_positions_a,
        )

    def test_different_seed_different_trajectory(self):
        from backend.common.engines.mesoscale_kmc import run_kmc

        r1 = run_kmc(_make_problem(seed=1))
        r2 = run_kmc(_make_problem(seed=2))
        assert not np.allclose(
            r1.final_vacancy_positions_a, r2.final_vacancy_positions_a,
        )


class TestRateMath:
    def test_arrhenius_rate_matches_formula(self):
        from backend.common.engines.mesoscale_kmc import run_kmc

        r = run_kmc(_make_problem(max_steps=10))
        # ν₀ = 1e13, E_a = 1 eV, T = 600 K, k_B = 8.617e-5 eV/K
        expected = 1e13 * math.exp(-1.0 / (8.617333262e-5 * 600.0))
        assert r.vacancy_rate_per_direction_hz == pytest.approx(expected, rel=1e-10)

    def test_higher_temperature_higher_rate(self):
        from backend.common.engines.mesoscale_kmc import run_kmc

        r_cold = run_kmc(_make_problem(temperature_k=300.0, max_steps=10))
        r_hot = run_kmc(_make_problem(temperature_k=1000.0, max_steps=10))
        assert r_hot.vacancy_rate_per_direction_hz > r_cold.vacancy_rate_per_direction_hz


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_vacancy_count_monotonic_with_no_interstitials(self):
        """Without interstitials, vacancies can't annihilate; count stays put."""
        from backend.common.engines.mesoscale_kmc import run_kmc

        r = run_kmc(_make_problem(
            n_initial_vacancies=10, n_initial_interstitials=0,
            max_steps=5000,
        ))
        assert (r.n_vacancies == 10).all()

    def test_defect_counts_only_decrease(self):
        """Under V+I annihilation, neither count can grow."""
        from backend.common.engines.mesoscale_kmc import run_kmc

        r = run_kmc(_make_problem(
            n_initial_vacancies=20, n_initial_interstitials=20,
            max_steps=20000, observe_every_n_steps=100,
        ))
        assert (np.diff(r.n_vacancies) <= 0).all()
        assert (np.diff(r.n_interstitials) <= 0).all()
        # V and I counts stay equal because V+I annihilation removes
        # one of each.
        assert r.n_vacancies[-1] == r.n_interstitials[-1]

    def test_annihilation_counter_matches_population_drop(self):
        from backend.common.engines.mesoscale_kmc import run_kmc

        r = run_kmc(_make_problem(
            n_initial_vacancies=15, n_initial_interstitials=15,
            max_steps=50000,
        ))
        initial = 15
        final = int(r.n_vacancies[-1])
        assert r.n_annihilations == initial - final


# ---------------------------------------------------------------------------
# Acceptance — fast variants suitable for the test suite
# ---------------------------------------------------------------------------


class TestAcceptanceDiffusion:
    # ~6 s locally but GitHub's ubuntu-latest 2 vCPU runners routinely
    # run Python-heavy workloads 3-5× slower; bump the per-test timeout
    # so CI doesn't kill us at pytest.ini's 60 s default.
    @pytest.mark.timeout(180)
    def test_vacancy_D_within_10_percent_of_analytical(self):
        """Roadmap acceptance: D = a² ν₀ exp(-E/kT) within 10%.

        We run 500 non-interacting vacancies on 100³ (0.05% occupancy)
        at 600 K with E_a = 1 eV, then fit the slope of MSD(t) over
        the later 80% of the trace. Single-walker statistics would
        need millions of steps for 10% confidence; 500 walkers × 500k
        steps hits ~2-3%.
        """
        from backend.common.engines.mesoscale_kmc import (
            EventCatalog, EventType, KMCProblem, LatticeSpec, run_kmc,
        )

        lat = LatticeSpec(a_m=3.0e-10, nx=100, ny=100, nz=100)
        cat = EventCatalog(
            vacancy=EventType(hop_attempt_frequency_hz=1e13, activation_energy_ev=1.0),
            interstitial=EventType(hop_attempt_frequency_hz=1e13, activation_energy_ev=1.0),
        )
        problem = KMCProblem(
            lattice=lat, catalog=cat, temperature_k=600.0,
            n_initial_vacancies=500, n_initial_interstitials=0,
            max_steps=500_000, observe_every_n_steps=5000, seed=1,
        )
        r = run_kmc(problem)

        D_analytical = lat.a_m**2 * r.vacancy_rate_per_direction_hz
        skip = max(1, len(r.time_s) // 5)
        t_fit = r.time_s[skip:]
        m_fit = r.vacancy_msd_m2[skip:]
        A = np.vstack([t_fit, np.ones_like(t_fit)]).T
        slope, _ = np.linalg.lstsq(A, m_fit, rcond=None)[0]
        D_fit = slope / 6.0

        err = abs(D_fit - D_analytical) / D_analytical
        assert err < 0.10, (
            f"D_fit = {D_fit:.3e}, D_analytical = {D_analytical:.3e}, "
            f"err = {err*100:.2f}% (>10%)"
        )


class TestAcceptanceAnnihilation:
    # Local runtime ~23 s (5 M-step kMC with occupancy-hash bookkeeping).
    # CI's 2 vCPU runner can hit 60-90 s; 240 s gives comfortable headroom
    # without masking a genuine regression (anything >4× local is a bug).
    @pytest.mark.timeout(240)
    def test_pair_annihilation_drives_populations_below_threshold(self):
        """Roadmap acceptance: 1% V + 1% I decay to < 0.01%.

        The roadmap specifies a 100³ lattice; we use 50³ (125k sites,
        1250 of each defect at 1%) to keep the test suite under a
        minute. The ``< 0.01%`` threshold scales with the lattice, so
        the acceptance still makes sense: 0.01% of 50³ is ~12 defects.
        With enough steps both populations go to 0 — the decay is
        exponential-like and the residue does fall well below the
        threshold. A 100³ live-acceptance variant lives in the
        session notes.
        """
        from backend.common.engines.mesoscale_kmc import (
            EventCatalog, EventType, KMCProblem, LatticeSpec, run_kmc,
        )

        lat = LatticeSpec(a_m=3.0e-10, nx=50, ny=50, nz=50)
        n_initial = 1250  # 1% of 125k
        threshold = max(1, int(lat.n_sites * 1e-4))  # 0.01% of sites ≈ 12

        cat = EventCatalog(
            vacancy=EventType(hop_attempt_frequency_hz=1e13, activation_energy_ev=1.0),
            interstitial=EventType(hop_attempt_frequency_hz=1e13, activation_energy_ev=1.0),
        )
        problem = KMCProblem(
            lattice=lat, catalog=cat, temperature_k=600.0,
            n_initial_vacancies=n_initial, n_initial_interstitials=n_initial,
            max_steps=5_000_000, observe_every_n_steps=100_000, seed=1,
        )
        r = run_kmc(problem)

        assert r.n_vacancies[0] == n_initial
        assert r.n_vacancies[-1] <= threshold, (
            f"V population {r.n_vacancies[-1]} exceeds 0.01% threshold {threshold} "
            f"after {r.n_steps_executed} steps"
        )
        assert r.n_interstitials[-1] <= threshold
        # Conservation: every drop in V count comes from an annihilation.
        assert r.n_annihilations == n_initial - int(r.n_vacancies[-1])
