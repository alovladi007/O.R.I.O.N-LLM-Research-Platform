"""Rejection-free kMC (Gillespie / BKL) core for Session 5.2.

Loop outline
------------

    while not stopping:
        W_v = n_vac * 6 * r_vac
        W_i = n_int * 6 * r_int
        W = W_v + W_i
        dt = -ln(u1) / W
        pick (species, defect_index, direction) proportional to rate
        move defect; wrap periodic index; update unwrapped coord
        check annihilation: if new site holds the opposite species → remove both
        t += dt; step += 1

Since every defect of a given species has the same per-direction rate
(we don't model defect-defect anisotropy in the MVP), the sampling
reduces to (a) pick species with prob W_s/W, (b) pick a uniform
random defect within the species, (c) pick a uniform random direction.
This is equivalent to full BKL sampling but costs O(1) per step.

Outputs
-------

The engine returns a :class:`KMCResult` containing:

- ``time_s`` — physical time at each observation point (s).
- ``n_vacancies`` / ``n_interstitials`` — defect counts over time.
- ``vacancy_msd_m2`` — single-walker mean-square displacement in m²
  averaged across the surviving vacancies at each observation, using
  unwrapped coordinates (no periodic-image artefact).
- ``final_vacancy_positions_a`` — unwrapped positions of surviving
  vacancies in lattice units. Useful for diagnostics; not for stats.

Numerical sanity
----------------

At small rates (e.g. E_a=1 eV @ 600 K gives r ≈ 5×10⁴ Hz per
direction for ν₀=10¹³ Hz), a 100k-step run takes roughly
100000 / (6 × 5×10⁴) ≈ 0.3 s of physical time. That's long enough
to see linear MSD growth but short enough to keep the test suite fast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .lattice import NEIGHBOR_OFFSETS, N_DIRECTIONS
from .params import EventCatalog, KMCProblem, LatticeSpec


# Boltzmann constant in eV/K — matches the convention in
# backend.common.reports.md (Arrhenius analyzer).
_K_B_EV_PER_K = 8.617333262e-5


@dataclass
class KMCResult:
    """Return of :func:`run_kmc`.

    All arrays are aligned so ``time_s[k]`` corresponds to
    ``n_vacancies[k]`` / ``n_interstitials[k]`` / ``vacancy_msd_m2[k]``.
    """

    time_s: np.ndarray
    n_vacancies: np.ndarray
    n_interstitials: np.ndarray
    vacancy_msd_m2: np.ndarray
    final_vacancy_positions_a: np.ndarray  # (n_surviving, 3), unwrapped, in a
    final_interstitial_positions_a: np.ndarray
    n_steps_executed: int
    stopped_reason: str  # "max_steps" | "max_time" | "no_events"
    # Derived constants — helpful for the caller's analytical check.
    vacancy_rate_per_direction_hz: float
    interstitial_rate_per_direction_hz: float
    n_annihilations: int = 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_kmc(problem: KMCProblem) -> KMCResult:
    """Execute a Session 5.2 rejection-free kMC run.

    Raises :class:`ValueError` on malformed stopping criteria (neither
    ``max_steps`` nor ``max_time_s`` set).
    """
    if problem.max_steps is None and problem.max_time_s is None:
        raise ValueError(
            "KMCProblem needs at least one of max_steps or max_time_s"
        )

    rng = np.random.default_rng(problem.seed)

    # Resolved per-direction rates.
    T = problem.temperature_k
    r_vac = _rate(problem.catalog.vacancy, T)
    r_int = _rate(problem.catalog.interstitial, T)

    # State: per-defect wrapped integer coords, unwrapped float coords.
    # Plus occupancy hash for O(1) defect-at-site lookup — critical
    # for the annihilation acceptance where ~20k defects need fast
    # collision checks each step.
    lat = problem.lattice
    vac_w, vac_u = _seed_defects(
        rng, problem.n_initial_vacancies, lat, forbidden=None,
    )
    int_w, int_u = _seed_defects(
        rng, problem.n_initial_interstitials, lat, forbidden=_to_occ_set(vac_w),
    )
    vac_occ: Dict[Tuple[int, int, int], int] = {
        (int(r[0]), int(r[1]), int(r[2])): k for k, r in enumerate(vac_w)
    }
    int_occ: Dict[Tuple[int, int, int], int] = {
        (int(r[0]), int(r[1]), int(r[2])): k for k, r in enumerate(int_w)
    }

    # Observation buffers.
    obs_time: List[float] = []
    obs_n_vac: List[int] = []
    obs_n_int: List[int] = []
    obs_msd: List[float] = []

    # Keep the initial positions for MSD.
    initial_vac_u = vac_u.copy() if vac_u.size else vac_u

    t = 0.0
    step = 0
    annihilations = 0
    stopped_reason = "max_steps"
    _observe(
        obs_time, obs_n_vac, obs_n_int, obs_msd,
        t, vac_u, initial_vac_u, int_u, lat.a_m,
    )

    while True:
        if problem.max_steps is not None and step >= problem.max_steps:
            stopped_reason = "max_steps"
            break
        if problem.max_time_s is not None and t >= problem.max_time_s:
            stopped_reason = "max_time"
            break

        n_v = vac_w.shape[0]
        n_i = int_w.shape[0]
        W_v = n_v * N_DIRECTIONS * r_vac
        W_i = n_i * N_DIRECTIONS * r_int
        W = W_v + W_i
        if W <= 0:
            stopped_reason = "no_events"
            break

        # Time advance (Gillespie's first reaction method).
        u1 = rng.random()
        # ``u1`` ∈ [0, 1); guard against the zero corner.
        if u1 <= 0.0:
            u1 = np.finfo(np.float64).tiny
        dt = -math.log(u1) / W

        # Pick species → defect → direction.
        pick_is_vac = rng.random() < (W_v / W)
        if pick_is_vac:
            result = _hop_defect(
                vac_w, vac_u, vac_occ, int_w, int_u, int_occ,
                initial_vac_u, rng, lat, is_vacancy=True,
            )
        else:
            result = _hop_defect(
                int_w, int_u, int_occ, vac_w, vac_u, vac_occ,
                initial_vac_u, rng, lat, is_vacancy=False,
            )
        vac_w, vac_u, int_w, int_u, initial_vac_u, annihilated = result
        if annihilated:
            annihilations += 1

        t += dt
        step += 1
        if (
            problem.observe_every_n_steps > 0
            and step % problem.observe_every_n_steps == 0
        ):
            _observe(
                obs_time, obs_n_vac, obs_n_int, obs_msd,
                t, vac_u, initial_vac_u, int_u, lat.a_m,
            )

    # Always record the final frame (even if cadence didn't land here).
    if not obs_time or obs_time[-1] != t:
        _observe(
            obs_time, obs_n_vac, obs_n_int, obs_msd,
            t, vac_u, initial_vac_u, int_u, lat.a_m,
        )

    return KMCResult(
        time_s=np.asarray(obs_time, dtype=np.float64),
        n_vacancies=np.asarray(obs_n_vac, dtype=np.int64),
        n_interstitials=np.asarray(obs_n_int, dtype=np.int64),
        vacancy_msd_m2=np.asarray(obs_msd, dtype=np.float64),
        final_vacancy_positions_a=vac_u.astype(np.float64, copy=True),
        final_interstitial_positions_a=int_u.astype(np.float64, copy=True),
        n_steps_executed=step,
        stopped_reason=stopped_reason,
        vacancy_rate_per_direction_hz=r_vac,
        interstitial_rate_per_direction_hz=r_int,
        n_annihilations=annihilations,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rate(event, T_k: float) -> float:
    return event.hop_attempt_frequency_hz * math.exp(
        -event.activation_energy_ev / (_K_B_EV_PER_K * T_k)
    )


def _seed_defects(
    rng: np.random.Generator,
    count: int,
    lat: LatticeSpec,
    forbidden: Optional[set],
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ``count`` unique integer sites avoiding ``forbidden``.

    Returns (wrapped_int_sites shape (count, 3), unwrapped_float_sites).
    Unwrapped coords start equal to wrapped at t=0.
    """
    if count == 0:
        empty = np.zeros((0, 3), dtype=np.int64)
        return empty, empty.astype(np.float64)
    n_sites = lat.nx * lat.ny * lat.nz
    max_attempts = max(count * 10, 100)
    picked: List[Tuple[int, int, int]] = []
    tried = 0
    # Sample without replacement by rejection — at dilute concentrations
    # this is near-optimal. For dense starts a flat index shuffle would
    # be faster but we cap count at 10% of the lattice.
    while len(picked) < count and tried < max_attempts * count:
        ix = int(rng.integers(0, lat.nx))
        iy = int(rng.integers(0, lat.ny))
        iz = int(rng.integers(0, lat.nz))
        key = (ix, iy, iz)
        if forbidden is not None and key in forbidden:
            tried += 1
            continue
        if key in set(picked):  # small; O(count) worst case
            tried += 1
            continue
        picked.append(key)
    if len(picked) < count:
        raise RuntimeError(
            f"unable to seed {count} defects in {n_sites} sites "
            f"(got {len(picked)} after {tried} attempts)"
        )
    wrapped = np.asarray(picked, dtype=np.int64)
    return wrapped, wrapped.astype(np.float64)


def _to_occ_set(arr: np.ndarray) -> set:
    return {(int(r[0]), int(r[1]), int(r[2])) for r in arr}


def _hop_defect(
    own_w: np.ndarray,
    own_u: np.ndarray,
    own_occ: Dict[Tuple[int, int, int], int],
    other_w: np.ndarray,
    other_u: np.ndarray,
    other_occ: Dict[Tuple[int, int, int], int],
    initial_vac_u: np.ndarray,
    rng: np.random.Generator,
    lat: LatticeSpec,
    is_vacancy: bool,
):
    """Hop one defect of the chosen species; check annihilation with the other.

    Uses the ``own_occ`` / ``other_occ`` occupancy dicts for O(1)
    collision lookup so the annihilation acceptance run stays under
    a few seconds. The function owns the bookkeeping (index rewiring
    after swap-pop) so the main loop stays short.

    Returns ``(vac_w, vac_u, int_w, int_u, initial_vac_u, annihilated_bool)``
    — the caller is expected to rebind all five because ``swap_pop``
    returns views.
    """
    n = own_w.shape[0]
    idx = int(rng.integers(0, n))
    dir_ = int(rng.integers(0, N_DIRECTIONS))
    dx, dy, dz = NEIGHBOR_OFFSETS[dir_]

    old_key = (int(own_w[idx, 0]), int(own_w[idx, 1]), int(own_w[idx, 2]))
    new_ix = (own_w[idx, 0] + dx) % lat.nx
    new_iy = (own_w[idx, 1] + dy) % lat.ny
    new_iz = (own_w[idx, 2] + dz) % lat.nz
    new_key = (int(new_ix), int(new_iy), int(new_iz))

    # Same-species collision: another defect of *our* species is on
    # the target site. In the MVP we treat this as a blocked hop —
    # the event is rejected, time still advances (this is Gillespie,
    # not BKL-strict); the alternative is to remove that direction
    # from the rate sum which complicates bookkeeping.
    if new_key in own_occ and own_occ[new_key] != idx:
        if is_vacancy:
            return own_w, own_u, other_w, other_u, initial_vac_u, False
        else:
            return other_w, other_u, own_w, own_u, initial_vac_u, False

    # Execute the hop in the wrapped + unwrapped tables.
    own_w[idx, 0] = new_ix
    own_w[idx, 1] = new_iy
    own_w[idx, 2] = new_iz
    own_u[idx, 0] += dx
    own_u[idx, 1] += dy
    own_u[idx, 2] += dz
    # Occupancy dict: remove the old site, register the new.
    # (own_occ[old_key] is ``idx`` by invariant.)
    del own_occ[old_key]
    own_occ[new_key] = idx

    annihilated = False
    if new_key in other_occ:
        other_idx = other_occ[new_key]
        # Remove both defects — swap-pop on each table.
        del own_occ[new_key]
        del other_occ[new_key]

        # Before popping own[idx], note the last-row index; if we
        # swap it into slot idx, we must update the occupancy entry
        # that previously pointed to the last row.
        last = own_w.shape[0] - 1
        if idx != last:
            moved_key = (
                int(own_w[last, 0]), int(own_w[last, 1]), int(own_w[last, 2]),
            )
            own_w[idx] = own_w[last]
            own_u[idx] = own_u[last]
            own_occ[moved_key] = idx
        own_w = own_w[:last]
        own_u = own_u[:last]

        last_o = other_w.shape[0] - 1
        if other_idx != last_o:
            moved_key_o = (
                int(other_w[last_o, 0]),
                int(other_w[last_o, 1]),
                int(other_w[last_o, 2]),
            )
            other_w[other_idx] = other_w[last_o]
            other_u[other_idx] = other_u[last_o]
            other_occ[moved_key_o] = other_idx
        other_w = other_w[:last_o]
        other_u = other_u[:last_o]

        # initial_vac_u tracks the *vacancy* species' initial positions.
        # Shrink it in lockstep with whichever species is the vacancy
        # table above. The caller decides by ``is_vacancy`` argument.
        if is_vacancy:
            # vacancy was ``own`` — we already popped index idx.
            # Rebuild parallel slice of initial_vac_u.
            initial_vac_u = _shrink_initial(initial_vac_u, idx)
        else:
            # vacancy was ``other`` — we popped other_idx.
            initial_vac_u = _shrink_initial(initial_vac_u, other_idx)
        annihilated = True

    if is_vacancy:
        return own_w, own_u, other_w, other_u, initial_vac_u, annihilated
    return other_w, other_u, own_w, own_u, initial_vac_u, annihilated


def _shrink_initial(initial: np.ndarray, idx: int) -> np.ndarray:
    """Swap-pop on the ``initial_vac_u`` buffer to mirror vac_u shrinkage."""
    if initial.shape[0] == 0:
        return initial
    last = initial.shape[0] - 1
    if idx != last:
        initial[idx] = initial[last]
    return initial[:last]


def _swap_pop(table: np.ndarray, idx: int) -> np.ndarray:
    """Remove row ``idx`` by swapping with the last row and truncating.

    Returns a new array (view into the same buffer via slicing).
    """
    n = table.shape[0]
    if n == 0:
        return table
    if idx != n - 1:
        table[idx] = table[n - 1]
    return table[: n - 1]


def _observe(
    obs_time: List[float],
    obs_n_vac: List[int],
    obs_n_int: List[int],
    obs_msd: List[float],
    t: float,
    vac_u: np.ndarray,
    initial_vac_u: np.ndarray,
    int_u: np.ndarray,
    a_m: float,
) -> None:
    obs_time.append(t)
    obs_n_vac.append(int(vac_u.shape[0]))
    obs_n_int.append(int(int_u.shape[0]))
    if vac_u.shape[0] == 0 or initial_vac_u.shape[0] == 0:
        obs_msd.append(0.0)
        return
    d = (vac_u - initial_vac_u) * a_m
    sq = np.einsum("ij,ij->i", d, d)
    obs_msd.append(float(sq.mean()))
