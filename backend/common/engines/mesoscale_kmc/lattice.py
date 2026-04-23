"""Simple-cubic lattice primitives for Session 5.2 kMC.

All sites are indexed by integer ``(ix, iy, iz)`` coordinates in
``[0, N_i)``. Periodic boundaries mean ``ix+nx ≡ ix``. For a rejection-
free engine we need cheap, allocation-free neighbor lookup: hence
the tuple of (±1 on one axis) offsets pre-baked into a constant.

We also carry an *unwrapped* coordinate per defect so MSD can be
computed without wraparound artefacts — the KMC analogue of the
Session 4.3b LAMMPS-dump ``xu yu zu`` fix. Unwrapped coords are
floats (in units of ``a_m``), wrapped coords are ints in
``[0, N_i)``; wrapping happens only when we test the occupancy lookup.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# 6 simple-cubic nearest-neighbour directions in (Δix, Δiy, Δiz) form.
NEIGHBOR_OFFSETS: np.ndarray = np.array(
    [
        (+1, 0, 0),
        (-1, 0, 0),
        (0, +1, 0),
        (0, -1, 0),
        (0, 0, +1),
        (0, 0, -1),
    ],
    dtype=np.int64,
)
N_DIRECTIONS = len(NEIGHBOR_OFFSETS)  # 6


def wrap(coord: np.ndarray, extent: int) -> np.ndarray:
    """Periodic wrap of an integer coordinate array to ``[0, extent)``.

    ``np.remainder`` handles negatives correctly for positive
    ``extent`` (result is always non-negative), so this is a one-
    liner; it exists as a named function for readability at call
    sites where ``% nx`` would be ambiguous.
    """
    return np.remainder(coord, extent)


def neighbor_site(
    ix: int, iy: int, iz: int, direction: int,
    nx: int, ny: int, nz: int,
) -> Tuple[int, int, int]:
    """Return the wrapped neighbour of ``(ix, iy, iz)`` in the given direction.

    ``direction`` indexes :data:`NEIGHBOR_OFFSETS`. The result is
    always in ``[0, N_i)`` — callers can use it directly as a dict /
    array key.
    """
    dx, dy, dz = NEIGHBOR_OFFSETS[direction]
    return (
        int((ix + dx) % nx),
        int((iy + dy) % ny),
        int((iz + dz) % nz),
    )
