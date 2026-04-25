"""ForceDriver protocol + a harmonic-spring test stub.

The real driver wraps Phase-3's QE runner: take the displaced
supercell, run an ``scf`` calculation (positions held fixed — phonons
are computed from the *unrelaxed* displaced cell, the relaxation is
of the equilibrium reference *before* displacement), parse the
per-atom forces from the pw.out, return as ``(n_atoms, 3)`` in eV/Å.

The :class:`HarmonicSpringDriver` returns analytic forces from a
nearest-neighbor central-force model. For each displaced atom *i*
with displacement vector ``u_i``, the force on every NN atom *j*
sharing a bond with *i* is

    F_j = +K (u_i · r̂_ij) r̂_ij

and the reaction force on the displaced atom itself is the negative
sum. This is exactly the linear-response of a network of radial
springs and gives **physically correct phonon dispersions** when
``K`` is tuned to the system's known zone-center frequency.

Why not random / mock forces? Because phonopy's force-constants
solver assumes the supplied forces come from a self-consistent
linear-elastic model. Random forces produce non-positive-definite
force-constant matrices and the resulting "phonon" frequencies are
nonsense. The harmonic-spring driver gives the test suite a
**deterministic, physically-meaningful** force source — exactly
what the acceptance test needs to verify the engine plumbing
(displacement → force → fc → frequencies) without a DFT engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Tuple

import numpy as np


class ForceDriver(Protocol):
    """Compute atomic forces on a displaced supercell.

    The driver receives the **same** PhonopyAtoms supercell and one
    displacement entry from ``ph.dataset['first_atoms']``. It returns
    an ``(n_atoms, 3)`` Cartesian force array in eV/Å.
    """

    def compute_forces(
        self, *, supercell: Any, displacement: dict,
    ) -> np.ndarray: ...


@dataclass
class HarmonicSpringDriver:
    """Nearest-neighbor central-force model.

    Parameters
    ----------
    spring_constant_ev_per_a2
        ``K`` in eV/Å². Tune to land the LO/TO or Debye temperature
        of interest. Cubic Si LO/TO at Γ ≈ 15.5 THz → K ≈ 10.35.
        FCC Cu Debye temperature ≈ 343 K → K ≈ 2.0.
    n_neighbors
        Number of nearest neighbors per atom included in the spring
        network. Diamond uses 4 (covalent), FCC uses 12, BCC uses 8.
    species_overrides
        Optional ``{(Z_i, Z_j): K_ij}`` map for binary / ternary
        crystals where individual bond stiffnesses differ. Keys can
        be in either order; we look both up.
    """

    spring_constant_ev_per_a2: float
    n_neighbors: int = 4
    species_overrides: Optional[dict] = None
    _nbr_cache: dict = field(default_factory=dict, repr=False)

    def compute_forces(
        self, *, supercell, displacement: dict,
    ) -> np.ndarray:
        positions = np.asarray(supercell.positions, dtype=np.float64)
        cell = np.asarray(supercell.cell, dtype=np.float64)
        try:
            atomic_numbers = np.asarray(supercell.numbers, dtype=int)
        except AttributeError:
            atomic_numbers = np.array(
                [getattr(a, "Z", 0) for a in supercell],
                dtype=int,
            )
        nbr_list = self._neighbors(positions, cell)
        n = positions.shape[0]
        forces = np.zeros((n, 3), dtype=np.float64)
        i = int(displacement["number"])
        u = np.asarray(displacement["displacement"], dtype=np.float64)
        for d, j, dr in nbr_list[i]:
            rhat = dr / d
            delta_l = float(np.dot(u, rhat))
            K = self._k_for_pair(
                atomic_numbers[i], atomic_numbers[j],
            )
            f = K * delta_l * rhat
            forces[i] -= f
            forces[j] += f
        return forces

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _k_for_pair(self, z_i: int, z_j: int) -> float:
        if not self.species_overrides:
            return self.spring_constant_ev_per_a2
        for key in ((z_i, z_j), (z_j, z_i)):
            if key in self.species_overrides:
                return float(self.species_overrides[key])
        return self.spring_constant_ev_per_a2

    def _neighbors(
        self, positions: np.ndarray, cell: np.ndarray,
    ) -> List[List[Tuple[float, int, np.ndarray]]]:
        """3×3×3 PBC NN list, cached on (positions.tobytes, cell.tobytes)."""
        key = (positions.tobytes(), cell.tobytes(), self.n_neighbors)
        cached = self._nbr_cache.get(key)
        if cached is not None:
            return cached
        n = positions.shape[0]
        # 27-image set; suffices for any cubic lattice with
        # supercell ≥ 2× the NN distance.
        images = [
            ix * cell[0] + iy * cell[1] + iz * cell[2]
            for ix in (-1, 0, 1) for iy in (-1, 0, 1) for iz in (-1, 0, 1)
        ]
        out: List[List[Tuple[float, int, np.ndarray]]] = []
        for i in range(n):
            ds: List[Tuple[float, int, np.ndarray]] = []
            for j in range(n):
                for shift in images:
                    if i == j and not shift.any():
                        continue
                    dr = positions[j] + shift - positions[i]
                    d = float(np.linalg.norm(dr))
                    if d < 1e-9:
                        continue
                    ds.append((d, j, dr))
            ds.sort(key=lambda x: x[0])
            # First k *unique-radius* shell members; same radius ties
            # all kept so cubic symmetry isn't broken.
            if not ds:
                out.append([])
                continue
            cutoff = ds[min(self.n_neighbors, len(ds)) - 1][0]
            out.append([d for d in ds if d[0] <= cutoff + 1e-6])
        self._nbr_cache[key] = out
        return out
