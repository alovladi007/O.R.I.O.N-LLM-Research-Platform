"""Strain-set generation for the 6-strain elastic-tensor scheme.

Convention
----------

We use the **Voigt notation** throughout: a 6-vector
``ε = (ε_xx, ε_yy, ε_zz, 2 ε_yz, 2 ε_xz, 2 ε_xy)``. The factor of 2
on the off-diagonal entries is the Voigt convention so that
``σ_i = Σ C_ij ε_j`` recovers the textbook strain-energy form
``U = ½ ε^T C ε`` without an extra factor of 2.

The deformation gradient ``F = I + ε_3x3`` is applied to the lattice
vectors as ``A_new = A_old · F^T`` (pymatgen's
:class:`Deformation` convention; see
``pymatgen.analysis.elasticity.strain.Strain``).

Six independent directions (the Voigt basis) cover the full strain
space; each is applied at two magnitudes (±1 %, ±2 %) so the linear-
elastic regime is well-sampled and the LSQ fit has 12 equations per
column of ``C``. The roadmap calls this "6-strain scheme (±1 %, ±2 %
isotropic + shear)" — implemented as 6 directions × 2 magnitudes,
plus their negative counterparts via signed magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# The six Voigt-basis strain directions. Each row is a unit-magnitude
# strain vector that picks out one Voigt component (1, 2, 3 are
# normal; 4, 5, 6 are engineering shears).
SIX_STRAIN_DIRECTIONS: np.ndarray = np.eye(6, dtype=np.float64)


@dataclass
class DeformationSpec:
    """One strained-cell prescription.

    Attributes
    ----------
    direction_index
        ``0..5`` — which Voigt-basis direction this strain perturbs.
    magnitude
        Signed strain magnitude (e.g. +0.01 = +1 %). Negative values
        give the symmetric counterpart used to centre the LSQ fit.
    voigt_strain
        The 6-vector (``unit_vector * magnitude``).
    """

    direction_index: int
    magnitude: float
    voigt_strain: np.ndarray  # shape (6,)


def voigt_strain_to_3x3(voigt: Sequence[float]) -> np.ndarray:
    """Inverse of the Voigt convention — build the 3×3 strain tensor.

    Recall the Voigt entries 4, 5, 6 carry a factor of 2 on the
    off-diagonals; the 3×3 tensor's off-diagonals are therefore
    ``voigt[i] / 2``.
    """
    v = np.asarray(voigt, dtype=np.float64).reshape(-1)
    if v.shape != (6,):
        raise ValueError(f"voigt strain must have 6 entries; got {v.shape}")
    eps = np.array([
        [v[0],     v[5] / 2, v[4] / 2],
        [v[5] / 2, v[1],     v[3] / 2],
        [v[4] / 2, v[3] / 2, v[2]],
    ], dtype=np.float64)
    return eps


def voigt_stress_from_3x3(sigma: np.ndarray) -> np.ndarray:
    """Pack a 3×3 Cauchy stress into the 6-vector Voigt form.

    No factor-of-2 here — stress is a "true" tensor; only strain
    needs the engineering convention.
    """
    s = np.asarray(sigma, dtype=np.float64).reshape(3, 3)
    return np.array([
        s[0, 0], s[1, 1], s[2, 2],
        s[1, 2], s[0, 2], s[0, 1],
    ], dtype=np.float64)


def generate_strain_set(
    *,
    magnitudes: Sequence[float] = (0.01, 0.02),
    include_negative: bool = True,
) -> List[DeformationSpec]:
    """Build the 6-strain × N-magnitude deformation list.

    With the default ``magnitudes=(0.01, 0.02)`` and
    ``include_negative=True``, returns 6 × 2 × 2 = **24** specs —
    six directions, each at ±1 % and ±2 %. The LSQ fit is over-
    determined by 4× per column of C, which gives a good check
    against numerical noise on the per-strain stress.

    The roadmap calls this "6-strain scheme (±1 %, ±2 %)"; we
    interpret the "+" and "-" as separate strain points so the fit
    can detect non-linearity (warning if the symmetric stresses
    disagree significantly).
    """
    if not magnitudes:
        raise ValueError("magnitudes must be non-empty")
    specs: List[DeformationSpec] = []
    signs = (-1.0, 1.0) if include_negative else (1.0,)
    for d in range(6):
        unit = SIX_STRAIN_DIRECTIONS[d]
        for mag in magnitudes:
            for s in signs:
                voigt = unit * (s * mag)
                specs.append(DeformationSpec(
                    direction_index=d,
                    magnitude=s * mag,
                    voigt_strain=voigt,
                ))
    return specs
