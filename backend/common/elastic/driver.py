"""Stress-driver protocol for the elastic-tensor workflow.

The real driver wraps a DFT engine — strain the cell, run an ionic
relax (atoms-only, fixed lattice), then run a static SCF and parse
the stress tensor from the output. The Phase-3 QE runner already
exposes the stress, so the QE driver is mostly plumbing.

For tests we need a stress source that doesn't require ``pw.x``.
The :class:`QEStressDriverStub` returns analytic linear-elastic
stresses for a caller-supplied reference ``C`` matrix:

    σ_voigt = C · ε_voigt + noise

where ``noise`` is small Gaussian noise to mimic the per-strain
SCF error (~ 0.01 GPa on a converged calc, ~ 0.1 GPa on a
poorly-converged one). The acceptance tests use known PBE
literature values for ``C`` on Al, Si, and a deliberately-unstable
fictitious cubic phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from .strain import DeformationSpec, voigt_strain_to_3x3, voigt_stress_from_3x3


class StressDriver(Protocol):
    """Compute Cauchy stress for one strained, then ionically-relaxed cell.

    The driver receives the strain (Voigt 6-vector) and is expected
    to internally:

    1. Apply the deformation to the reference structure's lattice.
    2. Relax atomic positions with the lattice held fixed.
    3. Compute the Cauchy stress of the relaxed cell.

    Returns the **3×3** stress in GPa. The fitter packs it into Voigt
    form. Returning a 6-vector directly is also accepted (we sniff
    the shape).
    """

    def compute_stress(
        self,
        *,
        spec: DeformationSpec,
        reference: object,  # opaque structure handle
    ) -> np.ndarray: ...


@dataclass
class QEStressDriverStub:
    """Test driver: σ = C · ε + Gaussian noise.

    Parameters
    ----------
    C_voigt
        6×6 reference elastic tensor in GPa. The driver returns the
        linear-elastic stress this would produce for the supplied
        strain.
    noise_std_gpa
        Gaussian noise stddev added to each Voigt stress entry.
        Defaults to 0.02 GPa, which is the noise level a reasonably-
        converged QE static run produces on a cubic ground-state
        cell.
    seed
        RNG seed for reproducibility.
    """

    C_voigt: np.ndarray
    noise_std_gpa: float = 0.02
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.C_voigt = np.asarray(self.C_voigt, dtype=np.float64)
        if self.C_voigt.shape != (6, 6):
            raise ValueError(
                f"C_voigt must be 6x6; got {self.C_voigt.shape}"
            )
        self._rng = np.random.default_rng(self.seed)

    def compute_stress(
        self, *, spec: DeformationSpec, reference: object,
    ) -> np.ndarray:
        sigma_voigt = self.C_voigt @ spec.voigt_strain
        if self.noise_std_gpa > 0.0:
            sigma_voigt = sigma_voigt + self._rng.normal(
                0.0, self.noise_std_gpa, size=6,
            )
        # Return as 3x3 to match the expected real-driver shape.
        return np.array([
            [sigma_voigt[0], sigma_voigt[5], sigma_voigt[4]],
            [sigma_voigt[5], sigma_voigt[1], sigma_voigt[3]],
            [sigma_voigt[4], sigma_voigt[3], sigma_voigt[2]],
        ], dtype=np.float64)
