"""Phase 8 / Session 8.1 — elastic-tensor workflow.

Three layers, glued together by :func:`compute_elastic_tensor`:

1. :func:`generate_strain_set` — 6-strain scheme (±1%, ±2% isotropic
   + shear, total 6 deformation directions × 2 magnitudes per
   direction = 12 strained structures by default).
2. :class:`StressDriver` — protocol for "compute the Cauchy stress of
   a strained, then ionically-relaxed cell." A real implementation
   (``QEStressDriver``, Session 8.1b) wraps the Phase-3 QE runner;
   the test driver returns analytic stresses for cubic crystals via
   known C_ij so the acceptance check runs without pw.x.
3. :func:`fit_elastic_tensor` — least-squares solve for the Voigt
   6×6 ``C`` from collected (strain, stress) pairs, with derived
   moduli (Voigt / Reuss / Hill bulk + shear, Young's, Poisson,
   universal anisotropy A_U) and Born stability check.

Result is a :class:`ElasticTensorResult` — a dataclass carrying the
full 6×6 tensor + scalars, ready to be persisted as a structured
property by Phase 1's property-ingestion pipeline.
"""

from .driver import (
    QEStressDriverStub,
    StressDriver,
)
from .strain import (
    DeformationSpec,
    SIX_STRAIN_DIRECTIONS,
    generate_strain_set,
    voigt_strain_to_3x3,
    voigt_stress_from_3x3,
)
from .tensor import (
    ElasticTensorResult,
    born_stability,
    compute_elastic_tensor,
    derive_moduli,
    fit_elastic_tensor,
)

__all__ = [
    "DeformationSpec",
    "ElasticTensorResult",
    "QEStressDriverStub",
    "SIX_STRAIN_DIRECTIONS",
    "StressDriver",
    "born_stability",
    "compute_elastic_tensor",
    "derive_moduli",
    "fit_elastic_tensor",
    "generate_strain_set",
    "voigt_strain_to_3x3",
    "voigt_stress_from_3x3",
]
