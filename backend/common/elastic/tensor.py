"""Elastic-tensor LSQ fit + derived moduli + Born-stability check.

Math
----

For a linear-elastic solid in Voigt form:
    σ_i = Σ_j C_ij ε_j        (Hooke's law)

Given ``N`` strain points {ε^(k)} and their stresses {σ^(k)}, we
stack the equations row-by-row:

    [σ^(1)_1, ..., σ^(1)_6,  σ^(2)_1, ...]^T  =  [ε^(1) ⊗ I_6,
                                                  ε^(2) ⊗ I_6, ...] · vec(C)

and solve in the least-squares sense. Equivalently — and what the
code does — for each component ``i`` of stress we have ``N`` linear
equations in 6 unknowns (the row ``C_i:``), so we solve six
independent 6-column LSQ systems.

We **symmetrize** the result (``C ← (C + C^T) / 2``) because energy
conservation requires C to be symmetric; small numerical noise on
the strain set otherwise leaves an asymmetric residual.

Voigt / Reuss / Hill
--------------------

Bulk and shear moduli have isotropic averages from the elastic
tensor (Hill 1952). Voigt is an upper bound (uniform-strain
assumption); Reuss is a lower bound (uniform-stress); Hill is the
arithmetic mean — what's used in materials databases.

Born stability
--------------

A crystal is mechanically stable iff all eigenvalues of C (as a 6×6
matrix in Voigt form) are strictly positive. For cubic crystals
this is also equivalent to the Born criteria
``C11 > |C12|``, ``C11 + 2 C12 > 0``, ``C44 > 0``; we report both
forms in :class:`ElasticTensorResult.stability_diagnostics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .driver import StressDriver
from .strain import (
    DeformationSpec,
    generate_strain_set,
    voigt_stress_from_3x3,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ElasticTensorResult:
    """Full elastic-tensor result: 6×6 + scalars + diagnostics.

    Persisted as a structured Property by Phase 1.3's importer:
    ``c_voigt`` lands in a JSONB column; the scalars become indexed
    Float columns alongside ``born_stable`` for quick stability
    filters.
    """

    c_voigt: np.ndarray              # 6×6 in GPa
    bulk_voigt: float                # GPa
    bulk_reuss: float
    bulk_hill: float
    shear_voigt: float
    shear_reuss: float
    shear_hill: float
    youngs_modulus: float            # Hill, GPa
    poisson_ratio: float             # Hill, dimensionless
    universal_anisotropy: float      # A_U, dimensionless (0 = isotropic)
    born_stable: bool
    stability_diagnostics: Dict[str, Any] = field(default_factory=dict)
    # Diagnostic data — kept for provenance / debugging.
    n_strain_points: int = 0
    fit_residual_norm: float = 0.0   # Frobenius norm of residual stresses

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict for the property store."""
        d = {
            "c_voigt_gpa": self.c_voigt.tolist(),
            "bulk_voigt_gpa": self.bulk_voigt,
            "bulk_reuss_gpa": self.bulk_reuss,
            "bulk_hill_gpa": self.bulk_hill,
            "shear_voigt_gpa": self.shear_voigt,
            "shear_reuss_gpa": self.shear_reuss,
            "shear_hill_gpa": self.shear_hill,
            "youngs_modulus_gpa": self.youngs_modulus,
            "poisson_ratio": self.poisson_ratio,
            "universal_anisotropy": self.universal_anisotropy,
            "born_stable": self.born_stable,
            "stability_diagnostics": self.stability_diagnostics,
            "n_strain_points": self.n_strain_points,
            "fit_residual_norm_gpa": self.fit_residual_norm,
        }
        return d


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def fit_elastic_tensor(
    strain_points: List[np.ndarray],
    stress_points: List[np.ndarray],
) -> tuple[np.ndarray, float]:
    """Least-squares fit ``σ = C · ε`` over ``N`` (strain, stress) pairs.

    Both inputs are lists of length ``N`` of 6-vectors (Voigt form).
    Returns ``(C, residual_norm)`` where ``C`` is the symmetric
    6×6 GPa tensor.
    """
    N = len(strain_points)
    if N != len(stress_points):
        raise ValueError(
            f"strain ({N}) / stress ({len(stress_points)}) length mismatch"
        )
    if N < 6:
        raise ValueError(
            f"need at least 6 strain points to determine C; got {N}"
        )
    E = np.stack(strain_points, axis=0)  # (N, 6)
    S = np.stack(stress_points, axis=0)  # (N, 6)
    # Solve E · C^T = S  (each column of C^T is the row C_i: of C).
    C_T, residuals, rank, sv = np.linalg.lstsq(E, S, rcond=None)
    C = C_T.T
    # Symmetrize — energy conservation requires C = C^T.
    C = 0.5 * (C + C.T)
    # Residual = || E · C^T - S ||_F  (numpy lstsq's residuals are
    # sum-of-squares per output column when full-rank; we recompute
    # to handle the rank-deficient case uniformly).
    pred = E @ C.T
    residual_norm = float(np.linalg.norm(pred - S))
    return C, residual_norm


# ---------------------------------------------------------------------------
# Derived moduli
# ---------------------------------------------------------------------------


def derive_moduli(C: np.ndarray) -> Dict[str, float]:
    """Voigt / Reuss / Hill bulk + shear, Young's, Poisson, A_U.

    Uses the standard formulas (Anderson 1963; Ranganathan & Ostoja-
    Starzewski 2008 for A_U). All inputs in GPa, all outputs in GPa
    except the dimensionless poisson + A_U.
    """
    C = np.asarray(C, dtype=np.float64)
    if C.shape != (6, 6):
        raise ValueError(f"C must be 6x6; got {C.shape}")
    S = np.linalg.inv(C)  # compliance, GPa^-1

    # Voigt averages (uniform strain): use the C tensor.
    K_V = (
        (C[0, 0] + C[1, 1] + C[2, 2])
        + 2.0 * (C[0, 1] + C[1, 2] + C[0, 2])
    ) / 9.0
    G_V = (
        (C[0, 0] + C[1, 1] + C[2, 2])
        - (C[0, 1] + C[1, 2] + C[0, 2])
        + 3.0 * (C[3, 3] + C[4, 4] + C[5, 5])
    ) / 15.0

    # Reuss averages (uniform stress): use the S = C^-1 compliance.
    K_R = 1.0 / (
        (S[0, 0] + S[1, 1] + S[2, 2])
        + 2.0 * (S[0, 1] + S[1, 2] + S[0, 2])
    )
    G_R = 15.0 / (
        4.0 * (S[0, 0] + S[1, 1] + S[2, 2])
        - 4.0 * (S[0, 1] + S[1, 2] + S[0, 2])
        + 3.0 * (S[3, 3] + S[4, 4] + S[5, 5])
    )

    # Hill averages — arithmetic mean of Voigt + Reuss.
    K_H = 0.5 * (K_V + K_R)
    G_H = 0.5 * (G_V + G_R)

    # Young's modulus + Poisson ratio (Hill).
    if (3.0 * K_H + G_H) <= 0.0:
        E_H = float("nan")
        nu_H = float("nan")
    else:
        E_H = 9.0 * K_H * G_H / (3.0 * K_H + G_H)
        nu_H = (3.0 * K_H - 2.0 * G_H) / (2.0 * (3.0 * K_H + G_H))

    # Universal anisotropy A_U = 5 (G_V/G_R) + (K_V/K_R) - 6.
    # 0 for isotropic materials (G_V == G_R, K_V == K_R).
    if K_R != 0.0 and G_R != 0.0:
        A_U = 5.0 * (G_V / G_R) + (K_V / K_R) - 6.0
    else:
        A_U = float("nan")

    return {
        "bulk_voigt": float(K_V),
        "bulk_reuss": float(K_R),
        "bulk_hill": float(K_H),
        "shear_voigt": float(G_V),
        "shear_reuss": float(G_R),
        "shear_hill": float(G_H),
        "youngs_modulus": float(E_H),
        "poisson_ratio": float(nu_H),
        "universal_anisotropy": float(A_U),
    }


# ---------------------------------------------------------------------------
# Born stability
# ---------------------------------------------------------------------------


def born_stability(C: np.ndarray) -> tuple[bool, Dict[str, Any]]:
    """Eigenvalue check + cubic-specific Born criteria.

    Returns ``(stable, diagnostics)``. The eigenvalue check is
    crystal-system agnostic — a strict positivity of the eigenvalues
    of the Voigt 6×6 is necessary and sufficient.
    """
    C = np.asarray(C, dtype=np.float64)
    eigs = np.linalg.eigvalsh(C)  # symmetric → use eigvalsh
    min_eig = float(eigs.min())
    eig_stable = bool(min_eig > 0.0)
    diagnostics: Dict[str, Any] = {
        "min_eigenvalue_gpa": min_eig,
        "eigenvalues_gpa": eigs.tolist(),
        "eigenvalue_check": eig_stable,
    }
    # Cubic-specific Born criteria (Born & Huang 1954). These are
    # equivalent to the eigenvalue check for cubic systems but more
    # commonly cited in materials papers.
    c11, c12, c44 = C[0, 0], C[0, 1], C[3, 3]
    born_cubic = {
        "c11_gt_abs_c12": bool(c11 > abs(c12)),
        "c11_plus_2c12_gt_0": bool(c11 + 2.0 * c12 > 0.0),
        "c44_gt_0": bool(c44 > 0.0),
    }
    diagnostics["born_cubic_criteria"] = born_cubic
    diagnostics["all_born_cubic_pass"] = all(born_cubic.values())
    return eig_stable, diagnostics


# ---------------------------------------------------------------------------
# Top-level workflow
# ---------------------------------------------------------------------------


def compute_elastic_tensor(
    *,
    reference: object,
    driver: StressDriver,
    magnitudes: tuple = (0.01, 0.02),
) -> ElasticTensorResult:
    """End-to-end: generate strain set → call driver → fit C → derive moduli.

    The reference structure is opaque — the driver is responsible
    for applying the strain to it. The roadmap's "+ relax" step
    happens inside the driver between strain application and
    stress measurement; the engine here doesn't know or care.
    """
    specs = generate_strain_set(
        magnitudes=magnitudes, include_negative=True,
    )
    strains: List[np.ndarray] = []
    stresses: List[np.ndarray] = []
    for spec in specs:
        sigma = driver.compute_stress(spec=spec, reference=reference)
        sigma = np.asarray(sigma, dtype=np.float64)
        if sigma.shape == (3, 3):
            sigma_v = voigt_stress_from_3x3(sigma)
        elif sigma.shape == (6,):
            sigma_v = sigma
        else:
            raise ValueError(
                f"driver returned stress with shape {sigma.shape}; "
                "expected (3, 3) or (6,)"
            )
        strains.append(spec.voigt_strain)
        stresses.append(sigma_v)

    C, residual = fit_elastic_tensor(strains, stresses)
    moduli = derive_moduli(C)
    stable, diag = born_stability(C)
    return ElasticTensorResult(
        c_voigt=C,
        bulk_voigt=moduli["bulk_voigt"],
        bulk_reuss=moduli["bulk_reuss"],
        bulk_hill=moduli["bulk_hill"],
        shear_voigt=moduli["shear_voigt"],
        shear_reuss=moduli["shear_reuss"],
        shear_hill=moduli["shear_hill"],
        youngs_modulus=moduli["youngs_modulus"],
        poisson_ratio=moduli["poisson_ratio"],
        universal_anisotropy=moduli["universal_anisotropy"],
        born_stable=stable,
        stability_diagnostics=diag,
        n_strain_points=len(specs),
        fit_residual_norm=residual,
    )
