"""Defect specification dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence, Tuple


class DefectKind(str, Enum):
    """Categorical defect type. Drives stoichiometry bookkeeping in
    :func:`compute_formation_energy` (which species are added /
    removed)."""

    VACANCY = "VACANCY"
    INTERSTITIAL = "INTERSTITIAL"
    SUBSTITUTIONAL = "SUBSTITUTIONAL"
    ANTISITE = "ANTISITE"


@dataclass
class DefectSpec:
    """One defect prescription.

    Attributes
    ----------
    kind
        Defect category — drives the stoichiometry vector ``n_i``.
    species_added
        Species **added** by the defect (e.g. an interstitial atom of
        type X, or the impurity for a substitutional). Empty list for
        a vacancy.
    species_removed
        Species **removed** by the defect (e.g. the host atom for a
        vacancy or substitutional). Empty for an interstitial.
    site
        Optional site label (e.g. "tetrahedral", "hexagonal" for a
        Si self-interstitial). Carried as metadata only.
    supercell_n_atoms
        Total atom count of the host supercell. The roadmap requires
        ≥ 64 to keep self-image effects manageable.
    charge_states
        Integer charge states to evaluate. The roadmap default is
        ``(-2, -1, 0, 1, 2)`` for cross-over plots.
    """

    kind: DefectKind
    species_added: Sequence[str] = ()
    species_removed: Sequence[str] = ()
    site: Optional[str] = None
    supercell_n_atoms: int = 64
    charge_states: Tuple[int, ...] = (-2, -1, 0, 1, 2)

    def __post_init__(self) -> None:
        if self.supercell_n_atoms < 64:
            raise ValueError(
                f"supercell_n_atoms must be ≥ 64 to bound finite-size "
                f"effects (Wright & Lewis 2019); got {self.supercell_n_atoms}"
            )
        if self.kind == DefectKind.VACANCY and not self.species_removed:
            raise ValueError(
                "vacancy defect must specify species_removed"
            )
        if self.kind == DefectKind.INTERSTITIAL and not self.species_added:
            raise ValueError(
                "interstitial defect must specify species_added"
            )
        if self.kind in (DefectKind.SUBSTITUTIONAL, DefectKind.ANTISITE):
            if not (self.species_added and self.species_removed):
                raise ValueError(
                    f"{self.kind.value} defect must specify both "
                    "species_added and species_removed"
                )
        if not self.charge_states:
            raise ValueError("charge_states must be non-empty")
