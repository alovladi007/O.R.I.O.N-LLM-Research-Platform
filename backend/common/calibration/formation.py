"""Formation energy from total energy + composition + reference table.

Math
----

For a compound with composition {element: n_i} and total energy E_total,

    E_formation = E_total - Σ_i n_i * E_ref(element_i)

Per-atom form:

    E_f_per_atom = E_formation / N_atoms

Where E_ref(element_i) comes from the reference_energies table, keyed
by (element, functional, pseudo_family).

What happens when a reference is missing
----------------------------------------

We raise :class:`MissingReferenceError` rather than silently substituting
zero. Session 3.4 specifically asked for this — cross-comparison with
a zeroed-out reference is silently wrong. Callers either run
calibration first or hand-supply a reference. The CLI and workflow
both surface this clearly.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MissingReferenceError(KeyError):
    """A reference energy was needed but not found in the table."""

    def __init__(self, element: str, functional: str, pseudo_family: str):
        self.element = element
        self.functional = functional
        self.pseudo_family = pseudo_family
        super().__init__(
            f"No reference_energy for ({element!r}, {functional!r}, "
            f"{pseudo_family!r}). Run calibration first: "
            f"`orion calibrate --element {element}`."
        )


@dataclass
class FormationEnergyResult:
    """Return of :meth:`FormationEnergyCalculator.compute`."""

    total_energy_ev: float
    reference_sum_ev: float
    formation_energy_ev: float
    formation_energy_per_atom_ev: float
    n_atoms: int
    composition: Dict[str, int]
    functional: str
    pseudo_family: str
    used_references: Dict[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_energy_ev": self.total_energy_ev,
            "reference_sum_ev": self.reference_sum_ev,
            "formation_energy_ev": self.formation_energy_ev,
            "formation_energy_per_atom_ev": self.formation_energy_per_atom_ev,
            "n_atoms": self.n_atoms,
            "composition": dict(self.composition),
            "functional": self.functional,
            "pseudo_family": self.pseudo_family,
            "used_references": dict(self.used_references),
        }


class FormationEnergyCalculator:
    """Compute formation energies against a stored reference library.

    Accepts references either as a preloaded dict (for tests — keyed
    by element symbol) or a callable ``(element, functional, pseudo_family)
    -> float`` for DB-backed lookup.
    """

    def __init__(
        self,
        functional: str,
        pseudo_family: str,
        *,
        references: Optional[Dict[str, float]] = None,
        lookup: Optional[Any] = None,
    ):
        if references is None and lookup is None:
            raise ValueError("Need either `references` dict or `lookup` callable.")
        if references is not None and lookup is not None:
            raise ValueError("Pass either `references` or `lookup`, not both.")
        self.functional = functional
        self.pseudo_family = pseudo_family
        self._references = references
        self._lookup = lookup

    def get_reference(self, element: str) -> float:
        if self._references is not None:
            if element not in self._references:
                raise MissingReferenceError(element, self.functional, self.pseudo_family)
            return self._references[element]
        try:
            return self._lookup(element, self.functional, self.pseudo_family)  # type: ignore[misc]
        except MissingReferenceError:
            raise
        except KeyError as exc:
            raise MissingReferenceError(
                element, self.functional, self.pseudo_family,
            ) from exc

    def compute(
        self,
        *,
        species: List[str],
        total_energy_ev: float,
    ) -> FormationEnergyResult:
        """Compute formation energy from a per-atom species list + total energy."""
        if not species:
            raise ValueError("empty species list")
        composition = dict(Counter(species))
        n_atoms = sum(composition.values())

        reference_sum_ev = 0.0
        used_refs: Dict[str, float] = {}
        for element, count in composition.items():
            e_ref = self.get_reference(element)
            used_refs[element] = e_ref
            reference_sum_ev += count * e_ref

        e_formation = total_energy_ev - reference_sum_ev
        return FormationEnergyResult(
            total_energy_ev=total_energy_ev,
            reference_sum_ev=reference_sum_ev,
            formation_energy_ev=e_formation,
            formation_energy_per_atom_ev=e_formation / n_atoms,
            n_atoms=n_atoms,
            composition=composition,
            functional=self.functional,
            pseudo_family=self.pseudo_family,
            used_references=used_refs,
        )


def make_db_lookup(session) -> Any:
    """Return a lookup callable that queries the reference_energies table."""
    from sqlalchemy import select

    from src.api.models import ReferenceEnergy

    def _lookup(element: str, functional: str, pseudo_family: str) -> float:
        stmt = (
            select(ReferenceEnergy)
            .where(ReferenceEnergy.element == element)
            .where(ReferenceEnergy.functional == functional)
            .where(ReferenceEnergy.pseudo_family == pseudo_family)
        )
        row = session.execute(stmt).scalar_one_or_none()
        if row is None:
            raise MissingReferenceError(element, functional, pseudo_family)
        return row.energy_per_atom_ev

    return _lookup
