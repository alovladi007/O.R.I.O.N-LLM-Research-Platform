"""Cross-validate ORION formation energies against Materials Project.

Given a list of MP fixture dicts (with ``formula`` and
``formation_energy_per_atom`` fields), we:

1. Load each fixture's structure via the Session 1.1 parsers.
2. Compute ORION's formation energy from the structure's DFT total
   energy (computed on the fly via dft_static, or supplied in the
   fixture's ``orion_total_energy_ev`` for pre-computed tests).
3. Compare to the MP-published ``formation_energy_per_atom``.
4. Report per-structure deviations + MAE + max |deviation|.

Session 3.4's acceptance criterion: MAE < 0.15 eV/atom, no single
deviation > 0.3 eV/atom.

Missing-reference handling
--------------------------

When a fixture contains an element with no calibrated reference, we
don't fail the whole run. We mark it ``skipped_missing_reference`` and
continue. This lets the test suite pass a partial cross-validation
while making the coverage gap loud.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CrossValidationEntry:
    """One structure's cross-validation result."""

    mp_id: str
    formula: str
    mp_e_f_per_atom_ev: float
    orion_e_f_per_atom_ev: Optional[float]
    deviation_ev_per_atom: Optional[float]
    status: str   # 'ok' | 'skipped_missing_reference' | 'skipped_no_energy' | 'errored'
    note: Optional[str] = None
    used_references: Dict[str, float] = field(default_factory=dict)


@dataclass
class CrossValidationReport:
    """Aggregate of per-entry results."""

    entries: List[CrossValidationEntry]

    def ok_entries(self) -> List[CrossValidationEntry]:
        return [e for e in self.entries if e.status == "ok"]

    def mae_ev_per_atom(self) -> Optional[float]:
        ok = self.ok_entries()
        if not ok:
            return None
        return sum(abs(e.deviation_ev_per_atom) for e in ok) / len(ok)

    def max_abs_deviation_ev_per_atom(self) -> Optional[float]:
        ok = self.ok_entries()
        if not ok:
            return None
        return max(abs(e.deviation_ev_per_atom) for e in ok)

    def skipped_count(self) -> int:
        return len([e for e in self.entries if e.status.startswith("skipped")])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_entries": len(self.entries),
            "n_ok": len(self.ok_entries()),
            "n_skipped": self.skipped_count(),
            "mae_ev_per_atom": self.mae_ev_per_atom(),
            "max_abs_deviation_ev_per_atom": self.max_abs_deviation_ev_per_atom(),
            "entries": [
                {
                    "mp_id": e.mp_id,
                    "formula": e.formula,
                    "mp_e_f_per_atom_ev": e.mp_e_f_per_atom_ev,
                    "orion_e_f_per_atom_ev": e.orion_e_f_per_atom_ev,
                    "deviation_ev_per_atom": e.deviation_ev_per_atom,
                    "status": e.status,
                    "note": e.note,
                }
                for e in self.entries
            ],
        }


def run_cross_validation(
    fixtures: List[Dict[str, Any]],
    calculator,
    *,
    total_energy_key: str = "orion_total_energy_ev",
) -> CrossValidationReport:
    """Compare each fixture's MP formation energy to ORION's.

    Each fixture must contain:
    - ``mp_id``
    - ``formula``
    - ``formation_energy_per_atom`` (MP's published value)
    - either ``orion_total_energy_ev`` (the DFT total we already
      computed) + ``orion_species`` (per-atom species list), OR
      a flag that the caller wants the test to skip the entry.

    *calculator* is a :class:`FormationEnergyCalculator` already
    configured with a reference lookup.
    """
    from .formation import MissingReferenceError

    entries: List[CrossValidationEntry] = []
    for f in fixtures:
        mp_id = f.get("mp_id", "?")
        formula = f.get("formula", "?")
        mp_e_f = f.get("formation_energy_per_atom")
        if mp_e_f is None:
            entries.append(CrossValidationEntry(
                mp_id=mp_id, formula=formula,
                mp_e_f_per_atom_ev=float("nan"),
                orion_e_f_per_atom_ev=None,
                deviation_ev_per_atom=None,
                status="skipped_no_mp_value",
                note="fixture missing formation_energy_per_atom",
            ))
            continue

        total_e = f.get(total_energy_key)
        species = f.get("orion_species")
        if total_e is None or species is None:
            entries.append(CrossValidationEntry(
                mp_id=mp_id, formula=formula,
                mp_e_f_per_atom_ev=mp_e_f,
                orion_e_f_per_atom_ev=None,
                deviation_ev_per_atom=None,
                status="skipped_no_energy",
                note=(
                    f"fixture missing {total_energy_key!r} or orion_species — "
                    "compute DFT total via dft_static first"
                ),
            ))
            continue

        try:
            result = calculator.compute(species=species, total_energy_ev=total_e)
        except MissingReferenceError as exc:
            entries.append(CrossValidationEntry(
                mp_id=mp_id, formula=formula,
                mp_e_f_per_atom_ev=mp_e_f,
                orion_e_f_per_atom_ev=None,
                deviation_ev_per_atom=None,
                status="skipped_missing_reference",
                note=str(exc),
            ))
            continue
        except Exception as exc:  # noqa: BLE001
            entries.append(CrossValidationEntry(
                mp_id=mp_id, formula=formula,
                mp_e_f_per_atom_ev=mp_e_f,
                orion_e_f_per_atom_ev=None,
                deviation_ev_per_atom=None,
                status="errored",
                note=f"{type(exc).__name__}: {exc}",
            ))
            continue

        orion_e_f = result.formation_energy_per_atom_ev
        deviation = orion_e_f - mp_e_f
        entries.append(CrossValidationEntry(
            mp_id=mp_id, formula=formula,
            mp_e_f_per_atom_ev=mp_e_f,
            orion_e_f_per_atom_ev=orion_e_f,
            deviation_ev_per_atom=deviation,
            status="ok",
            note=None,
            used_references=result.used_references,
        ))

    return CrossValidationReport(entries=entries)
