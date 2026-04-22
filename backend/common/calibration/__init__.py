"""QE calibration — reference energies + formation-energy calculator.

Session 3.4 — the piece that turns ORION's absolute DFT energies
into cross-comparable formation energies. Three parts:

- :mod:`.references` — build elemental reference cells + look up
  stored reference energies.
- :mod:`.formation` — compute formation energy from a structure's
  total energy + composition against the references table.
- :mod:`.workflow` — the calibration workflow that runs per-element
  vc-relax + scf and writes to the ``reference_energies`` table.
"""

from .formation import (
    FormationEnergyCalculator,
    FormationEnergyResult,
    MissingReferenceError,
    make_db_lookup,
)
from .references import (
    ATOMIC_SOLID_ONLY,
    DIATOMIC_GAS_ELEMENTS,
    ElementalReferenceCell,
    UnsupportedElement,
    build_elemental_reference_cell,
    is_triplet_diatomic,
    supported_elements,
)
from .cross_validate import (
    CrossValidationEntry,
    CrossValidationReport,
    run_cross_validation,
)
from .runner import (
    DEFAULT_FUNCTIONAL,
    DEFAULT_PSEUDO_FAMILY,
    CalibrationResult,
    run_element_calibration,
)
from .scf_compound import CompoundSCFResult, run_compound_scf

__all__ = [
    "ATOMIC_SOLID_ONLY",
    "CalibrationResult",
    "DEFAULT_FUNCTIONAL",
    "DEFAULT_PSEUDO_FAMILY",
    "DIATOMIC_GAS_ELEMENTS",
    "ElementalReferenceCell",
    "FormationEnergyCalculator",
    "FormationEnergyResult",
    "MissingReferenceError",
    "UnsupportedElement",
    "CompoundSCFResult",
    "CrossValidationEntry",
    "CrossValidationReport",
    "build_elemental_reference_cell",
    "is_triplet_diatomic",
    "make_db_lookup",
    "run_compound_scf",
    "run_cross_validation",
    "run_element_calibration",
    "supported_elements",
]
