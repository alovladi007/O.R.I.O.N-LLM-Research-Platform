"""Phase 8 / Session 8.3 — point defect workflow.

Computes defect formation energies as a function of charge state ``q``,
chemical potential ``μ``, and Fermi level ``E_F``:

    E_f(D^q) = E_tot(D^q) - E_tot(bulk)
             - Σ_i n_i μ_i
             + q (E_VBM + E_F)
             + ΔV_correction(q)

where the last term is the Freysoldt image-charge correction for
charged defects in periodic supercells. Transition levels
``ε(q/q')`` are the Fermi-level positions where two charge states
are equally stable.

Layout
------

- :mod:`spec` — :class:`DefectSpec` dataclass (defect kind, site,
  charge states, supercell size).
- :mod:`chempot` — chemical-potential bounds from a phase diagram
  via pymatgen ``PhaseDiagram``.
- :mod:`freysoldt` — image-charge correction wrapper. The full
  Freysoldt scheme requires the planar-averaged electrostatic
  potential from DFT; for tests / pre-DFT estimates we ship a
  Makov-Payne style monopole-only fallback that takes the
  static dielectric and supercell as inputs.
- :mod:`thermo` — formation energy + transition levels +
  charge-state cross-over data points (the roadmap's "charge state
  cross-over plot" is built from :func:`build_crossover_curve`).

The full pymatgen-analysis-defects ``FormationEnergyDiagram``
integration lands in 8.3b once we have DFT outputs to feed it; this
session ships a closed-form aggregator that works on any DFT engine
that returns total energies + the bulk dielectric.
"""

from .chempot import chemical_potential_bounds
from .freysoldt import (
    FreysoldtCorrection,
    makov_payne_monopole_correction,
)
from .spec import DefectKind, DefectSpec
from .thermo import (
    FormationEnergyResult,
    TransitionLevel,
    build_crossover_curve,
    compute_formation_energy,
    transition_levels,
)

__all__ = [
    "DefectKind",
    "DefectSpec",
    "FormationEnergyResult",
    "FreysoldtCorrection",
    "TransitionLevel",
    "build_crossover_curve",
    "chemical_potential_bounds",
    "compute_formation_energy",
    "makov_payne_monopole_correction",
    "transition_levels",
]
