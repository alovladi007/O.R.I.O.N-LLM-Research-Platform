"""Forcefield registry for the Session 4.1 LAMMPS input generator.

A *forcefield spec* is a declarative record that tells the renderer:

- which elements it covers (``applicable_elements``),
- the LAMMPS ``pair_style`` and ``pair_coeff`` lines to emit,
- the cutoff radius used by the potential (for RDF / neighbor skin
  sanity checks downstream),
- a recommended MD timestep in femtoseconds,
- LAMMPS ``units`` style (``metal`` for eV/Å/ps, ``real`` for
  kcal/mol/fs, ``lj`` for reduced units),
- a citation blob so we can credit the original authors in reports.

The registry is populated at import time with the canonical set from
the roadmap's Session 4.1 spec:

- **Lennard-Jones** — parametric toy ``pair_style lj/cut``, usable on
  any element via reduced units.
- **EAM** — Cu, Ni, Al via alloy files from NIST's Interatomic
  Potentials Repository (https://www.ctcms.nist.gov/potentials/). The
  ``.alloy`` files are shipped under
  ``backend/common/engines/lammps_input/forcefields/data/`` with
  ``LICENSE.txt`` documenting the public-domain status (US Federal
  Government work).
- **Tersoff** — Si and C, using the classic Tersoff 1988 parameters.
- **ReaxFF** — feature-flagged (``available=False`` unless the
  ``.reax`` file is present AND LAMMPS was compiled with the
  ``REAXFF`` package). The spec still exists so the auto-selector
  can report it as a degraded option.
- **ML (MACE / NequIP)** — feature-flagged placeholder for Phase 8.

Public API
----------

- :class:`ForcefieldSpec` — frozen dataclass.
- :class:`ForcefieldRegistry` — ``.get(name)``, ``.auto_select(elements)``,
  ``.list_compatible(elements)``.
- :exc:`NoCompatibleForcefieldError` — raised by ``auto_select`` when
  no registered forcefield covers the given element set.
"""

from .registry import (
    ForcefieldRegistry,
    ForcefieldSpec,
    NoCompatibleForcefieldError,
    _build_default_specs,
    default_registry,
)

__all__ = [
    "ForcefieldRegistry",
    "ForcefieldSpec",
    "NoCompatibleForcefieldError",
    "_build_default_specs",
    "default_registry",
]
