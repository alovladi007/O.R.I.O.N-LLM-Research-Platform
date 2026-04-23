"""Mesoscale kMC engine (Phase 5 / Session 5.2).

Rejection-free kinetic Monte Carlo for defect migration on a simple-
cubic lattice. Vacancy + interstitial hops + pair annihilation.

The legacy :mod:`backend.common.engines.mesoscale` ``MonteCarloEngine``
/ ``PhaseFieldEngine`` stubs still ship fake data; this package is
the canonical path. Migration of the legacy stub into the
:class:`SimulationEngine` interface is deferred — new callers should
route through :func:`run_kmc` directly.

Public API
----------

- :class:`LatticeSpec`, :class:`EventType`, :class:`EventCatalog`,
  :class:`KMCProblem` — pydantic schemas.
- :func:`run_kmc` — the Gillespie-style rejection-free engine.
- :class:`KMCResult` — time series of defect counts + vacancy MSD +
  final positions + bookkeeping.
- Lattice primitives :data:`NEIGHBOR_OFFSETS` / :func:`neighbor_site`
  exported in case callers want to compose custom event rules on top
  of the same lattice topology.

Limitations (Session 5.2 MVP, documented here and in the session
report)
------------------------------------------------------------------

- Simple-cubic only (``kind="simple_cubic"``).
- Isotropic hopping — no strain-field coupling, no local-environment
  rate modulation beyond the event catalog.
- Annihilation is tested only between a freshly-moved defect and the
  *exact* site of an opposite-species defect (recombination_radius = 1
  lattice spacing). Capture-radius-style variants are a follow-up.
- No cluster formation / break-up / chemistry beyond V+I → 0.
- Neighbour lookup is a linear scan in O(n_defects); acceptable for
  the acceptance cases (≤ 20k defects) but not for million-site
  irradiation cascades. A hash-based occupancy map closes that gap
  when a caller needs it.
"""

from .engine import KMCResult, run_kmc
from .lattice import NEIGHBOR_OFFSETS, N_DIRECTIONS, neighbor_site, wrap
from .params import (
    EventCatalog,
    EventType,
    KMCProblem,
    LatticeSpec,
    SpeciesName,
)

__all__ = [
    "EventCatalog",
    "EventType",
    "KMCProblem",
    "KMCResult",
    "LatticeSpec",
    "NEIGHBOR_OFFSETS",
    "N_DIRECTIONS",
    "SpeciesName",
    "neighbor_site",
    "run_kmc",
    "wrap",
]
