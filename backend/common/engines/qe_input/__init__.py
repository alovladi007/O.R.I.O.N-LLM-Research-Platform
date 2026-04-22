"""QE input generation (Phase 3 / Session 3.1).

Canonical pw.x input file generator. Separate from the legacy
:class:`backend.common.engines.qe.QuantumEspressoEngine` because:

- The legacy engine mixes input generation + subprocess execution +
  output parsing. Session 3.1 only does the first of those; Session
  3.2 migrates execution, 3.2+ migrates parsing.
- Pseudopotentials now come from a real SSSP directory instead of a
  hardcoded 33-element dict.
- Per-element cutoffs are read from UPF file headers, so we pick
  ``max(wfc_cutoff(e) for e in structure)`` as the roadmap convention
  requires.

Public API
----------

- :class:`PseudopotentialRegistry` — scan a UPF directory, map element
  → filename, extract recommended cutoffs.
- :func:`generate_pw_input` — the renderer. Takes a pymatgen
  ``Structure`` (or a dict-of-atoms shape) plus a ``QEInputParams``
  dict and returns the ``.in`` file text.
"""

from .params import QEInputParams
from .registry import (
    PseudopotentialRegistry,
    UnknownElementError,
    UPFFileNotFoundError,
)
from .renderer import (
    KPointGrid,
    RenderedInput,
    generate_pw_input,
    kgrid_from_structure,
)

__all__ = [
    "KPointGrid",
    "PseudopotentialRegistry",
    "QEInputParams",
    "RenderedInput",
    "UPFFileNotFoundError",
    "UnknownElementError",
    "generate_pw_input",
    "kgrid_from_structure",
]
