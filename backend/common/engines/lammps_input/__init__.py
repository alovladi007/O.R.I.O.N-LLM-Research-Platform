"""LAMMPS input generation (Phase 4 / Session 4.1).

Canonical ``in.lammps`` + ``structure.data`` generator. Separate from
the legacy :class:`backend.common.engines.lammps.LAMMPSEngine` because:

- The legacy engine hand-rolls ``in.lammps`` as a string and writes
  its own data file; Session 4.1 delegates the data file to pymatgen
  (``LammpsData``) and the input script to a Jinja template. The two
  decks are no longer a copy-paste hazard.
- Forcefields are now a registry (see ``forcefields/``) instead of
  a long if-elif in ``_write_potential_section``. Adding a new
  potential is now declarative: one :class:`~.forcefields.ForcefieldSpec`.
- Auto-selection: given a composition we pick the best compatible
  forcefield, or fail with a clear "no FF covers {X}" error.
- Units are audited: timesteps stay in fs through the API and are
  converted to the LAMMPS ``units`` style of the chosen forcefield
  at generation time.

Public API
----------

- :class:`LAMMPSInputParams` — Pydantic parameter schema.
- :class:`ForcefieldSpec`, :class:`ForcefieldRegistry`,
  :func:`default_registry` — the forcefield machinery.
- :class:`NoCompatibleForcefieldError` — raised on auto-select
  failure and on invalid explicit choices.
- :func:`generate_lammps_input` — the renderer.
- :func:`write_lammps_inputs` — write the rendered files to disk.
"""

from .forcefields import (
    ForcefieldRegistry,
    ForcefieldSpec,
    NoCompatibleForcefieldError,
    default_registry,
)
from .params import Ensemble, LAMMPSInputParams
from .renderer import (
    THERMO_COLUMNS_STRESS,
    RenderedLAMMPSInput,
    generate_lammps_input,
    strain_extra_commands,
    write_lammps_inputs,
)

__all__ = [
    "Ensemble",
    "ForcefieldRegistry",
    "ForcefieldSpec",
    "LAMMPSInputParams",
    "NoCompatibleForcefieldError",
    "RenderedLAMMPSInput",
    "THERMO_COLUMNS_STRESS",
    "default_registry",
    "generate_lammps_input",
    "strain_extra_commands",
    "write_lammps_inputs",
]
