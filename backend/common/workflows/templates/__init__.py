"""Reusable workflow DAG templates.

Organized by namespace: ``qe`` (Phase 3) and ``md`` (Phase 4).
The router endpoint ``POST /api/v1/workflow-runs/templates/{ns}/{name}``
looks up the builder via the ``SPEC_BUILDERS`` dict on each module.
"""

from .md import (
    diffusivity_vs_t_spec,
    elastic_constants_via_strain_spec,
    equilibrate_nvt_then_nve_spec,
    melting_curve_spec,
)
from .qe import (
    band_structure_spec,
    dos_spec,
    phonons_gamma_spec,
    relax_then_static_spec,
)

__all__ = [
    "band_structure_spec",
    "diffusivity_vs_t_spec",
    "dos_spec",
    "elastic_constants_via_strain_spec",
    "equilibrate_nvt_then_nve_spec",
    "melting_curve_spec",
    "phonons_gamma_spec",
    "relax_then_static_spec",
]
