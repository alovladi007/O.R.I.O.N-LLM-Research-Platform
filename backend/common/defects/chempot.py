"""Chemical-potential bounds from a phase diagram.

For a host crystal, the chemical potential of each constituent
species is bounded by the requirement that no other phase from the
same chemsys is more stable (otherwise the host would decompose).
We use ``pymatgen.analysis.phase_diagram.PhaseDiagram`` to compute
these bounds — the seeded MP data from Session 1.5 is the
standard source of competing-phase entries.

The bounds form a polytope in chemical-potential space; we report
the corners (vertices of the polytope) as a list of dicts. Each
vertex corresponds to a "growth condition" (e.g. "Si-rich" / "O-rich"
for SiO₂). Callers pick the vertex matching their experimental
condition and feed those ``μ_i`` into :func:`compute_formation_energy`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def chemical_potential_bounds(
    *,
    host_composition: Dict[str, int],
    phase_diagram_entries: Sequence[Any],
    tolerance: float = 1e-3,
) -> List[Dict[str, float]]:
    """Return the chemical-potential vertices of the host's stability region.

    Parameters
    ----------
    host_composition
        Reduced formula as ``{element: stoichiometry}``, e.g.
        ``{"Si": 1}`` for elemental silicon, ``{"Ba": 1, "Ti": 1, "O": 3}``
        for BaTiO₃.
    phase_diagram_entries
        List of ``pymatgen.entries.computed_entries.ComputedEntry``
        objects spanning the chemsys (must include the host as the
        most-stable entry at its composition).
    tolerance
        Slack in eV/atom for "at the hull" tests; entries within
        ``tolerance`` of the convex hull are considered competing.

    Returns
    -------
    vertices
        List of dicts ``{element: mu_eV_per_atom}``. Each dict is one
        vertex of the polytope. For a 1-element system the polytope
        is a single point: ``mu = energy_per_atom_of_host``.
    """
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core.composition import Composition

    if not host_composition:
        raise ValueError("host_composition must be non-empty")
    elements = sorted(host_composition)
    if len(elements) == 1:
        # Degenerate: μ of the single element is just its
        # ground-state energy/atom in the supplied entries.
        elt = elements[0]
        per_atom = [
            (e.composition.reduced_composition, e.energy_per_atom)
            for e in phase_diagram_entries
            if list(e.composition.reduced_composition) == [elt]
        ]
        if not per_atom:
            raise ValueError(
                f"no elemental entry for {elt!r} in phase_diagram_entries"
            )
        # Take the lowest-energy elemental entry.
        per_atom.sort(key=lambda x: x[1])
        return [{elt: float(per_atom[0][1])}]

    pd = PhaseDiagram(list(phase_diagram_entries))
    host_comp = Composition(host_composition)
    # pymatgen's PhaseDiagram.get_chempot_range_map works on lists
    # of element objects; collect them.
    elements_obj = [e for e in pd.elements if str(e) in host_composition]
    if len(elements_obj) != len(host_composition):
        missing = set(host_composition) - {str(e) for e in elements_obj}
        raise ValueError(
            f"phase_diagram_entries missing elements {sorted(missing)} "
            "from host_composition"
        )
    try:
        cp_map = pd.get_all_chempots(host_comp)
    except Exception as exc:
        raise ValueError(
            f"PhaseDiagram could not compute chempot vertices for "
            f"{host_composition}: {exc}"
        ) from exc
    vertices: List[Dict[str, float]] = []
    for vertex_label, mus in cp_map.items():
        vertices.append({str(elt): float(mus[elt]) for elt in mus})
    if not vertices:
        raise ValueError(
            f"no chempot vertices found for {host_composition} — host "
            "is likely above the convex hull"
        )
    return vertices
