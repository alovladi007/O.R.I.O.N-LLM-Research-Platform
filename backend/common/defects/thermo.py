"""Defect formation-energy formula + transition levels.

Formation-energy expression (the standard form, see Freysoldt et al.
2014 review or Walle 1989):

    E_f(D^q, μ, E_F) = E_tot(D^q) - E_tot(bulk)
                      - Σ_i n_i μ_i
                      + q (E_VBM + E_F)
                      + E_corr(q)

where ``n_i`` is the number of atoms of species *i* added by the
defect (negative for vacancies), ``μ_i`` is the species' chemical
potential, ``E_VBM`` is the host VBM (eV, from a DFT band-structure
calc on the bulk), ``E_F`` is the Fermi level relative to VBM
(0 ≤ E_F ≤ E_g), and ``E_corr(q)`` is the Freysoldt correction.

Transition levels ``ε(q/q')`` are the Fermi-level positions where
two charge states have equal formation energy:

    ε(q/q') = (E_f(q'; E_F=0) - E_f(q; E_F=0)) / (q - q')

These are the Fermi levels at which the defect's preferred charge
state changes; charge-state cross-over plots show the lowest-E_f
charge state at each E_F across the band gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .freysoldt import (
    FreysoldtCorrection,
    makov_payne_monopole_correction,
)
from .spec import DefectSpec


@dataclass
class FormationEnergyResult:
    """One (defect, charge_state) formation energy point."""

    charge: int
    e_form_at_efermi_zero: float          # eV — at E_F = 0 (VBM)
    correction: FreysoldtCorrection       # for provenance
    components: Dict[str, float] = field(default_factory=dict)
    # ``components`` records each term in the sum so the analysis can
    # be audited:
    #   {"e_tot_defect": ..., "e_tot_bulk": ..., "mu_correction": ...,
    #    "vbm_term": ..., "freysoldt": ...}

    def at_fermi(self, efermi_ev: float) -> float:
        """Return E_f at the supplied E_F (relative to VBM)."""
        return self.e_form_at_efermi_zero + self.charge * efermi_ev


@dataclass
class TransitionLevel:
    """One charge-state transition ``ε(q1/q2)``."""

    q1: int
    q2: int
    fermi_level_ev: float                 # relative to VBM


def compute_formation_energy(
    *,
    spec: DefectSpec,
    charge: int,
    e_tot_defect_ev: float,
    e_tot_bulk_ev: float,
    chemical_potentials: Dict[str, float],
    vbm_ev: float,
    static_dielectric: float,
    supercell_length_a: float,
) -> FormationEnergyResult:
    """One-shot formation energy at ``E_F = 0`` (VBM).

    Caller supplies the DFT total energies for the (defective,
    bulk) supercells, the chemical-potential dict for the relevant
    growth condition, and the bulk VBM. Freysoldt monopole
    correction is computed automatically; pass ``static_dielectric``
    and ``supercell_length_a`` for the host.

    The full formation-energy curve E_f(E_F) is recovered from the
    returned :class:`FormationEnergyResult` via :meth:`at_fermi`.
    """
    if charge not in spec.charge_states:
        raise ValueError(
            f"charge {charge} not in spec.charge_states {spec.charge_states}"
        )
    # Stoichiometry vector — what the defect adds (positive) or
    # removes (negative) relative to the bulk supercell.
    n: Dict[str, int] = {}
    for sp in spec.species_added:
        n[sp] = n.get(sp, 0) + 1
    for sp in spec.species_removed:
        n[sp] = n.get(sp, 0) - 1
    missing = [sp for sp in n if sp not in chemical_potentials]
    if missing:
        raise ValueError(
            f"chemical_potentials missing entries for {missing}; "
            f"have {sorted(chemical_potentials)}"
        )
    mu_term = sum(n_i * chemical_potentials[sp] for sp, n_i in n.items())
    correction = makov_payne_monopole_correction(
        charge=charge,
        static_dielectric=static_dielectric,
        supercell_length_a=supercell_length_a,
    )
    vbm_term = charge * vbm_ev
    e_form_zero = (
        (e_tot_defect_ev - e_tot_bulk_ev)
        - mu_term
        + vbm_term
        + correction.total_ev
    )
    return FormationEnergyResult(
        charge=charge,
        e_form_at_efermi_zero=float(e_form_zero),
        correction=correction,
        components={
            "e_tot_defect_ev": float(e_tot_defect_ev),
            "e_tot_bulk_ev": float(e_tot_bulk_ev),
            "mu_correction_ev": float(mu_term),
            "vbm_term_ev": float(vbm_term),
            "freysoldt_total_ev": float(correction.total_ev),
        },
    )


def transition_levels(
    results: List[FormationEnergyResult],
    *,
    efermi_min: float = 0.0,
    efermi_max: Optional[float] = None,
) -> List[TransitionLevel]:
    """All (q, q') pairs where ``E_f(q) == E_f(q')`` inside the gap.

    ``ε(q/q') = (E_f(q'; 0) - E_f(q; 0)) / (q - q')``.

    Filters to fermi levels inside ``[efermi_min, efermi_max]`` if
    ``efermi_max`` is supplied (typically set to the band gap).
    Results sorted ascending by Fermi level.
    """
    out: List[TransitionLevel] = []
    by_q = {r.charge: r for r in results}
    qs = sorted(by_q.keys())
    for i, q1 in enumerate(qs):
        for q2 in qs[i + 1:]:
            if q1 == q2:
                continue
            ef = (
                by_q[q2].e_form_at_efermi_zero - by_q[q1].e_form_at_efermi_zero
            ) / (q1 - q2)
            if ef < efermi_min:
                continue
            if efermi_max is not None and ef > efermi_max:
                continue
            out.append(TransitionLevel(q1=q1, q2=q2, fermi_level_ev=ef))
    out.sort(key=lambda t: t.fermi_level_ev)
    return out


def build_crossover_curve(
    results: List[FormationEnergyResult],
    *,
    efermi_min: float = 0.0,
    efermi_max: float = 1.0,
    n_points: int = 200,
) -> Dict[str, np.ndarray]:
    """Return ``(efermi_grid, min_eform, dominant_charge)`` arrays.

    The ``min_eform`` curve is the lower envelope of all charge
    states' E_f vs E_F lines — the canonical "charge state
    cross-over plot" the roadmap calls for. ``dominant_charge``
    records which charge state owns the envelope at each E_F.
    """
    if not results:
        raise ValueError("results must be non-empty")
    grid = np.linspace(efermi_min, efermi_max, int(n_points))
    eforms = np.stack(
        [np.array([r.at_fermi(ef) for ef in grid]) for r in results],
        axis=0,
    )  # (n_q, n_pts)
    min_eform = eforms.min(axis=0)
    argmin = eforms.argmin(axis=0)
    charges = np.array([r.charge for r in results], dtype=int)
    dominant_charge = charges[argmin]
    return {
        "efermi_ev": grid,
        "min_eform_ev": min_eform,
        "dominant_charge": dominant_charge,
        "all_eforms_ev": eforms,
    }
