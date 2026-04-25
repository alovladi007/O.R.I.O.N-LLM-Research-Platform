"""Image-charge corrections for charged-defect supercells.

The Freysoldt scheme (Freysoldt, Neugebauer, Van de Walle 2009) has
two parts:

1. **Monopole correction** ``E_lat = q² α / (2 ε L)`` — the
   self-energy of the charge in its periodic images. Closed form
   given the charge ``q``, static dielectric ``ε``, supercell linear
   size ``L``, and Madelung constant α. Equivalent to the older
   Makov-Payne 1995 leading-order term.

2. **Potential alignment** ``ΔV`` — the difference between the
   long-range planar-averaged electrostatic potential of the
   defective vs bulk supercells, far from the defect. Requires the
   actual DFT potentials.

For Session 8.3 we ship the monopole-only correction
(:func:`makov_payne_monopole_correction`) as a deterministic,
DFT-free aggregator that the formation-energy formula consumes.
Session 8.3b layers the potential alignment on top once a DFT
runner is wired.

The full pymatgen-analysis-defects
:func:`pymatgen.analysis.defects.corrections.freysoldt.get_freysoldt_correction`
is available for callers that have the LOCPOT data; we re-export
nothing from it here to keep the interface narrow and DFT-engine
agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Madelung constant for the supercell shape. For a cubic supercell
# in vacuum, the standard literature value is α ≈ 2.8373 (the
# Madelung constant of a simple cubic lattice with one charge per
# cell). Non-cubic supercells need the lattice-specific value;
# Freysoldt et al. tabulate these for the common Bravais lattices.
ALPHA_CUBIC = 2.8373

# Coulomb constant: e²/(4πε₀) in eV·Å. CODATA 2018.
E_SQUARED_OVER_4PI_EPS0_EV_A = 14.39964547842567


@dataclass
class FreysoldtCorrection:
    """Image-charge correction breakdown.

    Persisted alongside the formation energy so callers can audit
    which terms contributed.
    """

    monopole_ev: float
    alignment_ev: float = 0.0  # 0 in the monopole-only path

    @property
    def total_ev(self) -> float:
        return self.monopole_ev + self.alignment_ev


def makov_payne_monopole_correction(
    *,
    charge: int,
    static_dielectric: float,
    supercell_length_a: float,
    madelung_alpha: float = ALPHA_CUBIC,
) -> FreysoldtCorrection:
    """First-order monopole correction for a charged supercell.

    .. math::

        E_{\\text{corr}} = \\frac{q^2 \\, \\alpha \\, e^2}{2 \\, \\epsilon \\, L \\, 4 \\pi \\epsilon_0}

    Parameters
    ----------
    charge
        Defect charge state ``q`` (signed integer).
    static_dielectric
        Static dielectric constant ``ε`` of the host. For Si: 11.68;
        for SiO₂: 3.9; for diamond: 5.7.
    supercell_length_a
        Linear size ``L`` of the cubic supercell in Å. For a 64-atom
        cubic Si supercell (4×4×4 of the conventional cell) ``L`` is
        ~21.7 Å.
    madelung_alpha
        Madelung constant for the supercell shape. Defaults to the
        cubic value ``2.8373``.

    Returns
    -------
    correction
        :class:`FreysoldtCorrection` with the monopole term filled in
        and ``alignment_ev = 0``.
    """
    if static_dielectric <= 0:
        raise ValueError(
            f"static_dielectric must be > 0; got {static_dielectric}"
        )
    if supercell_length_a <= 0:
        raise ValueError(
            f"supercell_length_a must be > 0; got {supercell_length_a}"
        )
    # Neutral defects need no image-charge correction.
    if charge == 0:
        return FreysoldtCorrection(monopole_ev=0.0)
    monopole = (
        (charge ** 2) * madelung_alpha * E_SQUARED_OVER_4PI_EPS0_EV_A
        / (2.0 * static_dielectric * supercell_length_a)
    )
    return FreysoldtCorrection(monopole_ev=float(monopole))
