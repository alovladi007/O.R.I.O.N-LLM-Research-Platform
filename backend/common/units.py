"""
ORION unit conventions and conversions.

Policy (from ROADMAP_PROMPTS.md, "Scientific correctness global checklist"):

- **SI internally.** Every value stored in the database or passed between
  workers is in SI base units: meter, kilogram, second, kelvin, pascal,
  joule, ampere, coulomb, mole.
- **Boundary-only conversions.** Accept Å / eV / GPa / kcal·mol⁻¹ at the
  HTTP API boundary and at engine I/O points. Convert at exactly those
  two places; never in between.
- **Canonical units per property.** The :data:`PROPERTY_REGISTRY` below is
  the single source of truth for what unit a property is stored in and
  what physical bounds are acceptable. Session 1.3 (bulk property
  importer) and Session 3.2 (QE parser) both use it to validate.

This module depends only on :mod:`pint` and the standard library. It must
not import SQLAlchemy, FastAPI, or pymatgen — those come in at higher
layers that wrap this one.

Example::

    from backend.common.units import ureg, to_si, from_si, PROPERTY_REGISTRY

    # 2.0 eV → 3.204 × 10⁻¹⁹ J (stored)
    energy_j = to_si(2.0, "eV")              # 3.20435e-19

    # 10 GPa → 1e10 Pa
    pressure_pa = to_si(10.0, "GPa")         # 1.0e10

    # Round-trip with the registry
    spec = PROPERTY_REGISTRY["bandgap"]
    assert spec.canonical_unit == "J"
    assert spec.bounds_si == (0.0, None)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import pint

# ---------------------------------------------------------------------------
# pint unit registry.
# ---------------------------------------------------------------------------
# One shared registry for the whole process. Pint's default includes all
# SI + CGS + atomic / spectroscopy units; we only extend it for a couple
# of materials-science conveniences that are missing.
ureg = pint.UnitRegistry()
ureg.define("angstrom = 1e-10 meter = Å = Ang")

# Quantity shortcut. Usage: Q_(2.0, "eV") -> pint.Quantity
Q_ = ureg.Quantity


def to_si(value: float, unit: str) -> float:
    """
    Convert ``value`` expressed in ``unit`` to its SI-base magnitude.

    Rather than a hand-maintained dimension→unit table, we let pint do the
    work: ``Q.to_base_units()`` returns the quantity in SI base units
    (kilogram, meter, second, ampere, kelvin, mole, candela), which is the
    definition of "stored in SI" for our purposes.

    Raises
    ------
    pint.UndefinedUnitError
        If ``unit`` is not in the registry.
    """
    if value is None:
        raise ValueError("value cannot be None")
    q = Q_(value, unit)
    return float(q.to_base_units().magnitude)


def from_si(value_si: float, target_unit: str) -> float:
    """
    Convert a magnitude from its SI base unit into ``target_unit``.

    This is the counterpart of :func:`to_si`. The caller is responsible
    for knowing which SI unit the input is in (use the registry, not
    guesswork).
    """
    if value_si is None:
        raise ValueError("value_si cannot be None")
    q = Q_(value_si, "kilogram*meter**2/second**2")  # won't be used — we parse from target
    # Easier path: build a Q_ in target_unit and compare.
    # We take the SI base of the target, scale accordingly.
    target = Q_(1.0, target_unit)
    si_equivalent = target.to_base_units()
    return float(value_si / si_equivalent.magnitude)


# ---------------------------------------------------------------------------
# Property registry — single source of truth for stored units & bounds.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropertySpec:
    """
    Describes how ORION stores one physical property.

    Attributes
    ----------
    name
        Canonical identifier (matches database rows and API shape).
    canonical_unit
        Unit the numerical value is stored in at the DB / wire boundary.
        Internal computation should work in SI and convert on write; see
        module docstring.
    accepted_units
        Human-facing units the API will accept (and auto-convert) when
        property rows are created. ``canonical_unit`` is always included.
    bounds_si
        Optional ``(low, high)`` physical bounds in SI. ``None`` on
        either side means open. Rows outside these bounds are rejected
        by the bulk importer (Session 1.3) and engine parsers.
    description
        Short human-readable docstring.
    """

    name: str
    canonical_unit: str
    accepted_units: Tuple[str, ...]
    bounds_si: Tuple[Optional[float], Optional[float]]
    description: str = ""

    def __post_init__(self) -> None:
        if self.canonical_unit not in self.accepted_units:
            # Immutable dataclass → hack around frozen for init-time normalization
            object.__setattr__(
                self,
                "accepted_units",
                tuple(list(self.accepted_units) + [self.canonical_unit]),
            )


# Scientific bounds below are conservative — they reject obviously-broken
# inputs (negative bandgap, density zero) without rejecting unusual-but-real
# values. Tighter, method-specific checks belong in the QE / LAMMPS parsers.
_EV_TO_J = ureg("eV").to("J").magnitude       # ≈ 1.602176634e-19
_GPA_TO_PA = ureg("GPa").to("Pa").magnitude   # 1.0e9
_KCAL_MOL_TO_J = ureg("kcal/mol").to("J/mol").magnitude  # ≈ 4184

PROPERTY_REGISTRY: Dict[str, PropertySpec] = {
    "bandgap": PropertySpec(
        name="bandgap",
        canonical_unit="J",
        accepted_units=("eV", "J"),
        bounds_si=(0.0, 15.0 * _EV_TO_J),  # 0 (metal) → ~15 eV (hard insulator)
        description="Fundamental electronic bandgap.",
    ),
    "formation_energy_per_atom": PropertySpec(
        name="formation_energy_per_atom",
        canonical_unit="J",
        accepted_units=("eV", "J"),
        bounds_si=(-20.0 * _EV_TO_J, 20.0 * _EV_TO_J),
        description="Formation energy per atom, relative to elemental refs.",
    ),
    "total_energy": PropertySpec(
        name="total_energy",
        canonical_unit="J",
        accepted_units=("eV", "Ry", "Hartree", "J", "kcal/mol", "kJ/mol"),
        bounds_si=(None, None),
        description="Total electronic energy of a structure.",
    ),
    "density": PropertySpec(
        name="density",
        canonical_unit="kilogram / meter ** 3",
        accepted_units=("kg/m**3", "g/cm**3"),
        bounds_si=(1.0, 30000.0),  # 1 kg/m³ (low-density aerogel) → osmium-ish
        description="Mass density.",
    ),
    "bulk_modulus": PropertySpec(
        name="bulk_modulus",
        canonical_unit="pascal",
        accepted_units=("Pa", "GPa", "kbar"),
        bounds_si=(0.0, 1000.0 * _GPA_TO_PA),  # 0 → 1 TPa
        description="Bulk modulus (Voigt-Reuss-Hill average).",
    ),
    "shear_modulus": PropertySpec(
        name="shear_modulus",
        canonical_unit="pascal",
        accepted_units=("Pa", "GPa"),
        bounds_si=(0.0, 1000.0 * _GPA_TO_PA),
        description="Shear modulus.",
    ),
    "poisson_ratio": PropertySpec(
        name="poisson_ratio",
        canonical_unit="",
        accepted_units=("",),
        bounds_si=(-1.0, 0.5),  # thermodynamic limits
        description="Poisson ratio ν.",
    ),
    "lattice_parameter": PropertySpec(
        name="lattice_parameter",
        canonical_unit="meter",
        accepted_units=("m", "angstrom", "nm", "pm"),
        bounds_si=(1e-11, 1e-8),  # 0.1 Å → 100 Å
        description="Cubic lattice parameter.",
    ),
    "cell_volume": PropertySpec(
        name="cell_volume",
        canonical_unit="meter ** 3",
        accepted_units=("m**3", "angstrom**3", "nm**3"),
        bounds_si=(0.0, None),
        description="Unit-cell volume.",
    ),
    "pressure": PropertySpec(
        name="pressure",
        canonical_unit="pascal",
        accepted_units=("Pa", "GPa", "kbar", "bar", "atm"),
        bounds_si=(-1e12, 1e12),
        description="Mechanical / hydrostatic pressure.",
    ),
    "temperature": PropertySpec(
        name="temperature",
        canonical_unit="kelvin",
        accepted_units=("K", "degC", "degF"),
        bounds_si=(0.0, 1e5),  # 0 K → 100 000 K (plasma regime)
        description="Thermodynamic temperature.",
    ),
}


def lookup_property(name: str) -> PropertySpec:
    """Return the :class:`PropertySpec` for *name* or raise :class:`KeyError`."""
    spec = PROPERTY_REGISTRY.get(name)
    if spec is None:
        raise KeyError(
            f"Unknown property {name!r}. Known: {sorted(PROPERTY_REGISTRY)}"
        )
    return spec


def validate_value(name: str, value_si: float) -> None:
    """Raise :class:`ValueError` if *value_si* is outside the registered bounds."""
    spec = lookup_property(name)
    low, high = spec.bounds_si
    if low is not None and value_si < low:
        raise ValueError(
            f"{name}={value_si} SI is below the physical lower bound {low}."
        )
    if high is not None and value_si > high:
        raise ValueError(
            f"{name}={value_si} SI is above the physical upper bound {high}."
        )
    if not math.isfinite(value_si):
        raise ValueError(f"{name} must be finite; got {value_si!r}.")


def normalize_to_canonical(name: str, value: float, unit: str) -> float:
    """
    Convert *value* (in *unit*) to the canonical storage unit for *name*.

    Prefer this over raw :func:`to_si` when you know what property you're
    dealing with — it also runs the bounds check.
    """
    spec = lookup_property(name)
    if unit not in spec.accepted_units and unit != spec.canonical_unit:
        raise ValueError(
            f"{name} does not accept unit {unit!r}; "
            f"accepted: {sorted(spec.accepted_units)}."
        )
    if unit == spec.canonical_unit:
        canonical = value
    else:
        # Convert through pint: value [unit] -> canonical_unit
        canonical = float(Q_(value, unit).to(spec.canonical_unit).magnitude)

    # For bounds validation we compare in SI regardless of canonical_unit
    # (some canonical units aren't SI base, e.g. density in kg/m^3).
    try:
        value_si = float(Q_(value, unit).to_base_units().magnitude)
        validate_value(name, value_si)
    except pint.DimensionalityError:
        # Dimensionless properties short-circuit the pint conversion.
        validate_value(name, canonical)

    return canonical
