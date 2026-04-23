"""Forcefield specs and the registry that indexes them.

See ``__init__.py`` for context. This file declares:

- The :class:`ForcefieldSpec` dataclass.
- The canonical default registry (:data:`default_registry`) populated
  with LJ, EAM (Cu/Ni/Al), Tersoff (Si/C), plus feature-flagged
  ReaxFF and MACE entries.

A spec's ``pair_coeff_template`` is a string format with two named
slots: ``{potential_path}`` (absolute path to the potential file at
run time) and ``{elements}`` (whitespace-joined element list in the
order declared by the caller). The renderer substitutes both before
writing ``in.lammps``.

Why not Jinja for the pair_coeff line? Because it is always one line
of a very fixed form, and ``str.format`` makes the expected slots
explicit at spec-declaration time. Jinja is reserved for the outer
``in.lammps`` template which has conditional blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple


# Directory on disk that holds shipped potential files. The
# registry declares entries relative to this path; the renderer
# resolves the final absolute path when writing ``in.lammps``.
POTENTIAL_DATA_DIR = Path(__file__).parent / "data"


class NoCompatibleForcefieldError(ValueError):
    """No registered forcefield covers the requested element set."""


@dataclass(frozen=True)
class ForcefieldSpec:
    """Declarative spec for one LAMMPS forcefield.

    Fields
    ------
    name
        Short identifier used by auto-select and by user overrides
        (e.g. ``"eam_cu"``, ``"lj"``, ``"tersoff_si"``).
    kind
        One of ``"lj"``, ``"eam"``, ``"tersoff"``, ``"reaxff"``,
        ``"ml"``. Controls which code path the renderer takes.
    applicable_elements
        Frozen set of element symbols this potential parameterizes.
        The LJ spec declares an empty set and uses ``matches_all=True``
        to signal that it's a universal fallback.
    matches_all
        If True, auto-select treats the potential as universally
        applicable (LJ only). Ranked last.
    cutoff_angstrom
        Radial cutoff of the potential. For RDF / neighbor-skin
        defaults and for the LJ ``pair_style`` argument.
    timestep_fs_recommended
        Recommended MD timestep in **femtoseconds** (our canonical
        internal unit). The renderer converts to the appropriate
        LAMMPS ``units`` style at generation time.
    units
        LAMMPS ``units`` directive: ``"metal"`` for eV/Å/ps (most
        solid-state potentials), ``"real"`` for kcal/mol/fs
        (biomolecular forcefields, ReaxFF), ``"lj"`` for reduced
        units (parametric LJ only).
    pair_style_line
        The exact ``pair_style ...`` line, including cutoff if
        applicable. Example: ``"pair_style eam/alloy"``.
    pair_coeff_template
        Format-string with slots ``{potential_path}`` and
        ``{elements}``. For LJ (no potential file), we emit
        ``"pair_coeff * * 1.0 1.0"`` directly — ``potential_path`` is
        empty in that case.
    potential_file
        Bare filename of the shipped potential (e.g.
        ``"Cu_u3.eam"``). ``None`` for LJ. Resolved to
        ``POTENTIAL_DATA_DIR / potential_file`` at render time.
    citation
        BibTeX-style citation string for the report + any license
        restrictions the user should be aware of.
    available
        If False (ReaxFF, ML potentials on systems without the right
        build), the spec is listed but ``auto_select`` skips it and
        explicit ``get(name)`` raises. Phase 4.1 ships with
        ReaxFF/ML set to False by default.
    extra_setup_lines
        Optional LAMMPS commands emitted right after ``pair_coeff``
        but before the ``velocity`` line. Example: ReaxFF needs
        ``fix reaxff all qeq/reax ...``. LJ/EAM/Tersoff don't use
        this.
    """

    name: str
    kind: str  # "lj" | "eam" | "tersoff" | "reaxff" | "ml"
    applicable_elements: FrozenSet[str]
    cutoff_angstrom: float
    timestep_fs_recommended: float
    units: str
    pair_style_line: str
    pair_coeff_template: str
    citation: str
    potential_file: Optional[str] = None
    matches_all: bool = False
    available: bool = True
    extra_setup_lines: Tuple[str, ...] = field(default_factory=tuple)

    def covers(self, elements) -> bool:
        """Return True if this spec covers every element in ``elements``."""
        if self.matches_all:
            return True
        return set(elements) <= self.applicable_elements

    def resolve_potential_path(self, data_dir: Optional[Path] = None) -> Optional[Path]:
        """Return the absolute path to the potential file.

        ``data_dir`` defaults to :data:`POTENTIAL_DATA_DIR`. Callers
        writing into a staging dir may pass their own directory
        (e.g. a test ``tmp_path``).
        """
        if self.potential_file is None:
            return None
        return (data_dir or POTENTIAL_DATA_DIR) / self.potential_file


class ForcefieldRegistry:
    """Indexed collection of :class:`ForcefieldSpec` objects.

    Two lookup modes:

    1. Explicit — ``reg.get("eam_cu")`` returns exactly that spec,
       or raises ``KeyError`` if unknown (or ``NoCompatibleForcefieldError``
       if the spec exists but is marked unavailable).
    2. Auto-select — ``reg.auto_select({"Cu"})`` returns the highest
       priority *available* spec that covers the given elements.
       Priority is "first declared wins" with LJ (``matches_all``)
       ranked last.
    """

    def __init__(self, specs: Optional[List[ForcefieldSpec]] = None):
        self._specs: Dict[str, ForcefieldSpec] = {}
        if specs:
            for s in specs:
                self.register(s)

    def register(self, spec: ForcefieldSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"forcefield {spec.name!r} already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ForcefieldSpec:
        """Return the spec by name. Raises KeyError if unknown.

        Raises :exc:`NoCompatibleForcefieldError` if the spec exists
        but is flagged unavailable (e.g. ReaxFF without the right
        LAMMPS build).
        """
        if name not in self._specs:
            available = ", ".join(sorted(self._specs))
            raise KeyError(
                f"unknown forcefield {name!r}. Registered: {available}"
            )
        spec = self._specs[name]
        if not spec.available:
            raise NoCompatibleForcefieldError(
                f"forcefield {name!r} is declared but not available in this "
                f"environment (e.g. LAMMPS build missing required package)."
            )
        return spec

    def list_compatible(self, elements) -> List[ForcefieldSpec]:
        """Return all available specs whose ``covers(elements)`` is True.

        Order is: explicit element-match first, LJ (``matches_all``)
        last. Within a group, insertion order is preserved.
        """
        exact = [
            s for s in self._specs.values()
            if s.available and not s.matches_all and s.covers(elements)
        ]
        universal = [
            s for s in self._specs.values()
            if s.available and s.matches_all and s.covers(elements)
        ]
        return exact + universal

    def auto_select(self, elements) -> ForcefieldSpec:
        """Pick the highest-priority available spec covering ``elements``.

        Raises :exc:`NoCompatibleForcefieldError` if nothing matches.
        LJ counts as a match for any element set, so the error fires
        only if LJ has been explicitly removed or marked unavailable.
        """
        candidates = self.list_compatible(elements)
        if not candidates:
            raise NoCompatibleForcefieldError(
                f"no forcefield covers elements {sorted(set(elements))!r}. "
                f"Registered: {sorted(self._specs)}"
            )
        return candidates[0]

    def names(self) -> List[str]:
        return list(self._specs)

    def __contains__(self, name: str) -> bool:
        return name in self._specs


# ---------------------------------------------------------------------------
# Canonical default registry
# ---------------------------------------------------------------------------


def _build_default_specs() -> List[ForcefieldSpec]:
    """Return the specs loaded into ``default_registry``.

    Availability of EAM / Tersoff depends on whether the shipped
    potential files are present on disk. We check at build time and
    set ``available`` accordingly. This lets the registry still *list*
    the potentials (for docs + error messages) even if the data dir
    hasn't been populated — useful in CI before we've pulled the
    NIST files down.
    """
    specs: List[ForcefieldSpec] = []

    # -- EAM --------------------------------------------------------
    # NIST Interatomic Potentials Repository. Public-domain work of
    # US federal employees (17 USC §105). See
    # backend/common/engines/lammps_input/forcefields/data/LICENSE.txt.
    #
    # File choices (picked for coverage of roadmap Cu/Ni/Al):
    # - Cu: Foiles, Baskes, Daw (1986) — Cu_u3.eam (single-element)
    # - Ni: Mishin et al. (1999) — Ni99_v2.eam.alloy
    # - Al: Mishin et al. (1999) — Al99_v2.eam.alloy
    # LAMMPS has two EAM pair_styles with *different* pair_coeff syntax:
    # - ``pair_style eam`` (single-element, legacy): ``pair_coeff * * file``
    #   — no trailing element labels.
    # - ``pair_style eam/alloy``: ``pair_coeff * * file E1 E2 ...`` — one
    #   label per atom type, in data-file order.
    # The renderer formats {elements} into the template either way; the
    # template string differs per style so Cu_u3 (single-element) emits
    # just the path while Ni99/Al99 (alloy) append ``Ni`` / ``Al``.
    _SINGLE_EAM_TEMPLATE = "pair_coeff * * {potential_path}"
    _ALLOY_EAM_TEMPLATE = "pair_coeff * * {potential_path} {elements}"
    for element, filename, style, citation in [
        (
            "Cu",
            "Cu_u3.eam",
            "eam",
            "Foiles, Baskes, Daw 1986 Phys.Rev.B 33, 7983. NIST IPR.",
        ),
        (
            "Ni",
            "Ni99_v2.eam.alloy",
            "eam/alloy",
            "Mishin et al. 1999 Acta Mater. 47, 3273. NIST IPR.",
        ),
        (
            "Al",
            "Al99_v2.eam.alloy",
            "eam/alloy",
            "Mishin et al. 1999 Phys.Rev.B 59, 3393. NIST IPR.",
        ),
    ]:
        path = POTENTIAL_DATA_DIR / filename
        template = _ALLOY_EAM_TEMPLATE if style == "eam/alloy" else _SINGLE_EAM_TEMPLATE
        specs.append(
            ForcefieldSpec(
                name=f"eam_{element.lower()}",
                kind="eam",
                applicable_elements=frozenset({element}),
                cutoff_angstrom=6.0,  # typical EAM cutoff
                timestep_fs_recommended=1.0,
                units="metal",
                pair_style_line=f"pair_style {style}",
                pair_coeff_template=template,
                potential_file=filename,
                citation=citation,
                available=path.is_file(),
            )
        )

    # -- Tersoff ----------------------------------------------------
    # Si: J. Tersoff, Phys.Rev.B 38, 9902 (1988). Parameters shipped
    # in LAMMPS' own potentials/ dir; we ship a mirror.
    # C (diamond / graphene): J. Tersoff, Phys.Rev.B 39, 5566 (1989).
    for element, filename, citation in [
        (
            "Si",
            "Si.tersoff",
            "Tersoff 1988 Phys.Rev.B 38, 9902. LAMMPS-distributed.",
        ),
        (
            "C",
            "C.tersoff",
            "Tersoff 1989 Phys.Rev.B 39, 5566. LAMMPS-distributed.",
        ),
    ]:
        path = POTENTIAL_DATA_DIR / filename
        specs.append(
            ForcefieldSpec(
                name=f"tersoff_{element.lower()}",
                kind="tersoff",
                applicable_elements=frozenset({element}),
                cutoff_angstrom=3.2 if element == "Si" else 2.5,
                timestep_fs_recommended=0.5,  # Tersoff prefers tighter dt than EAM
                units="metal",
                pair_style_line="pair_style tersoff",
                pair_coeff_template="pair_coeff * * {potential_path} {elements}",
                potential_file=filename,
                citation=citation,
                available=path.is_file(),
            )
        )

    # -- ReaxFF (feature-flagged) -----------------------------------
    # Requires LAMMPS built with the REAXFF package AND a per-system
    # .reax parameter file. We register a placeholder so auto-select
    # can report it as "known but not available".
    specs.append(
        ForcefieldSpec(
            name="reaxff",
            kind="reaxff",
            # Broad coverage; typical CHONSSi sets ship with LAMMPS.
            applicable_elements=frozenset({"C", "H", "O", "N", "S", "Si"}),
            cutoff_angstrom=10.0,
            timestep_fs_recommended=0.25,  # ReaxFF needs very short dt
            units="real",
            pair_style_line="pair_style reaxff NULL",
            pair_coeff_template="pair_coeff * * {potential_path} {elements}",
            potential_file=None,  # user must supply
            citation="van Duin et al. 2001 J.Phys.Chem.A 105, 9396.",
            available=False,  # flip when a concrete .reax file + LAMMPS build is wired
            extra_setup_lines=("fix qeq all qeq/reax 1 0.0 10.0 1.0e-6 reax/c",),
        )
    )

    # -- ML (MACE / NequIP) placeholder — Phase 8 -------------------
    specs.append(
        ForcefieldSpec(
            name="mace",
            kind="ml",
            applicable_elements=frozenset(),  # spec-by-spec when trained
            matches_all=False,
            cutoff_angstrom=5.0,
            timestep_fs_recommended=1.0,
            units="metal",
            pair_style_line="pair_style mace",
            pair_coeff_template="pair_coeff * * {potential_path} {elements}",
            potential_file=None,
            citation="Batatia et al. 2022, MACE. Requires custom LAMMPS build.",
            available=False,
        )
    )

    # -- Lennard-Jones — parametric fallback ------------------------
    # Reduced units; cutoff=2.5σ is the textbook default. The
    # renderer emits ``pair_coeff * * 1.0 1.0`` so the system runs
    # in dimensionless form.
    specs.append(
        ForcefieldSpec(
            name="lj",
            kind="lj",
            applicable_elements=frozenset(),
            matches_all=True,
            cutoff_angstrom=2.5,  # interpreted in reduced units
            # LJ is dimensionless — the fs-named field is meaningless
            # for this spec. Renderer ignores it and uses the dedicated
            # ``timestep_lj_reduced`` field on LAMMPSInputParams instead
            # (default 0.005). Kept as 0.005 here so the value still
            # reflects the physics in case someone inspects the spec.
            timestep_fs_recommended=0.005,
            units="lj",
            pair_style_line="pair_style lj/cut 2.5",
            pair_coeff_template="pair_coeff * * 1.0 1.0",
            potential_file=None,
            citation="Lennard-Jones 1931 Proc.Phys.Soc. 43, 461.",
            available=True,
        )
    )

    return specs


default_registry = ForcefieldRegistry(_build_default_specs())
