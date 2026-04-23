"""Render a LAMMPS input deck from a structure + :class:`LAMMPSInputParams`.

Two artifacts are produced:

1. ``structure.data`` — a LAMMPS ``atom_style atomic`` data file
   generated through ``pymatgen.io.lammps.data.LammpsData``. That
   class already handles orthorhombic vs triclinic boxes, per-type
   masses, and the ``Atoms`` section in atomic coordinates.
2. ``in.lammps`` — the input script itself, rendered from a Jinja2
   template that pulls everything it needs from the params + the
   chosen :class:`~backend.common.engines.lammps_input.forcefields.ForcefieldSpec`.

Structure input
---------------

Accepts either:

- A pymatgen ``Structure`` (full-featured path).
- A ``dict`` with keys ``{lattice, species, frac_coords}`` — matches
  the shape the QE renderer accepts and the shape the Celery task
  passes through when it doesn't want to rehydrate pymatgen.

Unit handling at a glance
-------------------------

We keep timesteps in **fs** through the entire API. At render time:

- ``units metal`` ⇒ emit ``timestep`` in ps  (fs / 1000)
- ``units real``  ⇒ emit ``timestep`` in fs  (verbatim)
- ``units lj``    ⇒ emit ``timestep`` verbatim; user is responsible
  for meaning (reduced-time treatment).

Damping times (``t_damp``, ``p_damp``) are given to LAMMPS in the
same time unit as the timestep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from jinja2 import Environment

from .forcefields import (
    ForcefieldRegistry,
    ForcefieldSpec,
    NoCompatibleForcefieldError,
    default_registry,
)
from .params import LAMMPSInputParams


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------


@dataclass
class RenderedLAMMPSInput:
    """Return of :func:`generate_lammps_input`."""

    input_text: str
    data_text: str
    input_filename: str = "in.lammps"
    data_filename: str = "structure.data"
    forcefield: Optional[ForcefieldSpec] = None
    potential_file: Optional[str] = None  # bare filename, if any
    n_steps: int = 0
    timestep_in_units: float = 0.0  # emitted value (after unit conversion)


# ---------------------------------------------------------------------------
# Structure normalization
# ---------------------------------------------------------------------------


@dataclass
class _NormalizedStructure:
    species: List[str]
    # Lazily-built pymatgen Structure (we only need it to hand to LammpsData).
    pmg: Any


def _normalize_structure(structure: Any) -> _NormalizedStructure:
    # pymatgen Structure (duck-typed)
    if hasattr(structure, "lattice") and hasattr(structure, "frac_coords"):
        species = [str(s) for s in structure.species]
        return _NormalizedStructure(species=species, pmg=structure)

    if isinstance(structure, dict):
        needed = {"lattice", "species", "frac_coords"}
        if not needed <= set(structure):
            raise ValueError(
                f"structure dict must have keys {needed}; got {sorted(structure)}"
            )
        # Defer pymatgen import until here — the function still
        # raises for bad dicts before the import cost.
        from pymatgen.core import Lattice, Structure

        lat = Lattice(structure["lattice"])
        pmg = Structure(
            lat,
            species=list(structure["species"]),
            coords=list(structure["frac_coords"]),
            coords_are_cartesian=False,
        )
        return _NormalizedStructure(species=list(structure["species"]), pmg=pmg)

    raise ValueError(
        "generate_lammps_input: structure must be a pymatgen Structure "
        "or a dict with keys {lattice, species, frac_coords}."
    )


def _unique_species_in_order(species: Iterable[str]) -> List[str]:
    seen: Dict[str, None] = {}
    for s in species:
        if s not in seen:
            seen[s] = None
    return list(seen)


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


def _timestep_in_units(timestep_fs: float, units: str) -> float:
    """Convert fs to the LAMMPS time unit for the chosen ``units`` style.

    Only valid for physical-time styles (``metal``, ``real``). For
    ``lj`` there is no fs ↔ reduced time mapping; callers must use
    :func:`_resolve_lj_time_fields` instead. Calling this with
    ``units='lj'`` raises — Session 4.1 silently passed through,
    producing catastrophically large timesteps for LJ runs.
    """
    if units == "metal":
        return timestep_fs / 1000.0  # ps
    if units == "real":
        return timestep_fs  # fs
    if units == "lj":
        raise ValueError(
            "LJ time units are dimensionless; use timestep_lj_reduced "
            "/ duration_lj_reduced on LAMMPSInputParams instead of fs fields."
        )
    raise ValueError(f"unsupported LAMMPS units style: {units!r}")


# Canonical LJ defaults per the LAMMPS examples/ bench runs. dt* of
# 0.005 is the universally-cited choice for textbook LJ fluids.
_LJ_DEFAULT_DT_REDUCED = 0.005
_LJ_DEFAULT_DURATION_REDUCED = 500.0


def _resolve_lj_time_fields(params: LAMMPSInputParams) -> Dict[str, float]:
    """Compute (timestep, damping, duration, n_steps) for LJ runs.

    Everything here is in LJ reduced units. If the user left the
    dedicated LJ fields unset we fall back to community defaults so
    a naive ``LAMMPSInputParams()`` run of an Ar-like system doesn't
    crash LAMMPS.
    """
    dt = params.timestep_lj_reduced or _LJ_DEFAULT_DT_REDUCED
    duration = params.duration_lj_reduced or _LJ_DEFAULT_DURATION_REDUCED
    t_damp = params.t_damp_lj_reduced or 100.0 * dt
    p_damp = params.p_damp_lj_reduced or 1000.0 * dt
    n_steps = max(1, int(round(duration / dt)))
    return {
        "timestep": dt,
        "t_damp": t_damp,
        "p_damp": p_damp,
        "duration": duration,
        "n_steps": n_steps,
    }


# ---------------------------------------------------------------------------
# Jinja template
# ---------------------------------------------------------------------------


_IN_LAMMPS_TEMPLATE = """\
# LAMMPS input — generated by ORION Session 4.1.
# Prefix: {{ prefix }}
# Forcefield: {{ ff.name }} ({{ ff.kind }})
# Citation: {{ ff.citation }}

units           {{ ff.units }}
atom_style      atomic
boundary        p p p

read_data       {{ data_filename }}

{{ pair_style_line }}
{{ pair_coeff_line }}
{% for line in ff_extra_setup_lines -%}
{{ line }}
{% endfor %}
{% if init_velocity -%}
velocity        all create {{ '%.4f' % temperature_k }} {{ velocity_seed }} dist gaussian
{% endif %}
{% for line in extra_commands -%}
{{ line }}
{% endfor %}
timestep        {{ '%.8f' % timestep_in_units }}

{% if ensemble == "nve" -%}
fix             1 all nve
{%- elif ensemble == "nvt_nose_hoover" -%}
fix             1 all nvt temp {{ '%.4f' % temperature_k }} {{ '%.4f' % temperature_k }} {{ '%.6f' % t_damp_units }}
{%- elif ensemble == "nvt_langevin" -%}
fix             1 all nve
fix             2 all langevin {{ '%.4f' % temperature_k }} {{ '%.4f' % temperature_k }} {{ '%.6f' % t_damp_units }} {{ velocity_seed }}
{%- elif ensemble == "npt" -%}
fix             1 all npt temp {{ '%.4f' % temperature_k }} {{ '%.4f' % temperature_k }} {{ '%.6f' % t_damp_units }} iso {{ '%.4f' % pressure_bar }} {{ '%.4f' % pressure_bar }} {{ '%.6f' % p_damp_units }}
{%- endif %}

thermo          {{ thermo_every }}
thermo_style    custom {{ thermo_columns }}
dump            1 all custom {{ dump_every }} {{ dump_filename }} id type x y z xu yu zu vx vy vz

run             {{ n_steps }}
"""


# LAMMPS input is plain text, not HTML — autoescape must be off or
# ``&`` in paths gets turned into ``&amp;`` and LAMMPS chokes.
_jinja_env = Environment(
    autoescape=False,
    trim_blocks=False,
    lstrip_blocks=False,
    keep_trailing_newline=True,
)
_lammps_template = _jinja_env.from_string(_IN_LAMMPS_TEMPLATE)


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------


def generate_lammps_input(
    structure: Any,
    params: LAMMPSInputParams,
    registry: Optional[ForcefieldRegistry] = None,
    *,
    potential_dir: Optional[Path] = None,
) -> RenderedLAMMPSInput:
    """Render ``in.lammps`` and ``structure.data`` for a MD run.

    Parameters
    ----------
    structure
        pymatgen ``Structure`` or dict (see module docstring).
    params
        :class:`LAMMPSInputParams`.
    registry
        Optional custom :class:`ForcefieldRegistry`. Defaults to
        :data:`backend.common.engines.lammps_input.forcefields.default_registry`.
    potential_dir
        Absolute path used when rendering ``pair_coeff``. Defaults to
        ``ForcefieldSpec.resolve_potential_path()`` i.e. the shipped
        ``data/`` directory. Tests pass their own ``tmp_path``.

    Returns
    -------
    :class:`RenderedLAMMPSInput` with both file bodies and bookkeeping.
    """
    registry = registry or default_registry
    ns = _normalize_structure(structure)
    if not ns.species:
        raise ValueError("structure has no atoms")
    unique = _unique_species_in_order(ns.species)

    # Forcefield selection
    if params.forcefield_name:
        ff = registry.get(params.forcefield_name)
        # If the user chose an element-specific FF, it must cover them.
        if not ff.covers(unique):
            raise NoCompatibleForcefieldError(
                f"forcefield {ff.name!r} does not cover elements {unique!r}"
            )
    else:
        ff = registry.auto_select(unique)

    # Potential path written into pair_coeff. We default to the bare
    # filename so the rendered deck is relocatable — ``write_lammps_inputs``
    # copies the potential file alongside, and LAMMPS resolves the
    # reference relative to cwd. If the user passes ``potential_dir``
    # explicitly, we honor it (useful when the caller prefers a shared
    # potentials directory over a per-run copy).
    if ff.potential_file is None:
        pair_coeff_path_field = ""
    elif potential_dir is not None:
        pair_coeff_path_field = str(Path(potential_dir) / ff.potential_file)
    else:
        pair_coeff_path_field = ff.potential_file

    pair_coeff_line = ff.pair_coeff_template.format(
        potential_path=pair_coeff_path_field,
        elements=" ".join(unique),
    )

    # LAMMPS data file — via pymatgen.
    from pymatgen.io.lammps.data import LammpsData

    data_obj = LammpsData.from_structure(ns.pmg, atom_style="atomic")
    data_text = data_obj.get_str()

    # Unit conversion. LJ runs are dimensionless; resolve from the
    # dedicated reduced-unit fields. Physical-units forcefields use fs
    # fields as before.
    if ff.units == "lj":
        lj = _resolve_lj_time_fields(params)
        dt_units = lj["timestep"]
        t_damp_units = lj["t_damp"]
        p_damp_units = lj["p_damp"]
        n_steps = lj["n_steps"]
    else:
        dt_units = _timestep_in_units(params.timestep_fs, ff.units)
        t_damp_units = _timestep_in_units(params.t_damp_fs_resolved, ff.units)
        p_damp_units = _timestep_in_units(params.p_damp_fs_resolved, ff.units)
        n_steps = params.n_steps

    input_text = _lammps_template.render(
        prefix=params.run_prefix,
        ff=ff,
        pair_style_line=ff.pair_style_line,
        pair_coeff_line=pair_coeff_line,
        ff_extra_setup_lines=list(ff.extra_setup_lines),
        data_filename="structure.data",
        init_velocity=params.init_velocity,
        temperature_k=params.temperature_k,
        pressure_bar=params.pressure_bar,
        velocity_seed=params.velocity_seed,
        extra_commands=list(params.extra_commands),
        timestep_in_units=dt_units,
        t_damp_units=t_damp_units,
        p_damp_units=p_damp_units,
        ensemble=params.ensemble,
        thermo_every=params.thermo_every,
        thermo_columns=params.thermo_columns,
        dump_every=params.dump_every,
        dump_filename=params.dump_filename,
        n_steps=n_steps,
    )

    return RenderedLAMMPSInput(
        input_text=input_text,
        data_text=data_text,
        forcefield=ff,
        potential_file=ff.potential_file,
        n_steps=n_steps,
        timestep_in_units=dt_units,
    )


# ---------------------------------------------------------------------------
# Elastic strain helper (Session 4.3b)
# ---------------------------------------------------------------------------


# Voigt convention for the 6-component strain vector: (xx, yy, zz, yz, xz, xy).
_VOIGT_DIAGONAL_AXES = {0: "x", 1: "y", 2: "z"}


def strain_extra_commands(voigt_index: int, strain_value: float) -> List[str]:
    """Return the LAMMPS lines that apply a ±ε diagonal strain at run start.

    Uses ``change_box`` with the ``remap`` option so atom coordinates
    scale with the box — the canonical approach for elastic-constant
    workflows. Off-diagonal (shear, Voigt 3-5) strains aren't emitted
    here; they require the triclinic ``change_box tilt`` path plus a
    different thermo_style, and aren't on the Session 4.3b roadmap.

    Returns a list suitable for appending to
    :attr:`LAMMPSInputParams.extra_commands`. The caller should also
    set ``thermo_columns`` to include ``pxx pyy pzz`` so the analyzer
    can read per-component stress.
    """
    if voigt_index not in _VOIGT_DIAGONAL_AXES:
        raise ValueError(
            f"strain_extra_commands only supports diagonal Voigt indices "
            f"(0..2); got {voigt_index!r}"
        )
    axis = _VOIGT_DIAGONAL_AXES[voigt_index]
    # LAMMPS ``change_box all x scale 1.005 remap`` uniformly scales the
    # box x-extent by 1+ε. ``remap`` drags atoms so fractional coords
    # stay constant (equivalent to applying a deformation gradient).
    scale = 1.0 + strain_value
    return [
        f"# Apply diagonal strain ε_{voigt_index} = {strain_value:+.6f} along {axis}",
        f"change_box all {axis} scale {scale:.8f} remap",
    ]


# Thermo columns string that includes the per-component pressure tensor.
# Elastic workflows substitute this into ``params.thermo_columns`` so the
# analyzer can read ``final_thermo.Pxx`` etc.
THERMO_COLUMNS_STRESS = "step temp pe ke etotal press vol pxx pyy pzz pxy pxz pyz"


# ---------------------------------------------------------------------------
# Convenience helper — write both files to a directory
# ---------------------------------------------------------------------------


def write_lammps_inputs(
    rendered: RenderedLAMMPSInput,
    work_dir: Path,
    *,
    copy_potential: bool = True,
    potential_source_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Write ``in.lammps`` + ``structure.data`` into ``work_dir``.

    If ``copy_potential`` is True and the forcefield declares a
    potential file, the file is copied alongside so the input deck
    is self-contained. Callers can disable the copy when they'd
    rather reference the source directory directly (saves disk
    for big ML potentials).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    in_path = work_dir / rendered.input_filename
    in_path.write_text(rendered.input_text)
    paths["input"] = in_path

    data_path = work_dir / rendered.data_filename
    data_path.write_text(rendered.data_text)
    paths["data"] = data_path

    if copy_potential and rendered.forcefield and rendered.forcefield.potential_file:
        src = rendered.forcefield.resolve_potential_path(potential_source_dir)
        if src and src.is_file():
            dst = work_dir / rendered.forcefield.potential_file
            dst.write_bytes(src.read_bytes())
            paths["potential"] = dst

    return paths
