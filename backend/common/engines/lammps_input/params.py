"""Parameter schema for LAMMPS input generation (Session 4.1).

Everything the user supplies goes through :class:`LAMMPSInputParams`.
The renderer accepts nothing else — no stray kwargs, no dict-based
escape hatches except ``extra_commands`` which is an explicit list
of raw LAMMPS lines appended just before ``run``.

Unit convention
---------------

All parameters with a time dimension are in **femtoseconds**. The
renderer converts to the LAMMPS ``units`` style of the chosen
forcefield:

- ``units metal`` → timestep in **picoseconds** (divide fs by 1000).
- ``units real`` → timestep in **femtoseconds** (no conversion).
- ``units lj``   → timestep in reduced time units (user supplies
  ``timestep_fs`` but we treat it as the reduced value — LJ has no
  physical time scale without a mass/ε choice).

Temperature is always in Kelvin; pressure in bar (LAMMPS' default).

Ensembles
---------

- ``nve`` — microcanonical (no thermostat / barostat).
- ``nvt_nose_hoover`` — Nose-Hoover chain thermostat. Damping
  parameter defaults to ``100 * timestep_fs``.
- ``nvt_langevin`` — Langevin thermostat (stochastic). Same damping
  default.
- ``npt`` — Parrinello-Rahman barostat + Nose-Hoover thermostat.
  Pressure damping defaults to ``1000 * timestep_fs``.

Damping defaults mirror the LAMMPS manual's recommendations. They're
just defaults — the user can override via ``t_damp_fs`` / ``p_damp_fs``.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


Ensemble = Literal["nve", "nvt_nose_hoover", "nvt_langevin", "npt"]


class LAMMPSInputParams(BaseModel):
    """Parameters that drive ``in.lammps`` generation.

    Fields split into three groups: the simulation proper, the
    forcefield choice, and I/O.
    """

    model_config = ConfigDict(extra="forbid")

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    ensemble: Ensemble = Field(
        default="nvt_nose_hoover",
        description="Which integrator LAMMPS uses for the production run.",
    )
    temperature_k: float = Field(
        default=300.0,
        gt=0,
        description="Target temperature in Kelvin. Used for thermostat + initial velocities.",
    )
    pressure_bar: float = Field(
        default=1.0,
        description="Target pressure in bar. Used only by NPT.",
    )
    timestep_fs: float = Field(
        default=1.0,
        gt=0,
        description="Integration timestep in femtoseconds.",
    )
    duration_ps: float = Field(
        default=100.0,
        gt=0,
        description="Total simulation time in picoseconds. Converted to step count at render time.",
    )

    # Damping — None ⇒ "100*timestep" (T) or "1000*timestep" (P).
    t_damp_fs: Optional[float] = Field(
        default=None,
        gt=0,
        description="Thermostat damping time. None ⇒ 100 × timestep_fs.",
    )
    p_damp_fs: Optional[float] = Field(
        default=None,
        gt=0,
        description="Barostat damping time. None ⇒ 1000 × timestep_fs.",
    )

    # Initial velocities
    velocity_seed: int = Field(
        default=12345,
        ge=1,
        description="LAMMPS ``velocity create`` RNG seed.",
    )
    init_velocity: bool = Field(
        default=True,
        description="If True, emit a ``velocity all create`` line at the chosen temperature.",
    )

    # ------------------------------------------------------------------
    # Forcefield
    # ------------------------------------------------------------------
    forcefield_name: Optional[str] = Field(
        default=None,
        description=(
            "Explicit forcefield name (e.g. 'eam_cu'). When None, the "
            "renderer consults the forcefield registry's auto-select."
        ),
    )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    thermo_every: int = Field(
        default=100,
        ge=1,
        description="Emit a thermo line every N steps.",
    )
    dump_every: int = Field(
        default=1000,
        ge=1,
        description="Emit a trajectory frame every N steps.",
    )
    dump_filename: str = Field(default="dump.lammpstrj")
    log_filename: str = Field(default="log.lammps")
    run_prefix: str = Field(
        default="orion",
        max_length=32,
        description="Used as LAMMPS ``variable prefix string``; helps trace runs.",
    )

    # Extra commands inserted verbatim into ``in.lammps`` between
    # ``velocity`` and the ``fix`` for the ensemble. Use sparingly.
    extra_commands: List[str] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived values — resolved once at render time, not user-facing.
    # ------------------------------------------------------------------
    @property
    def t_damp_fs_resolved(self) -> float:
        return self.t_damp_fs if self.t_damp_fs is not None else 100.0 * self.timestep_fs

    @property
    def p_damp_fs_resolved(self) -> float:
        return self.p_damp_fs if self.p_damp_fs is not None else 1000.0 * self.timestep_fs

    @property
    def n_steps(self) -> int:
        """Total integration steps = duration_ps * 1000 / timestep_fs."""
        steps = (self.duration_ps * 1000.0) / self.timestep_fs
        # Round up to at least 1. duration_ps * 1000 is already in fs;
        # dividing by timestep_fs yields a float — we floor to int.
        return max(1, int(round(steps)))

    @model_validator(mode="after")
    def _validate_ensemble_needs(self):
        if self.ensemble == "npt" and self.pressure_bar is None:
            raise ValueError("npt ensemble requires pressure_bar")
        return self
