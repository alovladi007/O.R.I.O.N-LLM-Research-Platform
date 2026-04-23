"""Pydantic schemas for the Session 5.2 kMC engine.

Scope
-----

A *rejection-free* kinetic Monte Carlo (Gillespie / BKL-style) engine
for defect migration on a simple-cubic lattice. Events:

- **vacancy hop** — a vacancy exchanges with one of its 6 neighbors.
- **interstitial hop** — an interstitial exchanges with one of its 6
  neighbors.
- **pair annihilation** — a vacancy + interstitial at neighboring
  sites combine and vanish (instantaneous, rate ≫ hop rates).

Rates follow the Arrhenius form

    r = ν₀ · exp(−E_a / k_B T)

with ν₀ and E_a supplied per event type through :class:`EventCatalog`.

Rate convention
---------------

``hop_attempt_frequency_hz`` in :class:`EventType` is the **per-direction**
attempt rate — i.e. the rate at which a defect tries to jump into one
specific neighboring site. On a 6-coordinated simple-cubic lattice, the
total outgoing rate of a single defect is ``6 × r``. Under this
convention, the Einstein diffusion constant for a single walker is

    D = a² · ν_hop = a² · ν₀ · exp(−E_a / k_B T)

where ``a`` is the lattice parameter. Derivation: per second the
walker takes 6·ν_hop steps of mean-square displacement a²/6 per step
(each of 6 directions equally likely, each with ε = a), so
``⟨|Δr|²⟩/t = 6·ν_hop · a²/6 = a² ν_hop``, and for 3D isotropic
diffusion ``⟨|Δr|²⟩ = 6 D t`` gives ``D = a² ν_hop / 6 * 6 = a² ν_hop``.

The roadmap text gives ``D = a² ν₀ / 6 · exp(...)``; that form is
only consistent with a *total* attempt frequency convention. We pick
the per-direction convention because it makes the event catalog
and the expected analytical D both non-ambiguous. The live acceptance
test in :mod:`tests.test_mesoscale_kmc` checks against
``D = a² ν₀ exp(−E/kT)``.

Out of scope
------------

- Elastic strain fields / defect-defect long-range interactions.
- Non-cubic lattices.
- Variable coordination / grain boundaries / surfaces.
- Temperature swing during a run (``temperature_k`` is constant).
- Cluster tracking beyond size-1 single defects.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


SpeciesName = Literal["vacancy", "interstitial"]


class LatticeSpec(BaseModel):
    """Simple-cubic lattice geometry.

    Lengths are in metres. The lattice parameter ``a_m`` times the
    grid dimension gives the physical box length per axis; periodic
    boundaries are always on for Session 5.2.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["simple_cubic"] = "simple_cubic"
    a_m: float = Field(gt=0, description="Lattice parameter in metres.")
    nx: int = Field(ge=2, le=2048)
    ny: int = Field(ge=2, le=2048)
    nz: int = Field(ge=2, le=2048)

    @property
    def n_sites(self) -> int:
        return self.nx * self.ny * self.nz


class EventType(BaseModel):
    """One entry in the event catalog.

    Per-direction attempt rate + activation energy. At temperature T,
    the rate is ``hop_attempt_frequency_hz * exp(-E_a / k_B T)``.
    """

    model_config = ConfigDict(extra="forbid")

    hop_attempt_frequency_hz: float = Field(
        gt=0,
        description=(
            "Attempt frequency ν₀ (Hz) — per-direction. For simple cubic "
            "the total outgoing rate of a defect is 6 × this."
        ),
    )
    activation_energy_ev: float = Field(
        ge=0,
        description="Migration barrier E_a (eV).",
    )


class EventCatalog(BaseModel):
    """Per-species event parameters + optional annihilation.

    Vacancy and interstitial each hop with their own (ν₀, E_a). The
    annihilation event fires *instantaneously* whenever a vacancy
    and an interstitial end up on the same or adjacent site after a
    hop — modelled with an effectively infinite rate. In Session 5.2
    MVP we do the annihilation check immediately after a hop rather
    than as a queued event (keeps the rate table smaller and avoids
    modelling infinite rates directly).
    """

    model_config = ConfigDict(extra="forbid")

    vacancy: EventType
    interstitial: EventType
    # Recombination radius in units of the lattice parameter. 1.0 means
    # "vacancy and interstitial at the same site" (most common minimal
    # convention). Sqrt(3)≈1.732 would also catch body-diagonal
    # neighbours. We use 1.0 for Session 5.2; larger values are a
    # future extension.
    recombination_radius_a: float = Field(default=1.0, gt=0, le=2.0)


class KMCProblem(BaseModel):
    """Full spec for a Session 5.2 run.

    Stopping criteria: at least one of ``max_steps`` or ``max_time_s``
    must be set. The run halts the moment either cap is reached.
    ``observe_every_n_steps`` controls how often we snapshot for the
    concentration / MSD time series (0 disables the stream).
    """

    model_config = ConfigDict(extra="forbid")

    lattice: LatticeSpec
    catalog: EventCatalog
    temperature_k: float = Field(gt=0)

    # Initial populations as integer counts.
    n_initial_vacancies: int = Field(default=0, ge=0)
    n_initial_interstitials: int = Field(default=0, ge=0)

    # Stopping criteria.
    max_steps: Optional[int] = Field(default=None, ge=1)
    max_time_s: Optional[float] = Field(default=None, gt=0)

    # Observation cadence (steps). 0 → observe only start + end.
    observe_every_n_steps: int = Field(default=100, ge=0)

    # Reproducibility.
    seed: int = Field(default=0, ge=0)
