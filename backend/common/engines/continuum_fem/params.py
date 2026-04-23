"""Pydantic schemas for Session 5.1 FEM problems.

Two solvers ship in Session 5.1:

- :class:`LinearElasticityProblem` — 3D linear elastic stress/strain
  with Dirichlet (displacement) and Neumann (traction) BCs.
- :class:`SteadyHeatProblem` — ``∇·(k ∇T) = 0`` with Dirichlet and
  Neumann (heat flux) BCs.

Both problems share :class:`BoxGeometry` — a rectangular box meshed
with structured hexes. A future session will add :class:`MeshFile`
(explicit ``.msh`` / ``.vtu`` via meshio) for general geometries.

Boundary conditions are described by face tags. For a box the tags
are ``"x-"``, ``"x+"``, ``"y-"``, ``"y+"``, ``"z-"``, ``"z+"`` — the
two faces perpendicular to each axis. Tag strings are deliberately
keyed to the geometry to keep the BC dict trivially serializable.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


FaceTag = Literal["x-", "x+", "y-", "y+", "z-", "z+"]


class BoxGeometry(BaseModel):
    """Axis-aligned rectangular box with a structured hex mesh.

    Lengths are in SI metres. Element counts are per-axis; the total
    element count is ``n_x * n_y * n_z``, each element a hex of size
    ``(Lx/n_x, Ly/n_y, Lz/n_z)``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["box"] = "box"
    length_x_m: float = Field(gt=0)
    length_y_m: float = Field(gt=0)
    length_z_m: float = Field(gt=0)
    n_elements_x: int = Field(default=20, ge=1, le=400)
    n_elements_y: int = Field(default=4, ge=1, le=400)
    n_elements_z: int = Field(default=4, ge=1, le=400)


class IsotropicElasticMaterial(BaseModel):
    """Isotropic linear elastic material.

    Two independent constants are required. We accept (E, nu) — the
    engineering-convention pair. Lamé (λ, μ) are derived internally
    via the standard identities:

        λ = E ν / ((1 + ν)(1 − 2ν))
        μ = E / (2 (1 + ν))

    Future extension: anisotropic ``IsotropicElasticMaterial`` replaced
    by a full C_ij tensor. Not needed for Session 5.1's acceptance
    cases (isotropic steel cantilever, isotropic rod).
    """

    model_config = ConfigDict(extra="forbid")

    youngs_modulus_pa: float = Field(gt=0, description="Young's modulus E (Pa).")
    poisson_ratio: float = Field(gt=-1.0, lt=0.5)

    @property
    def lame_lambda(self) -> float:
        E, nu = self.youngs_modulus_pa, self.poisson_ratio
        return E * nu / ((1 + nu) * (1 - 2 * nu))

    @property
    def lame_mu(self) -> float:
        E, nu = self.youngs_modulus_pa, self.poisson_ratio
        return E / (2 * (1 + nu))


class DirichletBC(BaseModel):
    """Prescribed displacement (elasticity) or temperature (heat).

    For elasticity, ``components`` selects which Cartesian components
    of the displacement vector are constrained on the face. Missing
    components are free. For heat, ``components`` is ignored — the
    full scalar value is prescribed.

    The ``value`` field is the prescribed value. Time-dependent BCs
    are not supported in Session 5.1 (steady problems only).
    """

    model_config = ConfigDict(extra="forbid")

    face: FaceTag
    # Elasticity: value_xyz = (u_x, u_y, u_z). None entries mean "do
    # not constrain that component on this face". Heat: first entry
    # (scalar) is used; others ignored.
    value_xyz: Tuple[Optional[float], Optional[float], Optional[float]] = (
        None, None, None,
    )
    # Heat-only shortcut so users don't have to wrap a scalar in a tuple.
    value_scalar: Optional[float] = None


class NeumannBC(BaseModel):
    """Traction (elasticity) or heat-flux (heat) on a face.

    For elasticity, ``traction_xyz`` is the Cauchy traction vector
    (Pa) applied uniformly across the face. The total force is
    ``traction * face_area``.

    For heat, ``heat_flux_w_per_m2`` is the outward normal flux
    component (W/m²). Positive = flux leaving the domain through
    this face.
    """

    model_config = ConfigDict(extra="forbid")

    face: FaceTag
    traction_xyz: Optional[Tuple[float, float, float]] = None
    heat_flux_w_per_m2: Optional[float] = None


class LinearElasticityProblem(BaseModel):
    """Full spec for a linear-elastic FEM solve."""

    model_config = ConfigDict(extra="forbid")

    geometry: BoxGeometry
    material: IsotropicElasticMaterial
    dirichlet: List[DirichletBC] = Field(default_factory=list)
    neumann: List[NeumannBC] = Field(default_factory=list)
    # Element order: P1 (Hex1) is 8-node linear; P2 (Hex2) is 27-node
    # quadratic. Quadratic hits cantilever accuracy at far lower
    # element counts (0.14% @ 20x2x2 vs 3.5% @ 40x4x4 for P1).
    element_order: Literal[1, 2] = 2

    @model_validator(mode="after")
    def _require_some_bc(self):
        if not self.dirichlet:
            raise ValueError(
                "elasticity problem needs at least one Dirichlet BC "
                "(otherwise the stiffness matrix is singular — rigid-body modes)"
            )
        return self


class SteadyHeatProblem(BaseModel):
    """Full spec for a steady-state heat-conduction solve."""

    model_config = ConfigDict(extra="forbid")

    geometry: BoxGeometry
    thermal_conductivity_w_per_m_k: float = Field(gt=0)
    dirichlet: List[DirichletBC] = Field(default_factory=list)
    neumann: List[NeumannBC] = Field(default_factory=list)
    element_order: Literal[1, 2] = 1

    @model_validator(mode="after")
    def _require_some_dirichlet(self):
        # Pure-Neumann heat problem is ill-posed (T defined up to a
        # constant). At least one Dirichlet anchor is required.
        if not self.dirichlet:
            raise ValueError(
                "steady heat problem needs at least one Dirichlet temperature "
                "BC (otherwise the solution is defined only up to a constant)"
            )
        return self
