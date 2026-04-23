"""FEM continuum solver (Phase 5 / Session 5.1).

Pure-Python finite-element implementation using :mod:`scikit-fem`.
Session 5.1 ships two solvers against the roadmap's acceptance cases:

- :func:`solve_linear_elasticity` — 3D linear elasticity (Cauchy
  stress, Hex1/Hex2 elements, Dirichlet + Neumann BCs). Acceptance:
  steel cantilever beam deflection within 1% of δ = PL³/(3EI).
- :func:`solve_steady_heat` — ``∇·(k ∇T) = 0``. Acceptance: 1D rod
  between two fixed temperatures recovers the linear profile
  exactly.

The legacy :mod:`backend.common.engines.continuum` still ships a
stub ``FEMEngine`` class that returns fake data; this package is
the canonical solver. The legacy class will migrate in a follow-up
session.

Public API
----------

- :class:`BoxGeometry`, :class:`IsotropicElasticMaterial`,
  :class:`DirichletBC`, :class:`NeumannBC` — pydantic BC / geometry
  / material schemas.
- :class:`LinearElasticityProblem`, :class:`SteadyHeatProblem` —
  top-level problem specs.
- :class:`ElasticitySolution`, :class:`HeatSolution` — solver return types.
- :func:`solve_linear_elasticity`, :func:`solve_steady_heat` — the
  solvers.
- :func:`write_elasticity_vtu`, :func:`write_heat_vtu` — VTU
  exporters for ParaView.
- :func:`build_box_mesh`, :class:`BuiltMesh` — mesh helpers, for
  callers that want to compose their own workflow.
"""

from .mesh import BuiltMesh, build_box_mesh
from .params import (
    BoxGeometry,
    DirichletBC,
    FaceTag,
    IsotropicElasticMaterial,
    LinearElasticityProblem,
    NeumannBC,
    SteadyHeatProblem,
)
from .solvers import (
    ElasticitySolution,
    HeatSolution,
    solve_linear_elasticity,
    solve_steady_heat,
)
from .vtu import (
    read_vtu_point_data,
    write_elasticity_vtu,
    write_heat_vtu,
)

__all__ = [
    "BoxGeometry",
    "BuiltMesh",
    "DirichletBC",
    "ElasticitySolution",
    "FaceTag",
    "HeatSolution",
    "IsotropicElasticMaterial",
    "LinearElasticityProblem",
    "NeumannBC",
    "SteadyHeatProblem",
    "build_box_mesh",
    "read_vtu_point_data",
    "solve_linear_elasticity",
    "solve_steady_heat",
    "write_elasticity_vtu",
    "write_heat_vtu",
]
