"""Steady-state heat conduction solver for Session 5.1.

Solves ``∇·(k ∇T) = 0`` with:

- Dirichlet (prescribed T) BCs via :class:`DirichletBC`.
- Neumann (heat flux) BCs via :class:`NeumannBC`. The convention is
  q > 0 ⇒ heat flowing *out* of the domain. A non-zero Neumann BC
  contributes ``∫ -q · v dS`` to the load vector.

Linear Hex1 is the default here — the 1D rod acceptance case is
exact in P1 (constant k, linear T solution is in the FE space), so
there's no reason to pay the Hex2 cost unless the caller sets
``element_order=2`` explicitly (e.g. for curved isotherms).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from skfem import (
    Basis, BilinearForm, ElementHex1, ElementHex2,
    FacetBasis, LinearForm, asm, condense, solve,
)
from skfem.helpers import dot, grad

from ..mesh import BuiltMesh, build_box_mesh
from ..params import DirichletBC, NeumannBC, SteadyHeatProblem


@dataclass
class HeatSolution:
    """Return value of :func:`solve_steady_heat`.

    Fields
    ------
    temperature
        Per-vertex temperature (K). Length equals ``mesh.p.shape[1]``.
    min_temperature_k, max_temperature_k
        Convenience reductions over the nodal temperature.
    mesh
        The :class:`BuiltMesh` for VTU export.
    """

    temperature: np.ndarray
    min_temperature_k: float
    max_temperature_k: float
    mesh: BuiltMesh


def _make_element(order: int):
    if order == 1:
        return ElementHex1()
    if order == 2:
        return ElementHex2()
    raise ValueError(f"unsupported element order {order!r}")


def solve_steady_heat(problem: SteadyHeatProblem) -> HeatSolution:
    """Solve steady heat ``∇·(k ∇T) = 0`` with Dirichlet + Neumann BCs."""
    built = build_box_mesh(problem.geometry)
    mesh = built.skfem_mesh
    element = _make_element(problem.element_order)
    basis = Basis(mesh, element)
    k_cond = problem.thermal_conductivity_w_per_m_k

    @BilinearForm
    def k_form(u, v, w):
        return k_cond * dot(grad(u), grad(v))

    K = asm(k_form, basis)

    # Neumann: each face contributes -q to the load.
    b = np.zeros(basis.N)
    for nbc in problem.neumann:
        if nbc.heat_flux_w_per_m2 is None:
            continue
        q = nbc.heat_flux_w_per_m2
        face_facets = built.facets_for(nbc.face)
        fbasis = FacetBasis(mesh, element, facets=face_facets)

        @LinearForm
        def flux_load(v, w, _q=q):
            return -_q * v

        b = b + asm(flux_load, fbasis)

    # Dirichlet: per-face scalar temperature.
    u0 = np.zeros(basis.N)
    D_parts: List[np.ndarray] = []
    for dbc in problem.dirichlet:
        value = dbc.value_scalar
        if value is None:
            # Allow value_xyz[0] as a fallback so callers can share the
            # DirichletBC type between elasticity and heat.
            value = dbc.value_xyz[0]
        if value is None:
            raise ValueError(
                f"heat Dirichlet BC on face {dbc.face!r} has no value — "
                "set 'value_scalar' (or value_xyz[0])"
            )
        face_facets = built.facets_for(dbc.face)
        face_dofs = basis.get_dofs(facets=face_facets).all()
        D_parts.append(face_dofs)
        u0[face_dofs] = value

    D = np.unique(np.concatenate(D_parts)) if D_parts else np.array([], dtype=int)
    if D.size == 0:
        raise ValueError(
            "steady heat solver needs at least one Dirichlet BC — without "
            "a temperature anchor the solution is defined only up to a constant"
        )
    T_dofs = solve(*condense(K, b, D=D, x=u0))

    # Scatter onto vertex field for VTU / convenience. For P1 the
    # DOF array already matches the vertex count; for P2 we L²-
    # project to the scalar P1 basis.
    if problem.element_order == 1:
        T_vertices = T_dofs
    else:
        p1_basis = Basis(mesh, ElementHex1())
        T_interp = basis.interpolate(T_dofs)

        @LinearForm
        def rhs(v, w, _expr=T_interp.value):
            return _expr * v

        @BilinearForm
        def mass(u_, v, w):
            return u_ * v

        M = asm(mass, p1_basis)
        bp = asm(rhs, p1_basis)
        T_vertices = solve(M, bp)

    return HeatSolution(
        temperature=T_vertices,
        min_temperature_k=float(T_vertices.min()),
        max_temperature_k=float(T_vertices.max()),
        mesh=built,
    )
