"""Mesh construction for Session 5.1 FEM problems.

For now we only ship :class:`BoxGeometry` → structured hex mesh.
Explicit :mod:`meshio` import from ``.msh`` / ``.vtu`` is deferred;
it's a thin wrapper on top but the solver interface has to accept
heterogeneous face tagging first.

Face-tag convention
-------------------

For a box [0, Lx] × [0, Ly] × [0, Lz]:

- ``x-`` → the face at x = 0
- ``x+`` → the face at x = Lx
- (same for y, z)

The tag resolver returns a boolean mask that ``skfem.Mesh.facets_satisfying``
can consume. We keep this machinery out of the solver modules so
each BC's face-tag → facet-indices translation happens exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from skfem import MeshHex

from .params import BoxGeometry, FaceTag


# Tiny tolerance for floating-point face picks. The mesh coordinates
# come from ``np.linspace`` so exact comparison *does* work, but we
# prefer a small ε so the helpers remain robust if a user perturbs
# the box slightly.
_EPS = 1e-9


@dataclass
class BuiltMesh:
    """Result of meshing a geometry: the skfem mesh + tag predicates.

    The predicates are kept as ``Callable[[np.ndarray], np.ndarray]``
    closures over the box dimensions so the solver modules can
    invoke ``mesh.facets_satisfying(pred)`` without having to know
    which geometry produced the mesh.
    """

    skfem_mesh: MeshHex
    length_xyz_m: tuple
    face_predicates: Dict[FaceTag, Callable[[np.ndarray], np.ndarray]]

    def facets_for(self, tag: FaceTag) -> np.ndarray:
        """Return the facet indices of the face identified by *tag*."""
        return self.skfem_mesh.facets_satisfying(self.face_predicates[tag])

    def face_area(self, tag: FaceTag) -> float:
        """Geometric area (m²) of the requested face on the box.

        Convenience for converting a total-force BC into the uniform
        traction the FEM solver expects (``t = F / A``).
        """
        Lx, Ly, Lz = self.length_xyz_m
        if tag in ("x-", "x+"):
            return Ly * Lz
        if tag in ("y-", "y+"):
            return Lx * Lz
        return Lx * Ly


def build_box_mesh(geometry: BoxGeometry) -> BuiltMesh:
    """Mesh the requested box with a structured hex grid.

    Uses :meth:`skfem.MeshHex.init_tensor` which takes three 1-D
    coordinate arrays (the x, y, z grid lines) and produces a
    trilinear-hex mesh. All elements are regular parallelepipeds.
    """
    Lx, Ly, Lz = geometry.length_x_m, geometry.length_y_m, geometry.length_z_m
    mesh = MeshHex.init_tensor(
        np.linspace(0.0, Lx, geometry.n_elements_x + 1),
        np.linspace(0.0, Ly, geometry.n_elements_y + 1),
        np.linspace(0.0, Lz, geometry.n_elements_z + 1),
    )
    face_predicates: Dict[FaceTag, Callable[[np.ndarray], np.ndarray]] = {
        "x-": lambda p: np.isclose(p[0], 0.0, atol=_EPS),
        "x+": lambda p: np.isclose(p[0], Lx, atol=_EPS),
        "y-": lambda p: np.isclose(p[1], 0.0, atol=_EPS),
        "y+": lambda p: np.isclose(p[1], Ly, atol=_EPS),
        "z-": lambda p: np.isclose(p[2], 0.0, atol=_EPS),
        "z+": lambda p: np.isclose(p[2], Lz, atol=_EPS),
    }
    return BuiltMesh(
        skfem_mesh=mesh,
        length_xyz_m=(Lx, Ly, Lz),
        face_predicates=face_predicates,
    )
