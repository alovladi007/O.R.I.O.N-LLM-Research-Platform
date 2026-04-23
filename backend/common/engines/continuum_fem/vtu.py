"""Write FEM solutions to VTU files for ParaView / kitware tooling.

``.vtu`` is the XML-based unstructured-grid VTK format. ParaView
reads it directly; meshio handles the serialization.

skfem's ``MeshHex`` stores vertices in the VTK hex ordering already,
so we pass ``mesh.t.T`` straight into meshio. Field attachment uses
``point_data`` — tensor fields are stored component-wise because
meshio's VTK writer otherwise reshapes them into a shape ParaView
then re-interprets as vectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import meshio
import numpy as np

from .solvers import ElasticitySolution, HeatSolution


def write_elasticity_vtu(
    solution: ElasticitySolution, path: str | Path,
) -> Path:
    """Write displacement + stress fields to a VTU file.

    Fields written
    --------------
    - ``displacement`` (vector, 3 comps) — per-vertex u_x/u_y/u_z.
    - ``displacement_magnitude`` (scalar) — |u|.
    - ``stress_xx / yy / zz / yz / xz / xy`` (scalars) — Voigt
      components of the Cauchy stress tensor (Pa).
    - ``von_mises`` (scalar) — von Mises equivalent stress (Pa).
    """
    out = Path(path)
    mesh = solution.mesh.skfem_mesh
    points = mesh.p.T
    cells = [("hexahedron", mesh.t.T)]

    u = solution.displacement  # (3, n_nodes)
    mag = np.linalg.norm(u, axis=0)
    point_data = {
        "displacement": u.T,  # meshio expects (n_nodes, 3) for vectors
        "displacement_magnitude": mag,
        "stress_xx": solution.stress_nodal[0],
        "stress_yy": solution.stress_nodal[1],
        "stress_zz": solution.stress_nodal[2],
        "stress_yz": solution.stress_nodal[3],
        "stress_xz": solution.stress_nodal[4],
        "stress_xy": solution.stress_nodal[5],
        "von_mises": solution.von_mises_nodal,
    }
    mio = meshio.Mesh(points=points, cells=cells, point_data=point_data)
    out.parent.mkdir(parents=True, exist_ok=True)
    mio.write(out)
    return out


def write_heat_vtu(solution: HeatSolution, path: str | Path) -> Path:
    """Write the steady temperature field to a VTU file."""
    out = Path(path)
    mesh = solution.mesh.skfem_mesh
    points = mesh.p.T
    cells = [("hexahedron", mesh.t.T)]
    point_data = {"temperature": solution.temperature}
    mio = meshio.Mesh(points=points, cells=cells, point_data=point_data)
    out.parent.mkdir(parents=True, exist_ok=True)
    mio.write(out)
    return out


def read_vtu_point_data(path: str | Path) -> dict:
    """Round-trip helper for tests. Returns the point_data dict."""
    return dict(meshio.read(str(path)).point_data)
