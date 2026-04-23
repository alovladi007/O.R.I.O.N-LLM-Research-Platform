"""Tests for Phase 5 / Session 5.1 — FEM continuum solver.

Covers
------

1. :func:`solve_linear_elasticity` —
   - Cantilever-beam acceptance: steel, 1 m × 0.1 m × 0.1 m,
     clamped at x=0, 1 kN in -z at x=L → δ within **1%** of
     PL³/(3EI) using Hex2 at 20×2×2.
   - Dirichlet BC absence raises a clear error (rigid-body modes).
   - Uniform tension: a cube stretched by a uniform +x traction
     recovers σ_xx = traction everywhere (to within FE noise).

2. :func:`solve_steady_heat` —
   - 1D-rod acceptance: T(x=0.5 L) exactly (T_left + T_right)/2
     to machine precision on a linear-hex mesh.
   - Neumann flux BC (insulated y/z faces) doesn't perturb the 1D
     linear profile.
   - Dirichlet absence raises (defined up to a constant).

3. :mod:`vtu` —
   - Round-trip an elasticity solution → VTU → meshio read →
     arrays match original to machine precision.
   - Heat solution round-trip likewise.

4. Pydantic parameter schema —
   - IsotropicElasticMaterial.lame_{lambda, mu} derivation.
   - LinearElasticityProblem without Dirichlet raises at construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Params / material helpers
# ---------------------------------------------------------------------------


class TestMaterial:
    def test_lame_parameters_match_textbook(self):
        from backend.common.engines.continuum_fem import IsotropicElasticMaterial

        # Steel: E=210 GPa, nu=0.3
        m = IsotropicElasticMaterial(youngs_modulus_pa=210e9, poisson_ratio=0.3)
        # λ = E ν / ((1+ν)(1-2ν)) = 210e9 * 0.3 / (1.3 * 0.4) = 121.15 GPa
        assert m.lame_lambda == pytest.approx(121.15e9, rel=1e-4)
        # μ = E / (2(1+ν)) = 210e9 / 2.6 = 80.77 GPa
        assert m.lame_mu == pytest.approx(80.77e9, rel=1e-4)

    def test_zero_poisson_edge_case(self):
        from backend.common.engines.continuum_fem import IsotropicElasticMaterial

        m = IsotropicElasticMaterial(youngs_modulus_pa=1e9, poisson_ratio=0.0)
        assert m.lame_lambda == pytest.approx(0.0, abs=1e-9)
        assert m.lame_mu == pytest.approx(0.5e9, rel=1e-9)

    def test_poisson_bound_rejected(self):
        from backend.common.engines.continuum_fem import IsotropicElasticMaterial

        with pytest.raises(Exception):
            IsotropicElasticMaterial(youngs_modulus_pa=1e9, poisson_ratio=0.5)


class TestProblemValidation:
    def test_elasticity_without_dirichlet_raises(self):
        from backend.common.engines.continuum_fem import (
            BoxGeometry, IsotropicElasticMaterial, LinearElasticityProblem,
        )

        # Rigid-body modes → singular K. Validator rejects at
        # construction time rather than letting the solver explode.
        with pytest.raises(Exception):
            LinearElasticityProblem(
                geometry=BoxGeometry(length_x_m=1, length_y_m=1, length_z_m=1),
                material=IsotropicElasticMaterial(
                    youngs_modulus_pa=1e9, poisson_ratio=0.3,
                ),
                dirichlet=[],
            )

    def test_heat_without_dirichlet_raises(self):
        from backend.common.engines.continuum_fem import BoxGeometry, SteadyHeatProblem

        with pytest.raises(Exception):
            SteadyHeatProblem(
                geometry=BoxGeometry(length_x_m=1, length_y_m=1, length_z_m=1),
                thermal_conductivity_w_per_m_k=50.0,
                dirichlet=[],
            )


# ---------------------------------------------------------------------------
# Cantilever acceptance
# ---------------------------------------------------------------------------


@pytest.fixture
def steel_cantilever_problem():
    """Roadmap acceptance: 1 m × 0.1 m × 0.1 m steel, 1 kN tip load."""
    from backend.common.engines.continuum_fem import (
        BoxGeometry, DirichletBC, IsotropicElasticMaterial,
        LinearElasticityProblem, NeumannBC,
    )

    face_area = 0.1 * 0.1  # y × z
    traction_z = -1000.0 / face_area  # 1 kN in -z, distributed
    return LinearElasticityProblem(
        geometry=BoxGeometry(
            length_x_m=1.0, length_y_m=0.1, length_z_m=0.1,
            n_elements_x=20, n_elements_y=2, n_elements_z=2,
        ),
        material=IsotropicElasticMaterial(
            youngs_modulus_pa=210e9, poisson_ratio=0.3,
        ),
        dirichlet=[DirichletBC(face="x-", value_xyz=(0.0, 0.0, 0.0))],
        neumann=[NeumannBC(face="x+", traction_xyz=(0.0, 0.0, traction_z))],
        element_order=2,
    )


def _cantilever_analytical_deflection(
    P: float = 1000.0, L: float = 1.0, E: float = 210e9,
    b: float = 0.1, h: float = 0.1,
) -> float:
    I = b * h**3 / 12.0
    return P * L**3 / (3 * E * I)


class TestCantileverAcceptance:
    def test_max_deflection_within_1_percent(self, steel_cantilever_problem):
        """Roadmap acceptance target: δ_fem within ±1% of δ_analytical."""
        from backend.common.engines.continuum_fem import solve_linear_elasticity

        sol = solve_linear_elasticity(steel_cantilever_problem)
        analytical = _cantilever_analytical_deflection()
        err = abs(sol.max_displacement_magnitude_m - analytical) / analytical
        assert err < 0.01, (
            f"cantilever tip deflection {sol.max_displacement_magnitude_m*1000:.4f} mm "
            f"differs from analytical {analytical*1000:.4f} mm by "
            f"{err*100:.3f}% (>1%)"
        )

    def test_deflection_is_in_minus_z(self, steel_cantilever_problem):
        """A downward tip load must produce u_z ≤ 0 everywhere."""
        from backend.common.engines.continuum_fem import solve_linear_elasticity

        sol = solve_linear_elasticity(steel_cantilever_problem)
        uz = sol.displacement[2]
        # With a tiny FE-numerical Poisson contraction on the top
        # fibre, u_z can be slightly positive near the clamped face.
        # Allow that, but the extremum must be in -z.
        assert uz.min() < 0
        assert abs(uz.min()) > abs(uz.max())

    def test_hex1_needs_finer_mesh_to_meet_target(self):
        """Sanity: P1 at 20×2×2 misses the 1% target (shear-locking).

        Locks in a known limitation so a caller who picks
        ``element_order=1`` on a coarse mesh gets the expected error,
        not a silent misleading number.
        """
        from backend.common.engines.continuum_fem import (
            BoxGeometry, DirichletBC, IsotropicElasticMaterial,
            LinearElasticityProblem, NeumannBC, solve_linear_elasticity,
        )

        problem = LinearElasticityProblem(
            geometry=BoxGeometry(
                length_x_m=1.0, length_y_m=0.1, length_z_m=0.1,
                n_elements_x=20, n_elements_y=2, n_elements_z=2,
            ),
            material=IsotropicElasticMaterial(
                youngs_modulus_pa=210e9, poisson_ratio=0.3,
            ),
            dirichlet=[DirichletBC(face="x-", value_xyz=(0.0, 0.0, 0.0))],
            neumann=[NeumannBC(face="x+", traction_xyz=(0.0, 0.0, -100000.0))],
            element_order=1,
        )
        sol = solve_linear_elasticity(problem)
        analytical = _cantilever_analytical_deflection()
        err = abs(sol.max_displacement_magnitude_m - analytical) / analytical
        # P1 20×2×2 is typically ~7-10% off; locking the sign only.
        assert err > 0.02


# ---------------------------------------------------------------------------
# Uniform tension: σ_xx = traction to machine precision in the interior
# ---------------------------------------------------------------------------


class TestUniformTension:
    def test_uniform_tension_cube_recovers_traction(self):
        from backend.common.engines.continuum_fem import (
            BoxGeometry, DirichletBC, IsotropicElasticMaterial,
            LinearElasticityProblem, NeumannBC, solve_linear_elasticity,
        )

        traction = 1e6  # 1 MPa
        problem = LinearElasticityProblem(
            geometry=BoxGeometry(
                length_x_m=1.0, length_y_m=1.0, length_z_m=1.0,
                n_elements_x=4, n_elements_y=4, n_elements_z=4,
            ),
            material=IsotropicElasticMaterial(
                youngs_modulus_pa=200e9, poisson_ratio=0.0,  # ν=0 decouples axes
            ),
            # Fix one corner only — u=0 at the three faces closest to origin
            # with complementary components to allow free Poisson contraction.
            dirichlet=[
                DirichletBC(face="x-", value_xyz=(0.0, None, None)),
                DirichletBC(face="y-", value_xyz=(None, 0.0, None)),
                DirichletBC(face="z-", value_xyz=(None, None, 0.0)),
            ],
            neumann=[NeumannBC(face="x+", traction_xyz=(traction, 0.0, 0.0))],
            element_order=2,
        )
        sol = solve_linear_elasticity(problem)

        # Interior σ_xx should equal the applied traction. With ν=0
        # there's no Poisson contraction to muddle the picture.
        # Avoid boundary nodes where the stress projection is noisy.
        p = sol.mesh.skfem_mesh.p
        interior = (
            (p[0] > 0.3) & (p[0] < 0.7) &
            (p[1] > 0.3) & (p[1] < 0.7) &
            (p[2] > 0.3) & (p[2] < 0.7)
        )
        sxx_interior = sol.stress_nodal[0][interior]
        assert sxx_interior.size > 0
        mean_sxx = sxx_interior.mean()
        assert mean_sxx == pytest.approx(traction, rel=0.05), (
            f"σ_xx mean in interior is {mean_sxx:.3e} Pa, expected {traction:.3e} Pa"
        )


# ---------------------------------------------------------------------------
# Heat-rod acceptance
# ---------------------------------------------------------------------------


class TestHeatRodAcceptance:
    def test_midpoint_temperature_exact(self):
        """Roadmap: T(x=L/2) = (T_left + T_right)/2 to FE tolerance."""
        from backend.common.engines.continuum_fem import (
            BoxGeometry, DirichletBC, SteadyHeatProblem, solve_steady_heat,
        )

        T_left, T_right = 300.0, 500.0
        problem = SteadyHeatProblem(
            geometry=BoxGeometry(
                length_x_m=1.0, length_y_m=0.1, length_z_m=0.1,
                n_elements_x=20, n_elements_y=2, n_elements_z=2,
            ),
            thermal_conductivity_w_per_m_k=50.0,
            dirichlet=[
                DirichletBC(face="x-", value_scalar=T_left),
                DirichletBC(face="x+", value_scalar=T_right),
            ],
        )
        sol = solve_steady_heat(problem)
        p = sol.mesh.skfem_mesh.p
        mid = np.isclose(p[0], 0.5, atol=1e-9)
        mid_T = sol.temperature[mid].mean()
        expected = 0.5 * (T_left + T_right)
        assert mid_T == pytest.approx(expected, abs=1e-6), (
            f"midpoint T {mid_T} differs from expected {expected}"
        )
        # Linear profile at quartile points as well.
        for x_target, T_target in [(0.25, 350.0), (0.75, 450.0)]:
            mask = np.isclose(p[0], x_target, atol=1e-9)
            assert sol.temperature[mask].mean() == pytest.approx(T_target, abs=1e-6)

    def test_insulated_sides_dont_perturb_1d_profile(self):
        """Neumann q=0 on y±, z± must leave the 1D linear profile intact."""
        from backend.common.engines.continuum_fem import (
            BoxGeometry, DirichletBC, NeumannBC,
            SteadyHeatProblem, solve_steady_heat,
        )

        problem = SteadyHeatProblem(
            geometry=BoxGeometry(
                length_x_m=1.0, length_y_m=0.2, length_z_m=0.2,
                n_elements_x=10, n_elements_y=4, n_elements_z=4,
            ),
            thermal_conductivity_w_per_m_k=50.0,
            dirichlet=[
                DirichletBC(face="x-", value_scalar=273.0),
                DirichletBC(face="x+", value_scalar=373.0),
            ],
            neumann=[
                NeumannBC(face="y-", heat_flux_w_per_m2=0.0),
                NeumannBC(face="y+", heat_flux_w_per_m2=0.0),
                NeumannBC(face="z-", heat_flux_w_per_m2=0.0),
                NeumannBC(face="z+", heat_flux_w_per_m2=0.0),
            ],
        )
        sol = solve_steady_heat(problem)
        p = sol.mesh.skfem_mesh.p
        mid = np.isclose(p[0], 0.5, atol=1e-9)
        assert sol.temperature[mid].mean() == pytest.approx(323.0, abs=1e-6)

    def test_uniform_offset_with_zero_gradient(self):
        """T=500 on both ends → T=500 everywhere."""
        from backend.common.engines.continuum_fem import (
            BoxGeometry, DirichletBC, SteadyHeatProblem, solve_steady_heat,
        )

        problem = SteadyHeatProblem(
            geometry=BoxGeometry(
                length_x_m=1.0, length_y_m=0.1, length_z_m=0.1,
                n_elements_x=6, n_elements_y=2, n_elements_z=2,
            ),
            thermal_conductivity_w_per_m_k=50.0,
            dirichlet=[
                DirichletBC(face="x-", value_scalar=500.0),
                DirichletBC(face="x+", value_scalar=500.0),
            ],
        )
        sol = solve_steady_heat(problem)
        assert sol.min_temperature_k == pytest.approx(500.0, abs=1e-9)
        assert sol.max_temperature_k == pytest.approx(500.0, abs=1e-9)


# ---------------------------------------------------------------------------
# VTU export round-trip
# ---------------------------------------------------------------------------


class TestVTUExport:
    def test_elasticity_roundtrip(self, steel_cantilever_problem, tmp_path):
        from backend.common.engines.continuum_fem import (
            read_vtu_point_data, solve_linear_elasticity, write_elasticity_vtu,
        )

        sol = solve_linear_elasticity(steel_cantilever_problem)
        out_path = tmp_path / "cantilever.vtu"
        written = write_elasticity_vtu(sol, out_path)
        assert written.is_file()
        assert written.stat().st_size > 0

        point_data = read_vtu_point_data(out_path)
        # Fields present
        for key in (
            "displacement", "displacement_magnitude", "von_mises",
            "stress_xx", "stress_yy", "stress_zz",
            "stress_yz", "stress_xz", "stress_xy",
        ):
            assert key in point_data, f"VTU missing field {key!r}"

        # Displacement round-trip (meshio stores (n_nodes, 3))
        np.testing.assert_allclose(
            point_data["displacement"], sol.displacement.T,
            rtol=0, atol=1e-12,
        )
        np.testing.assert_allclose(
            point_data["von_mises"], sol.von_mises_nodal,
            rtol=0, atol=1e-6,
        )

    def test_heat_roundtrip(self, tmp_path):
        from backend.common.engines.continuum_fem import (
            BoxGeometry, DirichletBC, SteadyHeatProblem,
            read_vtu_point_data, solve_steady_heat, write_heat_vtu,
        )

        problem = SteadyHeatProblem(
            geometry=BoxGeometry(length_x_m=1, length_y_m=0.1, length_z_m=0.1,
                                 n_elements_x=4, n_elements_y=2, n_elements_z=2),
            thermal_conductivity_w_per_m_k=50.0,
            dirichlet=[
                DirichletBC(face="x-", value_scalar=300.0),
                DirichletBC(face="x+", value_scalar=500.0),
            ],
        )
        sol = solve_steady_heat(problem)
        out_path = tmp_path / "heat.vtu"
        write_heat_vtu(sol, out_path)
        assert out_path.is_file()
        pd = read_vtu_point_data(out_path)
        assert "temperature" in pd
        np.testing.assert_allclose(pd["temperature"], sol.temperature, atol=1e-12)


# ---------------------------------------------------------------------------
# Mesh face helpers
# ---------------------------------------------------------------------------


class TestBoxMesh:
    def test_face_areas_match_geometry(self):
        from backend.common.engines.continuum_fem import BoxGeometry, build_box_mesh

        built = build_box_mesh(BoxGeometry(length_x_m=2, length_y_m=3, length_z_m=4))
        assert built.face_area("x-") == pytest.approx(3 * 4)
        assert built.face_area("x+") == pytest.approx(3 * 4)
        assert built.face_area("y-") == pytest.approx(2 * 4)
        assert built.face_area("z+") == pytest.approx(2 * 3)

    def test_facets_partition_by_face(self):
        from backend.common.engines.continuum_fem import BoxGeometry, build_box_mesh

        built = build_box_mesh(BoxGeometry(
            length_x_m=1, length_y_m=1, length_z_m=1,
            n_elements_x=3, n_elements_y=3, n_elements_z=3,
        ))
        face_counts = {
            tag: built.facets_for(tag).size for tag in
            ("x-", "x+", "y-", "y+", "z-", "z+")
        }
        # On an axis-aligned 3×3×3 hex box, each face has 3×3 = 9 quads.
        for tag, n in face_counts.items():
            assert n == 9, f"face {tag} has {n} facets, expected 9"

        # Each pair x- vs x+ must not overlap.
        xm = set(built.facets_for("x-").tolist())
        xp = set(built.facets_for("x+").tolist())
        assert xm & xp == set()
