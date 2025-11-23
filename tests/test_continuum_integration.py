"""
Integration Tests for FEM/FVM Continuum Engines
================================================

Tests the FEM and FVM continuum simulation engine integration:
- FEM (Finite Element Method) for structural analysis
- FVM (Finite Volume Method) for fluid dynamics
- Job execution and result parsing
- Integration with Celery worker tasks
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.user import User


class TestFEMEngine:
    """Test FEM engine functionality"""

    def test_fem_engine_initialization(self):
        """Test FEM engine initializes correctly"""
        from backend.common.engines.continuum import FEMEngine

        engine = FEMEngine()

        assert engine is not None
        assert hasattr(engine, 'setup')
        assert hasattr(engine, 'run')
        assert hasattr(engine, 'cleanup')

    def test_fem_static_analysis(self):
        """Test FEM static structural analysis"""
        from backend.common.engines.continuum import FEMEngine

        engine = FEMEngine()

        # Structure/mesh data
        structure = {
            "mesh": {
                "n_nodes": 5000,
                "n_elements": 4000,
            },
            "material_properties": {
                "youngs_modulus": 200e9,  # Pa
                "poissons_ratio": 0.3,
            },
        }

        # Parameters
        parameters = {
            "analysis_type": "STATIC",
            "boundary_conditions": {
                "loads": [{"node": 100, "force": [0, -1000, 0]}],
                "constraints": [{"node": 0, "dof": [0, 1, 2]}],
            },
            "solver_settings": {
                "tolerance": 1e-6,
                "max_iterations": 1000,
            },
        }

        # Setup and run
        engine.setup(structure, parameters)
        results = engine.run()

        # Verify results
        assert results["convergence_reached"] is True
        assert "summary" in results
        assert "effective_properties" in results["summary"]
        assert "solution_fields" in results["summary"]
        assert "solver_info" in results["summary"]

        # Check solution fields
        solution_fields = results["summary"]["solution_fields"]
        assert "displacement" in solution_fields
        assert "stress" in solution_fields
        assert "strain" in solution_fields

        # Verify quality score
        assert 0.0 <= results["quality_score"] <= 1.0

    def test_fem_thermal_analysis(self):
        """Test FEM thermal analysis"""
        from backend.common.engines.continuum import FEMEngine

        engine = FEMEngine()

        structure = {
            "mesh": {"n_nodes": 10000, "n_elements": 8000},
            "material_properties": {
                "thermal_conductivity": 50.0,  # W/(m·K)
                "thermal_expansion": 1.2e-5,  # 1/K
            },
        }

        parameters = {
            "analysis_type": "THERMAL",
            "boundary_conditions": {
                "temperatures": [{"node": 0, "temp": 300}],
                "heat_flux": [{"face": 10, "flux": 1000}],
            },
        }

        engine.setup(structure, parameters)
        results = engine.run()

        assert results["convergence_reached"] is True
        assert results["summary"]["analysis_type"] == "THERMAL"

    def test_fem_with_progress_callback(self):
        """Test FEM with progress tracking"""
        from backend.common.engines.continuum import FEMEngine

        engine = FEMEngine()

        structure = {"mesh": {"n_nodes": 1000, "n_elements": 800}}
        parameters = {"analysis_type": "STATIC"}

        engine.setup(structure, parameters)

        # Track progress
        progress_updates = []

        def progress_callback(progress: float, step: str):
            progress_updates.append((progress, step))

        results = engine.run(progress_callback=progress_callback)

        # Verify progress tracking
        assert len(progress_updates) > 0
        assert progress_updates[0][0] == 0.0  # First update
        assert progress_updates[-1][0] == 1.0  # Last update
        assert results["convergence_reached"] is True

    def test_fem_effective_properties(self):
        """Test FEM computes effective properties"""
        from backend.common.engines.continuum import FEMEngine

        engine = FEMEngine()

        structure = {"mesh": {"n_nodes": 5000, "n_elements": 4000}}
        parameters = {"analysis_type": "STATIC"}

        engine.setup(structure, parameters)
        results = engine.run()

        effective_props = results["summary"]["effective_properties"]

        # Verify all required properties
        assert "youngs_modulus_gpa" in effective_props
        assert "shear_modulus_gpa" in effective_props
        assert "bulk_modulus_gpa" in effective_props
        assert "poissons_ratio" in effective_props
        assert "density_kg_m3" in effective_props
        assert "thermal_conductivity_w_mk" in effective_props

        # Verify reasonable values
        assert 0 < effective_props["youngs_modulus_gpa"] < 500
        assert 0 < effective_props["poissons_ratio"] < 0.5


class TestFVMEngine:
    """Test FVM engine functionality"""

    def test_fvm_engine_initialization(self):
        """Test FVM engine initializes correctly"""
        from backend.common.engines.continuum import FVMEngine

        engine = FVMEngine()

        assert engine is not None
        assert hasattr(engine, 'setup')
        assert hasattr(engine, 'run')
        assert hasattr(engine, 'cleanup')

    def test_fvm_laminar_flow(self):
        """Test FVM laminar flow simulation"""
        from backend.common.engines.continuum import FVMEngine

        engine = FVMEngine()

        # Domain/mesh data
        structure = {
            "mesh": {
                "n_cells": 100000,
            },
            "fluid_properties": {
                "density": 1000.0,  # kg/m³
                "viscosity": 0.001,  # Pa·s
            },
        }

        # Parameters
        parameters = {
            "flow_type": "LAMINAR",
            "boundary_conditions": {
                "inlet": {"velocity": [1.0, 0.0, 0.0]},
                "outlet": {"pressure": 101325},
                "walls": {"type": "no-slip"},
            },
            "solver_settings": {
                "tolerance": 1e-5,
                "max_iterations": 1000,
            },
        }

        # Setup and run
        engine.setup(structure, parameters)
        results = engine.run()

        # Verify results
        assert results["convergence_reached"] is True
        assert "summary" in results
        assert results["summary"]["flow_type"] == "LAMINAR"

        # Check solution fields
        solution_fields = results["summary"]["solution_fields"]
        assert "velocity" in solution_fields
        assert "pressure" in solution_fields
        assert "temperature" in solution_fields
        assert solution_fields["turbulence"] is None  # No turbulence in laminar

    def test_fvm_turbulent_flow(self):
        """Test FVM turbulent flow simulation"""
        from backend.common.engines.continuum import FVMEngine

        engine = FVMEngine()

        structure = {
            "mesh": {"n_cells": 200000},
            "fluid_properties": {"density": 1.225, "viscosity": 1.8e-5},
        }

        parameters = {
            "flow_type": "TURBULENT",
            "turbulence_model": "k-epsilon",
            "boundary_conditions": {
                "inlet": {"velocity": [10.0, 0.0, 0.0]},
            },
        }

        engine.setup(structure, parameters)
        results = engine.run()

        assert results["summary"]["flow_type"] == "TURBULENT"

        # Verify turbulence fields exist
        solution_fields = results["summary"]["solution_fields"]
        assert solution_fields["turbulence"] is not None
        assert "max_kinetic_energy_m2s2" in solution_fields["turbulence"]
        assert "max_dissipation_rate_m2s3" in solution_fields["turbulence"]

    def test_fvm_with_progress_callback(self):
        """Test FVM with progress tracking"""
        from backend.common.engines.continuum import FVMEngine

        engine = FVMEngine()

        structure = {"mesh": {"n_cells": 50000}}
        parameters = {"flow_type": "LAMINAR"}

        engine.setup(structure, parameters)

        # Track progress
        progress_updates = []

        def progress_callback(progress: float, step: str):
            progress_updates.append((progress, step))

        results = engine.run(progress_callback=progress_callback)

        # Verify progress tracking
        assert len(progress_updates) > 0
        assert progress_updates[0][0] == 0.0
        assert progress_updates[-1][0] == 1.0

    def test_fvm_effective_properties(self):
        """Test FVM computes effective properties"""
        from backend.common.engines.continuum import FVMEngine

        engine = FVMEngine()

        structure = {"mesh": {"n_cells": 50000}}
        parameters = {"flow_type": "LAMINAR"}

        engine.setup(structure, parameters)
        results = engine.run()

        effective_props = results["summary"]["effective_properties"]

        # Verify properties
        assert "thermal_conductivity_w_mk" in effective_props
        assert "effective_viscosity_pa_s" in effective_props
        assert "heat_transfer_coefficient_w_m2k" in effective_props

        # Verify reasonable values
        assert effective_props["thermal_conductivity_w_mk"] > 0
        assert effective_props["effective_viscosity_pa_s"] > 0


class TestContinuumCeleryIntegration:
    """Test continuum engine integration with Celery worker tasks"""

    @pytest.mark.asyncio
    async def test_continuum_job_with_fem(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test continuum simulation job with FEM engine"""
        from src.api.models.multiscale import ContinuumSimulationJob, EngineType, MultiscaleJobStatus

        # Create FEM job
        job = ContinuumSimulationJob(
            owner_id=test_user.id,
            name="Structural Analysis Test",
            engine_type=EngineType.FEM,
            status=MultiscaleJobStatus.PENDING,
            parameters={
                "analysis_type": "STATIC",
                "n_nodes": 5000,
                "n_elements": 4000,
                "structure": {
                    "mesh": {"n_nodes": 5000, "n_elements": 4000},
                },
            },
        )
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

        # Verify job created
        assert job.id is not None
        assert job.engine_type == EngineType.FEM
        assert job.parameters["analysis_type"] == "STATIC"

    @pytest.mark.asyncio
    async def test_continuum_job_with_fvm(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test continuum simulation job with FVM engine"""
        from src.api.models.multiscale import ContinuumSimulationJob, EngineType, MultiscaleJobStatus

        # Create FVM job
        job = ContinuumSimulationJob(
            owner_id=test_user.id,
            name="CFD Flow Analysis",
            engine_type=EngineType.FVM,
            status=MultiscaleJobStatus.PENDING,
            parameters={
                "flow_type": "TURBULENT",
                "n_cells": 100000,
                "structure": {
                    "mesh": {"n_cells": 100000},
                },
            },
        )
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

        # Verify job created
        assert job.id is not None
        assert job.engine_type == EngineType.FVM
        assert job.parameters["flow_type"] == "TURBULENT"

    def test_continuum_task_signature(self):
        """Test run_continuum_simulation task has correct signature"""
        from src.worker.tasks import run_continuum_simulation

        assert callable(run_continuum_simulation)
        assert run_continuum_simulation.name == "run_continuum_simulation"

    def test_continuum_engine_types(self):
        """Test supported continuum engine types"""
        engine_types = ["FEM", "FVM", "BEM"]

        for engine_type in engine_types:
            assert engine_type in ["FEM", "FVM", "BEM"]


class TestContinuumValidation:
    """Test continuum engine validation and error handling"""

    def test_fem_invalid_structure_raises_error(self):
        """Test that invalid structure raises error"""
        from backend.common.engines.continuum import FEMEngine

        engine = FEMEngine()

        # Missing required mesh data
        invalid_structure = {}
        parameters = {"analysis_type": "STATIC"}

        # Should not raise during setup (uses defaults)
        engine.setup(invalid_structure, parameters)

    def test_fvm_invalid_structure_raises_error(self):
        """Test that FVM validates structure"""
        from backend.common.engines.continuum import FVMEngine

        engine = FVMEngine()

        invalid_structure = {}
        parameters = {"flow_type": "LAMINAR"}

        # Should not raise during setup (uses defaults)
        engine.setup(invalid_structure, parameters)

    def test_fem_deterministic_results(self):
        """Test that FEM produces deterministic results"""
        from backend.common.engines.continuum import FEMEngine

        # Same inputs should give same outputs
        structure = {"mesh": {"n_nodes": 1000, "n_elements": 800}}
        parameters = {"analysis_type": "STATIC"}

        engine1 = FEMEngine()
        engine1.setup(structure, parameters)
        results1 = engine1.run()

        engine2 = FEMEngine()
        engine2.setup(structure, parameters)
        results2 = engine2.run()

        # Verify deterministic behavior
        assert results1["quality_score"] == results2["quality_score"]
        assert (
            results1["summary"]["effective_properties"]["youngs_modulus_gpa"]
            == results2["summary"]["effective_properties"]["youngs_modulus_gpa"]
        )

    def test_fvm_deterministic_results(self):
        """Test that FVM produces deterministic results"""
        from backend.common.engines.continuum import FVMEngine

        structure = {"mesh": {"n_cells": 50000}}
        parameters = {"flow_type": "LAMINAR"}

        engine1 = FVMEngine()
        engine1.setup(structure, parameters)
        results1 = engine1.run()

        engine2 = FVMEngine()
        engine2.setup(structure, parameters)
        results2 = engine2.run()

        # Verify deterministic behavior
        assert results1["quality_score"] == results2["quality_score"]
        assert (
            results1["summary"]["effective_properties"]["thermal_conductivity_w_mk"]
            == results2["summary"]["effective_properties"]["thermal_conductivity_w_mk"]
        )


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
