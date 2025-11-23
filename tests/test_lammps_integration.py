"""
Integration Tests for LAMMPS Engine
====================================

Tests the LAMMPS molecular dynamics engine integration:
- Input file generation (NVT, NPT, Annealing)
- Structure conversion to LAMMPS data format
- Job execution and result parsing
- Integration with Celery worker tasks
"""

import pytest
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.simulation import SimulationJob, SimulationResult, JobStatus
from src.api.models.user import User
from src.api.models.structure import Structure, StructureSource
from src.api.models.material import Material


class TestLAMMPSEngine:
    """Test LAMMPS engine functionality"""

    def test_lammps_engine_initialization(self):
        """Test LAMMPS engine initializes correctly"""
        from backend.common.engines.lammps import LAMMPSEngine

        engine = LAMMPSEngine()

        assert engine.name == "lammps"
        assert engine.lammps_command is not None
        assert engine.potential_dir is not None

    def test_lammps_nvt_input_generation(self):
        """Test LAMMPS NVT input file generation"""
        from backend.common.engines.lammps import LAMMPSEngine

        engine = LAMMPSEngine()

        # Create mock structure
        structure = Mock()
        structure.lattice_vectors = [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0]
        ]
        structure.atoms = [
            {"species": "C", "position": [0.0, 0.0, 0.0]},
            {"species": "C", "position": [2.5, 2.5, 2.5]},
        ]

        # Parameters
        parameters = {
            "workflow_type": "MD_NVT_LAMMPS",
            "temperature": 300.0,
            "timestep": 1.0,
            "num_steps": 10000,
            "potential": "lj"
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)

            # Generate input
            input_dir = engine.prepare_input(
                structure=structure,
                parameters=parameters,
                work_dir=work_dir
            )

            # Verify input files exist
            assert (input_dir / "in.lammps").exists()
            assert (input_dir / "structure.data").exists()

            # Verify input script content
            input_content = (input_dir / "in.lammps").read_text()
            assert "nvt temp 300" in input_content
            assert "timestep" in input_content
            assert "run 10000" in input_content
            assert "units metal" in input_content

            # Verify data file content
            data_content = (input_dir / "structure.data").read_text()
            assert "2 atoms" in data_content
            assert "Atoms" in data_content

    def test_lammps_npt_input_generation(self):
        """Test LAMMPS NPT input file generation"""
        from backend.common.engines.lammps import LAMMPSEngine

        engine = LAMMPSEngine()

        structure = Mock()
        structure.lattice_vectors = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
        structure.atoms = [{"species": "Al", "position": [0.0, 0.0, 0.0]}]

        parameters = {
            "workflow_type": "MD_NPT_LAMMPS",
            "temperature": 500.0,
            "pressure": 1.0,
            "timestep": 2.0,
            "num_steps": 50000,
            "potential": "eam",
            "potential_file": "Al.eam.alloy"
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            input_dir = engine.prepare_input(structure, parameters, work_dir)

            input_content = (input_dir / "in.lammps").read_text()

            # Verify NPT-specific content
            assert "npt temp 500" in input_content
            assert "iso 1.0 1.0" in input_content
            assert "pair_style eam/alloy" in input_content

    def test_lammps_anneal_input_generation(self):
        """Test LAMMPS simulated annealing input"""
        from backend.common.engines.lammps import LAMMPSEngine

        engine = LAMMPSEngine()

        structure = Mock()
        structure.lattice_vectors = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]
        structure.atoms = [{"species": "Si", "position": [i, i, i]} for i in range(4)]

        parameters = {
            "workflow_type": "MD_ANNEAL_LAMMPS",
            "temp_start": 1000.0,
            "temp_end": 0.0,
            "timestep": 1.0,
            "anneal_steps": 100000,
            "potential": "tersoff"
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            input_dir = engine.prepare_input(structure, parameters, work_dir)

            input_content = (input_dir / "in.lammps").read_text()

            # Verify annealing content
            assert "nvt temp 1000" in input_content
            assert "0.0" in input_content  # temp_end
            assert "minimize" in input_content  # Final minimization
            assert "run 100000" in input_content

    def test_lammps_ml_potential_snap(self):
        """Test LAMMPS with SNAP ML potential"""
        from backend.common.engines.lammps import LAMMPSEngine

        engine = LAMMPSEngine()

        structure = Mock()
        structure.lattice_vectors = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        structure.atoms = [{"species": "C", "position": [0, 0, 0]}]

        parameters = {
            "workflow_type": "MD_NVT_LAMMPS",
            "temperature": 300.0,
            "num_steps": 10000,
            "ml_potential_config": {
                "descriptor_type": "SNAP",
                "elements": ["C"],
                "files": {
                    "coefficients": "snap.coeffs",
                    "parameters": "snap.param"
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            input_dir = engine.prepare_input(structure, parameters, work_dir)

            input_content = (input_dir / "in.lammps").read_text()

            # Verify SNAP potential configuration
            assert "ML Potential: SNAP" in input_content
            assert "pair_style snap" in input_content
            assert "snap.param" in input_content
            assert "snap.coeffs" in input_content

    def test_lammps_output_parsing(self):
        """Test LAMMPS output parsing"""
        from backend.common.engines.lammps import LAMMPSEngine

        engine = LAMMPSEngine()

        # Create mock log file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            log_file = output_dir / "lammps.log"

            # Write mock thermo output
            log_content = """
LAMMPS (29 Sep 2021)
Step Temp PotEng KinEng TotEng Press Volume
0 300.0 -100.5 15.2 -85.3 0.5 1000.0
100 305.1 -98.2 15.5 -82.7 0.6 1001.2
200 298.5 -99.8 15.1 -84.7 0.4 999.8
300 302.0 -97.5 15.3 -82.2 0.7 1002.1
Loop time of 10.5 on 1 procs
"""
            log_file.write_text(log_content)

            # Parse output
            result = engine.parse_output(output_dir)

            # Verify parsed data
            assert result["success"] is True
            assert "avg_temperature" in result
            assert "avg_total_energy" in result
            assert "final_energy" in result
            assert result["num_steps"] == 4

            # Check averages
            assert 295 < result["avg_temperature"] < 310
            assert result["final_energy"] == -82.2


class TestLAMMPSCeleryIntegration:
    """Test LAMMPS integration with Celery worker tasks"""

    @pytest.mark.asyncio
    async def test_mesoscale_job_with_lammps(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test mesoscale simulation job with LAMMPS engine"""
        # Create material and structure
        material = Material(
            owner_id=test_user.id,
            formula="C",
            name="Carbon",
            description="Diamond structure"
        )
        db_session.add(material)
        await db_session.commit()
        await db_session.refresh(material)

        structure = Structure(
            material_id=material.id,
            name="Diamond C",
            format="INTERNAL",
            source=StructureSource.UPLOADED,
            raw_text="",
            formula="C",
            num_atoms=8,
            dimensionality=3,
            lattice_vectors=[
                [3.567, 0.0, 0.0],
                [0.0, 3.567, 0.0],
                [0.0, 0.0, 3.567]
            ],
            atoms=[
                {"species": "C", "position": [0.0, 0.0, 0.0]},
                {"species": "C", "position": [0.89, 0.89, 0.89]},
                {"species": "C", "position": [1.78, 1.78, 0.0]},
                {"species": "C", "position": [2.67, 2.67, 0.89]},
                {"species": "C", "position": [1.78, 0.0, 1.78]},
                {"species": "C", "position": [2.67, 0.89, 2.67]},
                {"species": "C", "position": [0.0, 1.78, 1.78]},
                {"species": "C", "position": [0.89, 2.67, 2.67]},
            ]
        )
        db_session.add(structure)
        await db_session.commit()
        await db_session.refresh(structure)

        # Create LAMMPS MD job
        job = SimulationJob(
            owner_id=test_user.id,
            structure_id=structure.id,
            engine="LAMMPS",
            status=JobStatus.PENDING,
            parameters={
                "workflow_type": "MD_NVT_LAMMPS",
                "temperature": 300.0,
                "timestep": 1.0,
                "num_steps": 1000,
                "potential": "lj",
                "dump_interval": 100,
                "thermo_interval": 10
            },
            priority=5
        )
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

        # Verify job created
        assert job.id is not None
        assert job.engine == "LAMMPS"
        assert job.parameters["workflow_type"] == "MD_NVT_LAMMPS"

    def test_lammps_task_signature(self):
        """Test run_mesoscale_simulation task has correct signature"""
        from src.worker.tasks import run_mesoscale_simulation

        assert callable(run_mesoscale_simulation)
        assert run_mesoscale_simulation.name == "run_mesoscale_simulation"

    def test_lammps_workflow_types(self):
        """Test different LAMMPS workflow types"""
        workflows = ["MD_NVT_LAMMPS", "MD_NPT_LAMMPS", "MD_ANNEAL_LAMMPS"]

        for workflow in workflows:
            assert "LAMMPS" in workflow
            assert workflow in ["MD_NVT_LAMMPS", "MD_NPT_LAMMPS", "MD_ANNEAL_LAMMPS"]


class TestLAMMPSInputValidation:
    """Test LAMMPS input validation and error handling"""

    def test_lammps_unknown_workflow_raises_error(self):
        """Test that unknown workflow type raises error"""
        from backend.common.engines.lammps import LAMMPSEngine

        engine = LAMMPSEngine()

        structure = Mock()
        structure.lattice_vectors = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
        structure.atoms = [{"species": "C", "position": [0, 0, 0]}]

        parameters = {
            "workflow_type": "MD_UNKNOWN_WORKFLOW",
            "temperature": 300.0
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError) as exc_info:
                engine.prepare_input(structure, parameters, Path(tmpdir))

            assert "Unknown workflow type" in str(exc_info.value)

    def test_lammps_execution_timeout(self):
        """Test LAMMPS execution with timeout"""
        from backend.common.engines.lammps import LAMMPSEngine

        engine = LAMMPSEngine()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            # Create minimal input file that will fail quickly
            (input_dir / "in.lammps").write_text("# Invalid LAMMPS input\n")

            # Execute with very short timeout (will fail)
            result = engine.execute(input_dir, output_dir, timeout=1)

            # Either fails immediately or times out
            assert result.success is False

    def test_lammps_missing_executable_error(self):
        """Test error when LAMMPS executable not found"""
        from backend.common.engines.lammps import LAMMPSEngine

        # Create engine with non-existent command
        engine = LAMMPSEngine()
        engine.lammps_command = "/nonexistent/lammps"

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            (input_dir / "in.lammps").write_text("# Test\n")

            result = engine.execute(input_dir, output_dir)

            assert result.success is False
            assert "not found" in result.stderr.lower()

    def test_lammps_output_parsing_missing_log(self):
        """Test output parsing when log file is missing"""
        from backend.common.engines.lammps import LAMMPSEngine

        engine = LAMMPSEngine()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # No log file created
            result = engine.parse_output(output_dir)

            assert result["success"] is False
            assert "not found" in result.get("error", "").lower()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
