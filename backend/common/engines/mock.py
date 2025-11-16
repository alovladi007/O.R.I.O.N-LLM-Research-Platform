"""
Mock Simulation Engine
======================

Mock simulation engine for testing and development.

This engine simulates realistic simulation behavior without actually running
computationally expensive calculations. It generates deterministic results
based on structure properties, making it ideal for:
- Testing the NANO-OS platform
- Developing frontend features
- Demonstrating workflows
- CI/CD testing
"""

import asyncio
import logging
import time
import random
import hashlib
import json
from typing import Dict, Any, Optional, Callable

from backend.common.engines.base import SimulationEngine

logger = logging.getLogger(__name__)


class MockSimulationEngine(SimulationEngine):
    """
    Mock simulation engine for testing.

    Simulates a real simulation with:
    - Input validation
    - Progress updates
    - Realistic timing
    - Deterministic results based on structure
    - Convergence checks
    """

    def __init__(self, engine: str = "MOCK"):
        """
        Initialize mock engine.

        Args:
            engine: Engine name to simulate (e.g., "VASP", "QE", "MOCK")
        """
        super().__init__()
        self.engine = engine
        self.structure = None
        self.parameters = None

    def setup(self, structure: Dict[str, Any], parameters: Dict[str, Any]) -> None:
        """
        Initialize engine with structure and parameters.

        Args:
            structure: Atomic structure data
            parameters: Simulation parameters

        Raises:
            ValueError: If structure or parameters are invalid
        """
        self.logger.info("Setting up mock simulation engine...")

        # Validate inputs
        self.validate_structure(structure)
        self.validate_parameters(parameters)

        # Store structure and parameters
        self.structure = structure
        self.parameters = parameters

        self.logger.info("Mock engine setup complete")

    def run(self, progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """
        Run the mock simulation.

        Args:
            progress_callback: Function to call with (progress, step) updates

        Returns:
            Simulation results dictionary
        """
        self.logger.info(f"Starting {self.engine} simulation...")

        # Check if we need async execution (for backwards compatibility)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, use coroutine
                return asyncio.create_task(self._run_async(progress_callback))
            else:
                # We're in sync context, run with asyncio.run
                return asyncio.run(self._run_async(progress_callback))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self._run_async(progress_callback))

    async def _run_async(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Async implementation of run simulation.

        Args:
            progress_callback: Function to call with (progress, step) updates

        Returns:
            Simulation results dictionary
        """
        # Simulate multi-step calculation with realistic but quick timing (1-3 seconds total)
        steps = [
            ("Geometry optimization", 0.4),
            ("SCF convergence", 0.6),
            ("Electronic structure calculation", 0.8),
            ("Property analysis", 0.3),
        ]

        total_steps = len(steps)

        for i, (step_name, duration) in enumerate(steps):
            self.logger.info(f"Step {i+1}/{total_steps}: {step_name}")

            # Update progress
            if progress_callback:
                progress = i / total_steps
                progress_callback(progress, step_name)

            # Simulate work (total: ~2.1 seconds)
            await asyncio.sleep(duration)

        # Final progress update
        if progress_callback:
            progress_callback(1.0, "Simulation complete")

        # Generate mock results
        results = self._generate_results()

        self.logger.info("Simulation completed successfully")

        return results

    def cleanup(self) -> None:
        """Clean up temporary files (no-op for mock engine)."""
        self.logger.info("Mock engine cleanup (no-op)")

    def _get_deterministic_seed(self, key: str) -> int:
        """
        Generate deterministic seed from structure data.

        This ensures results are reproducible for the same structure.

        Args:
            key: Key to differentiate different random streams

        Returns:
            Deterministic seed value
        """
        # Create a hash from structure data
        structure_str = json.dumps(self.structure, sort_keys=True)
        hash_obj = hashlib.md5(f"{structure_str}:{key}".encode())
        return int(hash_obj.hexdigest(), 16) % (2**32)

    def _generate_results(self) -> Dict[str, Any]:
        """
        Generate realistic mock simulation results based on structure properties.

        Results are deterministic based on structure ID and properties, making
        them reproducible and more realistic for testing.

        Returns:
            Dictionary with simulation results
        """
        # Extract structure properties
        structure_id = self.structure.get("id", "unknown")
        n_atoms = self.structure.get("n_atoms", self.structure.get("num_atoms", 10))
        formula = self.structure.get("formula", "UnknownMaterial")
        dimensionality = self.structure.get("dimensionality", 3)

        # Use deterministic random for reproducibility
        seed_energy = self._get_deterministic_seed("energy")
        seed_bandgap = self._get_deterministic_seed("bandgap")
        seed_stability = self._get_deterministic_seed("stability")

        # Calculate energy per atom (more negative for more atoms, with some variation)
        # Typical range: -5 to -15 eV/atom
        random.seed(seed_energy)
        base_energy_per_atom = -8.0  # eV
        energy_variation = random.uniform(-5.0, 5.0)
        size_factor = -0.1 * (n_atoms / 10.0)  # Larger systems slightly more stable
        energy_per_atom = base_energy_per_atom + energy_variation + size_factor
        total_energy = energy_per_atom * n_atoms

        # Calculate bandgap (deterministic based on structure)
        # 2D materials often have larger bandgaps
        random.seed(seed_bandgap)
        if dimensionality == 2:
            bandgap = random.uniform(0.5, 3.0)
        elif dimensionality == 3:
            bandgap = random.uniform(0.0, 2.5)
        else:
            bandgap = random.uniform(0.0, 2.0)

        # Determine stability (simple heuristics)
        random.seed(seed_stability)
        # More stable if: reasonable energy per atom, smaller systems
        is_stable = (
            -15.0 < energy_per_atom < -3.0 and
            n_atoms < 100 and
            random.random() > 0.2  # 80% chance of being stable
        )

        # Convergence iterations (deterministic)
        random.seed(seed_energy + seed_bandgap)
        iterations = random.randint(15, 45)
        converged = random.random() > 0.1  # 90% convergence rate

        # Build detailed results
        results = {
            "engine": self.engine,
            "energy_per_atom": round(energy_per_atom, 6),
            "total_energy": round(total_energy, 6),
            "bandgap": round(bandgap, 4),
            "is_stable": is_stable,
            "n_atoms": n_atoms,
            "formula": formula,
            "dimensionality": dimensionality,
            "convergence": {
                "reached": converged,
                "iterations": iterations,
                "threshold": 1e-6,
                "final_error": round(random.uniform(1e-7, 1e-5), 10) if converged else round(random.uniform(1e-4, 1e-3), 10)
            },
            "forces": {
                "max_force": round(random.uniform(0.001, 0.05), 6) if is_stable else round(random.uniform(0.1, 0.5), 6),
                "rms_force": round(random.uniform(0.0005, 0.02), 6) if is_stable else round(random.uniform(0.05, 0.2), 6)
            },
            "stress": {
                "max_stress": round(random.uniform(0.1, 2.0), 4),  # GPa
                "pressure": round(random.uniform(-1.0, 1.0), 4)
            },
            "magnetic_properties": {
                "is_magnetic": random.random() > 0.7,
                "total_magnetization": round(random.uniform(-2.0, 2.0), 4)
            },
            "timing": {
                "total_time_seconds": round(random.uniform(1.5, 2.5), 2),
                "scf_time_seconds": round(random.uniform(0.8, 1.5), 2),
                "postprocessing_time_seconds": round(random.uniform(0.2, 0.5), 2)
            },
            "calculation_details": {
                "functional": self.parameters.get("functional", "PBE"),
                "k_points": self.parameters.get("k_points", [4, 4, 4]),
                "energy_cutoff": self.parameters.get("ecutwfc", 500),
                "smearing": self.parameters.get("smearing", 0.01),
                "spin_polarized": self.parameters.get("spin_polarized", False)
            }
        }

        # Return in the format expected by tasks.py
        return {
            "summary": results,
            "convergence_reached": converged,
            "quality_score": round(0.9 if (converged and is_stable) else 0.7, 3),
            "metadata": {
                "engine": self.engine,
                "engine_version": "mock-2.0.0",
                "structure_id": str(structure_id)
            }
        }


# Legacy async wrapper for backwards compatibility
async def run_mock_simulation(
    structure: Dict[str, Any],
    parameters: Dict[str, Any],
    engine: str = "VASP",
    job_id: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Run a mock simulation for testing (legacy wrapper).

    This function provides backwards compatibility with the old API.

    Args:
        structure: Atomic structure data
        parameters: Simulation parameters
        engine: Simulation engine (VASP, QE, LAMMPS, etc.)
        job_id: Job ID for logging
        progress_callback: Function to call with (progress, step) updates

    Returns:
        Simulation results dictionary

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If simulation fails
    """
    logger.info(f"Running mock {engine} simulation for job {job_id}")

    try:
        # Create mock engine
        sim_engine = MockSimulationEngine(engine=engine)

        # Setup simulation
        sim_engine.setup(structure, parameters)

        # Run simulation
        results = await sim_engine._run_async(progress_callback=progress_callback)

        logger.info(f"Mock simulation for job {job_id} completed")

        return results

    except Exception as e:
        logger.error(f"Mock simulation for job {job_id} failed: {e}", exc_info=True)
        raise RuntimeError(f"Simulation failed: {e}") from e
