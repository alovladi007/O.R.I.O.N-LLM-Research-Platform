"""
LAMMPS Engine for Molecular Dynamics Simulations
=================================================

This module implements a LAMMPS-based engine for classical molecular dynamics.

Supported workflows:
- MD_NVT_LAMMPS: Canonical ensemble (constant N, V, T)
- MD_NPT_LAMMPS: Isothermal-isobaric ensemble (constant N, P, T)
- MD_ANNEAL_LAMMPS: Simulated annealing

Session 17: LAMMPS Integration
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import re
import numpy as np

from .base import SimulationEngine, ExecutionResult

logger = logging.getLogger(__name__)


class LAMMPSEngine(SimulationEngine):
    """
    LAMMPS molecular dynamics engine.

    Environment variables:
        NANO_OS_LAMMPS_COMMAND: Path to LAMMPS executable (default: lmp or lammps)
        NANO_OS_LAMMPS_POTENTIAL_DIR: Directory containing potential files
    """

    def __init__(self):
        self.name = "lammps"
        self.lammps_command = self._get_lammps_command()
        self.potential_dir = Path(
            os.environ.get("NANO_OS_LAMMPS_POTENTIAL_DIR", "/opt/potentials")
        )

        logger.info(f"LAMMPSEngine initialized")
        logger.info(f"  Command: {self.lammps_command}")
        logger.info(f"  Potential dir: {self.potential_dir}")

    def _get_lammps_command(self) -> str:
        """Get LAMMPS executable command."""
        import os
        import shutil

        # Check environment variable
        cmd = os.environ.get("NANO_OS_LAMMPS_COMMAND")
        if cmd:
            return cmd

        # Try common names
        for name in ["lmp", "lammps", "lmp_serial", "lmp_mpi"]:
            if shutil.which(name):
                return name

        # Default (may not exist, but prepare_input will fail gracefully)
        return "lmp"

    def prepare_input(
        self,
        structure: Any,
        parameters: Dict[str, Any],
        work_dir: Path
    ) -> Path:
        """
        Generate LAMMPS input files.

        Args:
            structure: Structure model object
            parameters: MD parameters
            work_dir: Working directory

        Returns:
            Path to input directory
        """
        input_dir = work_dir / "lammps_input"
        input_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Preparing LAMMPS input in {input_dir}")

        # Extract parameters
        workflow_type = parameters.get("workflow_type", "MD_NVT_LAMMPS")
        temperature = parameters.get("temperature", 300.0)
        timestep = parameters.get("timestep", 1.0)  # fs
        num_steps = parameters.get("num_steps", 100000)
        dump_interval = parameters.get("dump_interval", 1000)
        thermo_interval = parameters.get("thermo_interval", 100)
        potential = parameters.get("potential", "tersoff")
        potential_file = parameters.get("potential_file")

        # Generate data file (LAMMPS structure format)
        data_file = input_dir / "structure.data"
        self._write_data_file(structure, data_file)

        # Generate input script
        input_file = input_dir / "in.lammps"
        if "NVT" in workflow_type:
            self._write_nvt_input(
                input_file,
                data_file,
                temperature,
                timestep,
                num_steps,
                dump_interval,
                thermo_interval,
                potential,
                potential_file
            )
        elif "NPT" in workflow_type:
            pressure = parameters.get("pressure", 1.0)  # atm
            self._write_npt_input(
                input_file,
                data_file,
                temperature,
                pressure,
                timestep,
                num_steps,
                dump_interval,
                thermo_interval,
                potential,
                potential_file
            )
        elif "ANNEAL" in workflow_type:
            temp_start = parameters.get("temp_start", 1000.0)
            temp_end = parameters.get("temp_end", 0.0)
            anneal_steps = parameters.get("anneal_steps", 500000)
            self._write_anneal_input(
                input_file,
                data_file,
                temp_start,
                temp_end,
                timestep,
                anneal_steps,
                dump_interval,
                thermo_interval,
                potential,
                potential_file
            )
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        logger.info(f"LAMMPS input files created: {input_file}")
        return input_dir

    def _write_data_file(self, structure: Any, output_path: Path):
        """
        Write LAMMPS data file from structure.

        Args:
            structure: Structure model
            output_path: Path to write data file
        """
        # Extract structure data
        lattice_vectors = structure.lattice_vectors or [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        atoms = structure.atoms or []

        # Get unique atom types
        atom_types = {}
        type_counter = 1
        for atom in atoms:
            species = atom.get("species", "C")
            if species not in atom_types:
                atom_types[species] = type_counter
                type_counter += 1

        # Write LAMMPS data file
        with open(output_path, "w") as f:
            f.write("LAMMPS data file generated by NANO-OS\n\n")
            f.write(f"{len(atoms)} atoms\n")
            f.write(f"{len(atom_types)} atom types\n\n")

            # Box dimensions (from lattice vectors)
            a, b, c = lattice_vectors
            f.write(f"0.0 {a[0]:.6f} xlo xhi\n")
            f.write(f"0.0 {b[1]:.6f} ylo yhi\n")
            f.write(f"0.0 {c[2]:.6f} zlo zhi\n")

            # Tilts (for non-orthogonal boxes)
            if abs(a[1]) > 1e-6 or abs(a[2]) > 1e-6 or abs(b[2]) > 1e-6:
                f.write(f"{a[1]:.6f} {a[2]:.6f} {b[2]:.6f} xy xz yz\n")

            f.write("\nMasses\n\n")
            # Simplified masses (should use ELEMENT_PROPERTIES from features.py)
            for species, type_id in atom_types.items():
                mass = 12.0  # Default, should lookup actual mass
                f.write(f"{type_id} {mass}\n")

            f.write("\nAtoms\n\n")
            for i, atom in enumerate(atoms, start=1):
                species = atom.get("species", "C")
                type_id = atom_types[species]
                pos = atom.get("position", [0, 0, 0])
                f.write(f"{i} {type_id} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

    def _write_nvt_input(
        self,
        output_path: Path,
        data_file: Path,
        temperature: float,
        timestep: float,
        num_steps: int,
        dump_interval: int,
        thermo_interval: int,
        potential: str,
        potential_file: Optional[str]
    ):
        """Write NVT input script."""
        with open(output_path, "w") as f:
            f.write("# LAMMPS input script - NVT ensemble\n")
            f.write("# Generated by NANO-OS\n\n")

            f.write("units metal\n")
            f.write("atom_style atomic\n")
            f.write("boundary p p p\n\n")

            f.write(f"read_data {data_file}\n\n")

            # Potential
            self._write_potential_section(f, potential, potential_file)

            # Initialization
            f.write(f"velocity all create {temperature} 12345 dist gaussian\n\n")

            # NVT integrator
            f.write(f"fix 1 all nvt temp {temperature} {temperature} 100.0\n")
            f.write(f"timestep {timestep * 0.001}\n\n")  # Convert fs to ps

            # Output
            f.write(f"thermo {thermo_interval}\n")
            f.write("thermo_style custom step temp pe ke etotal press vol\n\n")

            f.write(f"dump 1 all custom {dump_interval} traj.lammpstrj id type x y z\n\n")

            # Run
            f.write(f"run {num_steps}\n")

    def _write_npt_input(
        self,
        output_path: Path,
        data_file: Path,
        temperature: float,
        pressure: float,
        timestep: float,
        num_steps: int,
        dump_interval: int,
        thermo_interval: int,
        potential: str,
        potential_file: Optional[str]
    ):
        """Write NPT input script."""
        with open(output_path, "w") as f:
            f.write("# LAMMPS input script - NPT ensemble\n")
            f.write("# Generated by NANO-OS\n\n")

            f.write("units metal\n")
            f.write("atom_style atomic\n")
            f.write("boundary p p p\n\n")

            f.write(f"read_data {data_file}\n\n")

            # Potential
            self._write_potential_section(f, potential, potential_file)

            # Initialization
            f.write(f"velocity all create {temperature} 12345 dist gaussian\n\n")

            # NPT integrator
            f.write(f"fix 1 all npt temp {temperature} {temperature} 100.0 iso {pressure} {pressure} 1000.0\n")
            f.write(f"timestep {timestep * 0.001}\n\n")

            # Output
            f.write(f"thermo {thermo_interval}\n")
            f.write("thermo_style custom step temp pe ke etotal press vol lx ly lz\n\n")

            f.write(f"dump 1 all custom {dump_interval} traj.lammpstrj id type x y z\n\n")

            # Run
            f.write(f"run {num_steps}\n")

    def _write_anneal_input(
        self,
        output_path: Path,
        data_file: Path,
        temp_start: float,
        temp_end: float,
        timestep: float,
        num_steps: int,
        dump_interval: int,
        thermo_interval: int,
        potential: str,
        potential_file: Optional[str]
    ):
        """Write simulated annealing input script."""
        with open(output_path, "w") as f:
            f.write("# LAMMPS input script - Simulated Annealing\n")
            f.write("# Generated by NANO-OS\n\n")

            f.write("units metal\n")
            f.write("atom_style atomic\n")
            f.write("boundary p p p\n\n")

            f.write(f"read_data {data_file}\n\n")

            # Potential
            self._write_potential_section(f, potential, potential_file)

            # Initialization
            f.write(f"velocity all create {temp_start} 12345 dist gaussian\n\n")

            # Annealing (gradually decrease temperature)
            f.write(f"fix 1 all nvt temp {temp_start} {temp_end} 100.0\n")
            f.write(f"timestep {timestep * 0.001}\n\n")

            # Output
            f.write(f"thermo {thermo_interval}\n")
            f.write("thermo_style custom step temp pe ke etotal press\n\n")

            f.write(f"dump 1 all custom {dump_interval} traj.lammpstrj id type x y z\n\n")

            # Run
            f.write(f"run {num_steps}\n")

            # Final minimization
            f.write("\n# Final energy minimization\n")
            f.write("unfix 1\n")
            f.write("minimize 1e-6 1e-8 1000 10000\n")

    def _write_potential_section(self, f, potential: str, potential_file: Optional[str]):
        """Write potential section of input script."""
        if potential == "tersoff":
            pot_file = potential_file or "C.tersoff"
            f.write(f"pair_style tersoff\n")
            f.write(f"pair_coeff * * {self.potential_dir / pot_file} C\n\n")
        elif potential == "eam":
            pot_file = potential_file or "Al.eam.alloy"
            f.write(f"pair_style eam/alloy\n")
            f.write(f"pair_coeff * * {self.potential_dir / pot_file} Al\n\n")
        elif potential == "lj":
            # Lennard-Jones (generic)
            f.write(f"pair_style lj/cut 10.0\n")
            f.write(f"pair_coeff * * 1.0 1.0\n\n")
        else:
            logger.warning(f"Unknown potential: {potential}, using LJ")
            f.write(f"pair_style lj/cut 10.0\n")
            f.write(f"pair_coeff * * 1.0 1.0\n\n")

    def execute(
        self,
        input_dir: Path,
        output_dir: Path,
        timeout: int = 3600
    ) -> ExecutionResult:
        """
        Execute LAMMPS simulation.

        Args:
            input_dir: Directory containing input files
            output_dir: Directory for output files
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult with success status and output
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        input_file = input_dir / "in.lammps"
        log_file = output_dir / "lammps.log"

        logger.info(f"Executing LAMMPS: {self.lammps_command}")

        try:
            result = subprocess.run(
                [self.lammps_command, "-in", str(input_file), "-log", str(log_file)],
                cwd=input_dir,
                capture_output=True,
                timeout=timeout,
                text=True
            )

            success = result.returncode == 0

            if success:
                logger.info("LAMMPS execution completed successfully")
            else:
                logger.error(f"LAMMPS execution failed with code {result.returncode}")

            return ExecutionResult(
                success=success,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr
            )

        except subprocess.TimeoutExpired:
            logger.error(f"LAMMPS execution timed out after {timeout}s")
            return ExecutionResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=f"Execution timed out after {timeout} seconds"
            )
        except FileNotFoundError:
            logger.error(f"LAMMPS executable not found: {self.lammps_command}")
            return ExecutionResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=f"LAMMPS executable not found: {self.lammps_command}"
            )

    def parse_output(
        self,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Parse LAMMPS output files.

        Args:
            output_dir: Directory containing output files

        Returns:
            Dictionary with parsed results
        """
        log_file = output_dir / "lammps.log"

        if not log_file.exists():
            logger.error("LAMMPS log file not found")
            return {"success": False, "error": "Log file not found"}

        # Parse thermo output
        thermo_data = self._parse_thermo(log_file)

        # Compute statistics
        if thermo_data:
            temps = thermo_data.get("temp", [])
            energies = thermo_data.get("etotal", [])

            avg_temp = np.mean(temps) if temps else 0.0
            std_temp = np.std(temps) if temps else 0.0
            avg_energy = np.mean(energies) if energies else 0.0
            final_energy = energies[-1] if energies else 0.0

            result = {
                "success": True,
                "avg_temperature": float(avg_temp),
                "std_temperature": float(std_temp),
                "avg_total_energy": float(avg_energy),
                "final_energy": float(final_energy),
                "md_trajectory_stats": {
                    "temp_vs_time": [[i, t] for i, t in enumerate(temps)],
                    "energy_vs_time": [[i, e] for i, e in enumerate(energies)]
                },
                "num_steps": len(temps)
            }

            logger.info(f"Parsed LAMMPS output: avg_temp={avg_temp:.2f} K")
            return result
        else:
            return {"success": False, "error": "Failed to parse thermo data"}

    def _parse_thermo(self, log_file: Path) -> Dict[str, List[float]]:
        """Parse thermodynamic output from log file."""
        data = {
            "step": [],
            "temp": [],
            "pe": [],
            "ke": [],
            "etotal": [],
            "press": [],
            "vol": []
        }

        with open(log_file, "r") as f:
            in_thermo = False

            for line in f:
                line = line.strip()

                # Start of thermo output
                if line.startswith("Step"):
                    in_thermo = True
                    continue

                # End of thermo output
                if line.startswith("Loop time"):
                    in_thermo = False
                    continue

                if in_thermo and line:
                    try:
                        parts = line.split()
                        if len(parts) >= 7:
                            data["step"].append(float(parts[0]))
                            data["temp"].append(float(parts[1]))
                            data["pe"].append(float(parts[2]))
                            data["ke"].append(float(parts[3]))
                            data["etotal"].append(float(parts[4]))
                            data["press"].append(float(parts[5]))
                            if len(parts) > 6:
                                data["vol"].append(float(parts[6]))
                    except (ValueError, IndexError):
                        pass

        return data


# Register engine
import os
