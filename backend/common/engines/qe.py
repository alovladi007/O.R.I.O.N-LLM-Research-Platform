"""
Quantum ESPRESSO Engine
========================

Quantum ESPRESSO (QE) engine for DFT calculations.

This engine:
- Generates pw.x input files from structure data
- Executes Quantum ESPRESSO calculations (or mocks if binary unavailable)
- Parses output for energies, forces, and convergence
- Handles relaxation, SCF, and band structure calculations

Configuration via environment variables:
- QE_EXECUTABLE: Path to pw.x executable (default: 'pw.x')
- QE_PSEUDO_DIR: Directory containing pseudopotentials (default: '/opt/qe/pseudo')
- QE_MOCK_MODE: If 'true', skip execution and return fake results (default: auto-detect)
"""

import os
import re
import subprocess
import tempfile
import shutil
import logging
import random
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from backend.common.engines.base import SimulationEngine

logger = logging.getLogger(__name__)


class QuantumEspressoEngine(SimulationEngine):
    """
    Quantum ESPRESSO simulation engine.

    Supports:
    - Single-point SCF calculations
    - Geometry relaxation
    - Band structure calculations
    - DOS calculations (future)
    """

    # Default pseudopotential mapping (PBE functional)
    DEFAULT_PSEUDOPOTENTIALS = {
        "H": "H.pbe-rrkjus_psl.1.0.0.UPF",
        "He": "He.pbe-n-kjpaw_psl.1.0.0.UPF",
        "Li": "Li.pbe-s-kjpaw_psl.1.0.0.UPF",
        "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
        "N": "N.pbe-n-kjpaw_psl.1.0.0.UPF",
        "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
        "F": "F.pbe-n-kjpaw_psl.1.0.0.UPF",
        "Na": "Na.pbe-spn-kjpaw_psl.1.0.0.UPF",
        "Mg": "Mg.pbe-spn-kjpaw_psl.1.0.0.UPF",
        "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
        "Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
        "P": "P.pbe-n-kjpaw_psl.1.0.0.UPF",
        "S": "S.pbe-n-kjpaw_psl.1.0.0.UPF",
        "Cl": "Cl.pbe-n-kjpaw_psl.1.0.0.UPF",
        "K": "K.pbe-spn-kjpaw_psl.1.0.0.UPF",
        "Ca": "Ca.pbe-spn-kjpaw_psl.1.0.0.UPF",
        "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
        "Cr": "Cr.pbe-spn-kjpaw_psl.1.0.0.UPF",
        "Fe": "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
        "Ni": "Ni.pbe-n-kjpaw_psl.1.0.0.UPF",
        "Cu": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF",
        "Zn": "Zn.pbe-dn-kjpaw_psl.1.0.0.UPF",
        "Ga": "Ga.pbe-dn-kjpaw_psl.1.0.0.UPF",
        "Ge": "Ge.pbe-dn-kjpaw_psl.1.0.0.UPF",
        "As": "As.pbe-n-kjpaw_psl.1.0.0.UPF",
        "Se": "Se.pbe-n-kjpaw_psl.1.0.0.UPF",
        "Mo": "Mo.pbe-spn-kjpaw_psl.1.0.0.UPF",
        "Ag": "Ag.pbe-n-kjpaw_psl.1.0.0.UPF",
        "In": "In.pbe-dn-kjpaw_psl.1.0.0.UPF",
        "Sn": "Sn.pbe-dn-kjpaw_psl.1.0.0.UPF",
        "W": "W.pbe-spn-kjpaw_psl.1.0.0.UPF",
        "Au": "Au.pbe-n-kjpaw_psl.1.0.0.UPF",
        "Pb": "Pb.pbe-dn-kjpaw_psl.1.0.0.UPF",
    }

    # Atomic masses (amu)
    ATOMIC_MASSES = {
        "H": 1.008, "He": 4.003, "Li": 6.941, "C": 12.011, "N": 14.007,
        "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305, "Al": 26.982,
        "Si": 28.086, "P": 30.974, "S": 32.065, "Cl": 35.453, "K": 39.098,
        "Ca": 40.078, "Ti": 47.867, "Cr": 51.996, "Fe": 55.845, "Ni": 58.693,
        "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.64, "As": 74.922,
        "Se": 78.96, "Mo": 95.94, "Ag": 107.868, "In": 114.818, "Sn": 118.710,
        "W": 183.84, "Au": 196.967, "Pb": 207.2,
    }

    def __init__(self):
        """Initialize Quantum ESPRESSO engine."""
        super().__init__()
        self.work_dir = None
        self.input_file = None
        self.output_file = None
        self.mock_mode = self._check_mock_mode()

        # Configuration from environment
        self.qe_executable = os.getenv("QE_EXECUTABLE", "pw.x")
        self.pseudo_dir = os.getenv("QE_PSEUDO_DIR", "/opt/qe/pseudo")

    def _check_mock_mode(self) -> bool:
        """
        Determine if we should run in mock mode.

        Mock mode is enabled if:
        - QE_MOCK_MODE environment variable is set to 'true'
        - QE executable is not found in PATH

        Returns:
            True if mock mode should be used
        """
        # Check environment variable
        env_mock = os.getenv("QE_MOCK_MODE", "").lower()
        if env_mock == "true":
            self.logger.info("Mock mode enabled via QE_MOCK_MODE environment variable")
            return True

        # Check if executable exists
        qe_executable = os.getenv("QE_EXECUTABLE", "pw.x")
        if shutil.which(qe_executable) is None:
            self.logger.warning(
                f"Quantum ESPRESSO executable '{qe_executable}' not found. "
                f"Using mock mode. Set QE_EXECUTABLE environment variable to specify path."
            )
            return True

        return False

    def setup(self, structure: Dict[str, Any], parameters: Dict[str, Any]) -> None:
        """
        Initialize engine with structure and parameters.

        Args:
            structure: Atomic structure data
            parameters: QE calculation parameters

        Raises:
            ValueError: If structure or parameters are invalid
        """
        self.logger.info("Setting up Quantum ESPRESSO engine...")

        # Validate inputs
        self.validate_structure(structure)
        self.validate_parameters(parameters)

        # Store structure and parameters
        self.structure = structure
        self.parameters = parameters

        # Create working directory
        self.work_dir = tempfile.mkdtemp(prefix="qe_")
        self.logger.info(f"Created working directory: {self.work_dir}")

        # Generate input file
        self.input_file = os.path.join(self.work_dir, "pw.in")
        self.output_file = os.path.join(self.work_dir, "pw.out")

        self._generate_input_file()

        self.logger.info("Quantum ESPRESSO setup complete")

    def _generate_input_file(self) -> None:
        """
        Generate Quantum ESPRESSO input file from structure and parameters.

        Creates a pw.in file with proper format for pw.x.
        """
        # Get calculation type
        calc_type = self.parameters.get("calculation", "scf")

        # Get parameters with defaults
        ecutwfc = self.parameters.get("ecutwfc", 50.0)
        ecutrho = self.parameters.get("ecutrho", ecutwfc * 4)  # Typically 4x ecutwfc
        conv_thr = self.parameters.get("conv_thr", 1.0e-8)
        mixing_beta = self.parameters.get("mixing_beta", 0.7)
        k_points = self.parameters.get("k_points", [4, 4, 4])

        # Extract structure information
        formula = self.structure.get("formula", "Unknown")
        atoms = self.structure.get("atoms", [])
        positions = self.structure.get("positions", [])
        cell = self.structure.get("cell", None)

        # If atoms/positions not provided, try to infer from structure_data
        if not atoms and "structure_data" in self.structure:
            sd = self.structure["structure_data"]
            atoms = sd.get("atoms", [])
            positions = sd.get("positions", [])
            cell = sd.get("cell", None)

        if not atoms:
            raise ValueError("No atomic species found in structure")

        # Get unique species
        unique_species = sorted(set(atoms))
        ntyp = len(unique_species)
        nat = len(atoms)

        # Build input file
        lines = []

        # CONTROL section
        lines.append("&CONTROL")
        lines.append(f"  calculation = '{calc_type}'")
        lines.append("  prefix = 'pwscf'")
        lines.append(f"  outdir = '{self.work_dir}/tmp'")
        lines.append(f"  pseudo_dir = '{self.pseudo_dir}'")
        lines.append("  verbosity = 'high'")
        if calc_type in ["relax", "vc-relax"]:
            lines.append("  forc_conv_thr = 1.0d-4")
        lines.append("/")
        lines.append("")

        # SYSTEM section
        lines.append("&SYSTEM")
        lines.append("  ibrav = 0")
        lines.append(f"  nat = {nat}")
        lines.append(f"  ntyp = {ntyp}")
        lines.append(f"  ecutwfc = {ecutwfc}")
        lines.append(f"  ecutrho = {ecutrho}")

        # Add spin polarization if specified
        if self.parameters.get("spin_polarized", False):
            lines.append("  nspin = 2")
            tot_magnetization = self.parameters.get("tot_magnetization", 0.0)
            lines.append(f"  tot_magnetization = {tot_magnetization}")

        lines.append("/")
        lines.append("")

        # ELECTRONS section
        lines.append("&ELECTRONS")
        lines.append(f"  conv_thr = {conv_thr}")
        lines.append(f"  mixing_beta = {mixing_beta}")
        lines.append("  electron_maxstep = 200")
        lines.append("/")
        lines.append("")

        # IONS section (for relaxation)
        if calc_type in ["relax", "vc-relax", "md"]:
            lines.append("&IONS")
            ion_dynamics = self.parameters.get("ion_dynamics", "bfgs")
            lines.append(f"  ion_dynamics = '{ion_dynamics}'")
            lines.append("/")
            lines.append("")

        # CELL section (for variable-cell relaxation)
        if calc_type == "vc-relax":
            lines.append("&CELL")
            cell_dynamics = self.parameters.get("cell_dynamics", "bfgs")
            lines.append(f"  cell_dynamics = '{cell_dynamics}'")
            lines.append("/")
            lines.append("")

        # ATOMIC_SPECIES
        lines.append("ATOMIC_SPECIES")
        for species in unique_species:
            mass = self.ATOMIC_MASSES.get(species, 1.0)
            pseudo = self.DEFAULT_PSEUDOPOTENTIALS.get(species, f"{species}.UPF")
            lines.append(f"  {species:<3} {mass:>8.3f}  {pseudo}")
        lines.append("")

        # ATOMIC_POSITIONS
        coord_type = self.parameters.get("coord_type", "crystal")
        lines.append(f"ATOMIC_POSITIONS {coord_type}")

        for i, (atom, pos) in enumerate(zip(atoms, positions)):
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                lines.append(f"  {atom:<3} {pos[0]:>12.8f} {pos[1]:>12.8f} {pos[2]:>12.8f}")
            else:
                self.logger.warning(f"Invalid position for atom {i}: {pos}")
                lines.append(f"  {atom:<3} {0.0:>12.8f} {0.0:>12.8f} {0.0:>12.8f}")
        lines.append("")

        # K_POINTS
        lines.append("K_POINTS automatic")
        if isinstance(k_points, (list, tuple)) and len(k_points) >= 3:
            lines.append(f"  {k_points[0]} {k_points[1]} {k_points[2]}  0 0 0")
        else:
            lines.append("  4 4 4  0 0 0")
        lines.append("")

        # CELL_PARAMETERS (if provided)
        if cell is not None and isinstance(cell, (list, tuple)) and len(cell) >= 3:
            lines.append("CELL_PARAMETERS angstrom")
            for row in cell:
                if isinstance(row, (list, tuple)) and len(row) >= 3:
                    lines.append(f"  {row[0]:>12.8f} {row[1]:>12.8f} {row[2]:>12.8f}")
            lines.append("")

        # Write to file
        input_content = "\n".join(lines)
        with open(self.input_file, 'w') as f:
            f.write(input_content)

        self.logger.info(f"Generated input file: {self.input_file}")
        self.logger.debug(f"Input file content:\n{input_content}")

    def run(self, progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """
        Run Quantum ESPRESSO simulation.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with simulation results

        Raises:
            RuntimeError: If simulation fails
        """
        self.logger.info("Running Quantum ESPRESSO simulation...")

        if self.mock_mode:
            return self._run_mock(progress_callback)
        else:
            return self._run_real(progress_callback)

    def _run_mock(self, progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """
        Run mock QE simulation.

        Generates realistic results without executing QE binary.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Mock simulation results
        """
        self.logger.info("Running in MOCK mode (QE binary not available)")

        # Simulate progress
        if progress_callback:
            progress_callback(0.2, "Reading input file")
            progress_callback(0.4, "Performing SCF iterations")
            progress_callback(0.7, "Computing forces and stress")
            progress_callback(0.9, "Writing output")

        # Generate mock results based on structure
        n_atoms = len(self.structure.get("atoms", []))
        formula = self.structure.get("formula", "Unknown")

        # Generate deterministic "random" results
        random.seed(hash(formula + str(n_atoms)))

        # Mock energy (typical DFT values)
        energy_per_atom = random.uniform(-8.0, -12.0)
        total_energy = energy_per_atom * n_atoms  # Ry
        total_energy_ev = total_energy * 13.6057  # Convert to eV

        # Mock convergence
        scf_iterations = random.randint(8, 25)
        converged = random.random() > 0.05  # 95% success rate

        # Mock forces
        max_force = random.uniform(0.001, 0.05) if converged else random.uniform(0.1, 0.5)
        rms_force = max_force * 0.6

        # Create mock output file
        mock_output = self._generate_mock_output(
            total_energy=total_energy,
            scf_iterations=scf_iterations,
            converged=converged,
            max_force=max_force
        )

        with open(self.output_file, 'w') as f:
            f.write(mock_output)

        # Parse the mock output
        return self._parse_output()

    def _generate_mock_output(
        self,
        total_energy: float,
        scf_iterations: int,
        converged: bool,
        max_force: float
    ) -> str:
        """
        Generate realistic mock QE output.

        Args:
            total_energy: Total energy in Ry
            scf_iterations: Number of SCF iterations
            converged: Whether calculation converged
            max_force: Maximum force on atoms

        Returns:
            Mock output file content
        """
        lines = []

        lines.append("     Program PWSCF v.7.0 starts on 16Nov2025 at 12:00:00")
        lines.append("")
        lines.append("     This program is part of the open-source Quantum ESPRESSO suite")
        lines.append("")
        lines.append("     bravais-lattice index     =            0")
        lines.append(f"     number of atoms/cell      =            {len(self.structure.get('atoms', []))}")
        lines.append(f"     number of atomic types    =            {len(set(self.structure.get('atoms', [])))}")
        lines.append("")

        # SCF iterations
        for i in range(scf_iterations):
            accuracy = 10.0 ** (-2 - i * 0.3)
            lines.append(f"     iteration # {i+1:3d}     ecut=    50.00 Ry     beta= 0.70")
            lines.append(f"     Davidson diagonalization with overlap")
            lines.append(f"     ethr =  {accuracy:.2E},  avg # of iterations =  4.0")
            lines.append("")

        if converged:
            lines.append("     convergence has been achieved in {scf_iterations} iterations")
            lines.append("")

        # Total energy
        lines.append(f"!    total energy              =    {total_energy:>15.8f} Ry")
        lines.append(f"     estimated scf accuracy    <       0.00000010 Ry")
        lines.append("")

        # Forces
        lines.append("     Forces acting on atoms (cartesian axes, Ry/au):")
        lines.append("")
        atoms = self.structure.get("atoms", [])
        for i, atom in enumerate(atoms):
            fx = random.uniform(-max_force, max_force) / 2
            fy = random.uniform(-max_force, max_force) / 2
            fz = random.uniform(-max_force, max_force) / 2
            lines.append(f"     atom    {i+1} type  1   force =    {fx:>12.8f}   {fy:>12.8f}   {fz:>12.8f}")
        lines.append("")
        lines.append(f"     Total force =    {max_force:>12.8f}     Total SCF correction =     0.00001000")
        lines.append("")

        # Final positions (if relaxation)
        if self.parameters.get("calculation", "scf") in ["relax", "vc-relax"]:
            lines.append("ATOMIC_POSITIONS (crystal)")
            positions = self.structure.get("positions", [])
            for i, (atom, pos) in enumerate(zip(atoms, positions)):
                if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    # Add small random displacement
                    new_pos = [p + random.uniform(-0.01, 0.01) for p in pos[:3]]
                    lines.append(f"{atom:<3} {new_pos[0]:>12.8f} {new_pos[1]:>12.8f} {new_pos[2]:>12.8f}")
            lines.append("")

        lines.append("     PWSCF        :      2.50s CPU      2.60s WALL")
        lines.append("")
        lines.append("     JOB DONE.")

        return "\n".join(lines)

    def _run_real(self, progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """
        Run real QE simulation.

        Executes pw.x binary and parses output.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Simulation results

        Raises:
            RuntimeError: If QE execution fails
        """
        self.logger.info(f"Executing {self.qe_executable}...")

        try:
            # Create tmp directory
            tmp_dir = os.path.join(self.work_dir, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)

            # Execute QE
            with open(self.output_file, 'w') as out_f:
                process = subprocess.Popen(
                    [self.qe_executable, "-in", self.input_file],
                    stdout=out_f,
                    stderr=subprocess.PIPE,
                    cwd=self.work_dir,
                    text=True
                )

                # Wait for completion (could add progress monitoring here)
                _, stderr = process.communicate()

                if process.returncode != 0:
                    error_msg = f"Quantum ESPRESSO failed with return code {process.returncode}"
                    if stderr:
                        error_msg += f"\nStderr: {stderr}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)

            self.logger.info("Quantum ESPRESSO execution completed")

            # Parse output
            return self._parse_output()

        except FileNotFoundError:
            raise RuntimeError(
                f"Quantum ESPRESSO executable '{self.qe_executable}' not found. "
                f"Please install QE or set QE_EXECUTABLE environment variable."
            )
        except Exception as e:
            self.logger.error(f"Error running Quantum ESPRESSO: {e}", exc_info=True)
            raise RuntimeError(f"Quantum ESPRESSO execution failed: {e}") from e

    def _parse_output(self) -> Dict[str, Any]:
        """
        Parse Quantum ESPRESSO output file.

        Extracts:
        - Total energy
        - Convergence status
        - Forces
        - Final coordinates (if relaxation)
        - SCF iterations

        Returns:
            Dictionary with parsed results

        Raises:
            ValueError: If output parsing fails
        """
        self.logger.info("Parsing Quantum ESPRESSO output...")

        if not os.path.exists(self.output_file):
            raise ValueError(f"Output file not found: {self.output_file}")

        with open(self.output_file, 'r') as f:
            output_text = f.read()

        # Extract total energy
        energy_match = re.search(r'!\s+total energy\s+=\s+([-\d.]+)\s+Ry', output_text)
        if energy_match:
            total_energy_ry = float(energy_match.group(1))
            total_energy_ev = total_energy_ry * 13.6057  # Ry to eV conversion
        else:
            self.logger.warning("Could not find total energy in output")
            total_energy_ry = 0.0
            total_energy_ev = 0.0

        # Check convergence
        converged = "convergence has been achieved" in output_text

        # Extract SCF iterations
        scf_iterations = len(re.findall(r'iteration #\s*\d+', output_text))

        # Extract forces
        force_matches = re.findall(r'force =\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', output_text)
        if force_matches:
            forces = [[float(x) for x in match] for match in force_matches]
            max_force = max(sum(f**2 for f in force)**0.5 for force in forces)
            rms_force = (sum(sum(f**2 for f in force) for force in forces) / len(forces))**0.5
        else:
            max_force = 0.0
            rms_force = 0.0

        # Extract final coordinates (if relaxation)
        final_positions = None
        if self.parameters.get("calculation", "scf") in ["relax", "vc-relax"]:
            # Look for ATOMIC_POSITIONS in output
            pos_match = re.search(r'ATOMIC_POSITIONS \((\w+)\)\n((?:.+\n)+?)(?:\n|\Z)', output_text)
            if pos_match:
                coord_type = pos_match.group(1)
                pos_lines = pos_match.group(2).strip().split('\n')
                final_positions = []
                for line in pos_lines:
                    parts = line.split()
                    if len(parts) >= 4:
                        final_positions.append({
                            "atom": parts[0],
                            "position": [float(parts[1]), float(parts[2]), float(parts[3])]
                        })

        # Build results
        n_atoms = len(self.structure.get("atoms", []))
        energy_per_atom = total_energy_ev / n_atoms if n_atoms > 0 else 0.0

        results = {
            "engine": "QE",
            "total_energy": round(total_energy_ev, 6),
            "total_energy_ry": round(total_energy_ry, 6),
            "energy_per_atom": round(energy_per_atom, 6),
            "n_atoms": n_atoms,
            "formula": self.structure.get("formula", "Unknown"),
            "convergence": {
                "reached": converged,
                "iterations": scf_iterations,
                "threshold": self.parameters.get("conv_thr", 1.0e-8),
            },
            "forces": {
                "max_force": round(max_force, 6),
                "rms_force": round(rms_force, 6),
            },
            "calculation_details": {
                "calculation_type": self.parameters.get("calculation", "scf"),
                "ecutwfc": self.parameters.get("ecutwfc", 50.0),
                "k_points": self.parameters.get("k_points", [4, 4, 4]),
                "functional": "PBE",
            }
        }

        if final_positions:
            results["final_positions"] = final_positions

        self.logger.info(f"Parsed QE output: E={total_energy_ev:.6f} eV, converged={converged}")

        # Return in standard format
        return {
            "summary": results,
            "convergence_reached": converged,
            "quality_score": 0.9 if converged else 0.5,
            "metadata": {
                "engine": "QE",
                "engine_version": "7.0+",
                "structure_id": str(self.structure.get("id", "unknown")),
                "work_dir": self.work_dir,
                "mock_mode": self.mock_mode,
            }
        }

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self.work_dir and os.path.exists(self.work_dir):
            try:
                # In production, might want to keep some files or archive them
                # For now, just log the location
                self.logger.info(f"Working directory preserved at: {self.work_dir}")
                # Uncomment to actually delete:
                # shutil.rmtree(self.work_dir)
                # self.logger.info(f"Cleaned up working directory: {self.work_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup working directory: {e}")
