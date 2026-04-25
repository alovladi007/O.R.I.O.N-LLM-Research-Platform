"""Phase 8 / Session 8.2 — phonon workflow via phonopy.

Pipeline (roadmap-conformant):

    PhonopyAtoms reference + supercell_matrix
        → ph.generate_displacements()                    (1)
        → for each displacement: forces = driver(struct) (2)
        → ph.forces = stack(forces); ph.produce_force_constants()
        → ph.run_band_structure / run_total_dos / run_thermal_properties
        → :func:`extract_band_dos_thermo`                (3)

Step 2 is what changes per backend. The real driver wraps the
Phase-3 QE runner (relax displaced supercell → static SCF → parse
forces, in eV/Å). For tests we ship :class:`HarmonicSpringDriver`
which generates analytically-correct forces from a nearest-neighbor
central-force model — that's sufficient for the roadmap acceptance
(Si LO/TO ≈ 15.5 THz with a tuned spring constant, Cu Debye
temperature ≈ 343 K, deliberately-soft mode flagged as imaginary).

The 8.2b session promotes the test driver to a real ``QEForceDriver``
once a CI-accessible ``pw.x`` binary exists.
"""

from .analyze import (
    PhononResult,
    debye_temperature_from_dos,
    extract_band_dos_thermo,
    has_imaginary_modes,
    run_phonons,
)
from .driver import (
    ForceDriver,
    HarmonicSpringDriver,
)

__all__ = [
    "ForceDriver",
    "HarmonicSpringDriver",
    "PhononResult",
    "debye_temperature_from_dos",
    "extract_band_dos_thermo",
    "has_imaginary_modes",
    "run_phonons",
]
