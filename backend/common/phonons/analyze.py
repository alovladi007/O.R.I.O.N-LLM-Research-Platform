"""Phonon analyzer: band, DOS, thermo + imaginary-mode flag.

Wraps phonopy's high-level driver so callers never touch
``ph.run_*`` / ``ph.get_*_dict`` directly. Returns a single
:class:`PhononResult` dataclass that's JSON-serializable for the
property store.

Imaginary-mode convention
-------------------------

Phonopy reports negative frequencies for imaginary modes. We
flag a structure as **dynamically unstable** when any frequency
across the supplied q-mesh falls below ``-IMAGINARY_TOLERANCE``
(default 0.05 THz) — small negative numbers near Γ are typical
acoustic-sum-rule numerical noise, not real instabilities, and
shouldn't trip the flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .driver import ForceDriver

# Frequencies below -IMAGINARY_TOLERANCE THz are considered real
# imaginary modes (dynamical instabilities). Anything above
# represents acoustic-sum-rule float noise near Γ.
IMAGINARY_TOLERANCE_THZ = 0.05


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PhononResult:
    """Frozen output of a phonon calculation.

    Carries the raw arrays + a few derived scalars (Debye T, max
    frequency, number of imaginary modes) suitable for ORION's
    structured-property store. The full band + DOS arrays are kept
    so the frontend can re-render without recomputing.
    """

    frequencies_thz: np.ndarray            # (n_q, n_bands)
    qpoints: np.ndarray                    # (n_q, 3) reduced coords
    dos_frequencies_thz: np.ndarray        # (n_pts,)
    dos_values: np.ndarray                 # (n_pts,)
    thermal_temperatures_k: np.ndarray     # (n_T,)
    thermal_free_energy_kj_per_mol: np.ndarray
    thermal_entropy_j_per_k_per_mol: np.ndarray
    thermal_heat_capacity_j_per_k_per_mol: np.ndarray
    debye_temperature_k: float
    max_frequency_thz: float
    n_imaginary_modes: int
    has_imaginary: bool

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict for the property store / API responses."""
        return {
            "frequencies_thz": self.frequencies_thz.tolist(),
            "qpoints": self.qpoints.tolist(),
            "dos_frequencies_thz": self.dos_frequencies_thz.tolist(),
            "dos_values": self.dos_values.tolist(),
            "thermal_temperatures_k": self.thermal_temperatures_k.tolist(),
            "thermal_free_energy_kj_per_mol":
                self.thermal_free_energy_kj_per_mol.tolist(),
            "thermal_entropy_j_per_k_per_mol":
                self.thermal_entropy_j_per_k_per_mol.tolist(),
            "thermal_heat_capacity_j_per_k_per_mol":
                self.thermal_heat_capacity_j_per_k_per_mol.tolist(),
            "debye_temperature_k": self.debye_temperature_k,
            "max_frequency_thz": self.max_frequency_thz,
            "n_imaginary_modes": self.n_imaginary_modes,
            "has_imaginary": self.has_imaginary,
        }


# ---------------------------------------------------------------------------
# Imaginary-mode flag
# ---------------------------------------------------------------------------


def has_imaginary_modes(
    frequencies_thz: np.ndarray,
    *,
    tolerance_thz: float = IMAGINARY_TOLERANCE_THZ,
) -> Tuple[bool, int]:
    """Return ``(has_imag, count)`` from a frequency array.

    Counts entries strictly below ``-tolerance_thz``. Acoustic-sum-rule
    noise (small negative numbers near Γ) is tolerated.
    """
    f = np.asarray(frequencies_thz)
    mask = f < -tolerance_thz
    return bool(mask.any()), int(mask.sum())


# ---------------------------------------------------------------------------
# Debye temperature from DOS
# ---------------------------------------------------------------------------


_PLANCK_J_S = 6.62607015e-34
_BOLTZMANN_J_PER_K = 1.380649e-23


def debye_temperature_from_dos(
    dos_frequencies_thz: np.ndarray,
    dos_values: np.ndarray,
) -> float:
    """Anderson 1965 second-moment formula:

        Θ_D = (h / k_B) · √(5 · ⟨ω²⟩ / 3) / (2 π)

    where ``⟨ω²⟩`` is the DOS-weighted average of ω². Drops modes at
    or below 0 THz to avoid contamination from the acoustic-sum-rule
    numerical zero at Γ.
    """
    f = np.asarray(dos_frequencies_thz, dtype=np.float64)
    d = np.asarray(dos_values, dtype=np.float64)
    valid = f > 0.0
    if not valid.any():
        return float("nan")
    f = f[valid]
    d = d[valid]
    omega = 2.0 * np.pi * f * 1e12  # rad/s
    norm = np.trapezoid(d, f) if hasattr(np, "trapezoid") else np.trapz(d, f)
    if norm <= 0:
        return float("nan")
    avg_omega2 = (
        np.trapezoid(omega ** 2 * d, f) if hasattr(np, "trapezoid")
        else np.trapz(omega ** 2 * d, f)
    ) / norm
    theta_D = (
        (_PLANCK_J_S / _BOLTZMANN_J_PER_K)
        * np.sqrt(5.0 * avg_omega2 / 3.0)
        / (2.0 * np.pi)
    )
    return float(theta_D)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run_phonons(
    *,
    phonopy_obj,                          # phonopy.Phonopy
    driver: ForceDriver,
    mesh: Tuple[int, int, int] = (20, 20, 20),
    band_qpoints: Optional[np.ndarray] = None,
    thermal_t_min: float = 10.0,
    thermal_t_max: float = 1000.0,
    thermal_t_step: float = 10.0,
    displacement_distance_a: float = 0.01,
) -> PhononResult:
    """End-to-end: generate displacements, fetch forces, run analyses.

    Parameters
    ----------
    phonopy_obj
        A pre-initialized :class:`phonopy.Phonopy` (caller chose the
        primitive + supercell). The function generates displacements
        if ``phonopy_obj.dataset is None``.
    driver
        :class:`ForceDriver` to compute per-displacement forces.
    mesh
        Q-point mesh for DOS + thermal properties.
    band_qpoints
        Optional ``(n_q, 3)`` reduced-coordinate q-points for the
        explicit band structure. Default is just the Γ point.
    """
    if phonopy_obj.dataset is None:
        phonopy_obj.generate_displacements(distance=displacement_distance_a)

    sc = phonopy_obj.supercell
    forces = []
    for entry in phonopy_obj.dataset["first_atoms"]:
        f = driver.compute_forces(supercell=sc, displacement=entry)
        f = np.asarray(f, dtype=np.float64)
        if f.shape != (len(sc), 3):
            raise ValueError(
                f"driver returned force shape {f.shape}; "
                f"expected ({len(sc)}, 3)"
            )
        forces.append(f)
    phonopy_obj.forces = np.array(forces)
    phonopy_obj.produce_force_constants()

    # Band (explicit q-points if supplied; else Γ for a sanity probe).
    qpts = (
        np.asarray(band_qpoints, dtype=np.float64)
        if band_qpoints is not None
        else np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    )
    phonopy_obj.run_qpoints(qpts, with_eigenvectors=False)
    band_freqs = np.asarray(phonopy_obj.qpoints.frequencies)

    # Mesh + DOS + thermal.
    phonopy_obj.run_mesh(mesh=list(mesh))
    phonopy_obj.run_total_dos()
    dos = phonopy_obj.total_dos
    phonopy_obj.run_thermal_properties(
        t_step=thermal_t_step, t_max=thermal_t_max, t_min=thermal_t_min,
    )
    thermo = phonopy_obj.get_thermal_properties_dict()

    dos_freqs = np.asarray(dos.frequency_points, dtype=np.float64)
    dos_vals = np.asarray(dos.dos, dtype=np.float64)
    theta_D = debye_temperature_from_dos(dos_freqs, dos_vals)

    # Aggregate imaginary-mode count across the full mesh.
    mesh_freqs = np.asarray(phonopy_obj.mesh.frequencies)
    has_imag, n_imag = has_imaginary_modes(mesh_freqs)
    max_freq = float(mesh_freqs.max())

    return PhononResult(
        frequencies_thz=band_freqs,
        qpoints=qpts,
        dos_frequencies_thz=dos_freqs,
        dos_values=dos_vals,
        thermal_temperatures_k=np.asarray(thermo["temperatures"]),
        thermal_free_energy_kj_per_mol=np.asarray(thermo["free_energy"]),
        thermal_entropy_j_per_k_per_mol=np.asarray(thermo["entropy"]),
        thermal_heat_capacity_j_per_k_per_mol=np.asarray(thermo["heat_capacity"]),
        debye_temperature_k=theta_D,
        max_frequency_thz=max_freq,
        n_imaginary_modes=n_imag,
        has_imaginary=has_imag,
    )


def extract_band_dos_thermo(result: PhononResult) -> Dict[str, Any]:
    """Convenience: shape the artifact JSON the way the API + frontend
    expect (band_json + dos_csv-shaped + thermo dict).

    Used by Session 8.2b's API/router to dump phonon_band.json /
    phonon_dos.csv / thermo.csv into MinIO under the job artifact
    bundle. The roadmap calls these names out explicitly.
    """
    return {
        "band_json": {
            "qpoints": result.qpoints.tolist(),
            "frequencies_thz": result.frequencies_thz.tolist(),
        },
        "dos_csv_rows": [
            {"frequency_thz": float(f), "dos": float(v)}
            for f, v in zip(result.dos_frequencies_thz, result.dos_values)
        ],
        "thermo_csv_rows": [
            {
                "temperature_k": float(T),
                "free_energy_kj_per_mol": float(F),
                "entropy_j_per_k_per_mol": float(S),
                "heat_capacity_j_per_k_per_mol": float(C),
            }
            for T, F, S, C in zip(
                result.thermal_temperatures_k,
                result.thermal_free_energy_kj_per_mol,
                result.thermal_entropy_j_per_k_per_mol,
                result.thermal_heat_capacity_j_per_k_per_mol,
            )
        ],
        "summary": {
            "debye_temperature_k": result.debye_temperature_k,
            "max_frequency_thz": result.max_frequency_thz,
            "n_imaginary_modes": result.n_imaginary_modes,
            "has_imaginary": result.has_imaginary,
        },
    }
