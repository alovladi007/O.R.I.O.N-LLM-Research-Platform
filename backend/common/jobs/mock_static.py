"""Pure mock-static computation used by the Celery task and by tests.

The math here is deliberately cheap so the Session 2.2 acceptance test
(20 jobs in parallel in ≤60 s) is comfortable. The formula matches the
Session 2.2 roadmap spec:

    energy_ev            = -sum(Z_i) * 1.5 + 0.01 * noise
    max_abs_force_ev/Å   ≈ 0.05 (per-atom Gaussian draws)
    trajectory           = 10 frames, small displacements from frame 0

Everything is deterministic given the seed. The seed is derived from
``structure_id`` so the same structure produces the same numbers across
workers — handy for reproducibility tests.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

from .schemas import MockForceEntry, MockStaticOutput, MockTrajectoryFrame


TRAJECTORY_N_FRAMES = 10
FORCE_NOISE_SIGMA_EV_PER_ANG = 0.05
ENERGY_NOISE_SCALE_EV = 0.01
COEFFICIENT_EV_PER_Z = -1.5
POSITION_NOISE_SIGMA_ANG = 0.01


def _seed_from_structure_id(structure_id: str) -> int:
    """Deterministic non-negative seed from a structure id."""
    digest = hashlib.sha1(structure_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _atomic_number(species: str) -> int:
    """Look up atomic number; fall back to 1 if pymatgen can't resolve it."""
    try:
        from pymatgen.core import Element  # lazy — keep import cost out of workers

        return int(Element(species).Z)
    except Exception:  # noqa: BLE001
        return 1


def _format_formula(species_counts: Dict[str, int]) -> str:
    """``{'Na': 1, 'Cl': 1}`` → ``'NaCl'`` (stable, alphabetical ordering)."""
    parts: List[str] = []
    for sym in sorted(species_counts):
        cnt = species_counts[sym]
        parts.append(f"{sym}{cnt}" if cnt != 1 else sym)
    return "".join(parts)


def _normalize_atoms(atoms: List[Dict[str, Any]]) -> List[Tuple[str, Tuple[float, float, float]]]:
    """Accept the loose atoms dict the DB stores and tighten it up."""
    normalized: List[Tuple[str, Tuple[float, float, float]]] = []
    for atom in atoms:
        species = atom.get("species") or atom.get("symbol") or atom.get("element")
        if not species:
            raise ValueError(f"atom entry missing species: {atom}")
        pos = atom.get("position") or atom.get("coords") or atom.get("xyz") or [0.0, 0.0, 0.0]
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        normalized.append((str(species), (x, y, z)))
    return normalized


def run_mock_static(
    *,
    structure_id: str,
    atoms: List[Dict[str, Any]],
    formula: Optional[str] = None,
) -> MockStaticOutput:
    """Compute the mock-static output for *atoms*.

    Parameters
    ----------
    structure_id
        Seed source — same structure id always returns the same numbers.
    atoms
        Sequence of ``{"species": "Si", "position": [x, y, z]}`` dicts. The
        Session 2.2 mock allows coordinates to be in any convention
        (fractional or Cartesian) since the formula only uses Z_i and the
        position noise for trajectory frames is isotropic.
    formula
        Optional chemical formula override; if omitted, derived from *atoms*.

    Returns
    -------
    MockStaticOutput
        Validated pydantic model — use ``.model_dump()`` to persist.
    """
    if not atoms:
        raise ValueError("mock_static requires at least one atom")

    normalized = _normalize_atoms(atoms)
    species_list = [s for s, _ in normalized]
    z_list = [_atomic_number(s) for s in species_list]

    seed = _seed_from_structure_id(structure_id)
    rng = random.Random(seed)

    # Energy: deterministic physics-shaped baseline + small noise.
    z_sum = sum(z_list)
    noise_energy = rng.gauss(0.0, 1.0)
    energy_ev = COEFFICIENT_EV_PER_Z * z_sum + ENERGY_NOISE_SCALE_EV * noise_energy
    energy_per_atom_ev = energy_ev / len(normalized)

    # Forces: per-atom zero vector + Gaussian noise (σ = FORCE_NOISE_SIGMA).
    forces: List[MockForceEntry] = []
    max_abs = 0.0
    for species in species_list:
        fx = rng.gauss(0.0, FORCE_NOISE_SIGMA_EV_PER_ANG)
        fy = rng.gauss(0.0, FORCE_NOISE_SIGMA_EV_PER_ANG)
        fz = rng.gauss(0.0, FORCE_NOISE_SIGMA_EV_PER_ANG)
        forces.append(MockForceEntry(species=species, fx=fx, fy=fy, fz=fz))
        max_abs = max(max_abs, abs(fx), abs(fy), abs(fz))

    # Trajectory: 10 frames, small random displacements from the input.
    frames: List[MockTrajectoryFrame] = []
    base_positions = [list(pos) for _, pos in normalized]
    for frame_idx in range(TRAJECTORY_N_FRAMES):
        if frame_idx == 0:
            positions = [list(p) for p in base_positions]
        else:
            positions = []
            for base in base_positions:
                dx = rng.gauss(0.0, POSITION_NOISE_SIGMA_ANG)
                dy = rng.gauss(0.0, POSITION_NOISE_SIGMA_ANG)
                dz = rng.gauss(0.0, POSITION_NOISE_SIGMA_ANG)
                positions.append([base[0] + dx, base[1] + dy, base[2] + dz])
        frames.append(MockTrajectoryFrame(index=frame_idx, positions=positions))

    # Formula
    if formula is None:
        counts: Dict[str, int] = {}
        for sym in species_list:
            counts[sym] = counts.get(sym, 0) + 1
        formula_value = _format_formula(counts)
    else:
        formula_value = formula

    return MockStaticOutput(
        engine="mock",
        n_atoms=len(normalized),
        formula=formula_value,
        energy_ev=energy_ev,
        energy_per_atom_ev=energy_per_atom_ev,
        max_abs_force_ev_per_ang=max_abs,
        forces=forces,
        trajectory_n_frames=len(frames),
        trajectory=frames,
        converged=True,
        seed=seed,
    )


def write_trajectory_xyz(frames: List[MockTrajectoryFrame], species: List[str]) -> str:
    """Render *frames* to XYZ multi-frame format.

    The returned string is safe to write with ``Path.write_text`` and is
    what gets tar'd into the run dir and uploaded to MinIO.
    """
    if not frames:
        return ""
    n_atoms = len(species)
    lines: List[str] = []
    for frame in frames:
        if len(frame.positions) != n_atoms:
            raise ValueError(
                f"frame {frame.index} has {len(frame.positions)} positions "
                f"but structure has {n_atoms} atoms"
            )
        lines.append(str(n_atoms))
        lines.append(f"frame={frame.index}")
        for sym, (x, y, z) in zip(species, frame.positions):
            lines.append(f"{sym} {x:.6f} {y:.6f} {z:.6f}")
    return "\n".join(lines) + "\n"
