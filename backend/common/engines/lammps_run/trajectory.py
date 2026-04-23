"""Parse LAMMPS ``dump`` files (``custom`` style).

Session 4.1's renderer uses the ``metal``/``real``/``lj`` units conventions
and emits dumps via::

    dump 1 all custom N traj.dump id type x y z

Format per frame::

    ITEM: TIMESTEP
    100
    ITEM: NUMBER OF ATOMS
    108
    ITEM: BOX BOUNDS pp pp pp
    0.0 10.8
    0.0 10.8
    0.0 10.8
    ITEM: ATOMS id type x y z
    1 1 0.0 0.0 0.0
    ...

This parser is a streaming reader — it yields one
:class:`TrajectoryFrame` per frame so a 1 GB trajectory doesn't have to
fit in memory. For RDF / MSD analysis we often iterate twice, so the
iterator is re-usable by construction (call :func:`parse_lammps_dump`
again rather than trying to reset the generator).

What we do *not* do here
------------------------

- Triclinic box handling (the ``xy xz yz`` line). The default
  ensembles Session 4.1 ships are all orthorhombic; we flag and refuse
  triclinic with a clear error. When Session 4.3's NPT workflows need
  it, this module will grow.
- extxyz / HDF5 conversion. Mentioned in the roadmap, deferred — the
  dump format is fine for the Session 4.2 acceptance analyzers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence


class TrajectoryParseError(ValueError):
    """Raised when a dump file can't be parsed as ``custom`` style."""


@dataclass
class TrajectoryFrame:
    """One frame from a LAMMPS custom dump."""

    timestep: int
    n_atoms: int
    # Orthorhombic box: [lo, hi] per axis in the dump's length units
    # (Å for metal, σ for LJ). Caller knows the units.
    box_lo: List[float]
    box_hi: List[float]
    # Column names from the "ITEM: ATOMS ..." header, in order.
    column_names: List[str] = field(default_factory=list)
    # Per-atom rows, same length as n_atoms. Each row is a list of
    # floats in the column order above. We keep as list-of-list rather
    # than numpy to keep the base parser dependency-free; analyzers
    # do the numpy conversion.
    rows: List[List[float]] = field(default_factory=list)

    def box_lengths(self) -> List[float]:
        return [self.box_hi[i] - self.box_lo[i] for i in range(3)]

    def box_volume(self) -> float:
        lx, ly, lz = self.box_lengths()
        return lx * ly * lz

    def coords(self) -> List[List[float]]:
        """Wrapped Cartesian (x, y, z), atom-id-sorted.

        Use this for pair-distance calculations (RDF) that rely on
        the minimum-image convention. For trajectory-style analyses
        where atoms cross periodic boundaries (MSD, VACF, diffusion),
        use :meth:`coords_unwrapped` instead.
        """
        return self._coords_for(("x", "y", "z"))

    def coords_unwrapped(self) -> List[List[float]]:
        """Unwrapped Cartesian (xu, yu, zu), atom-id-sorted.

        Falls back to wrapped (x, y, z) when the dump didn't include
        the ``xu yu zu`` columns — the Session 4.3b renderer emits
        them by default, but older trajectories won't have them and
        we prefer returning *something* over failing. Callers that
        need *guaranteed* unwrapping should check the presence of
        ``xu`` in :attr:`column_names` themselves.
        """
        if "xu" in self.column_names:
            return self._coords_for(("xu", "yu", "zu"))
        return self._coords_for(("x", "y", "z"))

    def _coords_for(self, keys) -> List[List[float]]:
        try:
            idxs = [self.column_names.index(k) for k in keys]
        except ValueError:
            raise TrajectoryParseError(
                f"dump columns missing {keys}: have {self.column_names}"
            )
        id_idx = (
            self.column_names.index("id") if "id" in self.column_names else None
        )
        rows = self.rows
        if id_idx is not None:
            rows = sorted(rows, key=lambda r: int(r[id_idx]))
        return [[r[idxs[0]], r[idxs[1]], r[idxs[2]]] for r in rows]

    def atom_types(self) -> List[int]:
        """Per-atom type IDs, atom-id-sorted."""
        if "type" not in self.column_names:
            return []
        t_idx = self.column_names.index("type")
        id_idx = (
            self.column_names.index("id") if "id" in self.column_names else None
        )
        rows = self.rows
        if id_idx is not None:
            rows = sorted(rows, key=lambda r: int(r[id_idx]))
        return [int(r[t_idx]) for r in rows]


def parse_lammps_dump(source: str | Path) -> Iterator[TrajectoryFrame]:
    """Yield :class:`TrajectoryFrame` instances from a custom-style dump.

    *source* may be a file path or raw text. The generator closes the
    underlying file after the last frame.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_file():
            file_obj = path.open("r", encoding="utf-8", errors="replace")
            try:
                yield from _iter_frames(file_obj)
            finally:
                file_obj.close()
            return
    # Treat source as raw text.
    from io import StringIO

    yield from _iter_frames(StringIO(str(source)))


def _iter_frames(handle) -> Iterator[TrajectoryFrame]:
    """Streaming frame parser."""
    line = handle.readline()
    while line:
        # Anchor: "ITEM: TIMESTEP"
        if line.startswith("ITEM: TIMESTEP"):
            frame = _read_one_frame(handle)
            if frame is not None:
                yield frame
        line = handle.readline()


def _read_one_frame(handle) -> Optional[TrajectoryFrame]:
    # TIMESTEP value
    ts_line = handle.readline().strip()
    if not ts_line:
        return None
    try:
        timestep = int(float(ts_line))
    except ValueError:
        raise TrajectoryParseError(f"bad TIMESTEP line: {ts_line!r}")

    # ITEM: NUMBER OF ATOMS
    hdr = handle.readline()
    if "NUMBER OF ATOMS" not in hdr:
        raise TrajectoryParseError(f"expected NUMBER OF ATOMS, got {hdr!r}")
    n_atoms = int(handle.readline().strip())

    # ITEM: BOX BOUNDS
    bb = handle.readline()
    if "BOX BOUNDS" not in bb:
        raise TrajectoryParseError(f"expected BOX BOUNDS, got {bb!r}")
    # Triclinic check: the box-bounds line mentions xy xz yz when triclinic.
    if "xy" in bb or "xz" in bb or "yz" in bb:
        raise TrajectoryParseError(
            "triclinic box not supported yet (Session 4.2 MVP is orthorhombic)"
        )
    lo: List[float] = []
    hi: List[float] = []
    for _ in range(3):
        parts = handle.readline().strip().split()
        if len(parts) < 2:
            raise TrajectoryParseError("malformed BOX BOUNDS row")
        lo.append(float(parts[0]))
        hi.append(float(parts[1]))

    # ITEM: ATOMS <columns>
    atoms_hdr = handle.readline()
    if not atoms_hdr.startswith("ITEM: ATOMS"):
        raise TrajectoryParseError(f"expected ATOMS, got {atoms_hdr!r}")
    column_names = atoms_hdr.strip().split()[2:]

    rows: List[List[float]] = []
    for _ in range(n_atoms):
        row = handle.readline().strip().split()
        if len(row) != len(column_names):
            raise TrajectoryParseError(
                f"atom row has {len(row)} fields, expected {len(column_names)}"
            )
        rows.append([float(x) for x in row])

    return TrajectoryFrame(
        timestep=timestep,
        n_atoms=n_atoms,
        box_lo=lo,
        box_hi=hi,
        column_names=column_names,
        rows=rows,
    )
