"""Parse ``log.lammps`` for thermo data + run metadata.

LAMMPS emits thermo output in a repeating pattern between ``run N``
commands::

    Per MPI rank memory allocation (min/avg/max) = 4.123 | 4.123 | 4.123 Mbytes
       Step          Temp          PotEng         TotEng         Press
           0           0           -405.96231     -405.96231      -4326.93
         100     299.89457        -405.56231     -404.40231      -2231.12
         ...

Different runs can use different ``thermo_style`` → different columns.
The parser reads each block separately and keeps all of them.

What we extract
---------------

- **Thermo table(s)**: one per ``run`` command, as a column-name →
  list-of-floats mapping (pandas-free to avoid a dependency).
- **Loop/wall time**: ``Total wall time:`` at the end + the per-run
  ``Loop time of N.NNN`` line.
- **Errors**: ``ERROR:`` lines (LAMMPS's convention for fatal messages).
- **Warnings**: ``WARNING:`` count (not the bodies; these are noisy).
- **LAMMPS version + git commit**: from the preamble for provenance.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


class LAMMPSLogParseError(ValueError):
    """The log file is missing or too malformed to extract anything useful."""


@dataclass
class ThermoTable:
    """One thermo block between ``run`` commands."""

    step: List[int] = field(default_factory=list)
    columns: Dict[str, List[float]] = field(default_factory=dict)
    # Some LAMMPS setups emit Step as a float; we coerce to int, but
    # keep the raw column names as the thermo_style line gave them.
    column_order: List[str] = field(default_factory=list)
    loop_time_seconds: Optional[float] = None

    def __len__(self) -> int:
        return len(self.step)

    def final(self, key: str) -> Optional[float]:
        """Return the last-step value for *key*, or None if not present."""
        col = self.columns.get(key)
        if not col:
            return None
        return col[-1]

    def get(self, key: str) -> List[float]:
        return self.columns.get(key, [])


@dataclass
class LAMMPSLog:
    """Structured view of one ``log.lammps`` file."""

    thermo_tables: List[ThermoTable] = field(default_factory=list)
    wall_time_seconds: Optional[float] = None
    lammps_version: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings_count: int = 0

    @property
    def last_table(self) -> Optional[ThermoTable]:
        return self.thermo_tables[-1] if self.thermo_tables else None

    def final_values(self) -> Dict[str, float]:
        """Last-step values from the last thermo table."""
        tbl = self.last_table
        if tbl is None:
            return {}
        out: Dict[str, float] = {}
        for k in tbl.column_order:
            if k == "Step":
                continue
            v = tbl.final(k)
            if v is not None:
                out[k] = v
        return out


# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------


# Version line: "LAMMPS (22 Jul 2025 - Update 4)" — we take the whole string.
_VERSION_RE = re.compile(r"^LAMMPS\s*\(([^)]+)\)", re.MULTILINE)

# "Loop time of 12.345 on 1 procs..."
_LOOP_TIME_RE = re.compile(r"^Loop time of\s+([\d.Ee+-]+)\s+on", re.MULTILINE)

# "Total wall time: 0:00:12"  (h:mm:ss)
_TOTAL_WALL_RE = re.compile(
    r"^Total wall time:\s*(\d+):(\d+):(\d+)", re.MULTILINE,
)

_ERROR_RE = re.compile(r"^ERROR[^\n]*", re.MULTILINE)
_WARNING_RE = re.compile(r"^WARNING[^\n]*", re.MULTILINE)

# Thermo block header: the line right after "Per MPI rank memory allocation"
# gives the column names. Example:
#    "   Step          Temp          PotEng         TotEng         Press"
_THERMO_HEADER_ANCHOR = "Per MPI rank memory allocation"

# End of a thermo block: either "Loop time of" or a blank followed by
# something that isn't a number row.
_NUMERIC_ROW_RE = re.compile(r"^\s*-?\d")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_lammps_log(source: str | Path) -> LAMMPSLog:
    """Parse a ``log.lammps`` (or the text) into :class:`LAMMPSLog`."""
    text = _load(source)
    if not text.strip():
        raise LAMMPSLogParseError("empty log")

    out = LAMMPSLog()

    m = _VERSION_RE.search(text)
    if m is not None:
        out.lammps_version = m.group(1).strip()

    # Total wall
    m = _TOTAL_WALL_RE.search(text)
    if m is not None:
        h, mm, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        out.wall_time_seconds = h * 3600 + mm * 60 + s

    # Errors + warnings
    out.errors = [e.strip() for e in _ERROR_RE.findall(text)]
    out.warnings_count = len(_WARNING_RE.findall(text))

    # Thermo tables: scan by anchor.
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if _THERMO_HEADER_ANCHOR in lines[i]:
            table, consumed = _parse_thermo_block(lines, i + 1)
            if table is not None:
                out.thermo_tables.append(table)
            i += consumed + 1
        else:
            i += 1

    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8", errors="replace")
    if (
        isinstance(source, str)
        and source.strip()
        and "\n" not in source
        and len(source) < 4096
    ):
        p = Path(source)
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")
    return source


def _parse_thermo_block(lines: List[str], start: int) -> tuple[Optional[ThermoTable], int]:
    """Parse one block starting at *start* (the header line).

    Returns (table or None, number of lines consumed after *start*).
    """
    if start >= len(lines):
        return None, 0
    header = lines[start].strip().split()
    if not header or header[0] != "Step":
        return None, 1
    table = ThermoTable(column_order=header)
    for col in header:
        table.columns[col] = []
    j = start + 1
    while j < len(lines):
        line = lines[j]
        stripped = line.strip()
        if not stripped:
            # Blank lines inside a thermo block are unusual but we
            # treat them as end.
            break
        if not _NUMERIC_ROW_RE.match(line):
            # Non-numeric row — end of block.
            m = _LOOP_TIME_RE.match(line)
            if m is not None:
                try:
                    table.loop_time_seconds = float(m.group(1))
                except ValueError:
                    pass
            break
        parts = stripped.split()
        if len(parts) != len(header):
            break
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            break
        # Step is always integer-valued; keep it as int for API sanity.
        try:
            table.step.append(int(nums[0]))
        except (ValueError, TypeError):
            table.step.append(int(round(nums[0])))
        for k, v in zip(header, nums):
            table.columns[k].append(v)
        j += 1
    return table, j - start
