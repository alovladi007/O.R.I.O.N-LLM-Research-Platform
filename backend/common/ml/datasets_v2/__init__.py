"""Versioned dataset registry + split strategies (Phase 6 / Session 6.2).

Fresh package sitting alongside the legacy Session-14
:mod:`backend.common.ml.datasets`. The legacy module is a one-shot
GNN dataset builder without splits, content hashes, or versioned
snapshots. New callers should prefer this package; the legacy
:mod:`datasets` path stays for the existing CGCNN-training code
until Session 6.4 migrates it.

Public API
----------

- :class:`PropertyRow` — ``(structure, property, method, conditions)``
  tuple; the input unit.
- :class:`FilterSpec`, :class:`SplitSpec`, :class:`DatasetConfig`,
  :class:`DatasetSnapshot` — pydantic schemas.
- :func:`apply_filter`, :func:`compile_filter`,
  :exc:`FilterParseError` — filter engine.
- :func:`random_split`, :func:`stratified_by_prototype_split`,
  :func:`structure_cluster_split`, :func:`split_rows`,
  :func:`scaffold_split` (deferred) — split strategies.
- :class:`DatasetRegistry`, :exc:`DatasetRegistryError` — the in-memory
  registry.

Storage note
------------

Session 6.2 ships an in-memory backend. A DB-backed variant (an
``orion_datasets`` table + an Alembic migration) is scoped for 6.2b
or 6.3, whichever needs it first. The snapshot API is deliberately
storage-agnostic — the content hash is identity, the rest is
bookkeeping.
"""

from .filter import FilterParseError, apply_filter, compile_filter
from .registry import DatasetRegistry, DatasetRegistryError
from .schema import (
    DatasetConfig,
    DatasetSnapshot,
    FilterSpec,
    PropertyRow,
    RowID,
    SplitKind,
    SplitSpec,
)
from .splits import (
    FingerprintFn,
    random_split,
    scaffold_split,
    split_rows,
    stratified_by_prototype_split,
    structure_cluster_split,
)

__all__ = [
    "DatasetConfig",
    "DatasetRegistry",
    "DatasetRegistryError",
    "DatasetSnapshot",
    "FilterParseError",
    "FilterSpec",
    "FingerprintFn",
    "PropertyRow",
    "RowID",
    "SplitKind",
    "SplitSpec",
    "apply_filter",
    "compile_filter",
    "random_split",
    "scaffold_split",
    "split_rows",
    "stratified_by_prototype_split",
    "structure_cluster_split",
]
