"""Dataset registry — Phase 6 / Session 6.2.

This module is the roadmap-specified import path
(``from backend.common.ml.datasets import ...``). The implementation
lives in :mod:`backend.common.ml.datasets_v2`; this file re-exports
the public API so the roadmap-canonical path and the
implementation-subpackage path refer to the same objects.

The earlier Session-14 ``datasets.py`` (a one-shot GNN dataset
builder with no splits, no snapshots, no content hashes) has been
moved to :mod:`_legacy_datasets_session14` for historical reference.
Nothing in the canonical codebase imports the legacy module, so the
rename is internal-only.

Why the v1/v2 split in :mod:`datasets_v2`
-----------------------------------------

Session 6.2 builds the real versioned registry while the legacy file
is still on disk. Keeping the implementation in ``datasets_v2/``
(a package) makes the internal organization — schema, filter DSL,
split strategies, registry — obvious. This shim module preserves
the roadmap's import path without forcing the implementation to
collapse into a single file.

Public API
----------

See :mod:`backend.common.ml.datasets_v2` for full documentation.
"""

from .datasets_v2 import (
    DatasetConfig,
    DatasetRegistry,
    DatasetRegistryError,
    DatasetSnapshot,
    FilterParseError,
    FilterSpec,
    FingerprintFn,
    PropertyRow,
    RowID,
    SplitKind,
    SplitSpec,
    apply_filter,
    compile_filter,
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
