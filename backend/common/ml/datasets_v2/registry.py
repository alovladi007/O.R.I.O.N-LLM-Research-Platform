"""Dataset registry: create, look up, re-materialize snapshots.

Storage for Session 6.2 is in-memory. Session 6.2b / 6.3 will add a
DB-backed variant (a new ``datasets`` table plus the alembic
migration), but the engine itself is storage-agnostic — it consumes
an iterable of :class:`PropertyRow`s and produces a
:class:`DatasetSnapshot`. That lets the unit tests pass without
Postgres and keeps the DB schema work scoped to a follow-up.

API
---

.. code:: python

    reg = DatasetRegistry()
    snap = reg.create(
        rows=my_property_rows,
        config=DatasetConfig(
            name="oxides_gap_v1",
            filter=FilterSpec(expression="property=bandgap_ev"),
            split=SplitSpec(kind="random", seed=42),
        ),
    )
    # Later:
    same = reg.create(...same args...)
    assert same.content_hash == snap.content_hash
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .filter import apply_filter
from .schema import (
    DatasetConfig,
    DatasetSnapshot,
    PropertyRow,
    SplitSpec,
)
from .splits import FingerprintFn, split_rows

logger = logging.getLogger(__name__)


class DatasetRegistryError(ValueError):
    """Raised on duplicate names, empty corpora, fraction errors, etc."""


@dataclass
class DatasetRegistry:
    """In-memory, name-keyed snapshot store.

    ``create(rows, config)`` filters rows, splits them, freezes the
    result into a :class:`DatasetSnapshot`, and stores it by
    ``(name, version)``. Re-creating with the same args returns a
    cache hit (and asserts the content hash matches; a mismatch is a
    bug report).
    """

    _snapshots: Dict[tuple, DatasetSnapshot] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Create / lookup
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        rows: Iterable[PropertyRow],
        config: DatasetConfig,
        fingerprint_fn: Optional[FingerprintFn] = None,
    ) -> DatasetSnapshot:
        """Materialize a :class:`DatasetSnapshot` from ``rows`` + ``config``.

        If a snapshot already exists for ``(name, version)``, verify
        that the new hash matches and return the cached one. That
        lets callers use ``create`` as get-or-create without worrying
        about duplicate side effects.
        """
        rows_list: List[PropertyRow] = list(rows)
        if not rows_list:
            raise DatasetRegistryError(
                f"corpus is empty for dataset {config.name!r}:{config.version}"
            )

        # 1. Filter.
        filtered = apply_filter(rows_list, config.filter.expression)
        if not filtered:
            raise DatasetRegistryError(
                f"filter expression {config.filter.expression!r} matched "
                f"0 of {len(rows_list)} rows for dataset {config.name!r}"
            )

        # 2. Split.
        config.split.validate_fractions()
        train_ids, val_ids, test_ids = split_rows(
            filtered, config.split, fingerprint_fn=fingerprint_fn,
        )

        # 3. Freeze → hash.
        snap = DatasetSnapshot(
            config=config,
            train_row_ids=train_ids,
            val_row_ids=val_ids,
            test_row_ids=test_ids,
        ).freeze()

        key = (config.name, config.version)
        existing = self._snapshots.get(key)
        if existing is not None and existing.content_hash != snap.content_hash:
            raise DatasetRegistryError(
                f"dataset {config.name!r}:{config.version} already exists "
                f"with a different content hash "
                f"(existing={existing.short_hash}, new={snap.short_hash}). "
                "Bump the version to create a fresh snapshot."
            )
        self._snapshots[key] = snap
        logger.info(
            "dataset %s:%s materialized — n_train=%d n_val=%d n_test=%d hash=%s",
            config.name, config.version,
            snap.n_train, snap.n_val, snap.n_test, snap.short_hash,
        )
        return snap

    def get(self, name: str, version: str = "v1") -> Optional[DatasetSnapshot]:
        return self._snapshots.get((name, version))

    def list(self) -> List[DatasetSnapshot]:
        return list(self._snapshots.values())

    def clear(self) -> None:
        self._snapshots.clear()

    # ------------------------------------------------------------------
    # Row-lookup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def rows_for_subset(
        rows: Iterable[PropertyRow],
        snapshot: DatasetSnapshot,
        subset: str,
    ) -> List[PropertyRow]:
        """Return the rows belonging to the named subset.

        ``subset`` ∈ ``{"train", "val", "test"}``. Accepts the same
        corpus the snapshot was created from (or a superset) and
        filters to the snapshot's row ids — caller doesn't need to
        keep the original filtered list around.
        """
        if subset == "train":
            wanted = set(snapshot.train_row_ids)
        elif subset == "val":
            wanted = set(snapshot.val_row_ids)
        elif subset == "test":
            wanted = set(snapshot.test_row_ids)
        else:
            raise ValueError(
                f"unknown subset {subset!r}; use train / val / test"
            )
        return [r for r in rows if r.row_id in wanted]
