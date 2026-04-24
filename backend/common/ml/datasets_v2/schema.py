"""Dataset-registry schemas for Phase 6 / Session 6.2.

A **Dataset** is a versioned, deterministic selection of
``(structure, property, method, conditions)`` tuples with a reproducible
train/val/test split.

Shape decisions
---------------

- :class:`PropertyRow` is the input unit. Not tied to the Postgres
  ``predicted_properties`` schema — the registry consumes any iterable
  of rows, so callers can build rows from a DB query, a pandas
  DataFrame, a JSONL dump of the Materials Project bulk-import path,
  etc. That keeps Session 6.2 operable before a `observed_properties`
  table exists.
- :class:`FilterSpec` is a tiny predicate language (``property=X AND
  method.functional='PBE'``) parsed by :mod:`.filter`. We stayed away
  from raw SQL so the same filter works against in-memory rows in tests
  AND against a future DB query builder.
- :class:`SplitSpec` names the strategy + seed. The split algorithms
  live in :mod:`.splits`; ``kind`` selects the function.
- :class:`DatasetSnapshot` freezes a materialized dataset — row IDs +
  per-subset membership + the content hash that the roadmap's
  "re-running the same dataset version returns identical splits"
  acceptance test keys on.

Content hash
------------

Computed over the dataset's *materialized* state (the sorted row-id
lists per subset + config + seed), not over the raw rows themselves.
That gives us two useful identities:

1. Same filter + same seed + same corpus → same hash.
2. Same filter + same seed + different corpus (row added/removed) →
   different hash.

The hash is SHA-256 hex, truncated to 16 chars in ``short_hash`` for
log output; full 64-char in ``content_hash`` for identity comparison.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Row identifier type — structure_id + property name uniquely names
# a datum in a Dataset. Using strings so UUIDs, MP-IDs, or hash keys
# all round-trip through JSON without custom encoders.
RowID = str


class PropertyRow(BaseModel):
    """One ``(structure, property, method, conditions)`` tuple.

    Fields
    ------
    row_id
        Deterministic unique identifier for this row. For DB-backed
        rows, the ``predicted_properties.id`` UUID as a string works.
        For in-memory test rows, any unique string.
    structure_id
        Foreign key to a ``Structure`` (or equivalent). Opaque string.
    structure_hash
        SHA-256 hash of the canonical structure representation (from
        :func:`backend.common.structures.hashing.structure_hash`).
        Used by the structure-cluster split to avoid re-featurizing.
    property_name
        ``"bandgap_ev"``, ``"formation_energy_ev_per_atom"``, etc.
    property_value
        The target value. Datasets are single-target at the moment —
        the roadmap's Session 6.3 baselines all regress a scalar —
        but callers who need multi-target can put a dict in
        ``extras`` and post-process.
    method
        ``{"functional": "PBE", "basis": "USPP", ...}``. Keys are
        free-form; the :class:`FilterSpec` syntax can index into this
        dict with ``method.functional='PBE'``.
    conditions
        Temperature / pressure / etc. that the property was measured
        at. Same free-form dict pattern.
    composition_formula
        Reduced-formula string ("Fe2O3"). Used by the stratified split.
    composition_prototype
        Optional coarser "prototype" label ("rocksalt", "perovskite",
        "wurtzite", ...). When absent, the stratified split falls back
        to the reduced formula.
    extras
        Free-form metadata that survives round-trip but isn't used
        by the registry itself.
    """

    model_config = ConfigDict(extra="forbid")

    row_id: RowID
    structure_id: str
    structure_hash: Optional[str] = None
    property_name: str
    property_value: float
    method: Dict[str, Any] = Field(default_factory=dict)
    conditions: Dict[str, Any] = Field(default_factory=dict)
    composition_formula: Optional[str] = None
    composition_prototype: Optional[str] = None
    extras: Dict[str, Any] = Field(default_factory=dict)


class FilterSpec(BaseModel):
    """Filter predicate over a sequence of :class:`PropertyRow`s.

    Uses a tiny expression language parsed in :mod:`.filter`. Syntax:

        <atom> := <field> <op> <literal>
                | <field>.<subkey> <op> <literal>
        <expr> := <atom> (AND <atom>)*

    Where:
    - ``<field>`` ∈ {property, property_name, method, conditions,
      composition_formula, composition_prototype, structure_id}.
    - ``<op>`` ∈ {``=``, ``!=``, ``>``, ``<``, ``>=``, ``<=``,
      ``in``} — the ``in`` form takes a comma-separated list.
    - String literals use single quotes; numeric literals use digits
      with optional decimal.

    Examples::

        property=bandgap
        property=bandgap AND method.functional='PBE'
        composition_prototype in 'rocksalt','wurtzite'
        property_value>=0 AND property_value<10

    OR / NOT are deliberately excluded in 6.2 — the corpora we expect
    are small enough that callers can post-filter. Pushed OR / NOT
    support to 6.2b.
    """

    model_config = ConfigDict(extra="forbid")

    expression: str = Field(
        default="",
        description="Filter expression string. Empty = keep everything.",
    )


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


SplitKind = Literal[
    "random",
    "stratified_by_prototype",
    "structure_cluster",
]


# Canonical-form → canonical-form identity map plus hyphen aliases.
# The roadmap's CLI example uses ``--split structure-cluster`` (hyphen),
# while the Python ``Literal`` type uses ``structure_cluster`` (underscore).
# We accept both on input and store the underscore form.
_SPLIT_KIND_ALIASES = {
    "random": "random",
    "stratified_by_prototype": "stratified_by_prototype",
    "stratified-by-prototype": "stratified_by_prototype",
    "structure_cluster": "structure_cluster",
    "structure-cluster": "structure_cluster",
}


class SplitSpec(BaseModel):
    """Split strategy + fractions + seed.

    Fractions must sum to exactly 1.0 (within 1e-6). We reject other
    totals rather than renormalizing because silent renormalization
    makes the "same seed → same split" acceptance test fragile.

    ``kind`` accepts either underscore (``structure_cluster``) or
    hyphen (``structure-cluster``) forms; the hyphen form matches the
    roadmap's CLI style. Both normalize to the canonical underscore
    value, so ``content_hash`` is stable regardless of input style.
    """

    model_config = ConfigDict(extra="forbid")

    kind: SplitKind = "random"
    train_fraction: float = Field(default=0.7, gt=0, lt=1)
    val_fraction: float = Field(default=0.15, ge=0, lt=1)
    test_fraction: float = Field(default=0.15, gt=0, lt=1)
    seed: int = Field(default=0, ge=0)

    # Kind-specific knobs. These are optional; the split function
    # reads only the ones it needs.

    # For structure_cluster: number of greedy k-center seeds to pull
    # into test. We default to the number of rows implied by
    # test_fraction; callers who want a specific cluster count can
    # override.
    cluster_n_centers: Optional[int] = Field(default=None, ge=1)

    @field_validator("kind", mode="before")
    @classmethod
    def _normalize_split_kind(cls, v):
        if isinstance(v, str) and v in _SPLIT_KIND_ALIASES:
            return _SPLIT_KIND_ALIASES[v]
        return v

    def validate_fractions(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"SplitSpec fractions must sum to 1.0 (got "
                f"{total:.6f}: train={self.train_fraction} "
                f"val={self.val_fraction} test={self.test_fraction})"
            )


# ---------------------------------------------------------------------------
# Dataset / Snapshot
# ---------------------------------------------------------------------------


class DatasetConfig(BaseModel):
    """Complete spec for creating a dataset.

    Two datasets created with the same config (same filter + split +
    seed) against the same corpus produce identical snapshots — that's
    the Session 6.2 acceptance contract.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    version: str = Field(default="v1", min_length=1, max_length=32)
    filter: FilterSpec = Field(default_factory=FilterSpec)
    split: SplitSpec = Field(default_factory=SplitSpec)
    # Optional human-readable description; ignored by the hash so
    # editing the description doesn't invalidate the snapshot.
    description: str = ""


class DatasetSnapshot(BaseModel):
    """Frozen, hashable dataset instance.

    The registry persists these (in-memory or on disk, per backend).
    ``content_hash`` is the identity: two snapshots with the same
    hash contain the same rows in the same splits, regardless of when
    they were materialized.
    """

    model_config = ConfigDict(extra="forbid")

    config: DatasetConfig
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Row IDs per subset, sorted. Sorting is the key to determinism:
    # ``sorted(list_of_ids)`` is canonical; the split functions
    # assign membership, and the snapshot re-sorts for hashing.
    train_row_ids: List[RowID] = Field(default_factory=list)
    val_row_ids: List[RowID] = Field(default_factory=list)
    test_row_ids: List[RowID] = Field(default_factory=list)

    # Content hash — 64-char SHA-256 hex. Populated by
    # :meth:`compute_hash` (called by the registry at create time).
    content_hash: str = ""

    @property
    def short_hash(self) -> str:
        """First 16 chars of the content hash — log-friendly."""
        return self.content_hash[:16] if self.content_hash else ""

    @property
    def n_train(self) -> int:
        return len(self.train_row_ids)

    @property
    def n_val(self) -> int:
        return len(self.val_row_ids)

    @property
    def n_test(self) -> int:
        return len(self.test_row_ids)

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_val + self.n_test

    def compute_hash(self) -> str:
        """Compute the content hash from the materialized state.

        Hash inputs (in order, separated by ``\\x00``):

        1. ``config.name``
        2. ``config.version``
        3. ``config.filter.expression`` (verbatim string)
        4. ``config.split.kind`` + fractions + seed +
           ``cluster_n_centers or ""``
        5. Sorted train row ids joined by ``\\x00``
        6. Sorted val row ids
        7. Sorted test row ids

        Not included: ``created_at`` (timestamps shouldn't
        invalidate identity), ``description`` (mutable metadata).
        """
        import hashlib

        parts: List[str] = [
            self.config.name,
            self.config.version,
            self.config.filter.expression,
            self.config.split.kind,
            f"{self.config.split.train_fraction:.10f}",
            f"{self.config.split.val_fraction:.10f}",
            f"{self.config.split.test_fraction:.10f}",
            str(self.config.split.seed),
            (
                str(self.config.split.cluster_n_centers)
                if self.config.split.cluster_n_centers is not None
                else ""
            ),
            "\x00".join(sorted(self.train_row_ids)),
            "\x00".join(sorted(self.val_row_ids)),
            "\x00".join(sorted(self.test_row_ids)),
        ]
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def freeze(self) -> "DatasetSnapshot":
        """Return a copy with sorted row id lists + populated hash."""
        sorted_snap = self.model_copy(
            update={
                "train_row_ids": sorted(self.train_row_ids),
                "val_row_ids": sorted(self.val_row_ids),
                "test_row_ids": sorted(self.test_row_ids),
            }
        )
        sorted_snap = sorted_snap.model_copy(
            update={"content_hash": sorted_snap.compute_hash()}
        )
        return sorted_snap
