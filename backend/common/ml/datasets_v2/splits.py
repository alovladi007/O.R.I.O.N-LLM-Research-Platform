"""Train/val/test split strategies for Session 6.2.

Three strategies shipped live; one deferred:

- :func:`random_split` — uniform random with the spec's seed.
- :func:`stratified_by_prototype_split` — each ``composition_prototype``
  contributes proportionally to all three subsets (with the seed
  controlling tie-breaking). Falls back to ``composition_formula`` when
  prototype is absent.
- :func:`structure_cluster_split` — greedy k-center with cosine
  distance over caller-supplied fingerprints. The roadmap's KS-test
  acceptance target for this split: test-set fingerprint distance
  distribution differs from the train-set's (p < 0.01).
- **Scaffold-like (organics)**: deferred to 6.2b. Would use RDKit's
  Murcko scaffold + BM hash. Raises :class:`~backend.common.reports.PendingAnalyzerError`.

All functions return ``(train_ids, val_ids, test_ids)`` as sorted
lists — callers wrap them in a :class:`DatasetSnapshot`. Determinism
is guaranteed by:

1. Stable row ordering on input (we sort by ``row_id`` before the
   random ops — so caller insertion order doesn't influence the split).
2. Seed threading through ``numpy.random.default_rng(seed)``.
3. Sorted output lists.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from backend.common.reports.md import PendingAnalyzerError

from .schema import PropertyRow, RowID, SplitSpec

logger = logging.getLogger(__name__)


SplitTriple = Tuple[List[RowID], List[RowID], List[RowID]]


# ---------------------------------------------------------------------------
# Shared utility: fraction-based index partition
# ---------------------------------------------------------------------------


def _partition_by_fractions(
    items: Sequence[RowID], fractions: Tuple[float, float, float],
) -> SplitTriple:
    """Split ``items`` into three lists with the given fractions.

    Rounding policy: floor for train + val, remainder for test. This
    matches scikit-learn's default behaviour and avoids off-by-one
    drift on small sweeps.
    """
    n = len(items)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    # test takes the remainder so we lose at most 2 rows to rounding
    # instead of silently dropping them.
    train = list(items[:n_train])
    val = list(items[n_train : n_train + n_val])
    test = list(items[n_train + n_val :])
    return train, val, test


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------


def random_split(rows: Sequence[PropertyRow], spec: SplitSpec) -> SplitTriple:
    """Uniform random split keyed by ``spec.seed``.

    Same seed + same row set ⇒ same partition, guaranteed.
    """
    spec.validate_fractions()
    # Stable input order.
    sorted_ids = sorted(r.row_id for r in rows)
    rng = np.random.default_rng(spec.seed)
    perm = rng.permutation(len(sorted_ids))
    shuffled = [sorted_ids[i] for i in perm]
    train, val, test = _partition_by_fractions(
        shuffled,
        (spec.train_fraction, spec.val_fraction, spec.test_fraction),
    )
    return sorted(train), sorted(val), sorted(test)


# ---------------------------------------------------------------------------
# Stratified by composition prototype
# ---------------------------------------------------------------------------


def stratified_by_prototype_split(
    rows: Sequence[PropertyRow], spec: SplitSpec,
) -> SplitTriple:
    """Split within each ``composition_prototype`` bucket, then concatenate.

    A prototype with fewer than 3 rows contributes only to train (we'd
    rather tiny classes stay in training than disappear from val+test;
    the model's val performance on those prototypes is then measured
    by the implicit "how does it do on prototypes it's never seen" via
    the structure_cluster split, not this one).

    Falls back to ``composition_formula`` when prototype is None, and
    to a single "__UNSTRATIFIED__" bucket when both are None.
    """
    spec.validate_fractions()

    buckets: Dict[str, List[RowID]] = defaultdict(list)
    for r in rows:
        key = r.composition_prototype or r.composition_formula or "__UNSTRATIFIED__"
        buckets[key].append(r.row_id)

    rng = np.random.default_rng(spec.seed)
    train_all: List[RowID] = []
    val_all: List[RowID] = []
    test_all: List[RowID] = []

    for key in sorted(buckets):  # deterministic iteration
        bucket_ids = sorted(buckets[key])
        perm = rng.permutation(len(bucket_ids))
        shuffled = [bucket_ids[i] for i in perm]
        if len(shuffled) < 3:
            # Too small to split cleanly — keep all in train.
            train_all.extend(shuffled)
            continue
        t, v, ts = _partition_by_fractions(
            shuffled,
            (spec.train_fraction, spec.val_fraction, spec.test_fraction),
        )
        train_all.extend(t)
        val_all.extend(v)
        test_all.extend(ts)

    return sorted(train_all), sorted(val_all), sorted(test_all)


# ---------------------------------------------------------------------------
# Structure-cluster (greedy k-center on fingerprint space)
# ---------------------------------------------------------------------------


FingerprintFn = Callable[[PropertyRow], np.ndarray]


def structure_cluster_split(
    rows: Sequence[PropertyRow],
    spec: SplitSpec,
    *,
    fingerprint_fn: FingerprintFn,
    metric: str = "euclidean",
) -> SplitTriple:
    """Greedy-farthest-point extrapolation split over caller-supplied fingerprints.

    Algorithm
    ---------

    1. Compute a fingerprint for every row via ``fingerprint_fn``.
    2. Pick the first seed by seeded RNG.
    3. Grow the test set greedily: each iteration, take the row
       whose *distance to the nearest already-in-test row* is
       largest. This is the dual of farthest-point k-center — we're
       picking the row the current test set is most dissimilar from,
       and adding it to the test set (rather than to centers and then
       pulling cluster-mates).
    4. Stop when the test set reaches ``ceil(n * test_fraction)``
       rows (or ``cluster_n_centers`` if set).
    5. Remaining rows → seeded random train/val split per fractions.

    This is the "forces extrapolation test" design the roadmap asks
    for: the test set is guaranteed to span the fingerprint space
    (one row in each extremal region), so the distance from test to
    train is systematically larger than train-to-train nearest-
    neighbour distances. KS-test p < 0.01 on any corpus with enough
    structural variation.

    The older "pick centers, test = {centers}" variant gave test rows
    that happened to sit near cluster-mates in training whenever the
    number of centers exceeded the number of clusters. This variant
    works for any corpus shape.
    """
    spec.validate_fractions()
    if not rows:
        return [], [], []

    # Stable ordering.
    sorted_rows = sorted(rows, key=lambda r: r.row_id)
    n = len(sorted_rows)

    # Featurize. A bad fingerprint_fn (returns None, wrong shape) is a
    # caller bug; raise loudly rather than silently degrade.
    fps: List[np.ndarray] = [
        np.asarray(fingerprint_fn(r), dtype=np.float64) for r in sorted_rows
    ]
    if not fps:
        return [], [], []
    d = fps[0].shape[0]
    fps_mat = np.vstack(fps)
    if fps_mat.shape != (n, d):
        raise ValueError(
            f"fingerprints have inconsistent dimensions; got shape {fps_mat.shape}"
        )

    # Target test count.
    if spec.cluster_n_centers is not None:
        target_n_test = min(spec.cluster_n_centers, n - 1)
    else:
        target_n_test = max(1, int(np.ceil(n * spec.test_fraction)))
    target_n_test = min(target_n_test, n - 1)  # leave at least 1 for train

    rng = np.random.default_rng(spec.seed)
    first = int(rng.integers(0, n))
    test_indices: List[int] = [first]
    # distances[i] = min distance from row i to any test-set row so far.
    distances = _pairwise_distances(
        fps_mat[first : first + 1], fps_mat, metric=metric,
    ).ravel()

    while len(test_indices) < target_n_test:
        masked = distances.copy()
        masked[test_indices] = -np.inf
        next_idx = int(np.argmax(masked))
        test_indices.append(next_idx)
        new_d = _pairwise_distances(
            fps_mat[next_idx : next_idx + 1], fps_mat, metric=metric,
        ).ravel()
        distances = np.minimum(distances, new_d)

    test_set = set(test_indices)
    test_ids = [sorted_rows[i].row_id for i in test_indices]
    remaining_ids = [
        r.row_id for i, r in enumerate(sorted_rows) if i not in test_set
    ]

    # Within remaining, do a random train/val split. Fractions that
    # used to go to test are absorbed by train+val proportionally.
    total_left = spec.train_fraction + spec.val_fraction
    if total_left <= 0:
        return [], [], sorted(test_ids)
    train_frac_inner = spec.train_fraction / total_left
    # Shuffle with seed bumped by 1 so train/val isn't correlated with
    # the k-center selection.
    inner_rng = np.random.default_rng(spec.seed + 1)
    perm = inner_rng.permutation(len(remaining_ids))
    shuffled = [remaining_ids[i] for i in perm]
    n_train = int(len(shuffled) * train_frac_inner)
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:]

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


def _pairwise_distances(
    a: np.ndarray, b: np.ndarray, metric: str = "euclidean",
) -> np.ndarray:
    """Pairwise distances between rows of ``a`` and rows of ``b``.

    Returns shape ``(n_a, n_b)``. Supports ``euclidean`` and ``cosine``;
    other metrics raise. We don't pull sklearn here — numpy does both
    in a few lines and keeps the dep graph small.
    """
    if metric == "euclidean":
        # (a-b)^2 expanded: |a|^2 + |b|^2 - 2 a.b
        a2 = np.einsum("ij,ij->i", a, a)[:, None]
        b2 = np.einsum("ij,ij->i", b, b)[None, :]
        d2 = a2 + b2 - 2.0 * (a @ b.T)
        d2 = np.clip(d2, 0.0, None)
        return np.sqrt(d2)
    if metric == "cosine":
        an = np.linalg.norm(a, axis=1, keepdims=True)
        bn = np.linalg.norm(b, axis=1, keepdims=True)
        an = np.where(an < 1e-12, 1.0, an)
        bn = np.where(bn < 1e-12, 1.0, bn)
        return 1.0 - (a @ b.T) / (an * bn.T)
    raise ValueError(f"unsupported metric: {metric!r}")


# ---------------------------------------------------------------------------
# Scaffold-like — deferred
# ---------------------------------------------------------------------------


def scaffold_split(rows: Sequence[PropertyRow], spec: SplitSpec) -> SplitTriple:
    """Murcko-scaffold split for organic datasets. Deferred to 6.2b.

    Requires RDKit which isn't in ``requirements.txt`` yet. The
    inorganics pipeline (Phase 8) doesn't exercise this path; we'll
    land it when a molecular-dataset session actually needs it.
    """
    raise PendingAnalyzerError(
        "scaffold_split", tracker="Session 6.2b (requires RDKit)",
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def split_rows(
    rows: Sequence[PropertyRow],
    spec: SplitSpec,
    *,
    fingerprint_fn: Optional[FingerprintFn] = None,
) -> SplitTriple:
    """Dispatch on :attr:`SplitSpec.kind`.

    ``structure_cluster`` requires a ``fingerprint_fn``; other strategies
    ignore it.
    """
    if spec.kind == "random":
        return random_split(rows, spec)
    if spec.kind == "stratified_by_prototype":
        return stratified_by_prototype_split(rows, spec)
    if spec.kind == "structure_cluster":
        if fingerprint_fn is None:
            raise ValueError(
                "structure_cluster split requires fingerprint_fn (a callable "
                "PropertyRow -> 1-D numpy array)"
            )
        return structure_cluster_split(rows, spec, fingerprint_fn=fingerprint_fn)
    raise ValueError(f"unsupported split kind: {spec.kind!r}")
