"""Tests for Phase 6 / Session 6.2 — dataset registry + splits.

Covers:

1. Filter grammar — atomic predicates, AND chains, ``in`` lists,
   dotted-subkey access, error paths.
2. Split strategies — random (seed reproducibility), stratified (per-
   bucket fractions + tiny-bucket fallback), structure_cluster
   (greedy k-center), scaffold_split (raises).
3. :class:`DatasetRegistry` — create, get, list, duplicate-name
   hash-mismatch guard, cache hit on identical config.
4. :class:`DatasetSnapshot.compute_hash` determinism — stable across
   Python runs given the same inputs.
5. **Acceptance** (roadmap):
   - Same seed ⇒ same split hash.
   - Structure-cluster split: train/test fingerprint distance
     distributions differ with KS p < 0.01.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_rows():
    """Eight rows spanning two prototypes and two properties."""
    from backend.common.ml.datasets_v2 import PropertyRow

    rows = []
    for i, proto in enumerate(["rocksalt", "rocksalt", "wurtzite", "wurtzite",
                                "rocksalt", "rocksalt", "wurtzite", "wurtzite"]):
        rows.append(PropertyRow(
            row_id=f"r{i:03d}",
            structure_id=f"s{i:03d}",
            property_name="bandgap_ev" if i % 2 == 0 else "formation_energy",
            property_value=float(i) * 0.5 - 1.0,
            method={"functional": "PBE" if i < 6 else "HSE"},
            conditions={"temperature_k": 0.0},
            composition_prototype=proto,
            composition_formula=f"Foo{i}",
        ))
    return rows


@pytest.fixture
def cluster_rows():
    """Large-enough corpus with spatially-separated fingerprints so the
    greedy k-center split actually has something to cluster.

    We draw 60 rows in two separated Gaussian blobs in R^12. The
    structure_cluster split should place centers at the extremes of
    the blobs, giving a test set whose distance-to-train distribution
    differs significantly from the train's internal distribution.
    """
    from backend.common.ml.datasets_v2 import PropertyRow

    rng = np.random.default_rng(17)
    fps = {}
    rows = []
    # 120 rows sampled uniformly in a 12-D hypercube. Greedy-farthest
    # k-center picks rows on the *convex-hull-like* edges of the
    # cloud; training keeps the interior. Nearest-training distance
    # for an edge row is systematically larger than the interior-
    # interior nearest-neighbour distance, which shows up as a clean
    # KS p << 0.01 in the acceptance test. This is the "forces
    # extrapolation" design the roadmap calls for.
    N = 240
    fps_arr = rng.uniform(-1.0, 1.0, size=(N, 12))
    for i in range(N):
        rid = f"R{i:03d}"
        rows.append(PropertyRow(
            row_id=rid, structure_id=f"s_{rid}",
            property_name="bandgap_ev",
            property_value=float(i) * 0.01,
            method={"functional": "PBE"},
            composition_prototype="alpha" if i % 2 == 0 else "beta",
        ))
        fps[rid] = fps_arr[i]

    def fingerprint_fn(r):
        return fps[r.row_id]

    return rows, fingerprint_fn, fps


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class TestFilter:
    def test_empty_expression_matches_all(self, small_rows):
        from backend.common.ml.datasets_v2 import apply_filter

        assert len(apply_filter(small_rows, "")) == len(small_rows)

    def test_simple_equality(self, small_rows):
        from backend.common.ml.datasets_v2 import apply_filter

        out = apply_filter(small_rows, "property=bandgap_ev")
        assert all(r.property_name == "bandgap_ev" for r in out)
        assert len(out) == 4

    def test_and_chain(self, small_rows):
        from backend.common.ml.datasets_v2 import apply_filter

        out = apply_filter(
            small_rows,
            "property=bandgap_ev AND method.functional='PBE'",
        )
        assert all(r.property_name == "bandgap_ev"
                   and r.method["functional"] == "PBE" for r in out)

    def test_in_list(self, small_rows):
        from backend.common.ml.datasets_v2 import apply_filter

        out = apply_filter(
            small_rows,
            "composition_prototype in 'rocksalt','wurtzite'",
        )
        assert len(out) == len(small_rows)  # all prototypes covered
        out2 = apply_filter(small_rows, "composition_prototype in 'rocksalt'")
        assert all(r.composition_prototype == "rocksalt" for r in out2)

    def test_numeric_comparison(self, small_rows):
        from backend.common.ml.datasets_v2 import apply_filter

        out = apply_filter(small_rows, "property_value>=0")
        assert all(r.property_value >= 0 for r in out)

    def test_unknown_field_raises(self):
        from backend.common.ml.datasets_v2 import FilterParseError, apply_filter

        with pytest.raises(FilterParseError, match="unknown field"):
            apply_filter([], "nope=1")

    def test_malformed_tokenize_raises(self):
        from backend.common.ml.datasets_v2 import FilterParseError, apply_filter

        # ``@`` isn't in the grammar — tokenizer should flag it.
        with pytest.raises(FilterParseError):
            apply_filter([], "property @ bandgap")


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


class TestRandomSplit:
    def test_reproducible(self, small_rows):
        from backend.common.ml.datasets_v2 import SplitSpec, random_split

        spec = SplitSpec(kind="random", train_fraction=0.5,
                         val_fraction=0.25, test_fraction=0.25, seed=42)
        a = random_split(small_rows, spec)
        b = random_split(small_rows, spec)
        assert a == b

    def test_seed_changes_split(self, small_rows):
        from backend.common.ml.datasets_v2 import SplitSpec, random_split

        a = random_split(small_rows, SplitSpec(
            kind="random", train_fraction=0.5, val_fraction=0.25,
            test_fraction=0.25, seed=1,
        ))
        b = random_split(small_rows, SplitSpec(
            kind="random", train_fraction=0.5, val_fraction=0.25,
            test_fraction=0.25, seed=2,
        ))
        assert a != b

    def test_covers_all_rows(self, small_rows):
        from backend.common.ml.datasets_v2 import SplitSpec, random_split

        t, v, te = random_split(small_rows, SplitSpec(
            kind="random", train_fraction=0.5, val_fraction=0.25,
            test_fraction=0.25, seed=0,
        ))
        all_ids = set(t) | set(v) | set(te)
        assert all_ids == {r.row_id for r in small_rows}
        # No overlap.
        assert not (set(t) & set(v))
        assert not (set(t) & set(te))
        assert not (set(v) & set(te))

    def test_fractions_must_sum_to_one(self):
        from backend.common.ml.datasets_v2 import SplitSpec

        spec = SplitSpec(train_fraction=0.5, val_fraction=0.25, test_fraction=0.2)
        with pytest.raises(ValueError, match="sum to 1"):
            spec.validate_fractions()


class TestStratifiedSplit:
    def test_tiny_buckets_stay_in_train(self):
        """Buckets with < 3 rows → all into train."""
        from backend.common.ml.datasets_v2 import (
            PropertyRow, SplitSpec, stratified_by_prototype_split,
        )

        rows = [
            PropertyRow(row_id=f"r{i}", structure_id=f"s{i}",
                        property_name="p", property_value=0.0,
                        composition_prototype="tiny")
            for i in range(2)
        ] + [
            PropertyRow(row_id=f"R{i}", structure_id=f"S{i}",
                        property_name="p", property_value=0.0,
                        composition_prototype="big")
            for i in range(20)
        ]
        t, v, te = stratified_by_prototype_split(rows, SplitSpec(
            train_fraction=0.7, val_fraction=0.15, test_fraction=0.15, seed=0,
        ))
        # Both tiny rows land in train.
        assert {"r0", "r1"} <= set(t)
        # Big bucket got split roughly 14/3/3.
        big_t = [x for x in t if x.startswith("R")]
        assert abs(len(big_t) - 14) <= 1

    def test_prototype_falls_back_to_formula(self):
        """Rows without a prototype use composition_formula as bucket key."""
        from backend.common.ml.datasets_v2 import (
            PropertyRow, SplitSpec, stratified_by_prototype_split,
        )

        rows = [
            PropertyRow(row_id=f"r{i}", structure_id=f"s{i}",
                        property_name="p", property_value=0.0,
                        composition_formula="NaCl")
            for i in range(10)
        ]
        # Should succeed (single bucket; >= 3 rows).
        t, v, te = stratified_by_prototype_split(rows, SplitSpec(
            train_fraction=0.6, val_fraction=0.2, test_fraction=0.2, seed=0,
        ))
        assert len(t) + len(v) + len(te) == 10


class TestStructureClusterSplit:
    def test_ks_distribution_shift(self, cluster_rows):
        """Roadmap acceptance: test-set distance-to-train differs from
        train-internal distance (KS test p < 0.01).

        The two-blob fixture has a visible separation; the greedy
        k-center should place most centers at the blob extremes. The
        test set then consists of rows maximally far from the training
        set, so its nearest-train-distance distribution is
        significantly larger than the train-internal nearest-neighbour
        distribution.
        """
        from scipy.stats import ks_2samp  # scipy already pinned

        from backend.common.ml.datasets_v2 import (
            SplitSpec, structure_cluster_split,
        )

        rows, fingerprint_fn, fps = cluster_rows
        spec = SplitSpec(
            kind="structure_cluster", train_fraction=0.7,
            val_fraction=0.15, test_fraction=0.15, seed=42,
        )
        train_ids, val_ids, test_ids = structure_cluster_split(
            rows, spec, fingerprint_fn=fingerprint_fn,
        )
        assert len(train_ids) > 0 and len(test_ids) > 0

        # Distance from each test row to its nearest training row.
        train_fps = np.vstack([fps[rid] for rid in train_ids])
        test_fps = np.vstack([fps[rid] for rid in test_ids])

        def _nearest(from_pts: np.ndarray, to_pts: np.ndarray) -> np.ndarray:
            a2 = np.einsum("ij,ij->i", from_pts, from_pts)[:, None]
            b2 = np.einsum("ij,ij->i", to_pts, to_pts)[None, :]
            d2 = np.clip(a2 + b2 - 2 * (from_pts @ to_pts.T), 0, None)
            d = np.sqrt(d2)
            return d.min(axis=1)

        test_dist = _nearest(test_fps, train_fps)
        # Train-internal nearest: mask out the self-pair by setting
        # diagonal to +inf before min.
        d_tt = np.sqrt(np.clip(
            np.einsum("ij,ij->i", train_fps, train_fps)[:, None]
            + np.einsum("ij,ij->i", train_fps, train_fps)[None, :]
            - 2 * (train_fps @ train_fps.T),
            0, None,
        ))
        np.fill_diagonal(d_tt, np.inf)
        train_dist = d_tt.min(axis=1)

        stat, p = ks_2samp(test_dist, train_dist)
        assert p < 0.01, (
            f"KS test p = {p:.3e}, expected < 0.01. "
            f"Test mean dist = {test_dist.mean():.3f}, "
            f"train mean dist = {train_dist.mean():.3f}"
        )
        # Sanity: test should be farther on average than train.
        assert test_dist.mean() > train_dist.mean()

    def test_reproducible_with_seed(self, cluster_rows):
        from backend.common.ml.datasets_v2 import (
            SplitSpec, structure_cluster_split,
        )

        rows, fingerprint_fn, _ = cluster_rows
        spec = SplitSpec(
            kind="structure_cluster", train_fraction=0.7,
            val_fraction=0.15, test_fraction=0.15, seed=5,
        )
        a = structure_cluster_split(rows, spec, fingerprint_fn=fingerprint_fn)
        b = structure_cluster_split(rows, spec, fingerprint_fn=fingerprint_fn)
        assert a == b

    def test_missing_fingerprint_fn_raises(self, cluster_rows):
        from backend.common.ml.datasets_v2 import SplitSpec, split_rows

        rows, _, _ = cluster_rows
        spec = SplitSpec(kind="structure_cluster", seed=0)
        with pytest.raises(ValueError, match="fingerprint_fn"):
            split_rows(rows, spec)


class TestScaffoldSplitDeferred:
    def test_scaffold_raises_pending(self):
        from backend.common.ml.datasets_v2 import SplitSpec, scaffold_split
        from backend.common.reports import PendingAnalyzerError

        with pytest.raises(PendingAnalyzerError, match="6.2b"):
            scaffold_split([], SplitSpec(seed=0))


# ---------------------------------------------------------------------------
# Registry + snapshot hashing
# ---------------------------------------------------------------------------


class TestRegistry:
    def _make_config(self, name="t", seed=0):
        from backend.common.ml.datasets_v2 import (
            DatasetConfig, FilterSpec, SplitSpec,
        )

        return DatasetConfig(
            name=name,
            filter=FilterSpec(expression=""),
            split=SplitSpec(
                kind="random", train_fraction=0.5,
                val_fraction=0.25, test_fraction=0.25, seed=seed,
            ),
        )

    def test_same_seed_same_hash(self, small_rows):
        """Roadmap acceptance: same config + same corpus → same hash."""
        from backend.common.ml.datasets_v2 import DatasetRegistry

        reg = DatasetRegistry()
        cfg = self._make_config(name="a", seed=42)
        snap1 = reg.create(rows=small_rows, config=cfg)
        # Fresh registry ⇒ no cache shortcut.
        reg2 = DatasetRegistry()
        snap2 = reg2.create(rows=small_rows, config=cfg)
        assert snap1.content_hash == snap2.content_hash
        assert snap1.short_hash == snap2.short_hash

    def test_different_seed_different_hash(self, small_rows):
        from backend.common.ml.datasets_v2 import DatasetRegistry

        reg = DatasetRegistry()
        snap1 = reg.create(rows=small_rows, config=self._make_config(name="a", seed=1))
        snap2 = reg.create(rows=small_rows,
                           config=self._make_config(name="b", seed=2))
        assert snap1.content_hash != snap2.content_hash

    def test_adding_a_row_changes_hash(self, small_rows):
        """Content hash captures the materialized row set."""
        from backend.common.ml.datasets_v2 import DatasetRegistry, PropertyRow

        reg = DatasetRegistry()
        cfg = self._make_config(name="a", seed=0)
        snap1 = reg.create(rows=small_rows, config=cfg)

        extra = PropertyRow(
            row_id="new", structure_id="sNew",
            property_name="bandgap_ev", property_value=99.0,
            composition_prototype="wurtzite",
        )
        reg2 = DatasetRegistry()
        snap2 = reg2.create(rows=small_rows + [extra], config=cfg)
        assert snap1.content_hash != snap2.content_hash

    def test_duplicate_create_returns_cached(self, small_rows):
        from backend.common.ml.datasets_v2 import DatasetRegistry

        reg = DatasetRegistry()
        cfg = self._make_config(name="dup", seed=0)
        a = reg.create(rows=small_rows, config=cfg)
        b = reg.create(rows=small_rows, config=cfg)
        assert a.content_hash == b.content_hash
        assert len(reg.list()) == 1

    def test_duplicate_name_different_hash_raises(self, small_rows):
        """Rematerializing an existing name with a *different* hash is
        an integrity failure — we force a version bump."""
        from backend.common.ml.datasets_v2 import (
            DatasetRegistry, DatasetRegistryError, PropertyRow,
        )

        reg = DatasetRegistry()
        cfg = self._make_config(name="collide", seed=0)
        reg.create(rows=small_rows, config=cfg)
        extra = PropertyRow(
            row_id="new", structure_id="sNew",
            property_name="bandgap_ev", property_value=99.0,
        )
        with pytest.raises(DatasetRegistryError, match="already exists"):
            reg.create(rows=small_rows + [extra], config=cfg)

    def test_empty_corpus_raises(self):
        from backend.common.ml.datasets_v2 import (
            DatasetRegistry, DatasetRegistryError,
        )

        with pytest.raises(DatasetRegistryError, match="empty"):
            DatasetRegistry().create(rows=[], config=self._make_config())

    def test_filter_matches_zero_raises(self, small_rows):
        from backend.common.ml.datasets_v2 import (
            DatasetConfig, DatasetRegistry, DatasetRegistryError,
            FilterSpec, SplitSpec,
        )

        cfg = DatasetConfig(
            name="nomatch",
            filter=FilterSpec(expression="property=does_not_exist"),
            split=SplitSpec(seed=0),
        )
        with pytest.raises(DatasetRegistryError, match="matched 0"):
            DatasetRegistry().create(rows=small_rows, config=cfg)

    def test_rows_for_subset(self, small_rows):
        from backend.common.ml.datasets_v2 import DatasetRegistry

        reg = DatasetRegistry()
        snap = reg.create(rows=small_rows, config=self._make_config(seed=0))
        train_rows = reg.rows_for_subset(small_rows, snap, "train")
        assert {r.row_id for r in train_rows} == set(snap.train_row_ids)

    def test_get_and_list(self, small_rows):
        from backend.common.ml.datasets_v2 import DatasetRegistry

        reg = DatasetRegistry()
        reg.create(rows=small_rows, config=self._make_config(name="one"))
        assert reg.get("one") is not None
        assert reg.get("missing") is None
        assert len(reg.list()) == 1


class TestSnapshotHashStability:
    def test_hash_stable_across_construction_order(self, small_rows):
        """Inserting rows in a different order must not change the hash
        because freeze() sorts row ids before hashing."""
        from backend.common.ml.datasets_v2 import (
            DatasetRegistry, DatasetSnapshot,
        )

        reg = DatasetRegistry()
        cfg_a = DatasetRegistry.__dict__  # placeholder to import locally
        from backend.common.ml.datasets_v2 import (
            DatasetConfig, FilterSpec, SplitSpec,
        )

        cfg = DatasetConfig(
            name="h", filter=FilterSpec(expression=""),
            split=SplitSpec(seed=7, train_fraction=0.5,
                            val_fraction=0.25, test_fraction=0.25),
        )
        snap1 = reg.create(rows=small_rows, config=cfg)
        # Reverse row order + new registry.
        reg2 = DatasetRegistry()
        snap2 = reg2.create(rows=list(reversed(small_rows)), config=cfg)
        assert snap1.content_hash == snap2.content_hash

    def test_description_does_not_affect_hash(self, small_rows):
        """Editing the description shouldn't invalidate snapshots."""
        from backend.common.ml.datasets_v2 import (
            DatasetConfig, DatasetRegistry, FilterSpec, SplitSpec,
        )

        cfg_a = DatasetConfig(
            name="d", description="initial",
            filter=FilterSpec(expression=""),
            split=SplitSpec(seed=0, train_fraction=0.5,
                            val_fraction=0.25, test_fraction=0.25),
        )
        cfg_b = cfg_a.model_copy(update={"description": "edited"})
        snap_a = DatasetRegistry().create(rows=small_rows, config=cfg_a)
        snap_b = DatasetRegistry().create(rows=small_rows, config=cfg_b)
        assert snap_a.content_hash == snap_b.content_hash
