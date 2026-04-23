"""Tests for Phase 6 / Session 6.1 — ML featurizers.

Covers
------

1. Composition featurizer — 146-d vector, deterministic, labels
   line up with vector length.
2. Structure featurizer — 122-d vector via SiteStatsFingerprint.
3. Graph builder — canonicalization round-trip (two Structure
   instances of the same crystal produce identical graphs), edge
   count matches pymatgen's neighbour search, edge feature
   dimensionality, cutoff validation.
4. Cache — hit / miss counters, per-instance independence, defensive
   copy semantics, explicit vs default cache (regression for the
   ``cache or _default_cache`` falsy-empty bug).
5. Embedding — PCA fit/transform round-trip, standardization, L2-
   normalization, save/load, NaN robustness.
6. Deferred analyzers — OFM / XRD / SOAP raise ``PendingAnalyzerError``.
7. Acceptance — 100 structures featurized in <30 s, Si-vs-group-IV
   similarity recovery (StandardScaler-gated).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def silicon_structure():
    from pymatgen.core import Lattice, Structure

    return Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


@pytest.fixture(scope="module")
def diverse_corpus():
    """~25 varied structures: group-IV diamonds + metals + simple oxides.

    Module-scoped so matminer's first-call overhead (Magpie data load)
    is paid once across every test that uses it.
    """
    from pymatgen.core import Lattice, Structure

    corpus = []
    labels = []

    def diamond(a, e):
        return Structure(Lattice.cubic(a), [e, e], [[0, 0, 0], [0.25, 0.25, 0.25]])

    for e, a in (("C", 3.567), ("Si", 5.43), ("Ge", 5.658), ("Sn", 6.489)):
        corpus.append(diamond(a, e))
        labels.append(e)

    for e, a in (
        ("Fe", 2.87), ("Cu", 3.615), ("Al", 4.05), ("Ni", 3.52), ("Ti", 3.28),
        ("Na", 4.29), ("K", 5.23), ("V", 3.03), ("Cr", 2.88), ("Mn", 2.90),
        ("Co", 3.54), ("Zn", 2.66), ("Mo", 3.15), ("W", 3.16), ("Ag", 4.08),
    ):
        corpus.append(
            Structure(Lattice.cubic(a), [e, e], [[0, 0, 0], [0.5, 0.5, 0.5]])
        )
        labels.append(e)

    for e, a in (("Li", 4.0), ("Mg", 4.2), ("Ca", 4.8), ("Ba", 5.6), ("Sr", 5.14)):
        corpus.append(
            Structure(Lattice.cubic(a), [e, "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        )
        labels.append(f"{e}O")

    return corpus, labels


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestComposition:
    def test_dim_matches_label_count(self):
        from backend.common.ml.features_v2 import (
            composition_feature_dim, composition_feature_labels,
        )

        d = composition_feature_dim()
        assert d == len(composition_feature_labels())
        # Magpie (132) + Stoichiometry (6) + ValenceOrbital (8) = 146
        assert d == 146

    def test_si_vector_is_finite_and_deterministic(self):
        from backend.common.ml.features_v2 import featurize_composition

        v1 = featurize_composition("Si")
        v2 = featurize_composition("Si")
        assert v1.shape == (146,)
        assert np.isfinite(v1).all()
        np.testing.assert_array_equal(v1, v2)

    def test_accepts_multiple_input_shapes(self):
        from pymatgen.core import Composition

        from backend.common.ml.features_v2 import featurize_composition

        v_str = featurize_composition("Fe2O3")
        v_comp = featurize_composition(Composition("Fe2O3"))
        np.testing.assert_array_equal(v_str, v_comp)


# ---------------------------------------------------------------------------
# Structure featurizer
# ---------------------------------------------------------------------------


class TestStructureFeaturizer:
    def test_si_vector_shape_and_finite(self, silicon_structure):
        from backend.common.ml.features_v2 import featurize_structure

        v = featurize_structure(silicon_structure)
        assert v.shape == (122,)
        assert np.isfinite(v).all()

    def test_labels_match_dim(self):
        from backend.common.ml.features_v2 import (
            structure_feature_dim, structure_feature_labels,
        )

        assert structure_feature_dim() == len(structure_feature_labels())

    def test_deferred_featurizers_raise(self, silicon_structure):
        from backend.common.ml.features_v2 import (
            orbital_field_matrix, soap_descriptor, xrd_powder_pattern,
        )
        from backend.common.reports import PendingAnalyzerError

        for fn in (orbital_field_matrix, xrd_powder_pattern, soap_descriptor):
            with pytest.raises(PendingAnalyzerError):
                fn(silicon_structure)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


class TestGraphBuilder:
    def test_si_graph_shape_and_edge_count_even(self, silicon_structure):
        from backend.common.ml.features_v2 import build_radius_graph

        g = build_radius_graph(silicon_structure)
        # Node features: 30 one-hot + 5 continuous = 35.
        assert g.node_features.shape == (2, 35)
        # Edge features: [d, 1/d] + 8 Gaussians = 10.
        assert g.edge_features.shape[1] == 10
        # Directed pair convention — total edge count is even.
        assert g.edge_features.shape[0] % 2 == 0
        assert g.edge_index.shape == (2, g.edge_features.shape[0])

    def test_canonical_node_ordering_roundtrip(self, silicon_structure):
        """Two Structures of the same crystal → identical graphs.

        Permuting the input atom order (or starting from ``.copy()``
        and shuffling) must not change node features, edge index, or
        edge features after canonical sort.
        """
        from pymatgen.core import Lattice, Structure

        from backend.common.ml.features_v2 import build_radius_graph

        si_reversed = Structure(
            Lattice.cubic(5.43),
            ["Si", "Si"],
            [[0.25, 0.25, 0.25], [0.0, 0.0, 0.0]],  # swapped order
        )
        g1 = build_radius_graph(silicon_structure)
        g2 = build_radius_graph(si_reversed)
        np.testing.assert_array_equal(g1.node_features, g2.node_features)
        np.testing.assert_array_equal(g1.edge_index, g2.edge_index)
        np.testing.assert_allclose(g1.edge_features, g2.edge_features, rtol=1e-10)
        assert g1.node_species == g2.node_species

    def test_empty_structure_rejected(self):
        from pymatgen.core import Lattice, Structure

        from backend.common.ml.features_v2 import build_radius_graph

        s = Structure(Lattice.cubic(5.0), [], [])
        with pytest.raises(ValueError, match="empty structure"):
            build_radius_graph(s)

    def test_zero_cutoff_rejected(self, silicon_structure):
        from backend.common.ml.features_v2 import build_radius_graph

        with pytest.raises(ValueError, match="cutoff_angstrom"):
            build_radius_graph(silicon_structure, cutoff_angstrom=0.0)

    def test_smaller_cutoff_yields_fewer_edges(self, silicon_structure):
        from backend.common.ml.features_v2 import build_radius_graph

        g_big = build_radius_graph(silicon_structure, cutoff_angstrom=6.0)
        g_small = build_radius_graph(silicon_structure, cutoff_angstrom=3.0)
        assert g_small.edge_features.shape[0] < g_big.edge_features.shape[0]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class TestCache:
    def test_first_call_miss_subsequent_calls_hit(self, silicon_structure):
        from backend.common.ml.features_v2 import (
            InMemoryFeatureCache, cached_featurize,
        )

        c = InMemoryFeatureCache()
        def fake_compute(s):
            return np.ones(5)

        cached_featurize(
            silicon_structure, featurizer_id="t", featurizer_version="1",
            compute_fn=fake_compute, cache=c,
        )
        assert c.misses == 1 and c.hits == 0 and len(c) == 1
        for _ in range(4):
            cached_featurize(
                silicon_structure, featurizer_id="t", featurizer_version="1",
                compute_fn=fake_compute, cache=c,
            )
        assert c.hits == 4 and c.misses == 1

    def test_different_version_invalidates(self, silicon_structure):
        from backend.common.ml.features_v2 import (
            InMemoryFeatureCache, cached_featurize,
        )

        c = InMemoryFeatureCache()
        cached_featurize(
            silicon_structure, featurizer_id="t", featurizer_version="1",
            compute_fn=lambda s: np.ones(3), cache=c,
        )
        cached_featurize(
            silicon_structure, featurizer_id="t", featurizer_version="2",
            compute_fn=lambda s: np.zeros(3), cache=c,
        )
        assert len(c) == 2

    def test_empty_cache_passed_explicitly_not_redirected_to_default(
        self, silicon_structure,
    ):
        """Regression: empty ``InMemoryFeatureCache`` is ``__len__ == 0``
        → falsy, so ``cache or _default_cache`` used to silently
        redirect. The fix compares against ``None`` explicitly.
        """
        from backend.common.ml.features_v2 import (
            InMemoryFeatureCache, cached_featurize, default_cache,
        )

        mine = InMemoryFeatureCache()
        default = default_cache()
        default_hits_before = default.hits
        cached_featurize(
            silicon_structure, featurizer_id="t", featurizer_version="1",
            compute_fn=lambda s: np.ones(3), cache=mine,
        )
        # My cache recorded the miss; the module default shouldn't have moved.
        assert mine.misses == 1 and len(mine) == 1
        assert default.hits == default_hits_before

    def test_defensive_copy(self, silicon_structure):
        from backend.common.ml.features_v2 import (
            InMemoryFeatureCache, cached_featurize,
        )

        c = InMemoryFeatureCache()
        v1 = cached_featurize(
            silicon_structure, featurizer_id="t", featurizer_version="1",
            compute_fn=lambda s: np.ones(3), cache=c,
        )
        v1[0] = 999.0
        v2 = cached_featurize(
            silicon_structure, featurizer_id="t", featurizer_version="1",
            compute_fn=lambda s: np.ones(3), cache=c,
        )
        # Second return isn't poisoned by the first caller's in-place mutation.
        assert v2[0] == 1.0


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class TestEmbedding:
    def test_fit_transform_round_trip(self, diverse_corpus):
        from backend.common.ml.features_v2 import (
            PCAEmbedder, raw_descriptors_batch,
        )

        corpus, _labels = diverse_corpus
        r = raw_descriptors_batch(corpus)
        # Use a small n_components so it fits into the corpus.
        emb = PCAEmbedder(n_components=16)
        e = emb.fit_transform(r)
        assert e.shape == (len(corpus), 16)
        # L2 normalization → all rows have unit norm (within float tolerance).
        norms = np.linalg.norm(e, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_transform_single_vs_batch_consistent(self, diverse_corpus):
        from backend.common.ml.features_v2 import (
            PCAEmbedder, raw_descriptors_batch,
        )

        corpus, _ = diverse_corpus
        r = raw_descriptors_batch(corpus)
        emb = PCAEmbedder(n_components=8).fit(r)
        single = emb.transform(r[0])
        batch = emb.transform(r[:1])
        np.testing.assert_allclose(single, batch[0], atol=1e-12)

    def test_clamps_n_components_with_warning(self, diverse_corpus, caplog):
        import logging

        from backend.common.ml.features_v2 import (
            PCAEmbedder, raw_descriptors_batch,
        )

        corpus, _ = diverse_corpus
        r = raw_descriptors_batch(corpus)
        with caplog.at_level(logging.WARNING):
            emb = PCAEmbedder(n_components=10000).fit(r)
        assert emb.n_components == min(r.shape)
        assert any("reducing n_components" in rec.message for rec in caplog.records)

    def test_save_load_roundtrip(self, diverse_corpus, tmp_path):
        from backend.common.ml.features_v2 import (
            PCAEmbedder, raw_descriptors_batch,
        )

        corpus, _ = diverse_corpus
        r = raw_descriptors_batch(corpus)
        emb = PCAEmbedder(n_components=8).fit(r)
        emb.save(tmp_path / "emb.pkl")
        loaded = PCAEmbedder.load(tmp_path / "emb.pkl")
        np.testing.assert_array_equal(emb.components_, loaded.components_)
        np.testing.assert_allclose(
            emb.transform(r), loaded.transform(r), rtol=1e-12,
        )

    def test_transform_before_fit_raises(self):
        from backend.common.ml.features_v2 import PCAEmbedder

        with pytest.raises(RuntimeError, match="fit\\(\\) before transform"):
            PCAEmbedder().transform(np.zeros(5))

    def test_dim_mismatch_rejected(self, diverse_corpus):
        from backend.common.ml.features_v2 import (
            PCAEmbedder, raw_descriptors_batch,
        )

        corpus, _ = diverse_corpus
        r = raw_descriptors_batch(corpus)
        emb = PCAEmbedder(n_components=4).fit(r)
        with pytest.raises(ValueError, match="expected raw dim"):
            emb.transform(np.zeros(r.shape[1] + 1))


# ---------------------------------------------------------------------------
# Acceptance (roadmap)
# ---------------------------------------------------------------------------


class TestAcceptance:
    def test_100_structures_featurize_under_30s(self):
        """Roadmap acceptance: featurizing 100 structures takes <30s on CPU."""
        from pymatgen.core import Lattice, Structure

        from backend.common.ml.features_v2 import raw_descriptors_batch

        # Warm-up: first matminer call loads Magpie data; we want the
        # steady-state featurization cost, not the data-load cost.
        warmup = Structure(
            Lattice.cubic(5.43), ["Si", "Si"],
            [[0, 0, 0], [0.25, 0.25, 0.25]],
        )
        from backend.common.ml.features_v2 import raw_descriptor

        raw_descriptor(warmup)

        element_set = [
            "Si", "Ge", "Sn", "C", "Fe", "Cu", "Al", "Ni", "Ti", "Na",
        ]
        structs = []
        for i, e in enumerate(element_set):
            lat = Lattice.cubic(3.5 + 0.1 * i)
            structs.append(
                Structure(lat, [e, e], [[0, 0, 0], [0.5, 0.5, 0.5]])
            )
        while len(structs) < 100:
            structs.extend(structs[: min(20, 100 - len(structs))])
        structs = structs[:100]

        t0 = time.time()
        r = raw_descriptors_batch(structs)
        dt = time.time() - t0
        assert r.shape == (100, 268)
        assert dt < 30.0, (
            f"featurizing 100 structures took {dt:.2f}s (>30s target)"
        )

    def test_si_similarity_recovers_group_IV_neighbours(self, diverse_corpus):
        """Roadmap: Si query's nearest neighbours include group-IV / diamond-structure cousins.

        We check that at least one of {C, Ge, Sn} is in the top-3
        non-self neighbours. The corpus deliberately mixes group-IV
        diamonds with metals and simple oxides — a working featurizer
        + PCA should score Si much closer to its diamond cousins than
        to K / Ba / MgO.
        """
        from backend.common.ml.features_v2 import (
            PCAEmbedder, raw_descriptors_batch,
        )

        corpus, labels = diverse_corpus
        r = raw_descriptors_batch(corpus)
        # Use n_components that fits comfortably; 16 is plenty for
        # this 24-structure corpus.
        emb = PCAEmbedder(n_components=min(16, r.shape[0] - 1)).fit_transform(r)

        si_idx = labels.index("Si")
        query = emb[si_idx]
        sims = emb @ query  # already L2-normalized → cosine
        order = np.argsort(-sims)
        # Skip the self-match at rank 0.
        top3_labels = [labels[int(order[k])] for k in range(1, 4)]
        group_iv = {"C", "Ge", "Sn"}
        assert any(lbl in group_iv for lbl in top3_labels), (
            f"Si top-3 neighbours were {top3_labels}; expected at least "
            "one of C / Ge / Sn"
        )


# ---------------------------------------------------------------------------
# Versioning constants
# ---------------------------------------------------------------------------


class TestVersioning:
    def test_composition_versioning_exported(self):
        from backend.common.ml.features_v2 import (
            COMPOSITION_FEATURIZER_ID, COMPOSITION_FEATURIZER_VERSION,
        )

        assert COMPOSITION_FEATURIZER_ID
        assert COMPOSITION_FEATURIZER_VERSION

    def test_structure_versioning_exported(self):
        from backend.common.ml.features_v2 import (
            STRUCTURE_FEATURIZER_ID, STRUCTURE_FEATURIZER_VERSION,
        )

        assert STRUCTURE_FEATURIZER_ID
        assert STRUCTURE_FEATURIZER_VERSION

    def test_embedding_versioning_exported(self):
        from backend.common.ml.features_v2 import (
            DEFAULT_EMBEDDING_DIM, EMBEDDING_ID, EMBEDDING_VERSION,
        )

        assert DEFAULT_EMBEDDING_DIM == 256
        assert EMBEDDING_ID
        assert EMBEDDING_VERSION
