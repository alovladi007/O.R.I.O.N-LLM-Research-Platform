"""ML featurizer pipeline (Phase 6 / Session 6.1).

Fresh package sitting alongside the legacy
:mod:`backend.common.ml.features` (Session 14's CGCNN-only
neighbour-list builder). The legacy module still works for
hand-rolled GNN features; new callers should prefer this one for:

- matminer-backed composition fingerprints (Magpie + Stoichiometry
  + ValenceOrbital),
- :class:`SiteStatsFingerprint` structure descriptor,
- a canonicalized radius graph suitable for PyG / DGL,
- an in-memory cache keyed by ``(structure_hash, featurizer_id,
  version)`` so repeated featurization is free,
- a PCA-based 256-d embedding trainable on a corpus + saveable for
  downstream pgvector ANN.

Deferred to Session 6.1b
------------------------

- SOAP (dscribe) — blocked on a numba/numpy env mismatch.
- OrbitalFieldMatrix, XRDPowderPattern — shippable but not in scope
  for 6.1's acceptance targets.

Public API
----------

- Composition: :func:`featurize_composition`, :func:`composition_feature_labels`,
  :func:`composition_feature_dim`.
- Structure: :func:`featurize_structure`, :func:`structure_feature_labels`.
- Graph: :func:`build_radius_graph` → :class:`RadiusGraph`.
- Cache: :class:`InMemoryFeatureCache`, :func:`cached_featurize`.
- Embedding: :class:`PCAEmbedder`, :func:`raw_descriptor`,
  :func:`raw_descriptors_batch`.
- Versioning constants: ``COMPOSITION_FEATURIZER_ID``,
  ``STRUCTURE_FEATURIZER_ID``, ``EMBEDDING_ID``, etc.
"""

from .cache import (
    CacheKey,
    InMemoryFeatureCache,
    cached_featurize,
    default_cache,
)
from .composition import (
    COMPOSITION_FEATURIZER_ID,
    COMPOSITION_FEATURIZER_VERSION,
    composition_feature_dim,
    composition_feature_labels,
    featurize_composition,
)
from .embedding import (
    DEFAULT_EMBEDDING_DIM,
    EMBEDDING_ID,
    EMBEDDING_VERSION,
    PCAEmbedder,
    raw_descriptor,
    raw_descriptors_batch,
)
from .graph import (
    DEFAULT_CUTOFF_A,
    DEFAULT_ELEMENT_ONE_HOT,
    DEFAULT_GAUSSIAN_N_CENTERS,
    RadiusGraph,
    build_radius_graph,
)
from .structure import (
    STRUCTURE_FEATURIZER_ID,
    STRUCTURE_FEATURIZER_VERSION,
    featurize_structure,
    orbital_field_matrix,
    soap_descriptor,
    structure_feature_dim,
    structure_feature_labels,
    xrd_powder_pattern,
)

__all__ = [
    "COMPOSITION_FEATURIZER_ID",
    "COMPOSITION_FEATURIZER_VERSION",
    "CacheKey",
    "DEFAULT_CUTOFF_A",
    "DEFAULT_ELEMENT_ONE_HOT",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_GAUSSIAN_N_CENTERS",
    "EMBEDDING_ID",
    "EMBEDDING_VERSION",
    "InMemoryFeatureCache",
    "PCAEmbedder",
    "RadiusGraph",
    "STRUCTURE_FEATURIZER_ID",
    "STRUCTURE_FEATURIZER_VERSION",
    "build_radius_graph",
    "cached_featurize",
    "composition_feature_dim",
    "composition_feature_labels",
    "default_cache",
    "featurize_composition",
    "featurize_structure",
    "orbital_field_matrix",
    "raw_descriptor",
    "raw_descriptors_batch",
    "soap_descriptor",
    "structure_feature_dim",
    "structure_feature_labels",
    "xrd_powder_pattern",
]
