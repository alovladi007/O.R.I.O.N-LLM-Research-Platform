"""PCA-based 256-d structure embedding for Phase 6 / Session 6.1.

The roadmap mandates a 256-d embedding stored in pgvector for
similarity search. Our raw descriptor is the concatenated
composition (146) + structure (122) vector — 268 features. PCA
down to 256 drops the least-informative 12 directions and produces
a dense, L2-normalized embedding suitable for cosine / L2 ANN.

We expose a trainable :class:`PCAEmbedder` class. Train once on a
corpus of a few hundred structures, save the fitted model, re-use
it for all downstream featurization. The ``embed`` path handles
single structures or batches uniformly.

Why L2-normalize?
-----------------

pgvector's IVF-flat index works in any metric, but cosine similarity
is the most common choice for "structurally similar" search and
L2-normalized PCA embeddings make cosine = ½ (2 - ‖u − v‖²), i.e.
both metrics line up. This is the same idiom used by sentence-
embedding retrieval stacks.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from pymatgen.core import Structure

from .composition import featurize_composition
from .structure import featurize_structure

logger = logging.getLogger(__name__)


EMBEDDING_ID = "orion-pca-composition-structure"
EMBEDDING_VERSION = "v1"
DEFAULT_EMBEDDING_DIM = 256


@dataclass
class PCAEmbedder:
    """Standardize → PCA → L2-normalize pipeline.

    The raw composition features (Magpie atomic statistics + orbital
    counts) span orders of magnitude — atomic weights around 100 vs
    orbital-fractions around 0.1. Feeding those into PCA untouched
    makes Z-scale features dominate the principal axes, so the
    top-k nearest-neighbour query collapses onto "elements with
    similar Z" rather than "structurally/chemically similar". We
    standardize (zero mean, unit variance per feature) before the
    PCA and L2-normalize the output so cosine / L2 distance align.

    Fields
    ------
    n_components
        Target dim. Default 256 (roadmap). Clamped at fit time to
        ``min(n_samples, d_raw)``; a warning fires if that clamping
        kicks in.
    mean_ / std_
        StandardScaler parameters (shape ``(d_raw,)``).
    components_
        PCA basis (shape ``(n_components, d_raw)``).
    raw_dim
        Dimensionality of the raw descriptor (146 + 122 = 268 with
        the default stack). Locked at fit time.
    normalize
        If True (default), L2-normalize the PCA output so cosine
        similarity is well-defined.
    """

    n_components: int = DEFAULT_EMBEDDING_DIM
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None
    components_: Optional[np.ndarray] = None
    raw_dim: Optional[int] = None
    normalize: bool = True

    def is_fit(self) -> bool:
        return (
            self.components_ is not None
            and self.mean_ is not None
            and self.std_ is not None
        )

    def fit(self, raw_features: np.ndarray) -> "PCAEmbedder":
        """Fit StandardScaler + PCA on a stack of raw descriptor vectors.

        ``raw_features`` is shape ``(n_samples, d_raw)``. NaN / inf
        entries are replaced with zero before fitting — matminer's
        impute_nan=True already prevents most of these, but guarding
        here means a pathological featurizer doesn't explode the
        PCA.
        """
        if raw_features.ndim != 2 or raw_features.shape[0] < 2:
            raise ValueError(
                f"PCAEmbedder.fit needs (n_samples, d) with n_samples>=2; "
                f"got shape {raw_features.shape}"
            )
        x = np.nan_to_num(
            np.asarray(raw_features, dtype=np.float64),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        n_samples, d_raw = x.shape

        # StandardScaler manually (avoids an extra sklearn class in the
        # saved model). Zero-std features get std = 1 so scaling is
        # a no-op on them.
        self.mean_ = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-12] = 1.0
        self.std_ = std
        x_std = (x - self.mean_) / self.std_

        from sklearn.decomposition import PCA

        n_comp = min(self.n_components, n_samples, d_raw)
        if n_comp < self.n_components:
            logger.warning(
                "PCAEmbedder.fit: reducing n_components from %d to %d "
                "(bound by min(n_samples=%d, d_raw=%d))",
                self.n_components, n_comp, n_samples, d_raw,
            )
        pca = PCA(n_components=n_comp)
        pca.fit(x_std)
        self.components_ = pca.components_.astype(np.float64)
        self.raw_dim = int(d_raw)
        self.n_components = int(n_comp)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Project a raw descriptor to the fitted embedding space.

        Accepts ``(d,)`` or ``(n, d)``; returns ``(n_components,)``
        or ``(n, n_components)`` respectively.
        """
        if not self.is_fit():
            raise RuntimeError("PCAEmbedder: call fit() before transform()")
        x = np.nan_to_num(
            np.asarray(raw, dtype=np.float64),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        squeezed = False
        if x.ndim == 1:
            x = x[None, :]
            squeezed = True
        if x.shape[1] != self.raw_dim:
            raise ValueError(
                f"PCAEmbedder.transform: expected raw dim {self.raw_dim}, "
                f"got {x.shape[1]}"
            )
        standardized = (x - self.mean_[None, :]) / self.std_[None, :]
        projected = standardized @ self.components_.T
        if self.normalize:
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            projected = projected / norms
        return projected[0] if squeezed else projected

    def fit_transform(self, raw_features: np.ndarray) -> np.ndarray:
        self.fit(raw_features)
        return self.transform(raw_features)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "n_components": self.n_components,
                    "mean_": self.mean_,
                    "std_": self.std_,
                    "components_": self.components_,
                    "raw_dim": self.raw_dim,
                    "normalize": self.normalize,
                    "embedding_id": EMBEDDING_ID,
                    "embedding_version": EMBEDDING_VERSION,
                },
                f,
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PCAEmbedder":
        with open(path, "rb") as f:
            data = pickle.load(f)
        emb = cls(
            n_components=data["n_components"],
            mean_=data["mean_"],
            std_=data["std_"],
            components_=data["components_"],
            raw_dim=data["raw_dim"],
            normalize=data.get("normalize", True),
        )
        return emb


# ---------------------------------------------------------------------------
# Convenience: raw descriptor = concatenated composition + structure
# ---------------------------------------------------------------------------


def raw_descriptor(structure: Structure) -> np.ndarray:
    """Return the 268-d concat of composition + structure features."""
    comp = featurize_composition(structure.composition)
    struct = featurize_structure(structure)
    return np.concatenate([comp, struct])


def raw_descriptors_batch(structures: Iterable[Structure]) -> np.ndarray:
    """Stack :func:`raw_descriptor` across a sequence of structures."""
    rows: List[np.ndarray] = []
    for s in structures:
        rows.append(raw_descriptor(s))
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.vstack(rows)
