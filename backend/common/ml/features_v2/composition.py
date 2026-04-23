"""Composition featurizers for Phase 6 / Session 6.1.

Wraps three matminer composition featurizers:

- :class:`matminer.featurizers.composition.ElementProperty` with the
  Magpie preset — 132 features across (min, max, range, mean, avg_dev,
  mode) × 22 atomic descriptors.
- :class:`matminer.featurizers.composition.Stoichiometry` — 6 features
  (L_p norms of atomic fractions, p ∈ {0, 2, 3, 5, 7, 10}).
- :class:`matminer.featurizers.composition.ValenceOrbital` — 8
  features (fraction of electrons in each s/p/d/f orbital + total
  valence electron count by orbital type).

Total composition feature vector: **146 features**.

The per-featurizer call costs are dominated by matminer's lazy
loading of the Magpie data on first use. We amortize that by caching
the featurizer instances at module scope; after the first call every
composition featurization is a few-ms numpy op.

NaN handling
------------

Matminer's ``impute_nan`` kwarg imputes dataset-mean values for
rare elements that lack full coverage. We default to
``impute_nan=True`` for robustness — the acceptance test exercises
~30 common oxides / simple compounds where imputation isn't needed,
but on a real inventory a few Lu / Pm / Fr entries will show up and
we'd rather get a finite feature vector than a NaN that quietly
breaks downstream scalers.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, List, Tuple

import numpy as np
from pymatgen.core import Composition

logger = logging.getLogger(__name__)


# Versioned featurizer identifier so the cache layer can invalidate
# when we change the stack. Bump on any change that alters feature
# dimensionality, order, or semantics.
COMPOSITION_FEATURIZER_ID = "matminer-magpie-stoich-valence"
COMPOSITION_FEATURIZER_VERSION = "v1"


@functools.lru_cache(maxsize=1)
def _featurizers() -> Tuple[Any, Any, Any]:
    """Lazy-construct the three featurizer objects (cached).

    Magpie data loading from disk is a few hundred ms the first time.
    We swallow that cost on module first-use, then every subsequent
    composition hits the cached instance.
    """
    from matminer.featurizers.composition import (
        ElementProperty, Stoichiometry, ValenceOrbital,
    )

    # Only ElementProperty + ValenceOrbital take ``impute_nan``;
    # Stoichiometry's feature vector is a pure composition aggregate
    # and can't produce NaNs regardless of element coverage.
    return (
        ElementProperty.from_preset("magpie", impute_nan=True),
        Stoichiometry(),
        ValenceOrbital(impute_nan=True),
    )


def featurize_composition(composition: Any) -> np.ndarray:
    """Return the 146-d composition feature vector for *composition*.

    Accepts any of:

    - ``pymatgen.core.Composition`` (fastest).
    - A formula string (``"Si"``, ``"Fe2O3"``).
    - A mapping like ``{"Si": 1}``.

    All three pass through :func:`pymatgen.core.Composition` on the
    way in, so the caller needn't hand-construct the composition.
    """
    if not isinstance(composition, Composition):
        composition = Composition(composition)
    ep, stoich, vo = _featurizers()
    parts = [
        np.asarray(ep.featurize(composition), dtype=np.float64),
        np.asarray(stoich.featurize(composition), dtype=np.float64),
        np.asarray(vo.featurize(composition), dtype=np.float64),
    ]
    return np.concatenate(parts)


def composition_feature_labels() -> List[str]:
    """Return the 146-entry feature-label list in concatenation order."""
    ep, stoich, vo = _featurizers()
    return [*ep.feature_labels(), *stoich.feature_labels(), *vo.feature_labels()]


def composition_feature_dim() -> int:
    """Convenience — total vector length (146)."""
    return len(composition_feature_labels())
