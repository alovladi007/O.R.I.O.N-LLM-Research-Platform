"""Cache layer for Session 6.1 featurizers.

Keys are ``(structure_hash, featurizer_id, featurizer_version)``. The
default backend is an in-process dict that persists for the lifetime
of the Python process — enough for the acceptance test and for
single-notebook exploration. Persisting across runs belongs in a
Session 6.2 follow-up (most likely as a JSON-B column on
``Structure`` or as Parquet-on-MinIO for bulk featurization output).

Why per-call hashing instead of passing ``structure_hash`` in?
---------------------------------------------------------------

We deliberately compute the hash here from the full pymatgen
``Structure`` so callers who build a ``Structure`` ad-hoc (not from
the ORION ``Structure`` table) get a correctly-keyed cache entry
without having to reach for
:func:`backend.common.structures.hashing.structure_hash` themselves.
The cost is a few ms per miss — amortized away by the cache hit
after the first featurization.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from pymatgen.core import Structure

logger = logging.getLogger(__name__)


CacheKey = Tuple[str, str, str]  # (structure_hash, featurizer_id, version)


@dataclass
class InMemoryFeatureCache:
    """Thread-safe in-process dict cache.

    Hit rate is tracked so callers / tests can confirm caching is
    working as intended.
    """

    _store: Dict[CacheKey, np.ndarray] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    hits: int = 0
    misses: int = 0

    def get(self, key: CacheKey) -> Optional[np.ndarray]:
        with self._lock:
            val = self._store.get(key)
            if val is None:
                self.misses += 1
                return None
            self.hits += 1
            return val.copy()  # defensive copy — cache stays immutable

    def put(self, key: CacheKey, value: np.ndarray) -> None:
        with self._lock:
            # Store contiguous copy so later mutations by the caller
            # don't leak into the cache.
            self._store[key] = np.ascontiguousarray(value).copy()

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        return len(self._store)


# Module-level default cache. Tests construct their own instances to
# avoid cross-test leakage; production code that wants a per-request
# cache should instantiate :class:`InMemoryFeatureCache` directly.
_default_cache = InMemoryFeatureCache()


def default_cache() -> InMemoryFeatureCache:
    return _default_cache


def _structure_hash(structure: Structure) -> str:
    """Compute the canonical ORION structure hash for a pymatgen Structure."""
    from backend.common.structures.hashing import structure_hash

    return structure_hash(pmg_structure=structure)


def cached_featurize(
    structure: Structure,
    *,
    featurizer_id: str,
    featurizer_version: str,
    compute_fn: Callable[[Structure], np.ndarray],
    cache: Optional[InMemoryFeatureCache] = None,
) -> np.ndarray:
    """Return cached features for *structure*, computing + storing on miss.

    ``compute_fn`` is called only on a cache miss. The return is a
    defensive copy so the caller can mutate the result without
    poisoning the cache.
    """
    # NB: ``cache or _default_cache`` would be wrong — an empty
    # ``InMemoryFeatureCache`` is falsy because ``__len__`` returns 0,
    # so a freshly-constructed per-test cache would silently redirect
    # to the module-level default. Compare against ``None`` explicitly.
    if cache is None:
        cache = _default_cache
    key: CacheKey = (_structure_hash(structure), featurizer_id, featurizer_version)
    cached = cache.get(key)
    if cached is not None:
        return cached
    value = np.asarray(compute_fn(structure), dtype=np.float64)
    cache.put(key, value)
    # Mirror the cache.get defensive-copy contract.
    return value.copy()
