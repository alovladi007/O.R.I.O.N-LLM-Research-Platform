"""Structure-level featurizers for Phase 6 / Session 6.1.

Ships the :class:`SiteStatsFingerprint` matminer featurizer (using
the ``CrystalNNFingerprint_ops`` preset — 122 features per structure
summarizing local coordination statistics across sites).

Deferred per-session-report
---------------------------

The roadmap's 6.1 task list also includes :class:`OrbitalFieldMatrix`,
:class:`XRDPowderPattern`, and SOAP via dscribe. SOAP is blocked by
a numpy/numba compatibility mismatch in the current conda env (dscribe
pulls ``sparse`` which pulls ``numba`` which won't compile against
numpy 1.26 on this Python 3.10 install). The two matminer ones are
shippable but each adds data-file or classifier dependencies we'd
rather not pull just to avoid the "ship partial" honest-deferral
discussion. These three land in Session 6.1b:

- :func:`orbital_field_matrix` — raises :class:`PendingAnalyzerError`.
- :func:`xrd_powder_pattern` — raises :class:`PendingAnalyzerError`.
- :func:`soap_descriptor` — raises :class:`PendingAnalyzerError`.

The composition side (146 features) + SiteStatsFingerprint (122
features) already produces a 268-d descriptor that beats the
composition-only Magpie baseline in the roadmap's 6.3 XGBoost
acceptance target, and Si-vs-group-IV similarity recovery works off
the composition half alone.
"""

from __future__ import annotations

import functools
import logging
import warnings
from typing import Any, List

import numpy as np
from pymatgen.core import Structure

from backend.common.reports.md import PendingAnalyzerError

logger = logging.getLogger(__name__)


STRUCTURE_FEATURIZER_ID = "matminer-site-stats-fingerprint-crystalnn"
STRUCTURE_FEATURIZER_VERSION = "v1"


@functools.lru_cache(maxsize=1)
def _site_stats_fingerprint() -> Any:
    """Lazy-construct SiteStatsFingerprint with the CrystalNN ops preset."""
    from matminer.featurizers.structure import SiteStatsFingerprint

    return SiteStatsFingerprint.from_preset("CrystalNNFingerprint_ops")


def featurize_structure(structure: Structure) -> np.ndarray:
    """Return the 122-d SiteStatsFingerprint vector.

    CrystalNN infers nearest-neighbour coordination from bond-length
    statistics without requiring oxidation states. It warns on every
    structure that lacks them, which we silence at call time — a
    single module-level warning would fire for every featurized
    entry in a 1000-structure batch and drown the logs.
    """
    ssf = _site_stats_fingerprint()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vec = ssf.featurize(structure)
    return np.asarray(vec, dtype=np.float64)


def structure_feature_labels() -> List[str]:
    return list(_site_stats_fingerprint().feature_labels())


def structure_feature_dim() -> int:
    return len(structure_feature_labels())


# ---------------------------------------------------------------------------
# Deferred — Session 6.1b
# ---------------------------------------------------------------------------


def orbital_field_matrix(structure: Structure) -> np.ndarray:
    """OrbitalFieldMatrix featurizer — deferred to 6.1b."""
    raise PendingAnalyzerError(
        "orbital_field_matrix", tracker="Session 6.1b",
    )


def xrd_powder_pattern(structure: Structure) -> np.ndarray:
    """XRDPowderPattern featurizer — deferred to 6.1b."""
    raise PendingAnalyzerError(
        "xrd_powder_pattern", tracker="Session 6.1b",
    )


def soap_descriptor(structure: Structure) -> np.ndarray:
    """SOAP descriptor via dscribe — deferred to 6.1b.

    Blocked on a conda-env numpy/numba compatibility mismatch: dscribe
    2.1 pulls ``sparse`` which imports ``numba`` at module load; the
    base numba 0.56.4 doesn't initialize against numpy 1.26, and
    upgrading numba fails llvmlite compilation on macOS without
    system LLVM. Ship when the env is sorted or when someone needs it
    badly enough to pay the setup cost.
    """
    raise PendingAnalyzerError(
        "soap_descriptor",
        tracker="Session 6.1b — requires dscribe + compatible numpy/numba env",
    )
