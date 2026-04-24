"""ML model implementations.

Phase 6 / Session 6.3 baselines:

- :class:`MeanRegressor` — constant-prediction sanity baseline.
- :class:`RandomForestRegressor` — sklearn RF with per-tree-std
  uncertainty.
- :class:`XGBoostQuantileRegressor` — three-quantile XGBoost,
  σ = ½(p84 − p16).

Legacy (Session 14/15):

- :class:`CGCNNModel` — simplified Crystal Graph Conv NN. Migration
  to the Phase 6.4 training infrastructure is a follow-up.
"""

from .baselines import (
    BaselineRegressor,
    MeanRegressor,
    RandomForestRegressor,
    XGBoostQuantileRegressor,
)
from .cgcnn_like import CGCNNModel

__all__ = [
    "BaselineRegressor",
    "CGCNNModel",
    "MeanRegressor",
    "RandomForestRegressor",
    "XGBoostQuantileRegressor",
]
