"""CGCNN v2 — Phase 6 / Session 6.4 Lightning-based GNN pipeline.

Fresh package sibling of the Session-14 :mod:`.cgcnn_like`. The
legacy ``CGCNNModel`` class pre-dates both PyG and Lightning; this
package rebuilds the stack with:

- A from-scratch CGCNN convolution (no PyG dep — keeps the install
  small).
- A ``pytorch_lightning.LightningModule`` wrapper with Huber / MSE
  loss selection, standard metric logging, and checkpoint round-trip.
- A ``pl.LightningDataModule`` that reads from
  :class:`~backend.common.ml.datasets_v2.DatasetSnapshot` and
  featurizes via :func:`~backend.common.ml.features_v2.build_radius_graph`.
- Deep-ensemble uncertainty (train N seeds → report (μ, σ)). The
  roadmap considered evidential regression as an alternative; the
  ensemble route is simpler and the σ calibration target is modest.
- A YAML config schema + reproducibility helpers (seed everything +
  deterministic cudnn + logged library versions).

The legacy :mod:`.cgcnn_like` module stays in place for callers of
the Session-14 prediction path; migrating them is a follow-up.
"""

from .config import (
    CGCNNTrainConfig,
    apply_reproducibility_flags,
    dump_config,
    library_versions,
    load_config,
)
from .datamodule import (
    CGCNNDataModule,
    CGCNNDataset,
    GraphSample,
    collate_graphs,
    featurize_row,
)
from .ensemble import (
    load_ensemble,
    predict_with_uncertainty,
    save_ensemble,
    train_ensemble,
)
from .lightning_module import CGCNNLitModule
from .model import CGCNN, CGCNNConv

__all__ = [
    "CGCNN",
    "CGCNNConv",
    "CGCNNDataModule",
    "CGCNNDataset",
    "CGCNNLitModule",
    "CGCNNTrainConfig",
    "GraphSample",
    "apply_reproducibility_flags",
    "collate_graphs",
    "dump_config",
    "featurize_row",
    "library_versions",
    "load_config",
    "load_ensemble",
    "predict_with_uncertainty",
    "save_ensemble",
    "train_ensemble",
]
