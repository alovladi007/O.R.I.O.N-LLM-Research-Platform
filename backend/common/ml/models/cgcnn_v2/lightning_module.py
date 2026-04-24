"""LightningModule wrapping :class:`CGCNN` for Session 6.4 training.

Responsibilities
----------------

1. Forward the batched graph tensors through CGCNN.
2. Compute the training loss — Huber for energies / formation, MSE
   for bandgaps (the roadmap's choice). ``loss_kind`` selects.
3. Log train / val / test MAE, RMSE, and the loss itself every
   epoch via Lightning's standard ``log`` so MLflow / TensorBoard
   /``runs/`` pick them up uniformly.
4. Expose ``save_hyperparameters`` so
   :meth:`LightningModule.load_from_checkpoint` round-trips without
   the caller needing to re-pass the model kwargs. Checkpoints carry
   all of (model shape + optimizer state + RNG state) so
   ``trainer.fit(ckpt_path=...)`` is bit-exact on CPU — that's the
   roadmap's resume acceptance test.

Why not a multi-target MTL head yet
-----------------------------------

The roadmap mentions "weighted per-target MTL if multi-property."
Session 6.4 ships single-target (scalar output, scalar target).
Multi-target adds a target_name → head_index mapping that the
DataModule needs to produce; we'd rather land the single-target
acceptance first and let a Session 6.4b extend. The LightningModule
signature is ready for it: the ``y`` field in the batch is already
per-sample scalar today.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .model import CGCNN

logger = logging.getLogger(__name__)


LossKind = Literal["huber", "mse"]


class CGCNNLitModule(pl.LightningModule):
    """Lightning wrapper around :class:`CGCNN`.

    Args
    ----
    node_feat_dim / edge_feat_dim
        Shapes of the node + edge feature vectors. Defaults match
        :func:`build_radius_graph` (35 / 10). Overriding these is
        how you'd plug in a different featurizer.
    hidden_dim / n_conv / mlp_dims / dropout
        Passed straight through to :class:`CGCNN`.
    loss_kind
        ``"huber"`` or ``"mse"``. Roadmap says Huber for energies,
        MSE for bandgaps.
    huber_delta
        Huber threshold (eV for energies, eV for bandgaps). 1.0 eV
        is a reasonable default — most residuals are sub-eV so the
        loss is mostly MSE-like with robustness on outliers.
    lr, weight_decay
        AdamW hyperparameters.
    """

    def __init__(
        self,
        *,
        node_feat_dim: int = 35,
        edge_feat_dim: int = 10,
        hidden_dim: int = 64,
        n_conv: int = 3,
        mlp_dims: tuple = (128, 64),
        dropout: float = 0.0,
        loss_kind: LossKind = "mse",
        huber_delta: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = CGCNN(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            n_conv=n_conv,
            mlp_dims=mlp_dims,
            dropout=dropout,
        )
        if loss_kind == "huber":
            self.loss_fn: nn.Module = nn.HuberLoss(delta=huber_delta)
        else:
            self.loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(
            node_features=batch["node_features"],
            edge_index=batch["edge_index"],
            edge_features=batch["edge_features"],
            graph_index=batch["graph_index"],
        )

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _step(
        self, batch: Dict[str, torch.Tensor], stage: str,
    ) -> torch.Tensor:
        y_hat = self(batch)
        y = batch["y"]
        loss = self.loss_fn(y_hat, y)
        mae = torch.mean(torch.abs(y_hat - y))
        rmse = torch.sqrt(torch.mean((y_hat - y) ** 2))
        # ``sync_dist`` is harmless on CPU + cheap on single-GPU;
        # it's the right default for future DDP runs.
        log_kwargs = dict(
            on_step=False, on_epoch=True, prog_bar=(stage == "val"),
            batch_size=y.shape[0], sync_dist=True,
        )
        self.log(f"{stage}/loss", loss, **log_kwargs)
        self.log(f"{stage}/mae", mae, **log_kwargs)
        self.log(f"{stage}/rmse", rmse, **log_kwargs)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, "test")

    def predict_step(
        self, batch, batch_idx, dataloader_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        y_hat = self(batch)
        return {"y_hat": y_hat, "y": batch["y"]}

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return opt
