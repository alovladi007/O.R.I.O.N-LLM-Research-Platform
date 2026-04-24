"""Deep-ensemble uncertainty for Session 6.4.

Trains ``N`` CGCNN models with different seeds, then aggregates
their predictions into ``(μ, σ)`` at inference time. Simpler +
more reliable than evidential regression (NIG output layer); the
roadmap's guidance ("pick ensemble for simplicity") matches.

API
---

- :func:`train_ensemble` — fit N models; returns a list of
  :class:`CGCNNLitModule` instances.
- :func:`predict_with_uncertainty` — run each model forward, stack
  the predictions, return ``(μ, σ)`` per sample.
- :func:`save_ensemble` / :func:`load_ensemble` — persist the full
  ensemble as a directory of Lightning checkpoints plus a
  manifest JSON.

Usage pattern (see ``scripts/orion_train_cgcnn.py`` for the full
pipeline):

.. code:: python

    from backend.common.ml.models.cgcnn_v2 import (
        CGCNNDataModule, CGCNNLitModule, train_ensemble,
        predict_with_uncertainty, save_ensemble,
    )

    dm = CGCNNDataModule(...)
    models = train_ensemble(
        dm=dm,
        n_models=5,
        module_kwargs=dict(hidden_dim=64, n_conv=3, lr=1e-3),
        max_epochs=300,
    )
    save_ensemble(models, "runs/cgcnn_oxides/ensemble/")

    # Later:
    loader = dm.test_dataloader()
    mu, sigma, y_true = predict_with_uncertainty(models, loader)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch

from .datamodule import CGCNNDataModule
from .lightning_module import CGCNNLitModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train_ensemble(
    *,
    dm: CGCNNDataModule,
    n_models: int = 5,
    module_kwargs: Optional[Dict[str, Any]] = None,
    max_epochs: int = 300,
    base_seed: int = 0,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
    deterministic: bool = True,
) -> List[CGCNNLitModule]:
    """Fit ``n_models`` CGCNN instances with ``seed = base_seed + i``.

    Each model sees the same train/val/test split from ``dm`` but
    different weight inits + dropout noise + mini-batch order, so
    their predictions de-correlate enough to give useful ensemble
    σ. The roadmap acceptance checks the σ-vs-|error| Spearman — a
    well-trained ensemble should give a positive correlation.
    """
    module_kwargs = dict(module_kwargs or {})
    trainer_kwargs = dict(trainer_kwargs or {})

    models: List[CGCNNLitModule] = []
    for i in range(n_models):
        seed = base_seed + i
        pl.seed_everything(seed, workers=True)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)

        model = CGCNNLitModule(**module_kwargs)
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            deterministic=deterministic,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            **trainer_kwargs,
        )
        logger.info("training ensemble member %d/%d (seed=%d)",
                    i + 1, n_models, seed)
        trainer.fit(model, datamodule=dm)
        models.append(model)
    return models


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


def predict_with_uncertainty(
    models: List[CGCNNLitModule],
    dataloader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(μ, σ, y_true)`` as 1-D numpy arrays over the loader.

    Each model's predictions are computed with ``torch.no_grad()``
    and the results stacked; ``μ`` is the mean, ``σ`` is the
    population std (``ddof=0``) over the ensemble axis. Targets are
    lifted from the batch dict so the acceptance test can compute
    |error| directly without re-walking the dataloader.
    """
    if not models:
        raise ValueError("empty ensemble — nothing to predict")

    per_model_preds: List[np.ndarray] = []
    y_all: Optional[np.ndarray] = None

    for model in models:
        model.eval()
        model.to(device)
        preds: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                y_hat = model(batch)
                preds.append(y_hat.detach().cpu().numpy())
                ys.append(batch["y"].detach().cpu().numpy())
        per_model_preds.append(np.concatenate(preds))
        if y_all is None:
            y_all = np.concatenate(ys)

    stacked = np.stack(per_model_preds, axis=0)   # (n_models, N)
    mu = stacked.mean(axis=0)
    sigma = stacked.std(axis=0, ddof=0)
    assert y_all is not None
    return mu, sigma, y_all


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


_MANIFEST_NAME = "ensemble_manifest.json"


def save_ensemble(
    models: List[CGCNNLitModule],
    out_dir: str | Path,
    *,
    dataset_hash: str = "",
    dataset_name: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save each model + a manifest JSON.

    The manifest records the model count, the dataset content hash
    it was trained against, and any extra JSON-safe metadata the
    caller wants to carry (e.g. hyperparameters, cli argv). Round-
    trips via :func:`load_ensemble`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for i, m in enumerate(models):
        ckpt_path = out_dir / f"member_{i:02d}.ckpt"
        # Lightning's Trainer.save_checkpoint takes a trainer — use
        # Lightning's helper instead: strip_default_keys lets us
        # round-trip without a Trainer.
        trainer = pl.Trainer(
            max_epochs=1, enable_progress_bar=False,
            enable_model_summary=False, logger=False,
        )
        trainer.strategy.connect(m)
        trainer.save_checkpoint(str(ckpt_path))
        paths.append(ckpt_path.name)

    manifest = {
        "schema": "cgcnn_ensemble.v1",
        "n_members": len(models),
        "members": paths,
        "dataset_hash": dataset_hash,
        "dataset_name": dataset_name,
        "extra": extra or {},
    }
    (out_dir / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return out_dir


def load_ensemble(in_dir: str | Path) -> List[CGCNNLitModule]:
    """Load an ensemble saved by :func:`save_ensemble`."""
    in_dir = Path(in_dir)
    manifest_path = in_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no ensemble manifest at {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "cgcnn_ensemble.v1":
        raise ValueError(
            f"unrecognized ensemble schema {manifest.get('schema')!r}"
        )
    models: List[CGCNNLitModule] = []
    for name in manifest["members"]:
        path = in_dir / name
        models.append(CGCNNLitModule.load_from_checkpoint(str(path)))
    return models
