"""YAML-backed training config for Session 6.4.

The roadmap calls out "training config surfaced via YAML;
reproducibility enforced (seeds + deterministic cudnn flag + logged
lib versions)". This module owns all three pieces:

- :class:`CGCNNTrainConfig` — pydantic schema for the YAML.
- :func:`load_config` / :func:`dump_config` — YAML round-trip.
- :func:`apply_reproducibility_flags` — seed everything + flip the
  deterministic cudnn flag + disable TF32 for exact reproducibility.
  Called once at the start of a training run.
- :func:`library_versions` — record the pinned versions of torch,
  lightning, torchmetrics, numpy, matminer, and the ORION commit
  SHA when available. Goes into the ensemble manifest so a future
  replay knows exactly which env produced the weights.
"""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field


LossKind = Literal["huber", "mse"]


class CGCNNTrainConfig(BaseModel):
    """One training run's configuration.

    Split into four sections — data, model, optimizer, trainer —
    matching how Lightning itself organizes things. Keeping them
    separate lets us log each independently.
    """

    model_config = ConfigDict(extra="forbid")

    # ------- data -------
    cutoff_angstrom: float = Field(default=6.0, gt=0)
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=0, ge=0)

    # ------- model -------
    node_feat_dim: int = Field(default=35, ge=1)
    edge_feat_dim: int = Field(default=10, ge=1)
    hidden_dim: int = Field(default=64, ge=1)
    n_conv: int = Field(default=3, ge=1)
    mlp_dims: Tuple[int, ...] = Field(default=(128, 64))
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    # ------- loss / optimizer -------
    loss_kind: LossKind = "mse"
    huber_delta: float = Field(default=1.0, gt=0)
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)

    # ------- trainer -------
    max_epochs: int = Field(default=300, ge=1)
    seed: int = Field(default=0, ge=0)
    n_ensemble: int = Field(default=5, ge=1, le=20)
    accelerator: Literal["cpu", "gpu", "auto"] = "cpu"
    # We use mixed-precision only on GPU; CPU stays fp32 because
    # the macOS CPU path doesn't support AMP.
    precision: Literal["32-true", "16-mixed", "bf16-mixed"] = "32-true"
    # Pass any extra Trainer kwargs through here if needed (e.g.
    # ``accumulate_grad_batches``, ``gradient_clip_val``).
    extra_trainer_kwargs: Dict[str, Any] = Field(default_factory=dict)


def load_config(path: str | Path) -> CGCNNTrainConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return CGCNNTrainConfig(**(raw or {}))


def dump_config(cfg: CGCNNTrainConfig, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
    return p


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def apply_reproducibility_flags(seed: int, *, deterministic: bool = True) -> None:
    """Seed everything + flip the deterministic flags.

    Lightning's ``seed_everything`` handles PyTorch / numpy / random /
    the ``PYTHONHASHSEED`` env var. We additionally set
    ``torch.use_deterministic_algorithms`` which the cudnn benchmark
    finder otherwise ignores. TF32 is disabled because it introduces
    sub-ulp floating-point drift that breaks bit-exact-resume tests.
    """
    import pytorch_lightning as pl

    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # TF32 / cudnn controls — no-op on CPU, relevant on GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False


def library_versions() -> Dict[str, str]:
    """Return a JSON-serializable snapshot of the training env.

    The ensemble manifest carries this so a future session can
    compare numbers across environments without guesswork. Git
    SHA is looked up best-effort; failures return ``"unknown"``.
    """
    out = {
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    for mod in ("pytorch_lightning", "torchmetrics", "matminer", "mlflow"):
        try:
            import importlib

            out[mod] = importlib.import_module(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = "not installed"

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[5],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        out["orion_git_sha"] = sha
    except Exception:  # noqa: BLE001
        out["orion_git_sha"] = "unknown"

    return out
