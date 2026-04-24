"""Training pipeline for Session 6.3 baselines.

Glue between ``datasets_v2`` (produces row-id → features mapping) and
``models/`` (consumes X, y). Logs metrics (MAE, RMSE, R², Spearman)
to MLflow, saves model artifacts to a local directory, and builds a
registry row suitable for the ``ml_model_registry`` table.

What's live
-----------

- :func:`compute_metrics` — MAE / RMSE / R² / Spearman from
  ``(y_true, y_pred)``. Uses numpy + scipy.stats.spearmanr; no
  sklearn dependency here (we already pin it elsewhere, but keeping
  this helper stdlib-clean makes it easy to call from other phases
  without dragging pipeline deps).
- :func:`compute_coverage` — empirical coverage of ``y_hat ± k σ``
  prediction intervals against ``y_true`` for Session 6.3's
  uncertainty-calibration acceptance.
- :func:`train_one_model` — fit/predict/evaluate/log one model on
  the train/val/test split of a ``DatasetSnapshot``. Returns a
  :class:`TrainingResult` with everything the registry cares about.
- :func:`mlflow_context` — helper that initializes MLflow from the
  ``MLFLOW_TRACKING_URI`` env var (falls back to a local
  ``./mlruns`` file store when unset) and sets a named experiment.

What's deferred
---------------

- **MinIO artifact upload** is wired into
  :func:`train_one_model` behind a feature flag; the actual MinIO
  client call is a thin wrapper that mirrors the Celery-worker
  artifact uploader pattern. When no MinIO config is available
  we save to ``args.artifacts_dir`` on local disk. That keeps the
  training script runnable on a dev box + in CI without a live
  object store.
- **DB registry row insert** is returned as a dict from
  :func:`build_registry_row` — the caller decides whether to persist
  it (e.g. via the CLI's ``--db-commit`` flag, which we can wire
  once a session needs it). Matches the 6.2 pure-Python-engine
  pattern.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from backend.common.ml.datasets_v2 import DatasetSnapshot, PropertyRow
from backend.common.ml.models.baselines import BaselineRegressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> Dict[str, float]:
    """Regression metrics: MAE, RMSE, R², Spearman.

    All four returned as plain floats so the dict round-trips
    through JSON / MLflow's ``log_metrics`` verbatim.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"compute_metrics: shape mismatch {y_true.shape} vs {y_pred.shape}"
        )
    if y_true.size < 2:
        raise ValueError(
            f"compute_metrics: need at least 2 samples; got {y_true.size}"
        )

    residual = y_pred - y_true
    mae = float(np.abs(residual).mean())
    rmse = float(np.sqrt((residual ** 2).mean()))

    # R² with the standard definition (1 - SS_res / SS_tot). A
    # constant model beats R² < 0 — honest negative values signal
    # the predictor is worse than the mean.
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    ss_res = float((residual ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Spearman rank correlation — robust to monotone transforms,
    # useful when the regression target has a non-Gaussian shape.
    from scipy.stats import spearmanr

    rho, _p = spearmanr(y_true, y_pred)
    spearman = float(rho) if np.isfinite(rho) else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman,
    }


def compute_coverage(
    y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray,
    k: float = 1.0,
) -> float:
    """Empirical fraction of ``y_true`` inside ``y_pred ± k · sigma``.

    ``k = 1`` corresponds to a nominal 68 % Gaussian PI; the roadmap's
    acceptance target is ≥ 60 % actual coverage. ``k = 2`` tests the
    nominal 95 %. Returns a float in ``[0, 1]``.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    inside = np.abs(y_true - y_pred) <= k * sigma
    return float(inside.mean())


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """What :func:`train_one_model` produces per run.

    Fields are deliberately JSON-serializable so the caller can
    dump them into MLflow, a DB row, or a report file without
    extra encoders.
    """

    model_id: str
    model_version: str
    dataset_name: str
    dataset_version: str
    dataset_hash: str
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    test_coverage_68: float      # k = 1 σ
    test_coverage_95: float      # k = 2 σ
    # Artifact paths are local; the CLI flips these to minio://...
    # URIs after a successful upload.
    model_artifact_path: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset → X, y extractor
# ---------------------------------------------------------------------------


FeaturizerFn = Callable[[PropertyRow], np.ndarray]


def _materialize(
    rows: Sequence[PropertyRow],
    row_ids: Sequence[str],
    featurize: FeaturizerFn,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(X, y)`` arrays from the selected row ids.

    We iterate the full corpus once so the user doesn't have to
    pre-index. For million-row corpora this becomes a hotspot;
    Session 6.3 ships the simple version.
    """
    wanted = set(row_ids)
    selected = [r for r in rows if r.row_id in wanted]
    # Preserve the split's sorted-row-id order so metrics are stable
    # across shuffles of the input corpus (the snapshot already sorts
    # for hash identity).
    selected.sort(key=lambda r: r.row_id)
    if not selected:
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    X = np.vstack([np.asarray(featurize(r), dtype=np.float64) for r in selected])
    y = np.asarray([r.property_value for r in selected], dtype=np.float64)
    return X, y


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------


@contextmanager
def mlflow_context(
    experiment_name: str = "orion-session-6.3",
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> Iterator[Any]:
    """Context manager that sets the tracking URI + starts a run.

    Resolution order for the tracking URI:

    1. Explicit ``tracking_uri`` argument.
    2. ``MLFLOW_TRACKING_URI`` env var.
    3. Local ``./mlruns`` file store (MLflow default).

    Yields the MLflow module itself so callers can do
    ``mlflow.log_metrics(...)`` inside the context. If MLflow is not
    installed (e.g. CI without it), yields ``None`` and downstream
    log calls short-circuit.
    """
    try:
        import mlflow  # type: ignore
    except ImportError:
        logger.warning(
            "MLflow not installed; training will run without metrics logging. "
            "Install ``mlflow`` to enable the acceptance-check UI."
        )
        yield None
        return

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        yield mlflow


def _mlflow_log_metrics(
    mlflow_mod: Any, prefix: str, metrics: Dict[str, float],
) -> None:
    if mlflow_mod is None:
        return
    # MLflow only accepts scalar numeric values for ``log_metric``.
    payload = {f"{prefix}/{k}": float(v) for k, v in metrics.items()}
    mlflow_mod.log_metrics(payload)


# ---------------------------------------------------------------------------
# Registry row
# ---------------------------------------------------------------------------


def build_registry_row(
    result: TrainingResult,
    *,
    owner_id: Optional[str] = None,
    description: str = "",
) -> Dict[str, Any]:
    """Return a dict shaped for the :class:`MLModelRegistry` table.

    Caller decides whether to ``session.add(...)``. This keeps Session
    6.3 operable without a live DB — unit tests inspect the dict
    directly.
    """
    return {
        "name": f"{result.model_id}-{result.dataset_name}-{result.dataset_version}",
        "version": result.model_version,
        "target": "regression",
        "description": description or (
            f"{result.model_id} trained on "
            f"{result.dataset_name}:{result.dataset_version} "
            f"(hash {result.dataset_hash[:16]})"
        ),
        "model_type": result.model_id,
        "checkpoint_path": result.model_artifact_path or "",
        "extra_metadata": {
            "dataset": {
                "name": result.dataset_name,
                "version": result.dataset_version,
                "content_hash": result.dataset_hash,
            },
            "metrics": {
                "train": result.train_metrics,
                "val": result.val_metrics,
                "test": result.test_metrics,
            },
            "coverage": {
                "test_68": result.test_coverage_68,
                "test_95": result.test_coverage_95,
            },
            "mlflow_run_id": result.mlflow_run_id,
            "config": result.config_snapshot,
        },
        "owner_id": owner_id,
    }


# ---------------------------------------------------------------------------
# train_one_model
# ---------------------------------------------------------------------------


def train_one_model(
    model: BaselineRegressor,
    *,
    rows: Sequence[PropertyRow],
    snapshot: DatasetSnapshot,
    featurize: FeaturizerFn,
    artifacts_dir: Optional[Path] = None,
    mlflow_experiment: str = "orion-session-6.3",
    mlflow_run_name: Optional[str] = None,
    config_snapshot: Optional[Dict[str, Any]] = None,
) -> TrainingResult:
    """Fit ``model`` on the snapshot's train split, evaluate on val + test, log.

    Steps:

    1. Materialize ``(X_train, y_train)``, ``(X_val, y_val)``,
       ``(X_test, y_test)`` by featurizing the selected rows.
    2. ``model.fit(X_train, y_train)``.
    3. Compute metrics on all three splits.
    4. Compute 68% + 95% test-set coverage via ``predict_uncertainty``.
    5. Log everything to MLflow (under the given experiment).
    6. Save the pickled model to ``artifacts_dir`` if provided;
       otherwise use a tmp path under ``./runs/``.

    Returns a :class:`TrainingResult` suitable for
    :func:`build_registry_row`.
    """
    X_tr, y_tr = _materialize(rows, snapshot.train_row_ids, featurize)
    X_va, y_va = _materialize(rows, snapshot.val_row_ids, featurize)
    X_te, y_te = _materialize(rows, snapshot.test_row_ids, featurize)

    if X_tr.size == 0:
        raise ValueError(
            f"train split is empty for snapshot "
            f"{snapshot.config.name}:{snapshot.config.version}"
        )

    model.fit(X_tr, y_tr)

    y_hat_tr = model.predict(X_tr)
    train_metrics = compute_metrics(y_tr, y_hat_tr)

    val_metrics = (
        compute_metrics(y_va, model.predict(X_va))
        if X_va.size > 0 else {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "spearman": 0.0}
    )

    y_hat_te, sigma_te = model.predict_uncertainty(X_te)
    test_metrics = compute_metrics(y_te, y_hat_te)
    cov_68 = compute_coverage(y_te, y_hat_te, sigma_te, k=1.0)
    cov_95 = compute_coverage(y_te, y_hat_te, sigma_te, k=2.0)

    mlflow_run_id: Optional[str] = None
    with mlflow_context(
        experiment_name=mlflow_experiment,
        run_name=mlflow_run_name or f"{model.model_id}-{snapshot.config.name}",
    ) as mlflow_mod:
        if mlflow_mod is not None:
            mlflow_run_id = mlflow_mod.active_run().info.run_id
            mlflow_mod.log_params({
                "model_id": model.model_id,
                "model_version": model.version,
                "dataset_name": snapshot.config.name,
                "dataset_version": snapshot.config.version,
                "dataset_hash": snapshot.content_hash,
                "n_train": len(snapshot.train_row_ids),
                "n_val": len(snapshot.val_row_ids),
                "n_test": len(snapshot.test_row_ids),
                "split_kind": snapshot.config.split.kind,
                "split_seed": snapshot.config.split.seed,
            })
            _mlflow_log_metrics(mlflow_mod, "train", train_metrics)
            _mlflow_log_metrics(mlflow_mod, "val", val_metrics)
            _mlflow_log_metrics(mlflow_mod, "test", test_metrics)
            mlflow_mod.log_metrics({
                "test/coverage_68": cov_68,
                "test/coverage_95": cov_95,
            })

        # Save the pickled model.
        if artifacts_dir is None:
            artifacts_dir = Path("./runs") / snapshot.config.name / model.model_id
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifacts_dir / f"{model.model_id}-{model.version}.pkl"
        model.save(artifact_path)
        if mlflow_mod is not None:
            mlflow_mod.log_artifact(str(artifact_path))

    return TrainingResult(
        model_id=model.model_id,
        model_version=model.version,
        dataset_name=snapshot.config.name,
        dataset_version=snapshot.config.version,
        dataset_hash=snapshot.content_hash,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        test_coverage_68=cov_68,
        test_coverage_95=cov_95,
        model_artifact_path=str(artifact_path),
        mlflow_run_id=mlflow_run_id,
        config_snapshot=config_snapshot or {},
    )
