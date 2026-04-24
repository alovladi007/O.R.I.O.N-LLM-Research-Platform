#!/usr/bin/env python3
"""Session 6.3 CLI: train the three Phase-6 baselines on a snapshot.

Usage
-----

Train all three baselines on a snapshot JSON produced by
``scripts/orion_dataset.py``::

    orion_train_baseline.py \\
        --rows-jsonl data/property_rows.jsonl \\
        --snapshot snapshots/oxides_gap_v1.json \\
        --feature composition \\
        --models mean,random_forest,xgboost \\
        --artifacts-dir runs/oxides_gap_v1

Per-run metrics land in MLflow (respects ``MLFLOW_TRACKING_URI``;
falls back to ``./mlruns`` file store otherwise). Model artifacts
pickle to ``--artifacts-dir`` so the MinIO uploader can pick them up
later (Session 6.3b wiring).

``--feature`` selects the row → ndarray featurizer:

- ``composition`` — the Session 6.1 146-d matminer composition
  vector. Requires ``row.structure`` isn't used; we reconstruct
  ``pymatgen.Composition`` from ``row.composition_formula`` or from
  the row's ``extras['composition_formula']`` fallback. For the
  roadmap's XGBoost target on oxides_gap_v1, composition is enough.
- ``pre-computed`` — read the feature vector from
  ``row.extras['features']``. Lets the caller use arbitrary
  descriptors (incl. ``features_v2.raw_descriptor``) without
  putting matminer on the CLI's critical path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List

import numpy as np

# Make the repo root importable when run directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.common.ml.datasets_v2 import DatasetSnapshot, PropertyRow  # noqa: E402
from backend.common.ml.models.baselines import (  # noqa: E402
    MeanRegressor, RandomForestRegressor, XGBoostQuantileRegressor,
)
from backend.common.ml.training import (  # noqa: E402
    TrainingResult, build_registry_row, train_one_model,
)


_MODEL_REGISTRY = {
    "mean": MeanRegressor,
    "random_forest": RandomForestRegressor,
    "xgboost": XGBoostQuantileRegressor,
}


def _composition_featurizer():
    """Lazy-construct the 146-d composition featurizer.

    We import matminer only on demand — calling the CLI with
    ``--feature pre-computed`` skips the ~1-2 s Magpie load.
    """
    from pymatgen.core import Composition

    from backend.common.ml.features_v2 import featurize_composition

    def _fn(row: PropertyRow) -> np.ndarray:
        formula = row.composition_formula or (row.extras or {}).get("composition_formula")
        if not formula:
            raise ValueError(
                f"row {row.row_id!r} has no composition_formula; set it or "
                "use --feature pre-computed with row.extras['features']"
            )
        return featurize_composition(Composition(formula))

    return _fn


def _precomputed_featurizer():
    def _fn(row: PropertyRow) -> np.ndarray:
        fp = (row.extras or {}).get("features")
        if fp is None:
            raise ValueError(
                f"row {row.row_id!r} has no extras['features'] (required for "
                "--feature pre-computed)"
            )
        return np.asarray(fp, dtype=np.float64)

    return _fn


def _load_rows(path: Path) -> List[PropertyRow]:
    rows: List[PropertyRow] = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(PropertyRow(**json.loads(line)))
            except Exception as exc:  # noqa: BLE001
                raise SystemExit(f"{path}:{lineno}: bad row: {exc}")
    return rows


def _print_result(r: TrainingResult) -> None:
    print(f"{r.model_id}  [dataset={r.dataset_name}:{r.dataset_version}]")
    for split in ("train", "val", "test"):
        m = getattr(r, f"{split}_metrics")
        print(
            f"  {split}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
            f"R²={m['r2']:.4f}  Spearman={m['spearman']:.4f}"
        )
    print(
        f"  test coverage: 68 %_PI={r.test_coverage_68:.2%}  "
        f"95 %_PI={r.test_coverage_95:.2%}"
    )
    if r.mlflow_run_id:
        print(f"  mlflow_run_id: {r.mlflow_run_id}")
    if r.model_artifact_path:
        print(f"  artifact:      {r.model_artifact_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="orion train-baseline")
    p.add_argument("--rows-jsonl", required=True,
                   help="PropertyRow JSONL, one per line")
    p.add_argument("--snapshot", required=True,
                   help="DatasetSnapshot JSON (from scripts/orion_dataset.py)")
    p.add_argument("--feature", choices=("composition", "pre-computed"),
                   default="composition")
    p.add_argument("--models", default="mean,random_forest,xgboost",
                   help="Comma-separated model kinds to train. "
                        f"Known: {sorted(_MODEL_REGISTRY)}")
    p.add_argument("--artifacts-dir", default=None)
    p.add_argument("--mlflow-experiment", default="orion-session-6.3")
    p.add_argument("--write-registry-json", default=None,
                   help="Optional path to dump the registry-row dicts "
                        "(for Phase 12 provenance ingest).")
    args = p.parse_args(argv)

    rows = _load_rows(Path(args.rows_jsonl))
    snapshot = DatasetSnapshot.model_validate_json(
        Path(args.snapshot).read_text()
    )

    featurize: Callable[[PropertyRow], np.ndarray]
    if args.feature == "composition":
        featurize = _composition_featurizer()
    else:
        featurize = _precomputed_featurizer()

    model_kinds = [s.strip() for s in args.models.split(",") if s.strip()]
    for k in model_kinds:
        if k not in _MODEL_REGISTRY:
            raise SystemExit(
                f"unknown model kind {k!r}. Known: {sorted(_MODEL_REGISTRY)}"
            )

    results: List[TrainingResult] = []
    for kind in model_kinds:
        cls = _MODEL_REGISTRY[kind]
        model = cls()
        result = train_one_model(
            model=model,
            rows=rows,
            snapshot=snapshot,
            featurize=featurize,
            artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
            mlflow_experiment=args.mlflow_experiment,
            config_snapshot={"feature": args.feature, "cli_argv": argv or sys.argv},
        )
        _print_result(result)
        results.append(result)

    if args.write_registry_json:
        rows_out = [build_registry_row(r) for r in results]
        Path(args.write_registry_json).write_text(json.dumps(rows_out, indent=2, default=str))
        print(f"Wrote {len(rows_out)} registry rows to {args.write_registry_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
