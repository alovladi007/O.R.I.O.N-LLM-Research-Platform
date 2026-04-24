#!/usr/bin/env python3
"""Session 6.4 CLI: train a CGCNN deep-ensemble on a DatasetSnapshot.

Usage
-----

::

    orion_train_cgcnn.py \\
        --rows-jsonl data/oxides_raw.jsonl \\
        --snapshot snapshots/oxides_gap_v1.json \\
        --structure-source materials_project_jsonl \\
        --mp-cache data/mp_structures.jsonl \\
        --config configs/cgcnn_oxides.yml \\
        --output runs/cgcnn_oxides/

The ``--structure-source`` selects how ``PropertyRow.structure_id``
resolves to a pymatgen ``Structure``. For JSONL ingests (the 6.3
oxides_gap_v1 path) we assume a sibling JSONL file where each line
carries the MP id + its JSON-encoded structure. See
``_build_resolver`` below for the schema.

Why CLI + library split
-----------------------

The :mod:`backend.common.ml.models.cgcnn_v2` package is purely
library code — it doesn't know what "materials_project_jsonl" or
"postgres_structure_fk" mean. Resolving structures from whatever
storage the caller uses is the CLI's job. Keeps the training code
reusable from the API layer (6.4b `POST /api/v1/ml/train`) without
CLI concerns leaking into it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.common.ml.datasets_v2 import DatasetSnapshot, PropertyRow  # noqa: E402
from backend.common.ml.models.cgcnn_v2 import (  # noqa: E402
    CGCNNDataModule,
    apply_reproducibility_flags,
    library_versions,
    load_config,
    predict_with_uncertainty,
    save_ensemble,
    train_ensemble,
)

logger = logging.getLogger(__name__)


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


def _build_mp_resolver(
    rows: List[PropertyRow],
    mp_cache_path: Optional[Path],
) -> Callable:
    """Resolve Materials Project rows to pymatgen Structures.

    Two strategies, in order of preference:

    1. If ``mp_cache_path`` exists, read MP id → CIF / structure JSON
       from it. Recommended for reproducibility — a cached JSONL is
       independent of network availability.
    2. Otherwise, hit the MP API live via ``mp_api.client.MPRester``.
       Requires ``$MP_API_KEY``.

    This matches how ``scripts/orion_fetch_mp_oxides.py`` works
    today and lets 6.4 training run offline once the user has cached
    the structures.
    """
    if mp_cache_path is not None and mp_cache_path.is_file():
        from pymatgen.core import Structure

        cache: Dict[str, Structure] = {}
        with mp_cache_path.open() as f:
            for line in f:
                obj = json.loads(line)
                cache[obj["material_id"]] = Structure.from_dict(obj["structure"])
        logger.info("mp cache: %d structures loaded from %s",
                    len(cache), mp_cache_path)

        def _resolver(row: PropertyRow):
            sid = (row.extras or {}).get("material_id") or row.structure_id
            if sid not in cache:
                raise KeyError(
                    f"structure {sid!r} not in MP cache; re-run "
                    "scripts/orion_fetch_mp_structures.py to populate."
                )
            return cache[sid]

        return _resolver

    # Live fallback — lazy import to avoid requiring mp_api when the
    # user ships a cache.
    try:
        from mp_api.client import MPRester
        from pymatgen.core import Structure
    except ImportError as exc:
        raise SystemExit(
            f"no MP cache at {mp_cache_path} and mp_api isn't installed. "
            f"Either populate the cache or install the MP API pair "
            f"(pip install 'mp-api<0.45' 'emmet-core<0.85'): {exc}"
        )

    import os

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise SystemExit("no MP cache and no MP_API_KEY — set one.")
    mpr = MPRester(api_key)

    fetched: Dict[str, Structure] = {}

    def _resolver(row: PropertyRow):
        sid = (row.extras or {}).get("material_id") or row.structure_id
        if sid not in fetched:
            struct = mpr.get_structure_by_material_id(sid)
            fetched[sid] = struct
        return fetched[sid]

    return _resolver


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
    )
    p = argparse.ArgumentParser(prog="orion train-cgcnn")
    p.add_argument("--rows-jsonl", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--config", default=None,
                   help="YAML training config; defaults used when omitted.")
    p.add_argument(
        "--structure-source", default="materials_project_jsonl",
        choices=("materials_project_jsonl",),
        help="How to resolve PropertyRow.structure_id → pymatgen Structure.",
    )
    p.add_argument(
        "--mp-cache", default=None,
        help="JSONL with {material_id, structure} per line. Recommended.",
    )
    p.add_argument(
        "--output", required=True,
        help="Directory for the saved ensemble + manifest.",
    )
    args = p.parse_args(argv)

    rows = _load_rows(Path(args.rows_jsonl))
    snapshot = DatasetSnapshot.model_validate_json(
        Path(args.snapshot).read_text()
    )

    from backend.common.ml.models.cgcnn_v2 import CGCNNTrainConfig

    cfg = load_config(args.config) if args.config else CGCNNTrainConfig()
    apply_reproducibility_flags(cfg.seed, deterministic=True)

    if args.structure_source == "materials_project_jsonl":
        resolver = _build_mp_resolver(
            rows, Path(args.mp_cache) if args.mp_cache else None,
        )
    else:
        raise SystemExit(
            f"unknown --structure-source {args.structure_source}"
        )

    dm = CGCNNDataModule(
        rows=rows,
        snapshot=snapshot,
        resolver=resolver,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        cutoff_angstrom=cfg.cutoff_angstrom,
    )
    dm.setup("fit")

    module_kwargs = dict(
        node_feat_dim=cfg.node_feat_dim,
        edge_feat_dim=cfg.edge_feat_dim,
        hidden_dim=cfg.hidden_dim,
        n_conv=cfg.n_conv,
        mlp_dims=tuple(cfg.mlp_dims),
        dropout=cfg.dropout,
        loss_kind=cfg.loss_kind,
        huber_delta=cfg.huber_delta,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    trainer_kwargs = dict(
        accelerator=cfg.accelerator,
        precision=cfg.precision,
        **cfg.extra_trainer_kwargs,
    )

    models = train_ensemble(
        dm=dm, n_models=cfg.n_ensemble,
        module_kwargs=module_kwargs,
        max_epochs=cfg.max_epochs,
        base_seed=cfg.seed,
        trainer_kwargs=trainer_kwargs,
        deterministic=True,
    )

    # Evaluate on test split.
    mu, sigma, y_true = predict_with_uncertainty(models, dm.test_dataloader())
    mae = float(np.abs(mu - y_true).mean())
    rmse = float(np.sqrt(((mu - y_true) ** 2).mean()))
    from scipy.stats import spearmanr

    rho, _ = spearmanr(np.abs(mu - y_true), sigma)
    spear_sigma = float(rho) if np.isfinite(rho) else 0.0

    out_dir = save_ensemble(
        models, args.output,
        dataset_hash=snapshot.content_hash,
        dataset_name=snapshot.config.name,
        extra={
            "config": cfg.model_dump(),
            "test_metrics": {"mae": mae, "rmse": rmse},
            "sigma_abs_err_spearman": spear_sigma,
            "library_versions": library_versions(),
        },
    )
    print(f"ensemble saved to {out_dir}")
    print(f"test MAE = {mae:.3f} eV  RMSE = {rmse:.3f} eV")
    print(f"σ vs |err| Spearman = {spear_sigma:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
