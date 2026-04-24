#!/usr/bin/env python3
"""Session 6.2 CLI: create / list / show versioned datasets.

Usage
-----

::

    orion_dataset.py create \\
        --name oxides_gap_v1 \\
        --rows-jsonl data/property_rows.jsonl \\
        --filter "property=bandgap_ev AND method.functional='PBE'" \\
        --split random \\
        --seed 42 \\
        --output out/snapshots/oxides_gap_v1.json

    orion_dataset.py show out/snapshots/oxides_gap_v1.json

Input rows are supplied as JSONL — one :class:`PropertyRow` per line
with the schema you get from ``PropertyRow(...).model_dump()``. This
keeps the CLI decoupled from any specific storage backend (DB query
result, pandas DataFrame, MP bulk import output, etc.).

The roadmap's acceptance call is just the ``create`` path with a
seed; the JSONL format is the obvious neutral serialization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Put repo root on sys.path so `import backend.common...` resolves
# when the script is run directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.common.ml.datasets_v2 import (  # noqa: E402
    DatasetConfig, DatasetRegistry, DatasetSnapshot, FilterSpec,
    PropertyRow, SplitSpec,
)


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
                raise SystemExit(
                    f"{path}:{lineno}: failed to parse PropertyRow: {exc}"
                )
    return rows


def _save_snapshot(snap: DatasetSnapshot, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snap.model_dump_json(indent=2))


def _print_snapshot(snap: DatasetSnapshot) -> None:
    print(f"{snap.config.name}:{snap.config.version}  [{snap.short_hash}]")
    print(f"  created: {snap.created_at.isoformat()}")
    print(f"  filter:  {snap.config.filter.expression or '(none)'}")
    print(
        f"  split:   {snap.config.split.kind} "
        f"train={snap.config.split.train_fraction} "
        f"val={snap.config.split.val_fraction} "
        f"test={snap.config.split.test_fraction} "
        f"seed={snap.config.split.seed}"
    )
    print(
        f"  counts:  train={snap.n_train} val={snap.n_val} test={snap.n_test} "
        f"total={snap.n_total}"
    )


def cmd_create(args: argparse.Namespace) -> int:
    rows = _load_rows(Path(args.rows_jsonl))
    if not rows:
        print(f"error: no rows in {args.rows_jsonl}", file=sys.stderr)
        return 2

    config = DatasetConfig(
        name=args.name,
        version=args.version,
        description=args.description,
        filter=FilterSpec(expression=args.filter),
        split=SplitSpec(
            kind=args.split,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            cluster_n_centers=args.cluster_n_centers,
        ),
    )

    reg = DatasetRegistry()
    fingerprint_fn = None
    # SplitSpec normalizes the kind, but the CLI branch below reads the
    # raw ``args.split`` value so we also accept the hyphen form.
    if args.split in ("structure_cluster", "structure-cluster"):
        # CLI users who want a structure_cluster split must ship
        # fingerprints on the PropertyRow's ``extras`` dict under the
        # key ``fingerprint`` (list[float]). This avoids the CLI
        # spinning up matminer on every invocation — which costs 1-2s
        # of Magpie data load.
        def fingerprint_fn(row):
            import numpy as np
            fp = (row.extras or {}).get("fingerprint")
            if fp is None:
                raise SystemExit(
                    f"structure_cluster requires row.extras['fingerprint']; "
                    f"row {row.row_id!r} has none"
                )
            return np.asarray(fp, dtype=np.float64)

    snap = reg.create(rows=rows, config=config, fingerprint_fn=fingerprint_fn)
    _print_snapshot(snap)
    if args.output:
        _save_snapshot(snap, Path(args.output))
        print(f"  wrote:   {args.output}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    path = Path(args.path)
    snap = DatasetSnapshot.model_validate_json(path.read_text())
    _print_snapshot(snap)
    if args.list_rows:
        print("  train row ids:")
        for rid in snap.train_row_ids:
            print(f"    {rid}")
        print("  val row ids:")
        for rid in snap.val_row_ids:
            print(f"    {rid}")
        print("  test row ids:")
        for rid in snap.test_row_ids:
            print(f"    {rid}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="orion dataset",
        description="Session 6.2 dataset registry CLI",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create", help="Materialize a new dataset snapshot")
    c.add_argument("--name", required=True)
    c.add_argument("--version", default="v1")
    c.add_argument("--description", default="")
    c.add_argument("--rows-jsonl", required=True,
                   help="Path to JSONL file, one PropertyRow per line")
    c.add_argument("--filter", default="",
                   help="Filter expression, e.g. \"property=bandgap_ev "
                        "AND method.functional='PBE'\"")
    # Accept both hyphen + underscore forms to match the roadmap's CLI
    # example (``--split structure-cluster``). SplitSpec's field
    # validator normalizes them to the canonical underscore kind.
    c.add_argument("--split", default="random",
                   choices=("random",
                            "stratified_by_prototype", "stratified-by-prototype",
                            "structure_cluster", "structure-cluster"))
    c.add_argument("--train-fraction", type=float, default=0.7)
    c.add_argument("--val-fraction", type=float, default=0.15)
    c.add_argument("--test-fraction", type=float, default=0.15)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--cluster-n-centers", type=int, default=None,
                   help="structure_cluster-only: override default "
                        "(ceil(n * test_fraction)) center count.")
    c.add_argument("--output",
                   help="Where to write the snapshot JSON "
                        "(optional; omit to only print).")
    c.set_defaults(func=cmd_create)

    s = sub.add_parser("show", help="Load a snapshot JSON and print it")
    s.add_argument("path")
    s.add_argument("--list-rows", action="store_true")
    s.set_defaults(func=cmd_show)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
