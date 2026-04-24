#!/usr/bin/env python3
"""Pull oxide bandgaps from the Materials Project into a PropertyRow JSONL.

Output format matches what ``scripts/orion_dataset.py create --rows-jsonl``
expects, so the complete pipeline is::

    export MP_API_KEY=<your key>  # next-gen.materialsproject.org/api

    python scripts/orion_fetch_mp_oxides.py \\
        --output data/oxides_raw.jsonl \\
        --max-rows 2000

    python scripts/orion_dataset.py create \\
        --name oxides_gap_v1 \\
        --rows-jsonl data/oxides_raw.jsonl \\
        --filter "property=bandgap_ev AND method.functional='PBE'" \\
        --split stratified_by_prototype \\
        --seed 42 \\
        --output snapshots/oxides_gap_v1.json

    python scripts/orion_train_baseline.py \\
        --rows-jsonl data/oxides_raw.jsonl \\
        --snapshot snapshots/oxides_gap_v1.json \\
        --feature composition \\
        --models mean,random_forest,xgboost \\
        --artifacts-dir runs/oxides_gap_v1

Filter defaults (overridable via CLI flags):

- ``--elements O`` — structures that contain oxygen.
- ``--min-elements 2 --max-elements 3`` — binaries + ternaries.
- ``--band-gap-min 0.01 --band-gap-max 6.0`` — exclude metals and
  above-UV outliers; both reflect PBE-GGA's reliability window.
- ``--energy-above-hull-max 0.05`` — keep thermodynamically-plausible
  structures only. MP's ``energy_above_hull`` is in eV/atom; 50 meV
  is the conventional "accessible phase" cutoff.

The script is deliberately defensive about the MP API — it pages 500
rows at a time, swallows per-document parse errors with a warning
(don't let one bad row tank a 2000-row pull), and prints progress so
long pulls don't look like hangs. The output is written atomically
via a ``.tmp`` → ``.rename()`` so a Ctrl-C mid-dump doesn't leave a
half-file for the dataset CLI to trip over.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(message)s",
    )


def _doc_to_property_row(doc: Any) -> Optional[Dict[str, Any]]:
    """Convert an ``MPSummaryDoc`` to a ``PropertyRow``-shaped dict.

    Returns ``None`` on any expected-but-missing field so the caller
    can skip the row rather than crash the run. Unexpected errors
    propagate.
    """
    mat_id = getattr(doc, "material_id", None)
    gap = getattr(doc, "band_gap", None)
    formula = getattr(doc, "formula_pretty", None)
    symmetry = getattr(doc, "symmetry", None)
    if mat_id is None or gap is None or formula is None:
        return None

    crystal_system = None
    if symmetry is not None:
        # symmetry can be an MPSymmetryDoc or a dict depending on API
        # version; be tolerant.
        cs = getattr(symmetry, "crystal_system", None)
        if cs is None and isinstance(symmetry, dict):
            cs = symmetry.get("crystal_system")
        if cs is not None:
            crystal_system = str(cs).lower()

    return {
        "row_id": str(mat_id),
        "structure_id": str(mat_id),
        "structure_hash": None,
        "property_name": "bandgap_ev",
        "property_value": float(gap),
        "method": {
            "functional": "PBE",
            "source": "materials_project",
            "mp_release": os.environ.get("MP_DB_VERSION", "current"),
        },
        "conditions": {},
        "composition_formula": str(formula),
        "composition_prototype": crystal_system,
        "extras": {
            "material_id": str(mat_id),
        },
    }


def fetch_oxide_bandgaps(
    api_key: Optional[str],
    *,
    elements: List[str],
    min_elements: int,
    max_elements: int,
    band_gap_min: float,
    band_gap_max: float,
    energy_above_hull_max: float,
    max_rows: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
    """Stream ``PropertyRow`` dicts from an MP summary search.

    The mp_api Python client handles paging internally; we iterate
    its results and yield converted rows. ``max_rows`` caps the
    output — set it small for dry runs, leave it ``None`` for a
    full pull.
    """
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise SystemExit(
            "mp_api isn't installed. Install the MP-compatible versions "
            "matching the current Python 3.10 pin:\n"
            "    pip install 'mp-api<0.45' 'emmet-core<0.85'\n"
            "See the comment block in requirements.txt for why these are "
            "not a hard dep.\n"
            f"Original error: {exc}"
        )

    # MPRester(None) picks up MP_API_KEY from the environment. We
    # pass the key explicitly when --api-key is provided so ad-hoc
    # CLI use works without `export`.
    with MPRester(api_key) as mpr:
        logger.info(
            "MP summary search: elements=%s num_elements=[%d,%d] "
            "band_gap=[%.2f,%.2f] e_above_hull<=%.3f eV/atom",
            elements, min_elements, max_elements,
            band_gap_min, band_gap_max, energy_above_hull_max,
        )
        docs = mpr.materials.summary.search(
            elements=elements,
            num_elements=(min_elements, max_elements),
            band_gap=(band_gap_min, band_gap_max),
            energy_above_hull=(0.0, energy_above_hull_max),
            fields=[
                "material_id",
                "formula_pretty",
                "band_gap",
                "symmetry",
                "energy_above_hull",
            ],
        )
        logger.info("retrieved %d documents from MP", len(docs))

        emitted = 0
        skipped = 0
        for doc in docs:
            try:
                row = _doc_to_property_row(doc)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "skipping document %s: %s",
                    getattr(doc, "material_id", "?"), exc,
                )
                skipped += 1
                continue
            if row is None:
                skipped += 1
                continue
            yield row
            emitted += 1
            if max_rows is not None and emitted >= max_rows:
                logger.info("reached --max-rows=%d; stopping", max_rows)
                break
        logger.info("emitted=%d skipped=%d", emitted, skipped)


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> int:
    """Atomic JSONL writer — writes to ``path.tmp`` then renames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            count += 1
            if count % 500 == 0:
                logger.info("wrote %d rows so far", count)
    tmp.replace(path)
    return count


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="orion fetch-mp-oxides",
        description="Fetch oxide bandgap rows from the Materials Project",
    )
    p.add_argument(
        "--output", required=True,
        help="Path to write the JSONL (e.g. data/oxides_raw.jsonl).",
    )
    p.add_argument(
        "--api-key", default=None,
        help="MP API key. Falls back to $MP_API_KEY env var.",
    )
    p.add_argument(
        "--elements", nargs="+", default=["O"],
        help="Elements that must be present in each structure.",
    )
    p.add_argument("--min-elements", type=int, default=2)
    p.add_argument("--max-elements", type=int, default=3)
    p.add_argument("--band-gap-min", type=float, default=0.01,
                   help="eV; excludes metals at the low end.")
    p.add_argument("--band-gap-max", type=float, default=6.0,
                   help="eV; excludes above-UV outliers where PBE is "
                        "unreliable.")
    p.add_argument("--energy-above-hull-max", type=float, default=0.05,
                   help="eV/atom; keeps thermodynamically-plausible "
                        "structures only.")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap the output row count (useful for dry runs).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)

    api_key = args.api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        print(
            "error: no MP API key. Get one from "
            "https://next-gen.materialsproject.org/api and export it:\n"
            "    export MP_API_KEY=<your-key>\n"
            "or pass --api-key.",
            file=sys.stderr,
        )
        return 2

    try:
        rows_iter = fetch_oxide_bandgaps(
            api_key=api_key,
            elements=args.elements,
            min_elements=args.min_elements,
            max_elements=args.max_elements,
            band_gap_min=args.band_gap_min,
            band_gap_max=args.band_gap_max,
            energy_above_hull_max=args.energy_above_hull_max,
            max_rows=args.max_rows,
        )
        count = write_jsonl(rows_iter, Path(args.output))
    except Exception as exc:  # noqa: BLE001
        logger.error("fetch failed: %s", exc, exc_info=args.verbose)
        return 1

    print(f"wrote {count} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
