"""Helper: merge new CalibrationResult JSON into the canonical fixture.

Read a per-run JSON dump (as emitted by `orion_calibrate.py --skip-db`)
from stdin, merge the entries into
`tests/fixtures/calibration/pbe_sssp_efficiency_1.3.0.json`
(keyed by element — replaces existing rows with the same element),
and write it back. Idempotent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


_TARGET = (
    Path(__file__).resolve().parent.parent
    / "tests" / "fixtures" / "calibration"
    / "pbe_sssp_efficiency_1.3.0.json"
)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: _merge_calibration_json.py <new_json>")
        return 2
    new_path = Path(argv[1])
    new = json.loads(new_path.read_text())
    if not isinstance(new, list):
        print("[err] new JSON must be a list of CalibrationResult dicts")
        return 1

    existing = []
    if _TARGET.is_file():
        existing = json.loads(_TARGET.read_text())

    by_element = {r["element"]: r for r in existing}
    for entry in new:
        by_element[entry["element"]] = entry

    merged = [by_element[e] for e in sorted(by_element)]
    _TARGET.parent.mkdir(parents=True, exist_ok=True)
    _TARGET.write_text(json.dumps(merged, indent=2))
    elements = [r["element"] for r in merged]
    print(f"[ok] wrote {_TARGET}")
    print(f"     elements: {elements}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
