"""
Bulk property importer.

Responsibilities
----------------
1. Parse a CSV body into rows (``parse_csv``).
2. For each row:
   - Resolve the structure reference (existing ORION UUID for now;
     Materials Project lookup wired in Phase 1.5 when MP_API_KEY is set).
   - Read the property name, raw value, and unit.
   - Validate against :data:`backend.common.units.PROPERTY_REGISTRY`:
     * unit must be one of ``accepted_units`` for the property
     * after converting to the canonical (SI / SI-derived) unit, the
       value must be inside ``bounds_si``.
   - Yield a ``(ok, ...)`` / ``(error, RowError)`` tuple.
3. Return a :class:`BulkImportResult` summarizing counts + per-row errors.

Scope boundary: this module does the **validation and normalization**.
DB persistence is the router's job in Session 1.3 (synchronous MVP);
Session 2.1 moves it into a Celery task but the validation logic stays
here.
"""

from __future__ import annotations

import csv
import io
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from backend.common.units import (
    PROPERTY_REGISTRY,
    lookup_property,
    normalize_to_canonical,
)

# Max per-row errors retained in the result object; mirrors
# BulkImportJob.ERROR_LOG_CAP so the two don't drift.
DEFAULT_ERROR_CAP: int = 500


@dataclass(frozen=True)
class RowError:
    """One rejected row with the reason + echo of the raw CSV dict."""

    row: int  # 1-indexed within the data rows (ignores header)
    reason: str
    raw: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {"row": self.row, "reason": self.reason, "raw": self.raw}


@dataclass
class AcceptedRow:
    """A validated, SI-normalized row ready for persistence."""

    row: int
    structure_ref: Dict[str, Any]
    property_name: str
    value_raw: float
    value_si: float
    value_canonical_unit: str
    input_unit: str
    method: Dict[str, Any]
    conditions: Dict[str, Any]


@dataclass
class BulkImportResult:
    """Aggregate result of a bulk import call."""

    rows_total: int
    rows_ok: int
    rows_failed: int
    accepted: List[AcceptedRow] = field(default_factory=list)
    errors: List[RowError] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "rows_total": self.rows_total,
            "rows_ok": self.rows_ok,
            "rows_failed": self.rows_failed,
            "errors": [e.as_dict() for e in self.errors[:DEFAULT_ERROR_CAP]],
        }


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def parse_csv(text: str) -> List[Dict[str, str]]:
    """Parse *text* as CSV into a list of per-row dicts.

    Uses :class:`csv.DictReader` with default dialect detection. Empty
    fields stay as empty strings (not ``None``); the importer treats
    them explicitly.
    """
    if not text.strip():
        return []
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader]


# ---------------------------------------------------------------------------
# Validation + normalization
# ---------------------------------------------------------------------------


def _resolve_structure_ref(ref: Dict[str, Any], row: Dict[str, str]) -> Dict[str, Any]:
    """
    Turn a mapping spec entry + a CSV row into a concrete structure
    reference dict of the form::

        {"kind": "uuid", "id": "..."}           # ORION internal
        {"kind": "external_id",
         "namespace": "materials_project",
         "external_id": "mp-1234"}              # resolved later

    Raises :class:`ValueError` on missing columns — caught by the caller.
    """
    column = ref.get("column")
    if not column or column not in row:
        raise ValueError(f"structure_ref column {column!r} missing from CSV row")

    kind = ref.get("kind", "uuid")
    value = (row[column] or "").strip()
    if not value:
        raise ValueError(f"structure_ref value in column {column!r} is empty")

    if kind == "uuid":
        try:
            uuid.UUID(value)
        except ValueError as exc:
            raise ValueError(f"{value!r} is not a valid UUID") from exc
        return {"kind": "uuid", "id": value}

    if kind == "external_id":
        namespace = ref.get("namespace") or "unknown"
        return {"kind": "external_id", "namespace": namespace, "external_id": value}

    raise ValueError(f"unsupported structure_ref.kind: {kind!r}")


def _parse_value(row: Dict[str, str], column: str) -> float:
    raw = (row.get(column) or "").strip()
    if not raw:
        raise ValueError(f"value column {column!r} is empty")
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{raw!r} is not a number (column {column!r})") from exc


def import_property_rows(
    *,
    rows: Iterable[Dict[str, str]],
    mapping: Dict[str, Any],
    error_cap: int = DEFAULT_ERROR_CAP,
) -> BulkImportResult:
    """
    Validate and normalize every row against the property registry.

    *mapping* shape matches the `BulkPropertyImportRequest` schema:

        {
          "structure_ref": {"column": "mp_id", "kind": "external_id",
                            "namespace": "materials_project"},
          "property": "bandgap",
          "value_column": "bandgap_ev",
          "unit": "eV",
          "method": {"kind": "DFT", "functional": "PBE"},
          "conditions": {"temperature_K": 0, "pressure_GPa": 0}
        }
    """
    prop_name = mapping.get("property")
    if not prop_name:
        raise ValueError("mapping.property is required")
    try:
        spec = lookup_property(prop_name)
    except KeyError as exc:
        raise ValueError(
            f"unknown property {prop_name!r}; known: "
            f"{sorted(PROPERTY_REGISTRY)}"
        ) from exc

    input_unit = mapping.get("unit")
    if not input_unit:
        raise ValueError("mapping.unit is required")
    if input_unit not in spec.accepted_units:
        raise ValueError(
            f"unit {input_unit!r} is not accepted for {prop_name!r}; "
            f"accepted: {sorted(spec.accepted_units)}"
        )

    structure_ref_spec = mapping.get("structure_ref") or {}
    value_column = mapping.get("value_column")
    if not value_column:
        raise ValueError("mapping.value_column is required")

    method = mapping.get("method") or {}
    conditions = mapping.get("conditions") or {}

    result = BulkImportResult(rows_total=0, rows_ok=0, rows_failed=0)

    for idx, row in enumerate(rows, start=1):
        result.rows_total += 1
        try:
            structure_ref = _resolve_structure_ref(structure_ref_spec, row)
            value_raw = _parse_value(row, value_column)
            value_canonical = normalize_to_canonical(prop_name, value_raw, input_unit)
            # Compute SI-base value explicitly for storage / downstream use;
            # canonical_unit is already SI for most registry entries.
            from pint import DimensionalityError
            from backend.common.units import to_si

            try:
                value_si = to_si(value_raw, input_unit)
            except DimensionalityError:
                value_si = value_canonical

            accepted = AcceptedRow(
                row=idx,
                structure_ref=structure_ref,
                property_name=prop_name,
                value_raw=value_raw,
                value_si=value_si,
                value_canonical_unit=spec.canonical_unit,
                input_unit=input_unit,
                method=method,
                conditions=conditions,
            )
            result.accepted.append(accepted)
            result.rows_ok += 1
        except Exception as exc:  # noqa: BLE001 — per-row failures are expected
            result.rows_failed += 1
            if len(result.errors) < error_cap:
                result.errors.append(
                    RowError(row=idx, reason=str(exc), raw=dict(row)),
                )

    return result


# ---------------------------------------------------------------------------
# CSV rendering for the error download endpoint.
# ---------------------------------------------------------------------------


def render_errors_csv(errors: List[Dict[str, Any]]) -> str:
    """
    Render the stored errors list back to a CSV the user can download.

    Columns: ``row,error,<original CSV columns...>`` — the union of keys
    seen across all error.raw dicts, so callers don't have to carry the
    original header around.
    """
    if not errors:
        return "row,error\n"

    buf = io.StringIO()
    raw_columns: List[str] = []
    seen: set = set()
    for e in errors:
        for k in (e.get("raw") or {}).keys():
            if k not in seen:
                raw_columns.append(k)
                seen.add(k)

    header = ["row", "error", *raw_columns]
    writer = csv.writer(buf)
    writer.writerow(header)
    for e in errors:
        raw = e.get("raw") or {}
        writer.writerow(
            [e.get("row"), e.get("reason"), *(raw.get(c, "") for c in raw_columns)]
        )
    return buf.getvalue()
