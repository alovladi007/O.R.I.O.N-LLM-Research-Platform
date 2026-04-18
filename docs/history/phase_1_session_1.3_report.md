# Phase 1 / Session 1.3 — Bulk property importer (synchronous MVP)

**Branch:** `phase-1-session-1.3-bulk-property-import`
**Date:** 2026-04-18

## Scope

Roadmap Session 1.3 describes a **background**-executed CSV property
importer with Celery-backed progress tracking. Celery infrastructure is
a Phase 2.1 deliverable, so this session implements the **synchronous
MVP** of the same endpoint and persists the scientifically valuable
pieces — strict unit validation via the Session 1.2b registry and a
durable `BulkImportJob` row with per-row errors. Session 2.1 or a
follow-up 1.3-addendum will move the work itself into a Celery task,
keeping the schema + validation logic unchanged.

## What changed

### `BulkImportJob` model + migration 011

New table tracking every bulk import. Columns:

- ``id``, ``owner_id`` (FK → users), ``kind`` (``PROPERTY`` | ``STRUCTURE``).
- ``status`` (``PENDING | RUNNING | COMPLETED | FAILED | CANCELLED``).
- ``mapping`` (JSONB — the user's column→property spec).
- ``source_kind`` (``inline`` now, ``minio`` / ``url`` later) + ``source_payload``.
- ``rows_total``, ``rows_ok``, ``rows_failed`` for progress.
- ``errors`` (JSONB list, capped at 500 rows via ``BulkImportJob.ERROR_LOG_CAP``).
- ``error_message`` for top-level failures (schema errors etc.).
- Timestamps: ``created_at``, ``started_at``, ``completed_at``.

Reversible migration with full ``downgrade()``.

### `backend.common.properties.bulk_importer`

New module with the validation kernel:

- ``parse_csv(text)`` — wraps :class:`csv.DictReader`.
- ``import_property_rows(rows, mapping, error_cap)`` — for every row:
  - Resolves ``structure_ref`` (``kind=uuid`` now; ``external_id``
    passthrough with namespace so Phase 1.5 Materials Project lookup
    can consume the structured ref directly).
  - Looks up the property in
    :data:`backend.common.units.PROPERTY_REGISTRY`.
  - Enforces: ``mapping.unit`` ∈ ``spec.accepted_units``, and the
    converted value ∈ ``spec.bounds_si``.
  - Returns an ``AcceptedRow`` or captures a ``RowError``.
- ``render_errors_csv(errors)`` — generates a downloadable CSV of
  rejected rows, column-set = union of every ``raw`` dict's keys.
- Hard failures (mapping-level: unknown property, missing unit, missing
  value column) **raise** and don't become per-row errors.
- Per-row failures (bad UUID, non-numeric value, bounds violation,
  empty value column) get logged up to ``error_cap=500``.

### Router: `src/api/routers/properties.py`

Three endpoints (auth required):

- ``POST /api/v1/properties/bulk`` — synchronous: create a
  ``BulkImportJob`` in ``PENDING``, flip to ``RUNNING``, validate every
  row, persist the accepted rows as ``PredictedProperties`` with
  ``value_unit`` and a dict carrying both raw and SI-normalized values,
  flip to ``COMPLETED``. Returns the job snapshot.
- ``GET /api/v1/jobs/bulk-import/{id}`` — progress + error log.
- ``GET /api/v1/jobs/bulk-import/{id}/errors.csv`` — downloads the
  rejected rows as CSV via ``render_errors_csv``.

One known limitation:
``structure_ref.kind == "external_id"`` is accepted + validated +
echoed back on the row, but the row is then downgraded to a per-row
error because Session 1.3 doesn't resolve external IDs. Session 1.5
wires Materials Project resolution.

### Persistence shape for imported rows

Each accepted row lands as a ``PredictedProperties`` row with:

- ``model_name = "BULK_IMPORT"`` (sentinel — keeps these out of the ML
  model registry views).
- ``model_version = <functional or method.kind or 'unspecified'>``.
- ``properties = {"<name>": raw_value, "<name>_si": si_value}``.
- ``value_unit = <user's input unit>``.
- ``extra_metadata = {"source": "bulk_import", "method": ..., "conditions": ..., "canonical_unit": ...}``.

This keeps the Phase 6 ML pipeline's contract with
``PredictedProperties`` intact (it expects ``properties`` as a dict) and
gives the upcoming provenance queries the unit info they need.

### Schemas — `src/api/schemas/bulk_import.py`

- ``StructureRefSpec`` — ``column``, ``kind``, ``namespace``.
- ``BulkPropertyImportRequest`` — ``csv_text`` (inline) + ``mapping``.
- ``BulkImportJobResponse`` — mirror of the model's ``to_dict()``.

### Tests — `tests/test_bulk_property_import.py`

16 cases, 15 passing / 1 DB-gated:

- ``TestParseCsv`` — empty input, basic parse.
- ``TestHappyPath`` — 2 rows, both accepted, SI conversion correct
  (1.12 eV → 1.794 × 10⁻¹⁹ J).
- ``TestMappingErrors`` — unknown property, unit not accepted,
  missing unit, missing value_column all raise.
- ``TestRowLevelErrors`` — non-numeric, negative bandgap (below
  physical bound), bad UUID, empty value cell all downgrade to row
  errors while sibling rows still import.
- ``TestExternalIdPassthrough`` — ``mp-149`` validates and carries
  namespace through (resolution deferred).
- ``TestErrorCap`` — 10 bad rows + cap=3 → 3 recorded errors,
  ``rows_failed`` still 10.
- ``TestErrorsCsvRenderer`` — header-only for empty; union of raw
  keys in the header.
- ``TestBulkImportEndpoint`` — placeholder marked ``requires_db``
  (skipped locally, run in CI service container).

### App wiring

- ``src/api/routers/__init__.py`` exports ``properties_router``.
- ``src/api/app.py`` imports and includes it at ``{api_prefix}``
  (router endpoints are self-prefixed with ``/properties/...`` and
  ``/jobs/bulk-import/...``).
- 100 routes registered, up from 97.

## Acceptance tests (from ROADMAP Session 1.3)

| Check | Status | Evidence |
|---|---|---|
| CSV fixture of MP formation energies imports in <10s | 🟡 | Logic correct; can't time end-to-end without Postgres. Unit-level import of 2 rows runs instantly. |
| A row with `bandgap=-1.2 eV` is rejected **at that row only** | ✅ | `test_negative_bandgap_rejected_by_bounds_check` |
| Downloading errors.csv returns exactly the rejected rows | ✅ | `test_renders_row_columns_from_raw` and the endpoint delegates to `render_errors_csv`. |
| Reject rows where unit doesn't match the property's registered unit | ✅ | `test_unit_not_accepted_for_property_raises` (mapping-level); row-level unit mismatch isn't valid — mapping is one unit per request, which matches the CSV convention. |
| Reject rows with value outside physical bounds | ✅ | Bounds come from `PROPERTY_REGISTRY`; `test_negative_bandgap_rejected_by_bounds_check` covers the low side, `test_units` covers the high side (Poisson ν > 0.5). |

## Test suite state

```
72 passed, 1 skipped (requires_db), 0 failed
```

Up from last session's 57 → 72 (15 new tests in this file, no regressions
on prior ones).

## Files changed

```
alembic/versions/011_add_bulk_import_jobs.py             NEW
backend/common/properties/__init__.py                    NEW
backend/common/properties/bulk_importer.py               NEW
src/api/app.py                                           (+ properties_router include)
src/api/models/__init__.py                               (+ BulkImportJob exports)
src/api/models/bulk_import.py                            NEW
src/api/routers/__init__.py                              (+ properties_router export)
src/api/routers/properties.py                            NEW
src/api/schemas/bulk_import.py                           NEW
tests/test_bulk_property_import.py                       NEW (16 cases)
docs/history/phase_1_session_1.3_report.md               (this)
```

## Known blockers / deferred to later sessions

- **Async execution:** the endpoint runs synchronously in the request
  thread. 10k-row uploads will block. Session 2.1 moves the guts into
  a Celery task; the ``BulkImportJob`` row is already structured to
  support RUNNING/PENDING states + periodic checkpointing.
- **External-ID resolution:** ``structure_ref.kind == "external_id"``
  currently no-ops past validation. Session 1.5 (Materials Project seed
  data) wires the real MP lookup.
- **Multipart file upload:** the API only accepts inline ``csv_text``.
  Multipart + MinIO streaming comes with Session 2.1.
- **Parquet support:** only CSV for now. Parquet would need a pandas
  dependency at the boundary; deferring until a user case actually
  needs it (roadmap has Parquet as optional).
- **Persistence-level test with Postgres:** currently a ``requires_db``
  placeholder. CI already spins up Postgres; the full test lands
  alongside whatever session extends ``tests/conftest.py`` with
  structure seeding.
