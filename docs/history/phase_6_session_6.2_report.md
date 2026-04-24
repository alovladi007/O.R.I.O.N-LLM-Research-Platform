# Phase 6 / Session 6.2 — Dataset registry + splits

**Branch:** `main`
**Date:** 2026-04-24

## Headline

Both roadmap acceptance targets hit. New
`backend.common.ml.datasets_v2` package ships a versioned dataset
registry with a pydantic schema, a small filter DSL, three live split
strategies + one deferred stub, content-hash-identity snapshots, and
a JSONL-fed CLI.

- **Same seed ⇒ same hash.** The content-hash identity property holds
  across runs: `reg.create(rows, config, seed=42)` run twice against
  the same corpus returns snapshots with identical `content_hash`.
- **Structure-cluster split forces extrapolation.** KS test of
  test-set-nearest-train vs train-internal-nearest-neighbour distance
  distributions gives p = 3.9 × 10⁻⁵ < 0.01 on the 240-row uniform-
  hypercube fixture.

Tests: 508 → 536 passing (+28 new), 6 infra/live skips unchanged.

## What shipped

### `backend/common/ml/datasets_v2/`

```
datasets_v2/
  __init__.py      # public API
  schema.py        # PropertyRow, FilterSpec, SplitSpec, DatasetConfig, DatasetSnapshot
  filter.py        # compile_filter / apply_filter + tokenizer + predicate builder
  splits.py        # random / stratified_by_prototype / structure_cluster / scaffold (deferred)
  registry.py      # DatasetRegistry (in-memory store with create/get/list + hash identity)
```

Sibling of the legacy Session-14 `backend.common.ml.datasets` (kept
for the existing CGCNN training path until Session 6.4 migrates it).

### Public API

- `PropertyRow(row_id, structure_id, property_name, property_value,
  method, conditions, composition_formula, composition_prototype,
  extras)` — the input unit. Deliberately **not** tied to the
  Postgres `predicted_properties` table shape — the registry
  consumes any iterable of rows, so callers can build them from a DB
  query, a pandas DataFrame, a Materials Project JSONL dump, etc.
- `FilterSpec(expression="...")` — tiny predicate DSL:
  - Atoms: `property=bandgap_ev`, `method.functional='PBE'`,
    `composition_prototype in 'rocksalt','wurtzite'`,
    `property_value>=0`.
  - Chains: `A AND B AND C` (OR / NOT deferred to 6.2b).
  - Dotted subkey access for `method` and `conditions` JSONB-shaped
    fields.
- `SplitSpec(kind, train_fraction, val_fraction, test_fraction, seed,
  cluster_n_centers)` — fractions must sum to 1.0 (± 1 e-6); we
  reject other totals rather than silently renormalizing, because
  silent renormalization breaks the same-seed-same-hash acceptance.
- `DatasetConfig(name, version, filter, split, description)` — the
  full recipe.
- `DatasetSnapshot.compute_hash()` / `.freeze()` — deterministic
  SHA-256 over config + sorted row-id lists. `short_hash` is the
  first 16 chars (log-friendly).
- `DatasetRegistry.create(rows, config, fingerprint_fn=None)` —
  filter → split → freeze → cache-keyed-by-`(name, version)`. Fresh
  call with the same config returns the cached snapshot (not a
  re-computed one), but a call with a changed corpus raises
  `DatasetRegistryError("already exists with a different content
  hash")` — you force a version bump.

### Split strategies

- **`random_split`** — uniform random with seeded RNG. Row order at
  the input is normalized (sorted by `row_id`) before the shuffle so
  caller insertion order doesn't drift the result.
- **`stratified_by_prototype_split`** — partition within each
  `composition_prototype` bucket, concatenate. Buckets < 3 rows go
  entirely to train (the alternative is a bucket contributing
  nothing to val/test, which isn't useful). Falls back to
  `composition_formula` when prototype is missing, and to a single
  `__UNSTRATIFIED__` bucket when both are missing.
- **`structure_cluster_split`** — greedy-farthest-point over caller-
  supplied fingerprints. Each iteration picks the row whose
  minimum-distance to any already-selected-test-row is largest. This
  pulls test rows to the extremes of the fingerprint cloud; training
  inherits the interior. The design guarantees the "forces
  extrapolation test" property the roadmap asks for — test rows are
  systematically far from training.
- **`scaffold_split`** — deferred to Session 6.2b. Requires RDKit
  (Murcko scaffold + BM hash); raises `PendingAnalyzerError` with the
  6.2b tracker hint. Same honest-deferral pattern as Sessions 4.3a /
  5.3 / 6.1.

### Content hash

SHA-256 over, in order:

1. `config.name`
2. `config.version`
3. `config.filter.expression` (verbatim)
4. `config.split.{kind, train_fraction, val_fraction, test_fraction,
   seed, cluster_n_centers}`
5. Sorted train row ids
6. Sorted val row ids
7. Sorted test row ids

Excluded: `created_at` (timestamps shouldn't invalidate identity) and
`description` (mutable metadata).

### CLI — `scripts/orion_dataset.py`

```
orion_dataset.py create \
    --name oxides_gap_v1 \
    --rows-jsonl path/to/rows.jsonl \
    --filter "property=bandgap_ev AND method.functional='PBE'" \
    --split stratified_by_prototype \
    --seed 42 \
    --output snapshots/oxides_gap_v1.json

orion_dataset.py show snapshots/oxides_gap_v1.json [--list-rows]
```

Rows are provided as JSONL (one `PropertyRow.model_dump()` per line).
This keeps the CLI decoupled from any single storage backend — DB
dump, MP bulk-import output, pandas DataFrame `.to_json(orient='records',
lines=True)`, all compatible.

For `--split structure_cluster`, the CLI reads the fingerprint from
`row.extras['fingerprint']` (list[float]). That avoids the CLI
spinning up matminer's 1-2 s Magpie data load on every invocation.

## Scope + honest framing

### Pure-Python engine, DB wiring deferred

Session 6.2 ships the **engine**. A DB-backed registry variant (an
`orion_datasets` table + an Alembic migration) is scoped for Session
6.2b or 6.3, whichever needs it first. Pattern matches Session 4.3a
(scaffold ships, physics wiring follows) and 5.3 (scaffold ships,
cross-scale coupling follows).

Why: there's no `observed_properties` table today (the bulk-importer
writes property CSVs into `PredictedProperties`, which is
ML-prediction-shaped, not ground-truth-shaped). Spending a session
designing + migrating a ground-truth table just to feed one dataset
test wasn't the best use of time; shipping a storage-agnostic engine
that consumes any iterable of `PropertyRow`s keeps 6.3 (baselines)
unblocked.

### OR / NOT filter operators deferred

The DSL is intentionally AND-only. Corpora at this scale (~hundreds
to low-thousands of rows) fit in memory, so a post-filter in Python
is easy when OR / NOT is needed. Adding them is a 6.2b follow-up
when a real user hits the limit.

### Scaffold (organic) split deferred

RDKit isn't in `requirements.txt`. The function exists; raises
`PendingAnalyzerError` with tracker = Session 6.2b. Phase 8
(materials workflows) is all inorganic; scaffold split gets
prioritized once a molecular session exists.

## Acceptance results (roadmap)

| Test | Target | Result |
|---|---|---|
| Same seed + same corpus → same content hash | exact equality | ✅ (3 tests lock it in) |
| Structure-cluster KS test | p < 0.01 | **p = 3.9 × 10⁻⁵** (240-row uniform hypercube, seed=42) |

## Known gaps / followups

### 1. Session 6.2b — DB-backed registry

Add an `orion_datasets` table + Alembic migration + an endpoint
(`POST /api/v1/datasets`) that wraps `DatasetRegistry.create`. Also
move the in-process `_snapshots` dict into a SQL store so snapshots
survive process restarts. Shape is in place — it's CRUD plumbing on
top of the existing schema.

### 2. RDKit + scaffold split

Install RDKit, implement `scaffold_split` with Murcko + BM hash. A
molecular session under Phase 6 or Phase 8 will motivate it.

### 3. OR / NOT / parentheses in filter

The tokenizer and parser have room for it; a couple hours in 6.2b.

### 4. Fingerprint builder integration

Session 6.1's `features_v2.PCAEmbedder` produces 256-d structure
embeddings ideal as the structure-cluster fingerprint. The CLI
currently reads precomputed fingerprints from `row.extras`; a
follow-up wires `PCAEmbedder` directly into the split call so users
don't need to precompute.

### 5. Persisted snapshots on disk

`DatasetSnapshot.model_dump_json()` works; the `show` CLI subcommand
already round-trips through it. A `--persist` flag on `create` that
writes to a configurable directory (plus a matching `load` CLI)
would close the "how do I keep a dataset between sessions" loop.
Deferred because Python notebook callers pickle `snap` directly and
the test suite uses `tmp_path`.

## Dependencies

No new deps. `numpy` and `scipy.stats.ks_2samp` (already pinned via
scipy) cover everything. `matminer` isn't a dependency of
`datasets_v2` itself — callers can plug any `PropertyRow → ndarray`
callable as the `fingerprint_fn`.

## Tests

- `tests/test_datasets_v2.py` — **28 new tests**:
  - `TestFilter` (6) — AND chains, `in` lists, numeric comparisons,
    dotted subkey, unknown-field + malformed-token error paths.
  - `TestRandomSplit` (4) — reproducibility by seed, seed changes
    split, covers all rows, fractions-sum-to-1 enforcement.
  - `TestStratifiedSplit` (2) — tiny-bucket fallback, formula
    fallback when prototype missing.
  - `TestStructureClusterSplit` (3) — KS test acceptance (p < 0.01),
    reproducible-by-seed, missing-fingerprint-fn raises.
  - `TestScaffoldSplitDeferred` (1) — `PendingAnalyzerError` with
    "6.2b" tracker.
  - `TestRegistry` (9) — same-seed-same-hash acceptance, different-
    seed-different-hash, corpus-change-changes-hash, cache-hit on
    duplicate create, hash-mismatch guard, empty-corpus error,
    filter-matches-zero error, `rows_for_subset`, `get`/`list`.
  - `TestSnapshotHashStability` (2) — hash stable across row
    insertion order, description doesn't affect hash.
- Full suite: **508 → 536 passed, 6 skipped** (infra/live gates
  unchanged).

## Phase 6 status

6.1 (featurizers) + 6.2 (dataset registry) done. Next per roadmap:
**Session 6.3 — Baseline models** (MeanRegressor, RandomForest on
Magpie, XGBoost on composition+structure), with MLflow for metric
tracking and MinIO for model-artifact storage. Depends on
`datasets_v2` + `features_v2` and introduces `mlflow` as a new dep.
