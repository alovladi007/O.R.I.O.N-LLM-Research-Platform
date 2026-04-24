#!/usr/bin/env bash
# Build the oxides_gap_v1 dataset from the Materials Project + train
# all three Session-6.3 baselines on it. End-to-end replacement for
# the synthetic-fixture acceptance path; see
# docs/history/phase_6_session_6.3_report.md "Honest framing" for
# why 6.3 shipped with synthetic data.
#
# Prereq: $MP_API_KEY from https://next-gen.materialsproject.org/api
#         $OMP_NUM_THREADS=1 so xgboost 2.x doesn't deadlock on macOS
#         (see backend/common/ml/models/baselines.py class comment).
#
# Optional: edit MAX_ROWS to cap the MP pull during dev (the full
#           oxides set is ~50k rows; the roadmap target is ~1000-2000).
#
# Output:
#   data/oxides_raw.jsonl          — raw MP rows in PropertyRow JSONL
#   snapshots/oxides_gap_v1.json   — DatasetSnapshot with hash + splits
#   runs/oxides_gap_v1/*.pkl       — pickled fitted models
#   mlruns/                        — MLflow file-store tracking
#
# Typical wall time:
#   MP fetch      ~30s for 2000 rows, ~2min for 10000
#   Dataset split <1s
#   Mean + RF     <5s
#   XGBoost       ~30-60s (single-threaded per the libomp pin)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Guard rails.
: "${MP_API_KEY:?set MP_API_KEY — get one from https://next-gen.materialsproject.org/api}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

MAX_ROWS="${MAX_ROWS:-2000}"

DATA=data/oxides_raw.jsonl
SNAP=snapshots/oxides_gap_v1.json
RUNS=runs/oxides_gap_v1

echo "→ fetching MP oxide bandgaps (max $MAX_ROWS rows)…"
python scripts/orion_fetch_mp_oxides.py \
    --output "$DATA" \
    --max-rows "$MAX_ROWS" \
    --band-gap-min 0.01 \
    --band-gap-max 6.0 \
    --energy-above-hull-max 0.05

echo "→ materializing oxides_gap_v1 snapshot…"
python scripts/orion_dataset.py create \
    --name oxides_gap_v1 \
    --rows-jsonl "$DATA" \
    --filter "property=bandgap_ev AND method.functional='PBE'" \
    --split stratified_by_prototype \
    --seed 42 \
    --output "$SNAP"

echo "→ training baselines…"
python scripts/orion_train_baseline.py \
    --rows-jsonl "$DATA" \
    --snapshot "$SNAP" \
    --feature composition \
    --models mean,random_forest,xgboost \
    --artifacts-dir "$RUNS" \
    --write-registry-json "$RUNS/registry.json"

echo
echo "Done. Inspect:"
echo "  python scripts/orion_dataset.py show $SNAP"
echo "  mlflow ui                           # http://localhost:5000"
echo "  ls $RUNS/*.pkl"
