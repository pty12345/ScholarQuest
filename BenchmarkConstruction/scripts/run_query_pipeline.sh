#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

SEED_OUTPUT="${SEED_OUTPUT:-$ROOT_DIR/data/seeds/topic_seeds.jsonl}"
RELEASE_OUTPUT="${RELEASE_OUTPUT:-$ROOT_DIR/data/release/query_pool_v0.1.jsonl}"
SEED_INPUT="${SEED_INPUT:-}"

if [[ -n "$SEED_INPUT" ]]; then
  echo "[1/3] Using external seed file: $SEED_INPUT"
else
  echo "[1/3] Generating topic seeds..."
  "$PYTHON_BIN" "$ROOT_DIR/scripts/generate_topic_seeds.py" --output "$SEED_OUTPUT"
fi

echo "[2/3] Running query pipeline..."
if [[ -n "$SEED_INPUT" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/run_pipeline.py" --root "$ROOT_DIR" --seed-file "$SEED_INPUT"
else
  "$PYTHON_BIN" "$ROOT_DIR/scripts/run_pipeline.py" --root "$ROOT_DIR"
fi

echo "[3/3] Inspecting release distribution..."
"$PYTHON_BIN" "$ROOT_DIR/scripts/inspect_distribution.py" --input "$RELEASE_OUTPUT"

echo "Query pipeline run completed."
