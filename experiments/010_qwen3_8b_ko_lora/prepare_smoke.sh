#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/data/processed/ko_triplet_smoke}"

if [[ -f "$ROOT/.env" ]]; then
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "$ROOT/.env" | tail -n 1)"
  export HF_TOKEN
fi

"$ROOT/.venv/bin/python" "$ROOT/scripts/prepare_ko_triplet.py" \
  --output-dir "$OUTPUT_DIR" \
  --limit "${LIMIT:-288}" \
  --val-size "${VAL_SIZE:-32}" \
  --seed "${SEED:-42}"

"$ROOT/.venv/bin/python" "$ROOT/scripts/validate_embedding_jsonl.py" \
  "$OUTPUT_DIR/train.jsonl" \
  "$OUTPUT_DIR/validation.jsonl"
