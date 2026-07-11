#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${OUT:-$ROOT/outputs/data/legal-performance-v1}"
PYTHON="${PYTHON:-$ROOT/.venv-train/bin/python}"
mkdir -p "$OUT/candidates"

extract() {
  local source="$1"
  local rows="$2"
  local output="$OUT/candidates/$source.jsonl"
  local manifest="$OUT/candidates/$source.manifest.json"
  [[ -s "$output" && -s "$manifest" ]] && return 0
  "$PYTHON" scripts/prepare_legal_embedding_data.py \
    --config configs/legal_data_sources_v1.json \
    --source "$source" \
    --max-records "$rows" \
    --output "$output" \
    --manifest "$manifest"
}

extract legalize_kr_statutes 50000
extract legalize_kr_administrative_rules 50000
extract legalize_kr_precedents 50000
extract legalize_kr_ordinances 100000

inputs=()
for path in "$OUT"/candidates/*.jsonl; do
  inputs+=(--input "$path")
done

"$PYTHON" scripts/compile_source_native_pairs.py \
  "${inputs[@]}" \
  --output "$OUT/train.bootstrap.jsonl" \
  --provenance-output "$OUT/provenance.jsonl" \
  --manifest-output "$OUT/manifest.json" \
  --max-rows 250000 \
  --negatives-per-row 1 \
  --seed 42

"$PYTHON" scripts/validate_embedding_jsonl.py "$OUT/train.bootstrap.jsonl"
