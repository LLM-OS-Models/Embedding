#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_ENV="${PYTHON_ENV:-$ROOT/.venv-mteb}"
INPUT="${INPUT:-$ROOT/data/processed/ko_triplet_smoke/train.jsonl}"
OUTPUT="${OUTPUT:-$ROOT/data/processed/ko_triplet_smoke/train.hn-qwen3-r095-n4.jsonl}"
BASE_REVISION="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"

if [[ -f "$ROOT/.env" ]]; then
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "$ROOT/.env" | tail -n 1)"
  export HF_TOKEN
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EXTRA_ARGS=(--assert-no-benchmark-data)
if [[ "${INCLUDE_SOURCE_NEGATIVES:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--include-source-negatives)
else
  EXTRA_ARGS+=(--no-include-source-negatives)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

"$PYTHON_ENV/bin/python" "$ROOT/scripts/mine_dense_hard_negatives.py" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model Qwen/Qwen3-Embedding-8B \
  --revision "$BASE_REVISION" \
  --device "${DEVICE:-cuda}" \
  --score-device "${SCORE_DEVICE:-cuda}" \
  --model-dtype "${MODEL_DTYPE:-bfloat16}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --max-seq-length "${MAX_SEQ_LENGTH:-512}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-8}" \
  --candidate-pool-size "${CANDIDATE_POOL_SIZE:-24}" \
  --num-negatives "${NUM_NEGATIVES:-4}" \
  --positive-relative-ratio "${POSITIVE_RELATIVE_RATIO:-0.95}" \
  --query-block-size "${QUERY_BLOCK_SIZE:-64}" \
  --corpus-block-size "${CORPUS_BLOCK_SIZE:-2048}" \
  --insufficient-policy "${INSUFFICIENT_POLICY:-drop}" \
  --duplicate-row-policy "${DUPLICATE_ROW_POLICY:-error}" \
  --seed "${SEED:-42}" \
  "${EXTRA_ARGS[@]}"
