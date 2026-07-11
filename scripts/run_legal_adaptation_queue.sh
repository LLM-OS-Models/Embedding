#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WAIT_PID="${WAIT_PID:-}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/legal-adaptation-20260711}"
DATA_DIR="$ROOT/outputs/data/legal-performance-v1"
BOOTSTRAP="$DATA_DIR/train.bootstrap.jsonl"
MINED="$DATA_DIR/train.faiss-r095-n7.jsonl"
AUDIT="$DATA_DIR/train.faiss-r095-n7.audit.jsonl"
MINING_MANIFEST="$DATA_DIR/train.faiss-r095-n7.manifest.json"
MINED_PROVENANCE="$DATA_DIR/provenance.faiss-r095-n7.jsonl"
ORDERED="$DATA_DIR/train.faiss-r095-n7.homogeneous-b16.jsonl"
ORDERED_PROVENANCE="$DATA_DIR/provenance.faiss-r095-n7.homogeneous-b16.jsonl"
ORDERED_MANIFEST="$DATA_DIR/faiss-r095-n7.homogeneous-b16.manifest.json"
GENERAL_DIR="$ROOT/outputs/data/performance-v1/performance-1m"
GENERAL_TRAIN="$GENERAL_DIR/train.homogeneous-b16.jsonl"
GENERAL_PROVENANCE="$GENERAL_DIR/provenance.homogeneous-b16.jsonl"
CURRICULUM="$DATA_DIR/train.faiss-r095-n7.legal25-replay75.jsonl"
CURRICULUM_PROVENANCE="$DATA_DIR/provenance.faiss-r095-n7.legal25-replay75.jsonl"
CURRICULUM_MANIFEST="$DATA_DIR/faiss-r095-n7.legal25-replay75.manifest.json"
VAL_FILE="$ROOT/data/processed/ko_triplet_pilot_10k/validation.hn-qwen3-r095-n4.jsonl"
RUN_NAME="qwen3-embedding-8b-ko-legal25-replay75-lora-r64"
MODEL_REL="artifacts/models/${RUN_NAME}-best-merged"
MODEL_DIR="$ROOT/$MODEL_REL"
SIONIC_OUT="$ROOT/outputs/evaluation/sionic9-legal250k"
OFFICIAL_OUT="$ROOT/outputs/evaluation/mteb-korean-v1-legal250k"
mkdir -p "$LOG_DIR" "$SIONIC_OUT" "$OFFICIAL_OUT"
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1

if [[ -f "$ROOT/.env" ]]; then
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "$ROOT/.env" | tail -n 1)"
  export HF_TOKEN
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$ROOT/scripts:$ROOT/third_party/mteb${PYTHONPATH:+:$PYTHONPATH}"
if "$ROOT/.venv-train/bin/python" -c 'import flash_attn' >/dev/null 2>&1; then
  export ATTN_IMPL=flash_attention_2
else
  export ATTN_IMPL=sdpa
fi

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
run_stage() {
  local name="$1"; shift
  echo "[$(timestamp)] START $name"
  "$@"
  local status=$?
  echo "[$(timestamp)] END $name status=$status"
  return "$status"
}

if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi
if [[ -s "$GENERAL_DIR/faiss-current-r095-n7.homogeneous-b16.manifest.json" \
    && -s "$GENERAL_DIR/train.faiss-current-r095-n7.homogeneous-b16.jsonl" \
    && -s "$GENERAL_DIR/provenance.faiss-current-r095-n7.homogeneous-b16.jsonl" ]]; then
  GENERAL_TRAIN="$GENERAL_DIR/train.faiss-current-r095-n7.homogeneous-b16.jsonl"
  GENERAL_PROVENANCE="$GENERAL_DIR/provenance.faiss-current-r095-n7.homogeneous-b16.jsonl"
fi
[[ -s "$BOOTSTRAP" && -s "$DATA_DIR/provenance.jsonl" && -s "$VAL_FILE" \
  && -s "$GENERAL_TRAIN" && -s "$GENERAL_PROVENANCE" ]] || exit 2

CONTINUAL_BASE="$ROOT/artifacts/models/qwen3-embedding-8b-ko-performance1m-lora-r64-best-merged"
if [[ ! -s "$CONTINUAL_BASE/merge_report.json" ]]; then
  CONTINUAL_BASE="$ROOT/artifacts/models/qwen3-embedding-8b-ko-performance1m-lora-r64-b8-best-merged"
fi
if [[ -s "$CONTINUAL_BASE/merge_report.json" ]]; then
  MINING_MODEL="$CONTINUAL_BASE"
  MINING_REVISION=""
  echo "[$(timestamp)] continuing from 1M model: $CONTINUAL_BASE"
else
  MINING_MODEL="Qwen/Qwen3-Embedding-8B"
  MINING_REVISION="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
  echo "[$(timestamp)] 1M merged model unavailable; using pinned Qwen base"
fi

if ! "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/check_mining_manifest.py" \
    --manifest "$MINING_MANIFEST" --model "$MINING_MODEL" \
    --revision "$MINING_REVISION" 2>/dev/null; then
  rm -f "$MINING_MANIFEST" "$MINED" "$AUDIT" "$MINED_PROVENANCE" \
    "$ORDERED" "$ORDERED_PROVENANCE" "$ORDERED_MANIFEST" \
    "$CURRICULUM" "$CURRICULUM_PROVENANCE" "$CURRICULUM_MANIFEST"
  run_stage legal-faiss-hard-negative-mining \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/mine_faiss_hard_negatives.py" \
    --input "$BOOTSTRAP" --output "$MINED" \
    --audit-output "$AUDIT" --manifest-output "$MINING_MANIFEST" \
    --work-dir "$DATA_DIR/faiss-work-current-student" \
    --model "$MINING_MODEL" --revision "$MINING_REVISION" \
    --encode-batch-size 128 --candidate-pool-size 24 --search-k 256 \
    --num-negatives 7 --positive-relative-ratio .95 \
    --nlist 512 --nprobe 32 --training-points 50000 \
    --keep-work-dir --allow-target-adapted || exit 3
fi

if [[ ! -s "$MINED_PROVENANCE" ]]; then
  run_stage project-legal-provenance \
    "$ROOT/.venv-train/bin/python" "$ROOT/scripts/project_mined_provenance.py" \
    --input-provenance "$DATA_DIR/provenance.jsonl" --mining-audit "$AUDIT" \
    --output "$MINED_PROVENANCE" \
    --manifest-output "$DATA_DIR/provenance.faiss-r095-n7.manifest.json" || exit 4
fi

if [[ ! -s "$ORDERED_MANIFEST" \
    || "$(jq -r '.length_bucketed // false' "$ORDERED_MANIFEST")" != true ]]; then
  run_stage order-legal-homogeneous-batches \
    "$ROOT/.venv-train/bin/python" "$ROOT/scripts/build_homogeneous_batches.py" \
    --train "$MINED" --provenance "$MINED_PROVENANCE" \
    --output "$ORDERED" --provenance-output "$ORDERED_PROVENANCE" \
    --manifest-output "$ORDERED_MANIFEST" --batch-size 16 --seed 42 \
    --length-bucketed || exit 5
fi

if [[ ! -s "$CURRICULUM_MANIFEST" ]]; then
  LEGAL_ROWS="$(jq -r '.output_rows' "$ORDERED_MANIFEST")"
  REPLAY_ROWS="$((LEGAL_ROWS * 3))"
  run_stage build-legal25-general75-curriculum \
    "$ROOT/.venv-train/bin/python" "$ROOT/scripts/build_replay_curriculum.py" \
    --primary-train "$ORDERED" --primary-provenance "$ORDERED_PROVENANCE" \
    --primary-rows "$LEGAL_ROWS" \
    --replay-train "$GENERAL_TRAIN" --replay-provenance "$GENERAL_PROVENANCE" \
    --replay-rows "$REPLAY_ROWS" --output "$CURRICULUM" \
    --provenance-output "$CURRICULUM_PROVENANCE" \
    --manifest-output "$CURRICULUM_MANIFEST" --batch-size 16 --seed 42 \
    --adaptation-label target-adapted-legal25-general75 || exit 6
fi

MAX_STEPS="$(jq -r '.output_rows / 64 | floor' "$CURRICULUM_MANIFEST")"
if (( MAX_STEPS < 1 )); then
  echo "[$(timestamp)] no complete legal training steps" >&2
  exit 6
fi
run_stage validate-legal-mined-data \
  "$ROOT/.venv-train/bin/python" "$ROOT/scripts/validate_embedding_jsonl.py" \
  "$CURRICULUM" "$VAL_FILE" || exit 6
train_legal() {
  local output_name="$1" batch="$2" accum="$3"
  run_stage "train-$output_name" env \
    RUN_NAME="$output_name" TRAIN_FILE="$CURRICULUM" VAL_FILE="$VAL_FILE" \
    MAX_STEPS="$MAX_STEPS" EVAL_STEPS=250 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=3 \
    TRAIN_BATCH_SIZE="$batch" GRAD_ACCUM_STEPS="$accum" \
    TRAIN_DATALOADER_SHUFFLE=false LEARNING_RATE=1e-5 \
    INFONCE_HARD_NEGATIVES=7 \
    BASE_MODEL="$MINING_MODEL" BASE_REVISION="$MINING_REVISION" \
    "$ROOT/experiments/020_hard_negative/train_pilot_lora_r64.sh"
}

primary_status=0
if ! find "$ROOT/outputs/$RUN_NAME" -maxdepth 3 -type d -name "checkpoint-$MAX_STEPS" -print -quit 2>/dev/null | grep -q .; then
  train_legal "$RUN_NAME" 8 8 || primary_status=$?
fi
checkpoint=""
if (( primary_status == 0 )); then
  checkpoint="$($ROOT/.venv-train/bin/python "$ROOT/scripts/select_best_checkpoint.py" "$ROOT/outputs/$RUN_NAME" --print-path 2>/dev/null)" || checkpoint=""
fi
if [[ -z "$checkpoint" ]]; then
  fallback="${RUN_NAME}-b4"
  train_legal "$fallback" 4 16 || exit 6
  RUN_NAME="$fallback"
  MODEL_REL="artifacts/models/${RUN_NAME}-best-merged"
  MODEL_DIR="$ROOT/$MODEL_REL"
  checkpoint="$($ROOT/.venv-train/bin/python "$ROOT/scripts/select_best_checkpoint.py" "$ROOT/outputs/$RUN_NAME" --print-path)" || exit 6
fi

run_stage verify-legal-adapter \
  "$ROOT/.venv-train/bin/python" "$ROOT/scripts/verify_adapter.py" \
  --adapter "$checkpoint" --data "$VAL_FILE" --model "$MINING_MODEL" \
  --output "$LOG_DIR/verification.json" || exit 7
if [[ ! -s "$MODEL_DIR/merge_report.json" ]]; then
  run_stage merge-legal-adapter \
    "$ROOT/.venv-train/bin/python" "$ROOT/scripts/merge_embedding_adapter.py" \
    --adapter "$checkpoint" --output-dir "$MODEL_DIR" \
    --base-model "$MINING_MODEL" --base-revision "$MINING_REVISION" \
    --device cuda --dtype bfloat16 --local-files-only || exit 8
fi

adapter_sha="$(jq -r '.adapter.weights_sha256' "$MODEL_DIR/merge_report.json")"
revision="adapter-${adapter_sha:0:12}"
run_stage sionic9-legal-target-adapted \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_sionic9.py" \
  --model "$MODEL_REL" --revision "$revision" --batch-size 192 --max-length 8192 \
  --attn-implementation flash_attention_2 --output-dir "$SIONIC_OUT" \
  --embedding-cache-dir "$ROOT/outputs/embedding-cache/sionic9-legal250k"

safe="${MODEL_REL//\//__}"
SIONIC_SUMMARY="$SIONIC_OUT/$safe/summary.json"
run_stage official-korean-legal-target-adapted \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_mteb_korean_v1.py" \
  --model "$MODEL_REL" --revision "$revision" --max-length 8192 \
  --batch-size 192 --attn-implementation flash_attention_2 \
  --output-dir "$OFFICIAL_OUT" \
  --embedding-cache-dir "$ROOT/outputs/embedding-cache/official-legal250k"

OFFICIAL_SUMMARY="$OFFICIAL_OUT/$safe/$revision/summary.json"
if [[ -s "$SIONIC_SUMMARY" && -s "$OFFICIAL_SUMMARY" ]]; then
  if run_stage publish-legal-target-adapted \
    "$ROOT/.venv-train/bin/python" "$ROOT/scripts/publish_best_embedding_model.py" \
    --model-dir "$MODEL_DIR" --sionic-summary "$SIONIC_SUMMARY" \
    --official-summary "$OFFICIAL_SUMMARY" --training-manifest "$CURRICULUM_MANIFEST" \
    --repo-id LLM-OS-Models/qwen3-embedding-8b-ko-legal-target-adapted-v1 \
    --upload --public; then
    run_stage record-legal-replay-result \
      "$ROOT/scripts/commit_campaign_result.sh" \
      --stage legal-replay --model "$MODEL_REL" \
      --repo-id LLM-OS-Models/qwen3-embedding-8b-ko-legal-target-adapted-v1 \
      --sionic-summary "$SIONIC_SUMMARY" --official-summary "$OFFICIAL_SUMMARY"
  fi
fi

echo "[$(timestamp)] legal target-adaptation queue complete"
