#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WAIT_PID="${WAIT_PID:-}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/scale-1m-20260711}"
DATA_DIR="$ROOT/outputs/data/performance-v1/performance-1m"
TRAIN_FILE="$DATA_DIR/train.jsonl"
DATA_MANIFEST="$DATA_DIR/manifest.json"
HOMOGENEOUS_TRAIN="$DATA_DIR/train.homogeneous-b16.jsonl"
HOMOGENEOUS_PROVENANCE="$DATA_DIR/provenance.homogeneous-b16.jsonl"
HOMOGENEOUS_MANIFEST="$DATA_DIR/homogeneous-b16.manifest.json"
MINED_TRAIN="$DATA_DIR/train.faiss-current-r095-n7.jsonl"
MINING_AUDIT="$DATA_DIR/train.faiss-current-r095-n7.audit.jsonl"
MINING_MANIFEST="$DATA_DIR/train.faiss-current-r095-n7.manifest.json"
MINED_PROVENANCE="$DATA_DIR/provenance.faiss-current-r095-n7.jsonl"
MINED_HOMOGENEOUS_TRAIN="$DATA_DIR/train.faiss-current-r095-n7.homogeneous-b16.jsonl"
MINED_HOMOGENEOUS_PROVENANCE="$DATA_DIR/provenance.faiss-current-r095-n7.homogeneous-b16.jsonl"
MINED_HOMOGENEOUS_MANIFEST="$DATA_DIR/faiss-current-r095-n7.homogeneous-b16.manifest.json"
VAL_FILE="$ROOT/data/processed/ko_triplet_pilot_10k/validation.hn-qwen3-r095-n4.jsonl"
RUN_NAME="qwen3-embedding-8b-ko-performance1m-lora-r64"
MODEL_REL="artifacts/models/${RUN_NAME}-best-merged"
MODEL_DIR="$ROOT/$MODEL_REL"
SIONIC_OUT="$ROOT/outputs/evaluation/sionic9-scale1m"
OFFICIAL_OUT="$ROOT/outputs/evaluation/mteb-korean-v1-scale1m"
POSTTRAIN_SELECTION="$ROOT/outputs/post-training-eval-20260711/sionic9-selection.json"
mkdir -p "$LOG_DIR" "$SIONIC_OUT" "$OFFICIAL_OUT"
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1

if [[ -f "$ROOT/.env" ]]; then
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "$ROOT/.env" | tail -n 1)"
  export HF_TOKEN
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$ROOT/third_party/mteb${PYTHONPATH:+:$PYTHONPATH}"
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
  echo "[$(timestamp)] waiting for post-training evaluation pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi

if [[ ! -s "$DATA_MANIFEST" || "$(jq -r '.phase + ":" + (.built_rows|tostring)' "$DATA_MANIFEST" 2>/dev/null)" != "performance_1m:1000000" ]]; then
  run_stage "build-performance-1m" \
    "$ROOT/.venv-train/bin/python" "$ROOT/scripts/build_performance_mix.py" \
    --phase performance_1m --output-dir "$DATA_DIR" || exit 2
fi
if [[ ! -s "$VAL_FILE" ]]; then
  echo "[$(timestamp)] missing mined validation data: $VAL_FILE" >&2
  exit 2
fi

CONTINUAL_BASE="Qwen/Qwen3-Embedding-8B"
CONTINUAL_REVISION="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
if [[ -s "$POSTTRAIN_SELECTION" ]]; then
  selected_rel="$(jq -r '.best.model // empty' "$POSTTRAIN_SELECTION")"
  selected_abs="$ROOT/$selected_rel"
  if [[ -n "$selected_rel" && ( -s "$selected_abs/merge_report.json" \
      || -s "$selected_abs/full_tuning_report.json" ) ]]; then
    CONTINUAL_BASE="$selected_abs"
    CONTINUAL_REVISION=""
    echo "[$(timestamp)] continuing 1M curriculum from post-training winner: $selected_abs"
  fi
fi
if [[ "$CONTINUAL_BASE" == Qwen/Qwen3-Embedding-8B ]]; then
  echo "[$(timestamp)] post-training winner unavailable; using pinned Qwen base"
fi

TRAINING_MANIFEST="$DATA_MANIFEST"
if [[ ! -s "$HOMOGENEOUS_MANIFEST" ]]; then
  run_stage "build-homogeneous-1m-batches" \
    "$ROOT/.venv-train/bin/python" "$ROOT/scripts/build_homogeneous_batches.py" \
    --train "$TRAIN_FILE" --provenance "$DATA_DIR/provenance.jsonl" \
    --output "$HOMOGENEOUS_TRAIN" \
    --provenance-output "$HOMOGENEOUS_PROVENANCE" \
    --manifest-output "$HOMOGENEOUS_MANIFEST" \
    --batch-size 16 --seed 42 || exit 2
fi
TRAIN_FILE="$HOMOGENEOUS_TRAIN"
TRAINING_MANIFEST="$HOMOGENEOUS_MANIFEST"
TRAIN_HARD_NEGATIVES=4

if [[ "${ENABLE_SCALE_HARD_NEGATIVE_MINING:-1}" == 1 ]]; then
  if [[ ! -s "$MINING_MANIFEST" ]]; then
    run_stage "mine-performance-1m-current-student" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/mine_faiss_hard_negatives.py" \
      --input "$DATA_DIR/train.jsonl" --output "$MINED_TRAIN" \
      --audit-output "$MINING_AUDIT" --manifest-output "$MINING_MANIFEST" \
      --work-dir "$DATA_DIR/faiss-work-current-student" --keep-work-dir \
      --model "$CONTINUAL_BASE" --revision "$CONTINUAL_REVISION" \
      --encode-batch-size 128 --candidate-pool-size 24 --search-k 256 \
      --num-negatives 7 --positive-relative-ratio .95 \
      --nlist 4096 --nprobe 32 --training-points 200000 \
      --allow-target-adapted || true
  fi
  if [[ -s "$MINING_MANIFEST" && ! -s "$MINED_PROVENANCE" ]]; then
    run_stage "project-performance-1m-mined-provenance" \
      "$ROOT/.venv-train/bin/python" "$ROOT/scripts/project_mined_provenance.py" \
      --input-provenance "$DATA_DIR/provenance.jsonl" \
      --mining-audit "$MINING_AUDIT" --output "$MINED_PROVENANCE" \
      --manifest-output "$DATA_DIR/provenance.faiss-current-r095-n7.manifest.json" || true
  fi
  if [[ -s "$MINED_TRAIN" && -s "$MINED_PROVENANCE" \
      && ! -s "$MINED_HOMOGENEOUS_MANIFEST" ]]; then
    run_stage "order-performance-1m-mined-batches" \
      "$ROOT/.venv-train/bin/python" "$ROOT/scripts/build_homogeneous_batches.py" \
      --train "$MINED_TRAIN" --provenance "$MINED_PROVENANCE" \
      --output "$MINED_HOMOGENEOUS_TRAIN" \
      --provenance-output "$MINED_HOMOGENEOUS_PROVENANCE" \
      --manifest-output "$MINED_HOMOGENEOUS_MANIFEST" --batch-size 16 --seed 42 \
      --benchmark-adaptation target-adapted-performance1m-current-student || true
  fi
  if [[ -s "$MINED_HOMOGENEOUS_MANIFEST" && -s "$MINED_HOMOGENEOUS_TRAIN" ]]; then
    TRAIN_FILE="$MINED_HOMOGENEOUS_TRAIN"
    TRAINING_MANIFEST="$MINED_HOMOGENEOUS_MANIFEST"
    TRAIN_HARD_NEGATIVES=7
    echo "[$(timestamp)] using current-student mined 1M curriculum"
  else
    echo "[$(timestamp)] current-student mining incomplete; using original homogeneous 1M"
  fi
fi

MAX_STEPS_1M="$(jq -r '.output_rows / 128 | floor' "$TRAINING_MANIFEST")"

train_scale() {
  local output_name="$1" batch="$2" accum="$3"
  run_stage "train-$output_name" env \
    RUN_NAME="$output_name" TRAIN_FILE="$TRAIN_FILE" VAL_FILE="$VAL_FILE" \
    MAX_STEPS="$MAX_STEPS_1M" EVAL_STEPS=500 SAVE_STEPS=500 SAVE_TOTAL_LIMIT=3 \
    TRAIN_BATCH_SIZE="$batch" GRAD_ACCUM_STEPS="$accum" \
    TRAIN_DATALOADER_SHUFFLE=false \
    LEARNING_RATE=2e-5 WARMUP_RATIO=.05 \
    INFONCE_HARD_NEGATIVES="$TRAIN_HARD_NEGATIVES" \
    BASE_MODEL="$CONTINUAL_BASE" BASE_REVISION="$CONTINUAL_REVISION" \
    "$ROOT/experiments/020_hard_negative/train_pilot_lora_r64.sh"
}

primary_status=0
if ! find "$ROOT/outputs/$RUN_NAME" -maxdepth 3 -type d -name "checkpoint-$MAX_STEPS_1M" -print -quit 2>/dev/null | grep -q .; then
  train_scale "$RUN_NAME" 16 8 || primary_status=$?
fi
checkpoint=""
if (( primary_status == 0 )); then
  checkpoint="$($ROOT/.venv-train/bin/python "$ROOT/scripts/select_best_checkpoint.py" \
    "$ROOT/outputs/$RUN_NAME" --print-path 2>/dev/null)" || checkpoint=""
fi
if [[ -z "$checkpoint" ]]; then
  fallback="${RUN_NAME}-b8"
  train_scale "$fallback" 8 16 || exit 3
  RUN_NAME="$fallback"
  MODEL_REL="artifacts/models/${RUN_NAME}-best-merged"
  MODEL_DIR="$ROOT/$MODEL_REL"
  checkpoint="$($ROOT/.venv-train/bin/python "$ROOT/scripts/select_best_checkpoint.py" \
    "$ROOT/outputs/$RUN_NAME" --print-path)" || exit 3
fi

run_stage "verify-$RUN_NAME" \
  "$ROOT/.venv-train/bin/python" "$ROOT/scripts/verify_adapter.py" \
  --adapter "$checkpoint" --data "$VAL_FILE" --model "$CONTINUAL_BASE" \
  --output "$LOG_DIR/adapter-verification.json" || exit 4

if [[ ! -s "$MODEL_DIR/merge_report.json" ]]; then
  run_stage "merge-$RUN_NAME" \
    "$ROOT/.venv-train/bin/python" "$ROOT/scripts/merge_embedding_adapter.py" \
    --adapter "$checkpoint" --output-dir "$MODEL_DIR" \
    --base-model "$CONTINUAL_BASE" --base-revision "$CONTINUAL_REVISION" \
    --device cuda --dtype bfloat16 --local-files-only || exit 5
fi

adapter_sha="$(jq -r '.adapter.weights_sha256' "$MODEL_DIR/merge_report.json")"
local_revision="adapter-${adapter_sha:0:12}"
run_stage "sionic9-$RUN_NAME" \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_sionic9.py" \
  --model "$MODEL_REL" --revision "$local_revision" --batch-size 192 --max-length 8192 \
  --attn-implementation flash_attention_2 --output-dir "$SIONIC_OUT" \
  --embedding-cache-dir "$ROOT/outputs/embedding-cache/sionic9-scale1m"

safe="${MODEL_REL//\//__}"
SIONIC_SUMMARY="$SIONIC_OUT/$safe/summary.json"
run_stage "official-korean-$RUN_NAME" \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_mteb_korean_v1.py" \
  --model "$MODEL_REL" --revision "$local_revision" --max-length 8192 \
  --batch-size 192 --attn-implementation flash_attention_2 \
  --output-dir "$OFFICIAL_OUT" \
  --embedding-cache-dir "$ROOT/outputs/embedding-cache/official-scale1m"

OFFICIAL_SUMMARY="$OFFICIAL_OUT/$safe/$local_revision/summary.json"
if [[ -s "$SIONIC_SUMMARY" && -s "$OFFICIAL_SUMMARY" ]]; then
  if run_stage "publish-$RUN_NAME" \
    "$ROOT/.venv-train/bin/python" "$ROOT/scripts/publish_best_embedding_model.py" \
    --model-dir "$MODEL_DIR" \
    --sionic-summary "$SIONIC_SUMMARY" \
    --official-summary "$OFFICIAL_SUMMARY" \
    --training-manifest "$TRAINING_MANIFEST" \
    --repo-id LLM-OS-Models/qwen3-embedding-8b-ko-performance-1m-v1 \
    --upload --public; then
    run_stage "record-scale-1m-result" \
      "$ROOT/scripts/commit_campaign_result.sh" \
      --stage scale-1m --model "$MODEL_REL" \
      --repo-id LLM-OS-Models/qwen3-embedding-8b-ko-performance-1m-v1 \
      --sionic-summary "$SIONIC_SUMMARY" --official-summary "$OFFICIAL_SUMMARY"
  fi
fi

echo "[$(timestamp)] 1M scale queue complete"
