#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
source "$ROOT/scripts/backend_admission.sh"
embedding_resolve_train_runtime
UTILITY_PYTHON="$EMBEDDING_TRAIN_PYTHON"
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
MINED_QUALITY_AUDIT="$DATA_DIR/faiss-current-r095-n7.homogeneous-b16.quality-audit.json"
MINED_OVERLAP_AUDIT="$DATA_DIR/faiss-current-r095-n7.homogeneous-b16.benchmark-overlap-audit.json"
BASE_QUALITY_AUDIT="$DATA_DIR/homogeneous-b16.quality-audit.json"
BASE_OVERLAP_AUDIT="$DATA_DIR/homogeneous-b16.benchmark-overlap-audit.json"
BLOCKLIST_ROOT="$ROOT/outputs/decontamination/benchmark_blocklist"
VAL_FILE="$ROOT/data/processed/ko_triplet_pilot_10k/validation.hn-qwen3-r095-n4.jsonl"
RUN_NAME="qwen3-embedding-8b-ko-performance1m-lora-r64"
MODEL_REL="artifacts/models/${RUN_NAME}-best-merged"
MODEL_DIR="$ROOT/$MODEL_REL"
SIONIC_OUT="$ROOT/outputs/evaluation/sionic9-scale1m"
OFFICIAL_OUT="$ROOT/outputs/evaluation/mteb-korean-v1-scale1m"
POSTTRAIN_SELECTION="${POSTTRAIN_SELECTION:-$ROOT/outputs/post-training-eval-20260717-frontier/clean-first-selection.json}"
mkdir -p "$LOG_DIR" "$SIONIC_OUT" "$OFFICIAL_OUT"
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1

if [[ -f "$ROOT/.env" ]]; then
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "$ROOT/.env" | tail -n 1)"
  export HF_TOKEN
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$ROOT/third_party/mteb${PYTHONPATH:+:$PYTHONPATH}"
FAISS_THREADS="${FAISS_THREADS:-$EFFECTIVE_CPU_COUNT}"

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
run_stage() {
  local name="$1"; shift
  echo "[$(timestamp)] START $name"
  "$@"
  local status=$?
  echo "[$(timestamp)] END $name status=$status"
  return "$status"
}

retry_stage() {
  local name="$1" attempts="$2" attempt status=1
  shift 2
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    run_stage "$name-attempt-$attempt" "$@" && return 0
    status=$?
    (( attempt == attempts )) || sleep 15
  done
  return "$status"
}

run_sionic_with_fallback() {
  local model="$1" revision="$2" cache="$3" batch
  for batch in "${CAMPAIGN_EVAL_BATCH_SIZE:-192}"; do
    if run_stage "sionic9-$RUN_NAME-b$batch" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_sionic9.py" \
      --model "$model" --revision "$revision" --batch-size "$batch" --max-length 8192 \
      --attn-implementation flash_attention_2 --output-dir "$SIONIC_OUT" \
      --embedding-cache-dir "$cache"; then
      return 0
    fi
  done
  return 1
}

run_official_with_fallback() {
  local model="$1" revision="$2" cache="$3" batch
  for batch in "${CAMPAIGN_EVAL_BATCH_SIZE:-192}"; do
    if run_stage "official-korean-$RUN_NAME-b$batch" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_mteb_korean_v1.py" \
      --model "$model" --revision "$revision" --max-length 8192 \
      --qwen3-instruction-loader --batch-size "$batch" \
      --attn-implementation flash_attention_2 --output-dir "$OFFICIAL_OUT" \
      --embedding-cache-dir "$cache"; then
      return 0
    fi
  done
  return 1
}

if [[ -n "$WAIT_PID" ]]; then
  echo "[$(timestamp)] waiting for post-training evaluation pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi

if [[ ! -s "$DATA_MANIFEST" || "$(jq -r '.phase + ":" + (.built_rows|tostring)' "$DATA_MANIFEST" 2>/dev/null)" != "performance_1m:1000000" ]]; then
  run_stage "build-performance-1m" \
    "$UTILITY_PYTHON" "$ROOT/scripts/build_performance_mix.py" \
    --phase performance_1m --output-dir "$DATA_DIR" \
    --critical-blocklist-root "$BLOCKLIST_ROOT" || exit 2
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
if [[ ! -s "$HOMOGENEOUS_MANIFEST" \
    || "$(jq -r '.length_bucketed // false' "$HOMOGENEOUS_MANIFEST")" != true ]]; then
  run_stage "build-homogeneous-1m-batches" \
    "$UTILITY_PYTHON" "$ROOT/scripts/build_homogeneous_batches.py" \
    --train "$TRAIN_FILE" --provenance "$DATA_DIR/provenance.jsonl" \
    --output "$HOMOGENEOUS_TRAIN" \
    --provenance-output "$HOMOGENEOUS_PROVENANCE" \
    --manifest-output "$HOMOGENEOUS_MANIFEST" \
    --batch-size 16 --seed 42 --length-bucketed || exit 2
fi
TRAIN_FILE="$HOMOGENEOUS_TRAIN"
TRAINING_MANIFEST="$HOMOGENEOUS_MANIFEST"
TRAIN_HARD_NEGATIVES=4

# The original homogeneous curriculum is the fallback whenever current-student
# mining cannot finish.  Audit that fallback unconditionally so a fresh rebuild
# can never bypass the same 15-task contamination gate as a mined curriculum.
run_stage "audit-performance-1m-base-curriculum" \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/audit_embedding_training_data.py" \
  --train "$HOMOGENEOUS_TRAIN" --provenance "$HOMOGENEOUS_PROVENANCE" \
  --output "$BASE_QUALITY_AUDIT" --expected-batch-size 16 || exit 3
run_stage "audit-performance-1m-base-benchmark-overlap" \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/audit_training_benchmark_overlap.py" \
  --train "$HOMOGENEOUS_TRAIN" --provenance "$HOMOGENEOUS_PROVENANCE" \
  --blocklist-root "$BLOCKLIST_ROOT" --output "$BASE_OVERLAP_AUDIT" \
  --fail-on-critical || exit 3

if [[ "${ENABLE_SCALE_HARD_NEGATIVE_MINING:-1}" == 1 ]]; then
  if ! "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/check_mining_manifest.py" \
      --manifest "$MINING_MANIFEST" --model "$CONTINUAL_BASE" \
      --revision "$CONTINUAL_REVISION" --selection-strategy score_rank_quantiles \
      --candidate-pool-size 24 --num-negatives 7 2>/dev/null; then
    rm -f "$MINING_MANIFEST" "$MINED_TRAIN" "$MINING_AUDIT" \
      "$MINED_PROVENANCE" "$MINED_HOMOGENEOUS_TRAIN" \
      "$MINED_HOMOGENEOUS_PROVENANCE" "$MINED_HOMOGENEOUS_MANIFEST"
    run_stage "mine-performance-1m-current-student" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/mine_faiss_hard_negatives.py" \
      --input "$DATA_DIR/train.jsonl" --output "$MINED_TRAIN" \
      --audit-output "$MINING_AUDIT" --manifest-output "$MINING_MANIFEST" \
      --work-dir "$DATA_DIR/faiss-work-current-student" --keep-work-dir \
      --model "$CONTINUAL_BASE" --revision "$CONTINUAL_REVISION" \
      --encode-batch-size 128 --candidate-pool-size 24 --search-k 256 \
      --num-negatives 7 --selection-strategy score_rank_quantiles \
      --positive-relative-ratio .95 \
      --nlist 1024 --nprobe 32 --training-points 50000 \
      --faiss-threads "$FAISS_THREADS" \
      --allow-target-adapted || true
  fi
  if [[ -s "$MINING_MANIFEST" && ! -s "$MINED_PROVENANCE" ]]; then
    run_stage "project-performance-1m-mined-provenance" \
      "$UTILITY_PYTHON" "$ROOT/scripts/project_mined_provenance.py" \
      --input-provenance "$DATA_DIR/provenance.jsonl" \
      --mining-audit "$MINING_AUDIT" --output "$MINED_PROVENANCE" \
      --manifest-output "$DATA_DIR/provenance.faiss-current-r095-n7.manifest.json" || true
  fi
  if [[ -s "$MINED_TRAIN" && -s "$MINED_PROVENANCE" \
      && ! -s "$MINED_HOMOGENEOUS_MANIFEST" ]]; then
    run_stage "order-performance-1m-mined-batches" \
      "$UTILITY_PYTHON" "$ROOT/scripts/build_homogeneous_batches.py" \
      --train "$MINED_TRAIN" --provenance "$MINED_PROVENANCE" \
      --output "$MINED_HOMOGENEOUS_TRAIN" \
      --provenance-output "$MINED_HOMOGENEOUS_PROVENANCE" \
      --manifest-output "$MINED_HOMOGENEOUS_MANIFEST" --batch-size 16 --seed 42 \
      --length-bucketed \
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

if [[ "$TRAINING_MANIFEST" == "$MINED_HOMOGENEOUS_MANIFEST" ]]; then
  run_stage "audit-performance-1m-mined-curriculum" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/audit_embedding_training_data.py" \
    --train "$MINED_HOMOGENEOUS_TRAIN" \
    --provenance "$MINED_HOMOGENEOUS_PROVENANCE" \
    --output "$MINED_QUALITY_AUDIT" --expected-batch-size 16 || exit 3
  run_stage "audit-performance-1m-mined-benchmark-overlap" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/audit_training_benchmark_overlap.py" \
    --train "$MINED_HOMOGENEOUS_TRAIN" \
    --provenance "$MINED_HOMOGENEOUS_PROVENANCE" \
    --blocklist-root "$BLOCKLIST_ROOT" \
    --output "$MINED_OVERLAP_AUDIT" --fail-on-critical || exit 3
fi

MAX_STEPS_1M="$(jq -r '.output_rows / 128 | floor' "$TRAINING_MANIFEST")"

train_scale() {
  local output_name="$1" batch="$2" accum="$3"
  local train_env="$EMBEDDING_TRAIN_ENV" train_attn=sdpa admission_report
  local admission_key="scale1m-lora-r64-b${batch}-a${accum}-m512-hn${TRAIN_HARD_NEGATIVES}"
  if embedding_select_fa2_backend "$TRAIN_FILE" "$admission_key" \
      "$batch" "$accum" 512 64 128 bfloat16 \
      "$CONTINUAL_BASE" "$CONTINUAL_REVISION" "$TRAIN_HARD_NEGATIVES" .05; then
    train_env="$BACKEND_ADMISSION_ENV"
    train_attn="$BACKEND_ADMISSION_ATTN"
  fi
  admission_report="$BACKEND_ADMISSION_REPORT"
  echo "[$(timestamp)] scale training backend=$train_attn env=$train_env admission=$admission_report"
  run_stage "train-$output_name" env \
    TRAIN_ENV="$train_env" ATTN_IMPL="$train_attn" \
    RUN_NAME="$output_name" TRAIN_FILE="$TRAIN_FILE" VAL_FILE="$VAL_FILE" \
    MAX_STEPS="$MAX_STEPS_1M" EVAL_STEPS=250 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=3 \
    TRAIN_BATCH_SIZE="$batch" GRAD_ACCUM_STEPS="$accum" \
    MAX_LENGTH=512 LORA_RANK=64 LORA_ALPHA=128 LORA_DROPOUT=.05 \
    DATASET_SHUFFLE=false TRAIN_DATALOADER_SHUFFLE=false \
    LEARNING_RATE=1e-5 WARMUP_RATIO=.05 \
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
  checkpoint="$("$UTILITY_PYTHON" "$ROOT/scripts/select_best_checkpoint.py" \
    "$ROOT/outputs/$RUN_NAME" --print-path 2>/dev/null)" || checkpoint=""
fi
if [[ -z "$checkpoint" ]]; then
  fallback="${RUN_NAME}-b8"
  train_scale "$fallback" 8 16 || exit 3
  RUN_NAME="$fallback"
  MODEL_REL="artifacts/models/${RUN_NAME}-best-merged"
  MODEL_DIR="$ROOT/$MODEL_REL"
  checkpoint="$("$UTILITY_PYTHON" "$ROOT/scripts/select_best_checkpoint.py" \
    "$ROOT/outputs/$RUN_NAME" --print-path)" || exit 3
fi

DATA_UPLOAD_PID=""
if [[ "$TRAINING_MANIFEST" == "$MINED_HOMOGENEOUS_MANIFEST" ]]; then
  retry_stage "upload-derived-performance-1m" 3 \
    "$UTILITY_PYTHON" "$ROOT/scripts/publish_derived_training_dataset.py" \
    --train "$MINED_HOMOGENEOUS_TRAIN" \
    --provenance "$MINED_HOMOGENEOUS_PROVENANCE" \
    --manifest "$MINED_HOMOGENEOUS_MANIFEST" \
    --mining-manifest "$MINING_MANIFEST" --mining-audit "$MINING_AUDIT" \
    --quality-audit "$MINED_QUALITY_AUDIT" \
    --benchmark-overlap-audit "$MINED_OVERLAP_AUDIT" \
    --repo-id LLM-OS-Models2/korean-embedding-performance-1m-quantile-hn7-v1 \
    --title "Korean Embedding Performance 1M Quantile HN7" \
    --source-dataset LLM-OS-Models/korean-embedding-performance-v1-performance-1m \
    --upload --public >"$LOG_DIR/derived-dataset-upload.log" 2>&1 &
  DATA_UPLOAD_PID=$!
  echo "[$(timestamp)] derived dataset upload started pid=$DATA_UPLOAD_PID"
fi

run_stage "verify-$RUN_NAME" \
  "$UTILITY_PYTHON" "$ROOT/scripts/verify_adapter.py" \
  --adapter "$checkpoint" --data "$VAL_FILE" --model "$CONTINUAL_BASE" \
  --output "$LOG_DIR/adapter-verification.json" || exit 4

if [[ ! -s "$MODEL_DIR/merge_report.json" ]]; then
  run_stage "merge-$RUN_NAME" \
    "$UTILITY_PYTHON" "$ROOT/scripts/merge_embedding_adapter.py" \
    --adapter "$checkpoint" --output-dir "$MODEL_DIR" \
    --base-model "$CONTINUAL_BASE" --base-revision "$CONTINUAL_REVISION" \
    --device cuda --dtype bfloat16 --local-files-only || exit 5
fi

model_sha="$(jq -r '.model.weights_sha256' "$MODEL_DIR/merge_report.json")"
local_revision="model-${model_sha:0:12}"
run_sionic_with_fallback "$MODEL_REL" "$local_revision" \
  "$ROOT/outputs/embedding-cache/sionic9-scale1m" || true

safe="${MODEL_REL//\//__}"
SIONIC_SUMMARY="$SIONIC_OUT/$safe/summary.json"
run_official_with_fallback "$MODEL_REL" "$local_revision" \
  "$ROOT/outputs/embedding-cache/official-scale1m" || true

OFFICIAL_SUMMARY="$OFFICIAL_OUT/$safe/$local_revision/summary.json"
CLEAN_OUT="$ROOT/outputs/evaluation/legal-source-heldout"
for batch in "${CAMPAIGN_EVAL_BATCH_SIZE:-192}"; do
  run_stage "clean-legal-$RUN_NAME-b$batch" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_legal_source_holdout.py" \
    --model "$MODEL_REL" --revision "$local_revision" --batch-size "$batch" \
    --max-length 8192 --attn-implementation flash_attention_2 \
    --output-dir "$CLEAN_OUT" \
    --embedding-cache-dir "$ROOT/outputs/embedding-cache/legal-source-heldout" && break
done
CLEAN_SUMMARY="$CLEAN_OUT/$safe/$local_revision/summary.json"
ROBUST_OUT="$ROOT/outputs/evaluation/conversational-noise-robustness"
for batch in "${CAMPAIGN_EVAL_BATCH_SIZE:-192}"; do
  run_stage "robustness-$RUN_NAME-b$batch" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_conversational_noise_robustness.py" \
    --model "$MODEL_REL" --revision "$local_revision" --batch-size "$batch" \
    --max-length 8192 --attn-implementation flash_attention_2 \
    --output-dir "$ROBUST_OUT" \
    --embedding-cache-dir "$ROOT/outputs/embedding-cache/legal-source-heldout" && break
done
ROBUST_SUMMARY="$ROBUST_OUT/$safe/$local_revision/summary.json"
if [[ -s "$SIONIC_SUMMARY" && -s "$OFFICIAL_SUMMARY" ]]; then
  clean_args=()
  [[ -s "$CLEAN_SUMMARY" ]] && clean_args+=(--clean-summary "$CLEAN_SUMMARY")
  robustness_args=()
  [[ -s "$ROBUST_SUMMARY" ]] && \
    robustness_args+=(--robustness-summary "$ROBUST_SUMMARY")
  if retry_stage "publish-$RUN_NAME" 3 \
    "$UTILITY_PYTHON" "$ROOT/scripts/publish_best_embedding_model.py" \
    --model-dir "$MODEL_DIR" \
    --sionic-summary "$SIONIC_SUMMARY" \
    --official-summary "$OFFICIAL_SUMMARY" \
    "${clean_args[@]}" \
    "${robustness_args[@]}" \
    --training-manifest "$TRAINING_MANIFEST" \
    --repo-id LLM-OS-Models2/qwen3-embedding-8b-ko-performance-1m-v1-private-candidate \
    --upload; then
    run_stage "record-scale-1m-result" \
      "$ROOT/scripts/commit_campaign_result.sh" \
      --stage scale-1m --model "$MODEL_REL" \
      --repo-id LLM-OS-Models2/qwen3-embedding-8b-ko-performance-1m-v1-private-candidate \
      --sionic-summary "$SIONIC_SUMMARY" --official-summary "$OFFICIAL_SUMMARY"
  fi
fi
run_stage "record-clean-legal-results" "$ROOT/scripts/commit_clean_legal_results.sh" || true
if [[ -n "$DATA_UPLOAD_PID" ]]; then
  if wait "$DATA_UPLOAD_PID"; then
    echo "[$(timestamp)] derived performance 1M dataset upload complete"
  else
    echo "[$(timestamp)] derived performance 1M dataset upload failed; see log" >&2
  fi
fi

if [[ "${ENABLE_SIONIC_RETRIEVAL_ADAPTATION:-1}" == 1 ]]; then
  run_stage "sionic-retrieval-train-family-adaptation" env WAIT_PID= \
    LOG_DIR="$ROOT/outputs/sionic-retrieval-family-adaptation-20260712" \
    bash "$ROOT/scripts/run_sionic_retrieval_adaptation_queue.sh" || true
fi
if [[ "${ENABLE_SIONIC_SQUAD_ADAPTATION:-1}" == 1 ]]; then
  run_stage "sionic-squad-train-family-adaptation" env WAIT_PID= \
    LOG_DIR="$ROOT/outputs/sionic-squad-adaptation-20260712" \
    bash "$ROOT/scripts/run_sionic_squad_adaptation_queue.sh" || true
fi
if [[ "${ENABLE_SIONIC_HEALTH_ADAPTATION:-1}" == 1 ]]; then
  run_stage "sionic-health-domain-adaptation" env WAIT_PID= \
    LOG_DIR="$ROOT/outputs/sionic-health-adaptation-20260712" \
    bash "$ROOT/scripts/run_sionic_health_adaptation_queue.sh" || true
fi
if [[ "${ENABLE_SIONIC_AUTORAG_ADAPTATION:-1}" == 1 ]]; then
  run_stage "sionic-autorag-domain-adaptation" env WAIT_PID= \
    LOG_DIR="$ROOT/outputs/sionic-autorag-adaptation-20260712" \
    bash "$ROOT/scripts/run_sionic_autorag_adaptation_queue.sh" || true
fi

echo "[$(timestamp)] 1M scale queue complete"
