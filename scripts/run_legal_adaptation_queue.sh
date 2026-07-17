#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
source "$ROOT/scripts/backend_admission.sh"
embedding_resolve_train_runtime
UTILITY_PYTHON="$EMBEDDING_TRAIN_PYTHON"
cd "$ROOT"
WAIT_PID="${WAIT_PID:-}"
ENABLE_PUBLIC_INTERMEDIATE_EVAL="${ENABLE_PUBLIC_INTERMEDIATE_EVAL:-0}"
GENERAL_SELECTION="${GENERAL_SELECTION:-$ROOT/outputs/reranker-kd-20260717-frontier/clean-first-selection.json}"
GENERAL_BASE_UPLOAD_REPORT="${GENERAL_BASE_UPLOAD_REPORT:-${GENERAL_SELECTION%/*}/private-clean-candidate-upload.json}"
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
CURRICULUM_QUALITY_AUDIT="$DATA_DIR/faiss-r095-n7.legal25-replay75.quality-audit.json"
CURRICULUM_OVERLAP_AUDIT="$DATA_DIR/faiss-r095-n7.legal25-replay75.benchmark-overlap-audit.json"
VAL_FILE="$ROOT/outputs/data/validation/legal-source-heldout-i-v2-text-strict-512/validation.jsonl"
RUN_NAME="qwen3-embedding-8b-ko-legal25-replay75-lora-r64"
MODEL_REL="artifacts/models/${RUN_NAME}-best-merged"
MODEL_DIR="$ROOT/$MODEL_REL"
SIONIC_OUT="$ROOT/outputs/evaluation/sionic9-legal250k"
OFFICIAL_OUT="$ROOT/outputs/evaluation/mteb-korean-v1-legal250k"
mkdir -p "$LOG_DIR" "$SIONIC_OUT" "$OFFICIAL_OUT"
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1

unset HF_TOKEN HUGGINGFACE_HUB_TOKEN
PUBLISH_HF_TOKEN_FILE="$ROOT/.env"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$ROOT/scripts:$ROOT/third_party/mteb${PYTHONPATH:+:$PYTHONPATH}"
FAISS_THREADS="${FAISS_THREADS:-$EFFECTIVE_CPU_COUNT}"

if [[ "$ENABLE_PUBLIC_INTERMEDIATE_EVAL" != 0 \
    && "$ENABLE_PUBLIC_INTERMEDIATE_EVAL" != 1 ]]; then
  echo "ENABLE_PUBLIC_INTERMEDIATE_EVAL must be 0 or 1" >&2
  exit 2
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
    if run_stage "sionic9-legal-target-adapted-b$batch" \
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
    if run_stage "official-korean-legal-target-adapted-b$batch" \
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

if embedding_resolve_general_base; then
  MINING_MODEL="$EMBEDDING_GENERAL_BASE"
  MINING_REVISION=""
  echo "[$(timestamp)] continuing from clean-selected general winner: $MINING_MODEL"
else
  MINING_MODEL="Qwen/Qwen3-Embedding-8B"
  MINING_REVISION="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
  echo "[$(timestamp)] 1M merged model unavailable; using pinned Qwen base"
fi

if ! "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/check_mining_manifest.py" \
    --manifest "$MINING_MANIFEST" --model "$MINING_MODEL" \
    --revision "$MINING_REVISION" --selection-strategy score_rank_quantiles \
    --candidate-pool-size 24 --num-negatives 7 2>/dev/null; then
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
    --num-negatives 7 --selection-strategy score_rank_quantiles \
    --positive-relative-ratio .95 \
    --nlist 512 --nprobe 32 --training-points 50000 \
    --faiss-threads "$FAISS_THREADS" \
    --keep-work-dir --allow-target-adapted || exit 3
fi

if [[ ! -s "$MINED_PROVENANCE" ]]; then
  run_stage project-legal-provenance \
    "$UTILITY_PYTHON" "$ROOT/scripts/project_mined_provenance.py" \
    --input-provenance "$DATA_DIR/provenance.jsonl" --mining-audit "$AUDIT" \
    --output "$MINED_PROVENANCE" \
    --manifest-output "$DATA_DIR/provenance.faiss-r095-n7.manifest.json" || exit 4
fi

if [[ ! -s "$ORDERED_MANIFEST" \
    || "$(jq -r '.length_bucketed // false' "$ORDERED_MANIFEST")" != true ]]; then
  run_stage order-legal-homogeneous-batches \
    "$UTILITY_PYTHON" "$ROOT/scripts/build_homogeneous_batches.py" \
    --train "$MINED" --provenance "$MINED_PROVENANCE" \
    --output "$ORDERED" --provenance-output "$ORDERED_PROVENANCE" \
    --manifest-output "$ORDERED_MANIFEST" --batch-size 16 --seed 42 \
    --length-bucketed || exit 5
fi

if [[ ! -s "$CURRICULUM_MANIFEST" ]]; then
  LEGAL_ROWS="$(jq -r '.output_rows' "$ORDERED_MANIFEST")"
  REPLAY_ROWS="$((LEGAL_ROWS * 3))"
  run_stage build-legal25-general75-curriculum \
    "$UTILITY_PYTHON" "$ROOT/scripts/build_replay_curriculum.py" \
    --primary-train "$ORDERED" --primary-provenance "$ORDERED_PROVENANCE" \
    --primary-rows "$LEGAL_ROWS" \
    --replay-train "$GENERAL_TRAIN" --replay-provenance "$GENERAL_PROVENANCE" \
    --replay-rows "$REPLAY_ROWS" --output "$CURRICULUM" \
    --provenance-output "$CURRICULUM_PROVENANCE" \
    --manifest-output "$CURRICULUM_MANIFEST" --batch-size 16 --seed 42 \
    --adaptation-label target-adapted-legal25-general75 || exit 6
fi

run_stage audit-legal25-general75-curriculum \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/audit_embedding_training_data.py" \
  --train "$CURRICULUM" --provenance "$CURRICULUM_PROVENANCE" \
  --output "$CURRICULUM_QUALITY_AUDIT" --expected-batch-size 16 || exit 6
run_stage audit-legal25-general75-benchmark-overlap \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/audit_training_benchmark_overlap.py" \
  --train "$CURRICULUM" --provenance "$CURRICULUM_PROVENANCE" \
  --blocklist-root "$ROOT/outputs/decontamination/benchmark_blocklist" \
  --output "$CURRICULUM_OVERLAP_AUDIT" --fail-on-critical || exit 6

MAX_STEPS="$(jq -r '.output_rows / 64 | floor' "$CURRICULUM_MANIFEST")"
if (( MAX_STEPS < 1 )); then
  echo "[$(timestamp)] no complete legal training steps" >&2
  exit 6
fi
run_stage validate-legal-mined-data \
  "$UTILITY_PYTHON" "$ROOT/scripts/validate_embedding_jsonl.py" \
  "$CURRICULUM" "$VAL_FILE" || exit 6

train_legal() {
  local output_name="$1" batch="$2" accum="$3"
  local train_env="$EMBEDDING_TRAIN_ENV" train_attn=sdpa admission_report
  local admission_key="legal-lora-r64-b${batch}-a${accum}-m512-hn7"
  if embedding_select_fa2_backend "$CURRICULUM" "$admission_key" \
      "$batch" "$accum" 512 64 128 bfloat16 \
      "$MINING_MODEL" "$MINING_REVISION" 7 .05; then
    train_env="$BACKEND_ADMISSION_ENV"
    train_attn="$BACKEND_ADMISSION_ATTN"
  fi
  admission_report="$BACKEND_ADMISSION_REPORT"
  echo "[$(timestamp)] legal training backend=$train_attn env=$train_env admission=$admission_report"
  run_stage "train-$output_name" env \
    EMBEDDING_OFFLINE=1 ENABLE_VALIDATED_CONTINUAL_BASE=0 \
    ENABLE_PRIVATE_CHECKPOINT_WATCHER=1 \
    CHECKPOINT_TRAINING_MANIFEST="$CURRICULUM_MANIFEST" \
    CHECKPOINT_BASE_UPLOAD_REPORT="$GENERAL_BASE_UPLOAD_REPORT" \
    PRIVATE_CHECKPOINT_REPO_ID="LLM-OS-Models2/${output_name}-candidates" \
    TRAIN_ENV="$train_env" ATTN_IMPL="$train_attn" \
    RUN_NAME="$output_name" TRAIN_FILE="$CURRICULUM" VAL_FILE="$VAL_FILE" \
    MAX_STEPS="$MAX_STEPS" EVAL_STEPS=250 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=5 \
    TRAIN_BATCH_SIZE="$batch" GRAD_ACCUM_STEPS="$accum" \
    MAX_LENGTH=512 LORA_RANK=64 LORA_ALPHA=128 LORA_DROPOUT=.05 \
    DATASET_SHUFFLE=false TRAIN_DATALOADER_SHUFFLE=false LEARNING_RATE=1e-5 \
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
  checkpoint="$("$UTILITY_PYTHON" "$ROOT/scripts/select_best_checkpoint.py" "$ROOT/outputs/$RUN_NAME" --print-path 2>/dev/null)" || checkpoint=""
fi
if [[ -z "$checkpoint" ]]; then
  fallback="${RUN_NAME}-b4"
  train_legal "$fallback" 4 16 || exit 6
  RUN_NAME="$fallback"
  MODEL_REL="artifacts/models/${RUN_NAME}-best-merged"
  MODEL_DIR="$ROOT/$MODEL_REL"
  checkpoint="$("$UTILITY_PYTHON" "$ROOT/scripts/select_best_checkpoint.py" "$ROOT/outputs/$RUN_NAME" --print-path)" || exit 6
fi

DATA_UPLOAD_PID=""
if [[ -f "$PUBLISH_HF_TOKEN_FILE" ]]; then
  (
    set -a
    source "$PUBLISH_HF_TOKEN_FILE"
    set +a
    retry_stage upload-derived-legal-replay 3 \
      "$UTILITY_PYTHON" "$ROOT/scripts/publish_derived_training_dataset.py" \
      --train "$CURRICULUM" --provenance "$CURRICULUM_PROVENANCE" \
      --manifest "$CURRICULUM_MANIFEST" \
      --mining-manifest "$MINING_MANIFEST" --mining-audit "$AUDIT" \
      --quality-audit "$CURRICULUM_QUALITY_AUDIT" \
      --benchmark-overlap-audit "$CURRICULUM_OVERLAP_AUDIT" \
      --repo-id LLM-OS-Models2/korean-legal-quantile-hn7-replay-v1 \
      --title "Korean Legal Quantile HN7 with General Replay" \
      --source-dataset LLM-OS-Models/korean-legal-retrieval-source-native-250k \
      --source-dataset LLM-OS-Models/korean-embedding-performance-v1-performance-1m \
      --upload --public
  ) >"$LOG_DIR/derived-dataset-upload.log" 2>&1 &
  DATA_UPLOAD_PID=$!
  echo "[$(timestamp)] derived legal dataset upload started pid=$DATA_UPLOAD_PID"
else
  echo "[$(timestamp)] no token file; derived legal dataset upload skipped" >&2
fi

run_stage verify-legal-adapter \
  "$UTILITY_PYTHON" "$ROOT/scripts/verify_adapter.py" \
  --adapter "$checkpoint" --data "$VAL_FILE" --model "$MINING_MODEL" \
  --output "$LOG_DIR/verification.json" || exit 7
if [[ ! -s "$MODEL_DIR/merge_report.json" ]]; then
  run_stage merge-legal-adapter \
    "$UTILITY_PYTHON" "$ROOT/scripts/merge_embedding_adapter.py" \
    --adapter "$checkpoint" --output-dir "$MODEL_DIR" \
    --base-model "$MINING_MODEL" --base-revision "$MINING_REVISION" \
    --device cuda --dtype bfloat16 --local-files-only || exit 8
fi

model_sha="$(jq -r '.model.weights_sha256' "$MODEL_DIR/merge_report.json")"
revision="model-${model_sha:0:12}"
safe="${MODEL_REL//\//__}"
SIONIC_SUMMARY="$SIONIC_OUT/$safe/summary.json"
OFFICIAL_SUMMARY="$OFFICIAL_OUT/$safe/$revision/summary.json"
if [[ "$ENABLE_PUBLIC_INTERMEDIATE_EVAL" == 1 ]]; then
  run_sionic_with_fallback "$MODEL_REL" "$revision" \
    "$ROOT/outputs/embedding-cache/sionic9-legal250k" || true
  run_official_with_fallback "$MODEL_REL" "$revision" \
    "$ROOT/outputs/embedding-cache/official-legal250k" || true
else
  echo "[$(timestamp)] public intermediate evaluation disabled for $RUN_NAME"
fi
CLEAN_OUT="$ROOT/outputs/evaluation/legal-source-heldout"
for batch in "${CAMPAIGN_EVAL_BATCH_SIZE:-192}"; do
  run_stage "clean-legal-legal-target-adapted-b$batch" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_legal_source_holdout.py" \
    --model "$MODEL_REL" --revision "$revision" --batch-size "$batch" \
    --max-length 8192 --attn-implementation flash_attention_2 \
    --output-dir "$CLEAN_OUT" \
    --embedding-cache-dir "$ROOT/outputs/embedding-cache/legal-source-heldout" && break
done
CLEAN_SUMMARY="$CLEAN_OUT/$safe/$revision/summary.json"
ROBUST_OUT="$ROOT/outputs/evaluation/conversational-noise-robustness"
for batch in "${CAMPAIGN_EVAL_BATCH_SIZE:-192}"; do
  run_stage "robustness-legal-target-adapted-b$batch" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_conversational_noise_robustness.py" \
    --model "$MODEL_REL" --revision "$revision" --batch-size "$batch" \
    --max-length 8192 --attn-implementation flash_attention_2 \
    --output-dir "$ROBUST_OUT" \
    --embedding-cache-dir "$ROOT/outputs/embedding-cache/legal-source-heldout" && break
done
ROBUST_SUMMARY="$ROBUST_OUT/$safe/$revision/summary.json"
if [[ "$ENABLE_PUBLIC_INTERMEDIATE_EVAL" == 1 \
    && -s "$SIONIC_SUMMARY" && -s "$OFFICIAL_SUMMARY" ]]; then
  clean_args=()
  [[ -s "$CLEAN_SUMMARY" ]] && clean_args+=(--clean-summary "$CLEAN_SUMMARY")
  robustness_args=()
  [[ -s "$ROBUST_SUMMARY" ]] && \
    robustness_args+=(--robustness-summary "$ROBUST_SUMMARY")
  if [[ ! -f "$PUBLISH_HF_TOKEN_FILE" ]]; then
    echo "[$(timestamp)] no token file; private legal model upload skipped" >&2
  elif retry_stage publish-legal-target-adapted 3 \
    "$UTILITY_PYTHON" "$ROOT/scripts/publish_best_embedding_model.py" \
    --model-dir "$MODEL_DIR" --sionic-summary "$SIONIC_SUMMARY" \
    --official-summary "$OFFICIAL_SUMMARY" --training-manifest "$CURRICULUM_MANIFEST" \
    "${clean_args[@]}" \
    "${robustness_args[@]}" \
    --repo-id LLM-OS-Models2/qwen3-embedding-8b-ko-legal-target-adapted-v1-private-candidate \
    --hf-token-file "$PUBLISH_HF_TOKEN_FILE" --upload; then
    run_stage record-legal-replay-result \
      "$ROOT/scripts/commit_campaign_result.sh" \
      --stage legal-replay --model "$MODEL_REL" \
      --repo-id LLM-OS-Models2/qwen3-embedding-8b-ko-legal-target-adapted-v1-private-candidate \
      --sionic-summary "$SIONIC_SUMMARY" --official-summary "$OFFICIAL_SUMMARY"
  fi
fi
run_stage record-clean-legal-results "$ROOT/scripts/commit_clean_legal_results.sh" || true
if [[ -n "$DATA_UPLOAD_PID" ]]; then
  if wait "$DATA_UPLOAD_PID"; then
    echo "[$(timestamp)] derived legal dataset upload complete"
  else
    echo "[$(timestamp)] derived legal dataset upload failed; see log" >&2
  fi
fi
if [[ "${ENABLE_SIONIC_COMBINED_ADAPTATION:-1}" == 1 ]]; then
  run_stage sionic-combined-target-adaptation env \
    LOG_DIR="$ROOT/outputs/sionic-combined-adaptation-20260712" \
    bash "$ROOT/scripts/run_sionic_combined_adaptation_queue.sh" || true
fi

echo "[$(timestamp)] legal target-adaptation queue complete"
