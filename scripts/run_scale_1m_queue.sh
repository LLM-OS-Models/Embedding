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
VAL_FILE="$ROOT/outputs/data/validation/legal-source-heldout-i-v2-text-strict-512/validation.jsonl"
RUN_NAME="qwen3-embedding-8b-ko-performance1m-lora-r64"
MODEL_REL="artifacts/models/${RUN_NAME}-best-merged"
MODEL_DIR="$ROOT/$MODEL_REL"
SIONIC_OUT="$ROOT/outputs/evaluation/sionic9-scale1m"
OFFICIAL_OUT="$ROOT/outputs/evaluation/mteb-korean-v1-scale1m"
MULTIDOMAIN_OUT="$ROOT/outputs/evaluation/multidomain-selection"
MULTIDOMAIN_DATASET="$ROOT/outputs/evaluation/multidomain-selection-heldout-v1"
POSTTRAIN_SELECTION="${POSTTRAIN_SELECTION:-$ROOT/outputs/post-capacity-eval-20260717-frontier/clean-first-selection.json}"
POSTTRAIN_UPLOAD_REPORT="${POSTTRAIN_UPLOAD_REPORT:-${POSTTRAIN_SELECTION%/*}/public-clean-candidate-upload.json}"
SCALE_SELECTION="$LOG_DIR/clean-first-selection.json"
SCALE_UPLOAD_REPORT="$LOG_DIR/public-clean-candidate-upload.json"
mkdir -p "$LOG_DIR" "$SIONIC_OUT" "$OFFICIAL_OUT" "$MULTIDOMAIN_OUT"
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1

unset HF_TOKEN HUGGINGFACE_HUB_TOKEN
PUBLISH_HF_TOKEN_FILE="$ROOT/.env"
OFFLINE_ENV=(
  env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN
  EMBEDDING_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
)
read -r -a EVAL_BATCHES <<< "${CAMPAIGN_EVAL_BATCH_SIZES:-192 128 64 32 16 8 4 2}"
for batch in "${EVAL_BATCHES[@]}"; do
  [[ "$batch" =~ ^[1-9][0-9]*$ ]] || {
    echo "Invalid evaluation batch size: $batch" >&2
    exit 2
  }
done
if ! "${OFFLINE_ENV[@]}" "$UTILITY_PYTHON" \
    "$ROOT/scripts/build_multidomain_selection_holdout.py" \
    --output-dir "$MULTIDOMAIN_DATASET" --verify-only \
    >"$LOG_DIR/multidomain-dataset-verification.json"; then
  echo "fixed multidomain selection dataset verification failed" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$ROOT/third_party/mteb${PYTHONPATH:+:$PYTHONPATH}"
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

verified_public_report() {
  local report="$1" expected_model="$2" expected_weights_sha="$3"
  local contract report_model report_weights_sha report_commit
  contract="$(jq -r \
    '.visibility + ":" + (.remote_manifest_exact|tostring) + ":" + (.remote_file_set_exact|tostring)' \
    "$report" 2>/dev/null || true)"
  report_model="$(jq -r '.model // empty' "$report" 2>/dev/null || true)"
  report_weights_sha="$(jq -r '.weights_sha256 // empty' "$report" 2>/dev/null || true)"
  report_commit="$(jq -r '.commit_sha // empty' "$report" 2>/dev/null || true)"
  [[ "$contract" == "public:true:true" \
      && "$report_model" == "$expected_model" \
      && "$report_weights_sha" == "$expected_weights_sha" \
      && "$report_commit" =~ ^[0-9a-f]{40}$ ]]
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
      --mining-audit "$MINING_AUDIT" --mined-train "$MINED_TRAIN" \
      --output "$MINED_PROVENANCE" \
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
    EMBEDDING_OFFLINE=1 ENABLE_VALIDATED_CONTINUAL_BASE=0 \
    ENABLE_PRIVATE_CHECKPOINT_WATCHER=1 CHECKPOINT_REPO_PUBLIC=1 \
    CHECKPOINT_TRAINING_MANIFEST="$TRAINING_MANIFEST" \
    CHECKPOINT_BASE_UPLOAD_REPORT="$POSTTRAIN_UPLOAD_REPORT" \
    PRIVATE_CHECKPOINT_REPO_ID="LLM-OS-Models2/${output_name}-candidates" \
    TRAIN_ENV="$train_env" ATTN_IMPL="$train_attn" \
    RUN_NAME="$output_name" TRAIN_FILE="$TRAIN_FILE" VAL_FILE="$VAL_FILE" \
    MAX_STEPS="$MAX_STEPS_1M" EVAL_STEPS=250 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=5 \
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
  if [[ -f "$PUBLISH_HF_TOKEN_FILE" ]]; then
    (
      embedding_load_hf_credential "$PUBLISH_HF_TOKEN_FILE"
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
        --upload --public
    ) >"$LOG_DIR/derived-dataset-upload.log" 2>&1 &
    DATA_UPLOAD_PID=$!
    echo "[$(timestamp)] derived dataset upload started pid=$DATA_UPLOAD_PID"
  else
    echo "[$(timestamp)] token file unavailable for required derived dataset upload" >&2
    exit 6
  fi
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
else
  run_stage "validate-reused-merge-$RUN_NAME" \
    "${OFFLINE_ENV[@]}" \
    "$UTILITY_PYTHON" "$ROOT/scripts/merge_embedding_adapter.py" \
    --adapter "$checkpoint" --output-dir "$MODEL_DIR" \
    --base-model "$CONTINUAL_BASE" --base-revision "$CONTINUAL_REVISION" \
    --dtype bfloat16 --local-files-only --validate-existing || exit 5
fi

model_sha="$(jq -r '.model.weights_sha256' "$MODEL_DIR/merge_report.json")"
local_revision="model-${model_sha:0:12}"
safe="${MODEL_REL//\//__}"
SIONIC_SUMMARY="$SIONIC_OUT/$safe/summary.json"
OFFICIAL_SUMMARY="$OFFICIAL_OUT/$safe/$local_revision/summary.json"
if [[ "$ENABLE_PUBLIC_INTERMEDIATE_EVAL" == 1 ]]; then
  run_sionic_with_fallback "$MODEL_REL" "$local_revision" \
    "$ROOT/outputs/embedding-cache/sionic9-scale1m" || true
  run_official_with_fallback "$MODEL_REL" "$local_revision" \
    "$ROOT/outputs/embedding-cache/official-scale1m" || true
else
  echo "[$(timestamp)] public intermediate evaluation disabled for $RUN_NAME"
fi
CLEAN_OUT="$ROOT/outputs/evaluation/legal-source-heldout"
for batch in "${EVAL_BATCHES[@]}"; do
  run_stage "clean-legal-$RUN_NAME-b$batch" \
    "${OFFLINE_ENV[@]}" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_legal_source_holdout.py" \
    --model "$MODEL_REL" --revision "$local_revision" --batch-size "$batch" \
    --max-length 8192 --attn-implementation flash_attention_2 \
    --output-dir "$CLEAN_OUT" \
    --embedding-cache-dir "$ROOT/outputs/embedding-cache/legal-source-heldout" && break
done
CLEAN_SUMMARY="$CLEAN_OUT/$safe/$local_revision/summary.json"
ROBUST_OUT="$ROOT/outputs/evaluation/conversational-noise-robustness"
for batch in "${EVAL_BATCHES[@]}"; do
  run_stage "robustness-$RUN_NAME-b$batch" \
    "${OFFLINE_ENV[@]}" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_conversational_noise_robustness.py" \
    --model "$MODEL_REL" --revision "$local_revision" --batch-size "$batch" \
    --max-length 8192 --attn-implementation flash_attention_2 \
    --output-dir "$ROBUST_OUT" \
    --embedding-cache-dir "$ROOT/outputs/embedding-cache/legal-source-heldout" && break
done
ROBUST_SUMMARY="$ROBUST_OUT/$safe/$local_revision/summary.json"
for batch in "${EVAL_BATCHES[@]}"; do
  run_stage "multidomain-$RUN_NAME-b$batch" \
    "${OFFLINE_ENV[@]}" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_multidomain_selection.py" \
    --model "$MODEL_REL" --revision "$local_revision" --batch-size "$batch" \
    --max-length 8192 --attn-implementation flash_attention_2 \
    --dataset-dir "$MULTIDOMAIN_DATASET" --output-dir "$MULTIDOMAIN_OUT" \
    --embedding-cache-dir "$ROOT/outputs/embedding-cache/multidomain-selection" && break
done
MULTIDOMAIN_SUMMARY="$MULTIDOMAIN_OUT/$safe/$local_revision/summary.json"
if [[ ! -s "$CLEAN_SUMMARY" || ! -s "$ROBUST_SUMMARY" \
    || ! -s "$MULTIDOMAIN_SUMMARY" ]]; then
  echo "[$(timestamp)] 1M clean/robustness/multidomain evidence is incomplete" >&2
  exit 6
fi
run_stage "select-clean-$RUN_NAME" \
  "${OFFLINE_ENV[@]}" \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/select_best_clean_model.py" \
  "$CLEAN_OUT" "$ROBUST_OUT" --multidomain-root "$MULTIDOMAIN_OUT" \
  --workspace-root "$ROOT" --clean-epsilon 0.005 --multidomain-epsilon 0.002 \
  --output "$SCALE_SELECTION" --disqualification-root "$ROOT/outputs" \
  --candidate-model "$MODEL_REL" || exit 6
selected_scale_model="$(jq -r '.best.model // empty' "$SCALE_SELECTION")"
selected_scale_weights_sha="$(jq -r '.best.weights_sha256 // empty' "$SCALE_SELECTION")"
if ! verified_public_report \
    "$SCALE_UPLOAD_REPORT" "$selected_scale_model" "$selected_scale_weights_sha"; then
  if [[ ! -f "$PUBLISH_HF_TOKEN_FILE" ]]; then
    echo "[$(timestamp)] token file unavailable for required 1M private backup" >&2
    exit 6
  fi
  retry_stage "publish-clean-$RUN_NAME" 3 \
    env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN \
    "$UTILITY_PYTHON" "$ROOT/scripts/publish_private_clean_candidate.py" \
    --model-dir "$MODEL_DIR" --selection "$SCALE_SELECTION" \
    --training-manifest "$TRAINING_MANIFEST" \
    --repo-id LLM-OS-Models2/qwen3-embedding-8b-ko-performance1m-clean-winner-v1 \
    --hf-token-file "$PUBLISH_HF_TOKEN_FILE" \
    --report-output "$SCALE_UPLOAD_REPORT" --upload --public || exit 6
fi
if ! verified_public_report \
    "$SCALE_UPLOAD_REPORT" "$selected_scale_model" "$selected_scale_weights_sha"; then
  echo "[$(timestamp)] 1M public clean-winner remote verification is incomplete" >&2
  exit 6
fi
if [[ "$ENABLE_PUBLIC_INTERMEDIATE_EVAL" == 1 \
    && -s "$SIONIC_SUMMARY" && -s "$OFFICIAL_SUMMARY" ]]; then
  clean_args=()
  [[ -s "$CLEAN_SUMMARY" ]] && clean_args+=(--clean-summary "$CLEAN_SUMMARY")
  robustness_args=()
  [[ -s "$ROBUST_SUMMARY" ]] && \
    robustness_args+=(--robustness-summary "$ROBUST_SUMMARY")
  run_stage "record-scale-1m-result" \
    "$ROOT/scripts/commit_campaign_result.sh" \
    --stage scale-1m --model "$MODEL_REL" \
    --repo-id LLM-OS-Models2/qwen3-embedding-8b-ko-performance1m-clean-winner-v1 \
    --sionic-summary "$SIONIC_SUMMARY" --official-summary "$OFFICIAL_SUMMARY"
fi
run_stage "record-clean-legal-results" "$ROOT/scripts/commit_clean_legal_results.sh" || true
if [[ -n "$DATA_UPLOAD_PID" ]]; then
  if wait "$DATA_UPLOAD_PID"; then
    echo "[$(timestamp)] derived performance 1M dataset upload complete"
  else
    echo "[$(timestamp)] derived performance 1M dataset upload failed; see log" >&2
    exit 6
  fi
fi

GENERAL_SELECTION="$SCALE_SELECTION"
GENERAL_BASE_UPLOAD_REPORT="$SCALE_UPLOAD_REPORT"
if [[ "${ENABLE_RERANKER_KD_ABLATION:-1}" == 1 ]]; then
  GENERAL_SELECTION="$ROOT/outputs/reranker-kd-20260717-frontier/clean-first-selection.json"
  GENERAL_BASE_UPLOAD_REPORT="${GENERAL_SELECTION%/*}/public-clean-candidate-upload.json"
  run_stage "qwen3-reranker-listwise-kd-ablation" env \
    LOG_DIR="$ROOT/outputs/reranker-kd-20260717-frontier" \
    GENERAL_BASE_MODEL="$MODEL_DIR" \
    GENERAL_TRAINING_MANIFEST="$TRAINING_MANIFEST" \
    GENERAL_BASE_UPLOAD_REPORT="$SCALE_UPLOAD_REPORT" \
    bash "$ROOT/scripts/run_reranker_kd_ablation_queue.sh" || exit 7
  if [[ ! -s "$GENERAL_SELECTION" ]]; then
    echo "[$(timestamp)] reranker KD A/B produced no clean selection" >&2
    exit 7
  fi
  general_model="$(jq -r '.best.model // empty' "$GENERAL_SELECTION")"
  general_weights_sha="$(jq -r '.best.weights_sha256 // empty' "$GENERAL_SELECTION")"
  if ! verified_private_report \
      "$GENERAL_BASE_UPLOAD_REPORT" "$general_model" "$general_weights_sha"; then
    echo "[$(timestamp)] reranker KD winner backup is not exact" >&2
    exit 7
  fi
else
  echo "[$(timestamp)] reranker KD A/B disabled; using clean-selected 1M base"
fi

if [[ "${ENABLE_SIONIC_RETRIEVAL_ADAPTATION:-1}" == 1 ]]; then
  run_stage "sionic-retrieval-train-family-adaptation" env WAIT_PID= \
    GENERAL_SELECTION="$GENERAL_SELECTION" \
    GENERAL_BASE_UPLOAD_REPORT="$GENERAL_BASE_UPLOAD_REPORT" \
    LOG_DIR="$ROOT/outputs/sionic-retrieval-family-adaptation-20260712" \
    bash "$ROOT/scripts/run_sionic_retrieval_adaptation_queue.sh" || exit 8
fi
if [[ "${ENABLE_SIONIC_SQUAD_ADAPTATION:-1}" == 1 ]]; then
  run_stage "sionic-squad-train-family-adaptation" env WAIT_PID= \
    GENERAL_SELECTION="$GENERAL_SELECTION" \
    GENERAL_BASE_UPLOAD_REPORT="$GENERAL_BASE_UPLOAD_REPORT" \
    LOG_DIR="$ROOT/outputs/sionic-squad-adaptation-20260712" \
    bash "$ROOT/scripts/run_sionic_squad_adaptation_queue.sh" || exit 9
fi
if [[ "${ENABLE_SIONIC_HEALTH_ADAPTATION:-1}" == 1 ]]; then
  run_stage "sionic-health-domain-adaptation" env WAIT_PID= \
    GENERAL_SELECTION="$GENERAL_SELECTION" \
    GENERAL_BASE_UPLOAD_REPORT="$GENERAL_BASE_UPLOAD_REPORT" \
    LOG_DIR="$ROOT/outputs/sionic-health-adaptation-20260712" \
    bash "$ROOT/scripts/run_sionic_health_adaptation_queue.sh" || exit 10
fi
if [[ "${ENABLE_SIONIC_AUTORAG_ADAPTATION:-1}" == 1 ]]; then
  run_stage "sionic-autorag-domain-adaptation" env WAIT_PID= \
    GENERAL_SELECTION="$GENERAL_SELECTION" \
    GENERAL_BASE_UPLOAD_REPORT="$GENERAL_BASE_UPLOAD_REPORT" \
    LOG_DIR="$ROOT/outputs/sionic-autorag-adaptation-20260712" \
    bash "$ROOT/scripts/run_sionic_autorag_adaptation_queue.sh" || exit 11
fi

echo "[$(timestamp)] 1M scale queue complete"
