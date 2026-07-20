#!/usr/bin/env bash
set -uo pipefail

# Generic target-domain adaptation engine. Defaults reproduce the isolated
# KorQuAD train-family experiment; a thin wrapper may override TARGET_* for
# other audited datasets such as the multilingual health shard.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
source "$ROOT/scripts/backend_admission.sh"
embedding_resolve_train_runtime
UTILITY_PYTHON="$EMBEDDING_TRAIN_PYTHON"
cd "$ROOT"
WAIT_PID="${WAIT_PID:-}"
ENABLE_PUBLIC_INTERMEDIATE_EVAL="${ENABLE_PUBLIC_INTERMEDIATE_EVAL:-0}"
GENERAL_SELECTION="${GENERAL_SELECTION:-$ROOT/outputs/reranker-kd-20260717-frontier/clean-first-selection.json}"
GENERAL_BASE_UPLOAD_REPORT="${GENERAL_BASE_UPLOAD_REPORT:-${GENERAL_SELECTION%/*}/public-clean-candidate-upload.json}"
TARGET_KIND="${TARGET_KIND:-squad}"
TARGET_PHASE="${TARGET_PHASE:-sionic_squad_train_60k}"
TARGET_DATA_REL="${TARGET_DATA_REL:-outputs/data/performance-v1/sionic-squad-train-60k}"
TARGET_ADAPTATION="${TARGET_ADAPTATION:-target-adapted-squad}"
TARGET_NLIST="${TARGET_NLIST:-128}"
TARGET_TRAINING_POINTS="${TARGET_TRAINING_POINTS:-9606}"
TARGET_MAX_LENGTH="${TARGET_MAX_LENGTH:-512}"
TARGET_ENCODE_BATCH_SIZE="${TARGET_ENCODE_BATCH_SIZE:-128}"
TARGET_TRAIN_BATCH_SIZE="${TARGET_TRAIN_BATCH_SIZE:-8}"
TARGET_GRAD_ACCUM_STEPS="${TARGET_GRAD_ACCUM_STEPS:-8}"
TARGET_FALLBACK_BATCH_SIZE="${TARGET_FALLBACK_BATCH_SIZE:-4}"
TARGET_FALLBACK_GRAD_ACCUM_STEPS="${TARGET_FALLBACK_GRAD_ACCUM_STEPS:-16}"
TARGET_EVAL_BATCH_SIZE="${TARGET_EVAL_BATCH_SIZE:-4}"
TARGET_SOURCE_DATASET="${TARGET_SOURCE_DATASET:-LLM-OS-Models/korean-embedding-performance-v1-sionic-squad-train-60k}"
DERIVED_REPO="${DERIVED_REPO:-LLM-OS-Models2/korean-embedding-sionic-squad-quantile-hn7-replay-v1}"
DERIVED_TITLE="${DERIVED_TITLE:-Korean Sionic SQuAD Quantile HN7 with General Replay}"
CAMPAIGN_STAGE="${CAMPAIGN_STAGE:-sionic-squad-target}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/sionic-${TARGET_KIND}-adaptation-20260712}"
DATA_DIR="$ROOT/$TARGET_DATA_REL"
BOOTSTRAP="$DATA_DIR/train.jsonl"
PROVENANCE="$DATA_DIR/provenance.jsonl"
MINED="$DATA_DIR/train.faiss-current-r095-n7.jsonl"
MINING_AUDIT="$DATA_DIR/train.faiss-current-r095-n7.audit.jsonl"
MINING_MANIFEST="$DATA_DIR/train.faiss-current-r095-n7.manifest.json"
MINED_PROVENANCE="$DATA_DIR/provenance.faiss-current-r095-n7.jsonl"
ORDERED="$DATA_DIR/train.faiss-current-r095-n7.homogeneous-b16.jsonl"
ORDERED_PROVENANCE="$DATA_DIR/provenance.faiss-current-r095-n7.homogeneous-b16.jsonl"
ORDERED_MANIFEST="$DATA_DIR/faiss-current-r095-n7.homogeneous-b16.manifest.json"
GENERAL_DIR="$ROOT/outputs/data/performance-v1/performance-1m"
GENERAL_TRAIN="$GENERAL_DIR/train.homogeneous-b16.jsonl"
GENERAL_PROVENANCE="$GENERAL_DIR/provenance.homogeneous-b16.jsonl"
CURRICULUM="$DATA_DIR/train.faiss-current-r095-n7.${TARGET_KIND}50-replay50.jsonl"
CURRICULUM_PROVENANCE="$DATA_DIR/provenance.faiss-current-r095-n7.${TARGET_KIND}50-replay50.jsonl"
CURRICULUM_MANIFEST="$DATA_DIR/faiss-current-r095-n7.${TARGET_KIND}50-replay50.manifest.json"
CURRICULUM_QUALITY="$DATA_DIR/faiss-current-r095-n7.${TARGET_KIND}50-replay50.quality-audit.json"
CURRICULUM_OVERLAP="$DATA_DIR/faiss-current-r095-n7.${TARGET_KIND}50-replay50.benchmark-overlap-audit.json"
VAL_FILE="$ROOT/outputs/data/validation/legal-source-heldout-i-v2-text-strict-512/validation.jsonl"
RUN_NAME="${TARGET_RUN_NAME:-qwen3-embedding-8b-ko-sionic-squad50-replay50-lora-r64}"
MODEL_REL="artifacts/models/${RUN_NAME}-best-merged"
MODEL_DIR="$ROOT/$MODEL_REL"
SIONIC_OUT="$ROOT/outputs/evaluation/sionic9-${TARGET_KIND}-target-adapted"
OFFICIAL_OUT="$ROOT/outputs/evaluation/mteb-korean-v1-${TARGET_KIND}-target-adapted"
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
run_sionic() {
  local model="$1" revision="$2" cache="$3" batch
  for batch in "${CAMPAIGN_EVAL_BATCH_SIZE:-192}"; do
    run_stage "sionic9-${TARGET_KIND}-target-b$batch" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_sionic9.py" \
      --model "$model" --revision "$revision" --batch-size "$batch" \
      --max-length 8192 --attn-implementation flash_attention_2 \
      --output-dir "$SIONIC_OUT" --embedding-cache-dir "$cache" && return 0
  done
  return 1
}
run_official() {
  local model="$1" revision="$2" cache="$3" batch
  for batch in "${CAMPAIGN_EVAL_BATCH_SIZE:-192}"; do
    run_stage "official-korean-${TARGET_KIND}-target-b$batch" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_mteb_korean_v1.py" \
      --model "$model" --revision "$revision" --batch-size "$batch" \
      --max-length 8192 --qwen3-instruction-loader \
      --attn-implementation flash_attention_2 --output-dir "$OFFICIAL_OUT" \
      --embedding-cache-dir "$cache" && return 0
  done
  return 1
}

if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi

if [[ ! -s "$BOOTSTRAP" || ! -s "$PROVENANCE" ]]; then
  if [[ "$TARGET_KIND" == retrieval_family ]]; then
    run_stage "build-sionic-${TARGET_KIND}-data" \
      "$UTILITY_PYTHON" "$ROOT/scripts/extract_training_source_subset.py" \
      --train "$GENERAL_DIR/train.jsonl" \
      --provenance "$GENERAL_DIR/provenance.jsonl" \
      --output-dir "$DATA_DIR" \
      --source f2_miracl_ko_train --source f2_mrtidy_korean_train \
      --source f2_mldr_ko_train --phase "$TARGET_PHASE" \
      --expected-rows 4146 || exit 2
  else
    run_stage "build-sionic-${TARGET_KIND}-data" \
      "$UTILITY_PYTHON" "$ROOT/scripts/build_performance_mix.py" \
      --phase "$TARGET_PHASE" --output-dir "$DATA_DIR" || exit 2
  fi
fi
[[ -s "$VAL_FILE" && -s "$GENERAL_TRAIN" && -s "$GENERAL_PROVENANCE" ]] || exit 2
if [[ -s "$GENERAL_DIR/faiss-current-r095-n7.homogeneous-b16.manifest.json" \
    && -s "$GENERAL_DIR/train.faiss-current-r095-n7.homogeneous-b16.jsonl" \
    && -s "$GENERAL_DIR/provenance.faiss-current-r095-n7.homogeneous-b16.jsonl" ]]; then
  GENERAL_TRAIN="$GENERAL_DIR/train.faiss-current-r095-n7.homogeneous-b16.jsonl"
  GENERAL_PROVENANCE="$GENERAL_DIR/provenance.faiss-current-r095-n7.homogeneous-b16.jsonl"
fi

MINING_REVISION=""
if embedding_resolve_general_base; then
  MINING_MODEL="$EMBEDDING_GENERAL_BASE"
  echo "[$(timestamp)] continuing from clean-selected general winner: $MINING_MODEL"
else
  MINING_MODEL="Qwen/Qwen3-Embedding-8B"
  MINING_REVISION="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
  echo "[$(timestamp)] 1M winner unavailable; using pinned Qwen base"
fi

if ! "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/check_mining_manifest.py" \
    --manifest "$MINING_MANIFEST" --model "$MINING_MODEL" \
    --revision "$MINING_REVISION" --selection-strategy score_rank_quantiles \
    --candidate-pool-size 24 --num-negatives 7 2>/dev/null; then
  rm -f "$MINED" "$MINING_AUDIT" "$MINING_MANIFEST" "$MINED_PROVENANCE" \
    "$ORDERED" "$ORDERED_PROVENANCE" "$ORDERED_MANIFEST" \
    "$CURRICULUM" "$CURRICULUM_PROVENANCE" "$CURRICULUM_MANIFEST"
  run_stage "mine-sionic-${TARGET_KIND}-current-student" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/mine_faiss_hard_negatives.py" \
    --input "$BOOTSTRAP" --output "$MINED" --audit-output "$MINING_AUDIT" \
    --manifest-output "$MINING_MANIFEST" \
    --work-dir "$DATA_DIR/faiss-work-current-student" --keep-work-dir \
    --model "$MINING_MODEL" --revision "$MINING_REVISION" \
    --encode-batch-size "$TARGET_ENCODE_BATCH_SIZE" \
    --max-seq-length "$TARGET_MAX_LENGTH" \
    --candidate-pool-size 24 --search-k 256 \
    --num-negatives 7 --selection-strategy score_rank_quantiles \
    --positive-relative-ratio .95 --nlist "$TARGET_NLIST" --nprobe 32 \
    --training-points "$TARGET_TRAINING_POINTS" \
    --faiss-threads "$FAISS_THREADS" \
    --allow-target-adapted || exit 3
fi

if [[ ! -s "$MINED_PROVENANCE" ]]; then
  run_stage "project-sionic-${TARGET_KIND}-mined-provenance" \
    "$UTILITY_PYTHON" "$ROOT/scripts/project_mined_provenance.py" \
    --input-provenance "$PROVENANCE" --mining-audit "$MINING_AUDIT" \
    --mined-train "$MINED" --output "$MINED_PROVENANCE" \
    --manifest-output "$DATA_DIR/provenance.faiss-current-r095-n7.manifest.json" || exit 4
fi
if [[ ! -s "$ORDERED_MANIFEST" ]]; then
  run_stage "order-sionic-${TARGET_KIND}-homogeneous" \
    "$UTILITY_PYTHON" "$ROOT/scripts/build_homogeneous_batches.py" \
    --train "$MINED" --provenance "$MINED_PROVENANCE" \
    --output "$ORDERED" --provenance-output "$ORDERED_PROVENANCE" \
    --manifest-output "$ORDERED_MANIFEST" --batch-size 16 --seed 42 \
    --length-bucketed --benchmark-adaptation "${TARGET_ADAPTATION}-source" || exit 5
fi

PRIMARY_ROWS="$(jq -r '.output_rows' "$ORDERED_MANIFEST")"
PRIMARY_ROWS="$((PRIMARY_ROWS / 16 * 16))"
if [[ ! -s "$CURRICULUM_MANIFEST" ]]; then
  run_stage "build-${TARGET_KIND}50-general50-curriculum" \
    "$UTILITY_PYTHON" "$ROOT/scripts/build_replay_curriculum.py" \
    --primary-train "$ORDERED" --primary-provenance "$ORDERED_PROVENANCE" \
    --primary-rows "$PRIMARY_ROWS" --replay-train "$GENERAL_TRAIN" \
    --replay-provenance "$GENERAL_PROVENANCE" --replay-rows "$PRIMARY_ROWS" \
    --output "$CURRICULUM" --provenance-output "$CURRICULUM_PROVENANCE" \
    --manifest-output "$CURRICULUM_MANIFEST" --batch-size 16 --seed 42 \
    --adaptation-label "${TARGET_ADAPTATION}50-general50" || exit 6
fi
run_stage "audit-${TARGET_KIND}50-general50-curriculum" \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/audit_embedding_training_data.py" \
  --train "$CURRICULUM" --provenance "$CURRICULUM_PROVENANCE" \
  --output "$CURRICULUM_QUALITY" --expected-batch-size 16 || exit 6
run_stage "audit-${TARGET_KIND}50-general50-benchmark-overlap" \
  "$UTILITY_PYTHON" "$ROOT/scripts/audit_training_benchmark_overlap.py" \
  --train "$CURRICULUM" --provenance "$CURRICULUM_PROVENANCE" \
  --blocklist-root "$ROOT/outputs/decontamination/benchmark_blocklist" \
  --output "$CURRICULUM_OVERLAP" --fail-on-critical || exit 6

MAX_STEPS="$(jq -r '.output_rows / 64 | floor' "$CURRICULUM_MANIFEST")"
(( MAX_STEPS > 0 )) || exit 6
train_target() {
  local output_name="$1" batch="$2" accum="$3"
  local train_env="$EMBEDDING_TRAIN_ENV" train_attn=sdpa admission_report
  local admission_key="sionic-${TARGET_KIND}-lora-r64-b${batch}-a${accum}-m${TARGET_MAX_LENGTH}-hn7"
  if embedding_select_fa2_backend "$CURRICULUM" "$admission_key" \
      "$batch" "$accum" "$TARGET_MAX_LENGTH" 64 128 bfloat16 \
      "$MINING_MODEL" "$MINING_REVISION" 7 .05; then
    train_env="$BACKEND_ADMISSION_ENV"
    train_attn="$BACKEND_ADMISSION_ATTN"
  fi
  admission_report="$BACKEND_ADMISSION_REPORT"
  echo "[$(timestamp)] ${TARGET_KIND} training backend=$train_attn env=$train_env admission=$admission_report"
  run_stage "train-$output_name" env \
    EMBEDDING_OFFLINE=1 ENABLE_VALIDATED_CONTINUAL_BASE=0 \
    ENABLE_PRIVATE_CHECKPOINT_WATCHER=1 CHECKPOINT_REPO_PUBLIC=1 \
    CHECKPOINT_TRAINING_MANIFEST="$CURRICULUM_MANIFEST" \
    CHECKPOINT_BASE_UPLOAD_REPORT="$GENERAL_BASE_UPLOAD_REPORT" \
    PRIVATE_CHECKPOINT_REPO_ID="LLM-OS-Models2/${output_name}-candidates" \
    TRAIN_ENV="$train_env" ATTN_IMPL="$train_attn" \
    RUN_NAME="$output_name" TRAIN_FILE="$CURRICULUM" VAL_FILE="$VAL_FILE" \
    MAX_STEPS="$MAX_STEPS" EVAL_STEPS=125 SAVE_STEPS=125 SAVE_TOTAL_LIMIT=5 \
    MAX_LENGTH="$TARGET_MAX_LENGTH" \
    EVAL_BATCH_SIZE="$TARGET_EVAL_BATCH_SIZE" \
    TRAIN_BATCH_SIZE="$batch" GRAD_ACCUM_STEPS="$accum" \
    LORA_RANK=64 LORA_ALPHA=128 LORA_DROPOUT=.05 \
    DATASET_SHUFFLE=false TRAIN_DATALOADER_SHUFFLE=false \
    LEARNING_RATE=5e-6 WARMUP_RATIO=.05 \
    INFONCE_HARD_NEGATIVES=7 BASE_MODEL="$MINING_MODEL" \
    BASE_REVISION="$MINING_REVISION" \
    "$ROOT/experiments/020_hard_negative/train_pilot_lora_r64.sh"
}

checkpoint="$("$UTILITY_PYTHON" "$ROOT/scripts/select_best_checkpoint.py" \
  "$ROOT/outputs/$RUN_NAME" --print-path 2>/dev/null)" || checkpoint=""
status=0
if [[ -z "$checkpoint" ]]; then
  train_target "$RUN_NAME" "$TARGET_TRAIN_BATCH_SIZE" "$TARGET_GRAD_ACCUM_STEPS" || status=$?
  if (( status == 0 )); then
    checkpoint="$("$UTILITY_PYTHON" "$ROOT/scripts/select_best_checkpoint.py" \
      "$ROOT/outputs/$RUN_NAME" --print-path 2>/dev/null)" || checkpoint=""
  fi
else
  echo "[$(timestamp)] reusing completed target-adaptation checkpoint: $checkpoint"
fi
if [[ -z "$checkpoint" ]]; then
  fallback="${RUN_NAME}-b${TARGET_FALLBACK_BATCH_SIZE}"
  train_target "$fallback" "$TARGET_FALLBACK_BATCH_SIZE" \
    "$TARGET_FALLBACK_GRAD_ACCUM_STEPS" || exit 7
  RUN_NAME="$fallback"
  MODEL_REL="artifacts/models/${RUN_NAME}-best-merged"
  MODEL_DIR="$ROOT/$MODEL_REL"
  checkpoint="$("$UTILITY_PYTHON" "$ROOT/scripts/select_best_checkpoint.py" \
    "$ROOT/outputs/$RUN_NAME" --print-path)" || exit 7
fi

DATA_UPLOAD_PID=""
if [[ -f "$PUBLISH_HF_TOKEN_FILE" ]]; then
  (
    embedding_load_hf_credential "$PUBLISH_HF_TOKEN_FILE"
    retry_stage "upload-derived-${TARGET_KIND}-replay" 3 \
      "$UTILITY_PYTHON" "$ROOT/scripts/publish_derived_training_dataset.py" \
      --train "$CURRICULUM" --provenance "$CURRICULUM_PROVENANCE" \
      --manifest "$CURRICULUM_MANIFEST" --mining-manifest "$MINING_MANIFEST" \
      --mining-audit "$MINING_AUDIT" --quality-audit "$CURRICULUM_QUALITY" \
      --benchmark-overlap-audit "$CURRICULUM_OVERLAP" \
      --repo-id "$DERIVED_REPO" --title "$DERIVED_TITLE" \
      --source-dataset "$TARGET_SOURCE_DATASET" \
      --source-dataset LLM-OS-Models/korean-embedding-performance-v1-performance-1m \
      --upload --public
  ) >"$LOG_DIR/derived-dataset-upload.log" 2>&1 &
  DATA_UPLOAD_PID=$!
else
  echo "[$(timestamp)] token file unavailable for required derived ${TARGET_KIND} dataset upload" >&2
  exit 10
fi

run_stage "verify-${TARGET_KIND}-target-adapter" \
  "$UTILITY_PYTHON" "$ROOT/scripts/verify_adapter.py" \
  --adapter "$checkpoint" --data "$VAL_FILE" --model "$MINING_MODEL" \
  --output "$LOG_DIR/verification.json" || exit 8
if [[ ! -s "$MODEL_DIR/merge_report.json" ]]; then
  run_stage "merge-${TARGET_KIND}-target-adapter" \
    "$UTILITY_PYTHON" "$ROOT/scripts/merge_embedding_adapter.py" \
    --adapter "$checkpoint" --output-dir "$MODEL_DIR" --base-model "$MINING_MODEL" \
    --base-revision "$MINING_REVISION" --device cuda --dtype bfloat16 \
    --local-files-only || exit 9
else
  run_stage "validate-reused-${TARGET_KIND}-merge" \
    "$UTILITY_PYTHON" "$ROOT/scripts/merge_embedding_adapter.py" \
    --adapter "$checkpoint" --output-dir "$MODEL_DIR" --base-model "$MINING_MODEL" \
    --base-revision "$MINING_REVISION" --dtype bfloat16 --local-files-only \
    --validate-existing || exit 9
fi

model_sha="$(jq -r '.model.weights_sha256' "$MODEL_DIR/merge_report.json")"
revision="model-${model_sha:0:12}"
safe="${MODEL_REL//\//__}"
SIONIC_SUMMARY="$SIONIC_OUT/$safe/summary.json"
OFFICIAL_SUMMARY="$OFFICIAL_OUT/$safe/$revision/summary.json"
if [[ "$ENABLE_PUBLIC_INTERMEDIATE_EVAL" == 1 ]]; then
  run_sionic "$MODEL_REL" "$revision" \
    "$ROOT/outputs/embedding-cache/sionic9-${TARGET_KIND}-target-adapted" || true
  run_official "$MODEL_REL" "$revision" \
    "$ROOT/outputs/embedding-cache/official-${TARGET_KIND}-target-adapted" || true
else
  echo "[$(timestamp)] public intermediate evaluation disabled for $RUN_NAME"
fi

CLEAN_OUT="$ROOT/outputs/evaluation/legal-source-heldout"
for batch in "${CAMPAIGN_EVAL_BATCH_SIZE:-192}"; do
  run_stage "clean-${TARGET_KIND}-target-b$batch" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_legal_source_holdout.py" \
    --model "$MODEL_REL" --revision "$revision" --batch-size "$batch" \
    --max-length 8192 --attn-implementation flash_attention_2 \
    --output-dir "$CLEAN_OUT" \
    --embedding-cache-dir "$ROOT/outputs/embedding-cache/legal-source-heldout" && break
done
CLEAN_SUMMARY="$CLEAN_OUT/$safe/$revision/summary.json"

if [[ "$ENABLE_PUBLIC_INTERMEDIATE_EVAL" == 1 \
    && -s "$SIONIC_SUMMARY" && -s "$OFFICIAL_SUMMARY" ]]; then
  run_stage "record-${TARGET_KIND}-target-result" "$ROOT/scripts/commit_campaign_result.sh" \
    --stage "$CAMPAIGN_STAGE" --model "$MODEL_REL" \
    --repo-id "LLM-OS-Models2/${RUN_NAME}-candidates" \
    --sionic-summary "$SIONIC_SUMMARY" --official-summary "$OFFICIAL_SUMMARY"
fi
if [[ -n "$DATA_UPLOAD_PID" ]]; then
  if ! wait "$DATA_UPLOAD_PID"; then
    echo "[$(timestamp)] derived $TARGET_KIND dataset upload failed" >&2
    exit 10
  fi
fi
echo "[$(timestamp)] Sionic $TARGET_KIND target-adaptation queue complete"
