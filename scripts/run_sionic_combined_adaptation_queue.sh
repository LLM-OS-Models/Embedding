#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
source "$ROOT/scripts/backend_admission.sh"
embedding_resolve_train_runtime
UTILITY_PYTHON="$EMBEDDING_TRAIN_PYTHON"
cd "$ROOT"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/sionic-combined-adaptation-20260712}"
OUT_DIR="$ROOT/outputs/data/sionic-combined-target-v1"
CURRICULUM="$OUT_DIR/train.multidomain.jsonl"
PROVENANCE="$OUT_DIR/provenance.multidomain.jsonl"
MANIFEST="$OUT_DIR/multidomain.manifest.json"
QUALITY="$OUT_DIR/multidomain.quality-audit.json"
OVERLAP="$OUT_DIR/multidomain.benchmark-overlap-audit.json"
VAL_FILE="$ROOT/data/processed/ko_triplet_pilot_10k/validation.hn-qwen3-r095-n4.jsonl"
RUN_NAME="qwen3-embedding-8b-ko-sionic-combined-target-lora-r64"
MODEL_REL="artifacts/models/${RUN_NAME}-best-merged"
MODEL_DIR="$ROOT/$MODEL_REL"
SIONIC_OUT="$ROOT/outputs/evaluation/sionic9-combined-target-adapted"
OFFICIAL_OUT="$ROOT/outputs/evaluation/mteb-korean-v1-combined-target-adapted"
mkdir -p "$LOG_DIR" "$OUT_DIR" "$SIONIC_OUT" "$OFFICIAL_OUT"
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1

if [[ -f "$ROOT/.env" ]]; then
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "$ROOT/.env" | tail -n 1)"
  export HF_TOKEN
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$ROOT/scripts:$ROOT/third_party/mteb${PYTHONPATH:+:$PYTHONPATH}"

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
    run_stage "sionic9-combined-b$batch" \
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
    run_stage "official-korean-combined-b$batch" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_mteb_korean_v1.py" \
      --model "$model" --revision "$revision" --batch-size "$batch" \
      --max-length 8192 --qwen3-instruction-loader \
      --attn-implementation flash_attention_2 --output-dir "$OFFICIAL_OUT" \
      --embedding-cache-dir "$cache" && return 0
  done
  return 1
}
component() {
  local role="$1" train="$2" provenance="$3" manifest="$4" desired="$5"
  [[ -s "$train" && -s "$provenance" && -s "$manifest" ]] || return 1
  local available rows
  available="$(jq -r '.output_rows // 0' "$manifest")"
  (( available > 0 )) || return 1
  rows="$desired"
  (( rows > available )) && rows="$available"
  rows="$((rows / 16 * 16))"
  (( rows > 0 )) || return 1
  printf '%s=%s=%s=%s' "$role" "$train" "$provenance" "$rows"
}

SQUAD="$ROOT/outputs/data/performance-v1/sionic-squad-train-60k"
HEALTH="$ROOT/outputs/data/performance-v1/sionic-health-multilingual-100k"
AUTORAG="$ROOT/outputs/data/performance-v1/sionic-autorag-domain-100k"
RETRIEVAL="$ROOT/outputs/data/performance-v1/sionic-retrieval-train-family-4146"
LEGAL="$ROOT/outputs/data/legal-performance-v1"
GENERAL="$ROOT/outputs/data/performance-v1/performance-1m"
GENERAL_TRAIN="$GENERAL/train.homogeneous-b16.jsonl"
GENERAL_PROVENANCE="$GENERAL/provenance.homogeneous-b16.jsonl"
GENERAL_MANIFEST="$GENERAL/homogeneous-b16.manifest.json"
if [[ -s "$GENERAL/faiss-current-r095-n7.homogeneous-b16.manifest.json" ]]; then
  GENERAL_TRAIN="$GENERAL/train.faiss-current-r095-n7.homogeneous-b16.jsonl"
  GENERAL_PROVENANCE="$GENERAL/provenance.faiss-current-r095-n7.homogeneous-b16.jsonl"
  GENERAL_MANIFEST="$GENERAL/faiss-current-r095-n7.homogeneous-b16.manifest.json"
fi

components=()
components+=("$(component squad \
  "$SQUAD/train.faiss-current-r095-n7.homogeneous-b16.jsonl" \
  "$SQUAD/provenance.faiss-current-r095-n7.homogeneous-b16.jsonl" \
  "$SQUAD/faiss-current-r095-n7.homogeneous-b16.manifest.json" 40000)") || exit 2
components+=("$(component health \
  "$HEALTH/train.faiss-current-r095-n7.homogeneous-b16.jsonl" \
  "$HEALTH/provenance.faiss-current-r095-n7.homogeneous-b16.jsonl" \
  "$HEALTH/faiss-current-r095-n7.homogeneous-b16.manifest.json" 40000)") || exit 2
components+=("$(component autorag \
  "$AUTORAG/train.faiss-current-r095-n7.homogeneous-b16.jsonl" \
  "$AUTORAG/provenance.faiss-current-r095-n7.homogeneous-b16.jsonl" \
  "$AUTORAG/faiss-current-r095-n7.homogeneous-b16.manifest.json" 40000)") || exit 2
components+=("$(component retrieval_family \
  "$RETRIEVAL/train.faiss-current-r095-n7.homogeneous-b16.jsonl" \
  "$RETRIEVAL/provenance.faiss-current-r095-n7.homogeneous-b16.jsonl" \
  "$RETRIEVAL/faiss-current-r095-n7.homogeneous-b16.manifest.json" 4128)") || exit 2
components+=("$(component legal \
  "$LEGAL/train.faiss-r095-n7.homogeneous-b16.jsonl" \
  "$LEGAL/provenance.faiss-r095-n7.homogeneous-b16.jsonl" \
  "$LEGAL/faiss-r095-n7.homogeneous-b16.manifest.json" 60000)") || exit 2
components+=("$(component general "$GENERAL_TRAIN" "$GENERAL_PROVENANCE" \
  "$GENERAL_MANIFEST" 215872)") || exit 2

if [[ ! -s "$MANIFEST" ]]; then
  component_args=()
  for value in "${components[@]}"; do component_args+=(--component "$value"); done
  run_stage build-sionic-combined-curriculum \
    "$UTILITY_PYTHON" "$ROOT/scripts/build_multidomain_curriculum.py" \
    "${component_args[@]}" --output "$CURRICULUM" \
    --provenance-output "$PROVENANCE" --manifest-output "$MANIFEST" \
    --batch-size 16 --seed 42 \
    --adaptation-label target-adapted-sionic-combined-v1 || exit 3
fi
run_stage audit-sionic-combined-quality \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/audit_embedding_training_data.py" \
  --train "$CURRICULUM" --provenance "$PROVENANCE" --output "$QUALITY" \
  --expected-batch-size 16 || exit 3
run_stage audit-sionic-combined-overlap \
  "$UTILITY_PYTHON" "$ROOT/scripts/audit_training_benchmark_overlap.py" \
  --train "$CURRICULUM" --provenance "$PROVENANCE" \
  --blocklist-root "$ROOT/outputs/decontamination/benchmark_blocklist" \
  --output "$OVERLAP" --fail-on-critical || exit 3

BASE_MODEL="$ROOT/artifacts/models/qwen3-embedding-8b-ko-performance1m-lora-r64-best-merged"
if [[ ! -s "$BASE_MODEL/merge_report.json" ]]; then
  BASE_MODEL="$ROOT/artifacts/models/qwen3-embedding-8b-ko-performance1m-lora-r64-b8-best-merged"
fi
[[ -s "$BASE_MODEL/merge_report.json" && -s "$VAL_FILE" ]] || exit 4
MAX_STEPS="$(jq -r '.output_rows / 64 | floor' "$MANIFEST")"

train_combined() {
  local name="$1" batch="$2" accum="$3"
  local train_env="$ROOT/.venv-train" train_attn=sdpa admission_report
  local admission_key="sionic-combined-lora-r64-b${batch}-a${accum}-m512-hn7"
  if embedding_select_fa2_backend "$CURRICULUM" "$admission_key" \
      "$batch" "$accum" 512 64 128 bfloat16 "$BASE_MODEL" "" 7 .05; then
    train_env="$BACKEND_ADMISSION_ENV"
    train_attn="$BACKEND_ADMISSION_ATTN"
  fi
  admission_report="$BACKEND_ADMISSION_REPORT"
  echo "[$(timestamp)] combined training backend=$train_attn env=$train_env admission=$admission_report"
  run_stage "train-$name" env TRAIN_ENV="$train_env" ATTN_IMPL="$train_attn" \
    RUN_NAME="$name" TRAIN_FILE="$CURRICULUM" VAL_FILE="$VAL_FILE" \
    MAX_STEPS="$MAX_STEPS" EVAL_STEPS=250 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=3 \
    TRAIN_BATCH_SIZE="$batch" GRAD_ACCUM_STEPS="$accum" \
    MAX_LENGTH=512 LORA_RANK=64 LORA_ALPHA=128 LORA_DROPOUT=.05 \
    DATASET_SHUFFLE=false TRAIN_DATALOADER_SHUFFLE=false \
    LEARNING_RATE=5e-6 WARMUP_RATIO=.05 \
    INFONCE_HARD_NEGATIVES=7 BASE_MODEL="$BASE_MODEL" BASE_REVISION= \
    "$ROOT/experiments/020_hard_negative/train_pilot_lora_r64.sh"
}

checkpoint="$("$UTILITY_PYTHON" "$ROOT/scripts/select_best_checkpoint.py" \
  "$ROOT/outputs/$RUN_NAME" --print-path 2>/dev/null)" || checkpoint=""
if [[ -z "$checkpoint" ]]; then
  train_combined "$RUN_NAME" 8 8 || true
  checkpoint="$("$UTILITY_PYTHON" "$ROOT/scripts/select_best_checkpoint.py" \
    "$ROOT/outputs/$RUN_NAME" --print-path 2>/dev/null)" || checkpoint=""
fi
if [[ -z "$checkpoint" ]]; then
  fallback="${RUN_NAME}-b4"
  train_combined "$fallback" 4 16 || exit 5
  RUN_NAME="$fallback"
  MODEL_REL="artifacts/models/${RUN_NAME}-best-merged"
  MODEL_DIR="$ROOT/$MODEL_REL"
  checkpoint="$("$UTILITY_PYTHON" "$ROOT/scripts/select_best_checkpoint.py" \
    "$ROOT/outputs/$RUN_NAME" --print-path)" || exit 5
fi

retry_stage upload-combined-curriculum 3 \
  "$UTILITY_PYTHON" "$ROOT/scripts/publish_derived_training_dataset.py" \
  --train "$CURRICULUM" --provenance "$PROVENANCE" --manifest "$MANIFEST" \
  --quality-audit "$QUALITY" --benchmark-overlap-audit "$OVERLAP" \
  --repo-id LLM-OS-Models2/korean-embedding-sionic-combined-replay-v1 \
  --title "Korean Sionic Combined Target Domains with General Replay" \
  --source-dataset LLM-OS-Models2/korean-embedding-sionic-squad-quantile-hn7-replay-v1 \
  --source-dataset LLM-OS-Models2/korean-embedding-sionic-health-quantile-hn7-replay-v1 \
  --source-dataset LLM-OS-Models2/korean-embedding-sionic-autorag-quantile-hn7-replay-v1 \
  --source-dataset LLM-OS-Models2/korean-embedding-sionic-retrieval-family-quantile-hn7-replay-v1 \
  --source-dataset LLM-OS-Models2/korean-legal-quantile-hn7-replay-v1 \
  --source-dataset LLM-OS-Models/korean-embedding-performance-v1-performance-1m \
  --upload --public >"$LOG_DIR/dataset-upload.log" 2>&1 &
DATA_UPLOAD_PID=$!

run_stage verify-combined-adapter \
  "$UTILITY_PYTHON" "$ROOT/scripts/verify_adapter.py" \
  --adapter "$checkpoint" --data "$VAL_FILE" --model "$BASE_MODEL" \
  --output "$LOG_DIR/verification.json" || exit 6
if [[ ! -s "$MODEL_DIR/merge_report.json" ]]; then
  run_stage merge-combined-adapter \
    "$UTILITY_PYTHON" "$ROOT/scripts/merge_embedding_adapter.py" \
    --adapter "$checkpoint" --output-dir "$MODEL_DIR" --base-model "$BASE_MODEL" \
    --base-revision "" --device cuda --dtype bfloat16 --local-files-only || exit 7
fi
model_sha="$(jq -r '.model.weights_sha256' "$MODEL_DIR/merge_report.json")"
revision="model-${model_sha:0:12}"
run_sionic "$MODEL_REL" "$revision" \
  "$ROOT/outputs/embedding-cache/sionic9-combined-target-adapted" || true
safe="${MODEL_REL//\//__}"
SIONIC_SUMMARY="$SIONIC_OUT/$safe/summary.json"
run_official "$MODEL_REL" "$revision" \
  "$ROOT/outputs/embedding-cache/official-combined-target-adapted" || true
OFFICIAL_SUMMARY="$OFFICIAL_OUT/$safe/$revision/summary.json"

CLEAN_OUT="$ROOT/outputs/evaluation/legal-source-heldout"
for batch in "${CAMPAIGN_EVAL_BATCH_SIZE:-192}"; do
  run_stage "clean-combined-b$batch" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_legal_source_holdout.py" \
    --model "$MODEL_REL" --revision "$revision" --batch-size "$batch" \
    --max-length 8192 --attn-implementation flash_attention_2 \
    --output-dir "$CLEAN_OUT" \
    --embedding-cache-dir "$ROOT/outputs/embedding-cache/legal-source-heldout" && break
done
CLEAN_SUMMARY="$CLEAN_OUT/$safe/$revision/summary.json"
if [[ -s "$SIONIC_SUMMARY" && -s "$OFFICIAL_SUMMARY" ]]; then
  clean_args=()
  [[ -s "$CLEAN_SUMMARY" ]] && clean_args+=(--clean-summary "$CLEAN_SUMMARY")
  if retry_stage publish-combined-model 3 \
    "$UTILITY_PYTHON" "$ROOT/scripts/publish_best_embedding_model.py" \
    --model-dir "$MODEL_DIR" --sionic-summary "$SIONIC_SUMMARY" \
    --official-summary "$OFFICIAL_SUMMARY" "${clean_args[@]}" \
    --training-manifest "$MANIFEST" \
    --repo-id LLM-OS-Models2/qwen3-embedding-8b-ko-sionic-combined-v1-private-candidate \
    --upload; then
    run_stage record-combined-result "$ROOT/scripts/commit_campaign_result.sh" \
      --stage sionic-combined --model "$MODEL_REL" \
      --repo-id LLM-OS-Models2/qwen3-embedding-8b-ko-sionic-combined-v1-private-candidate \
      --sionic-summary "$SIONIC_SUMMARY" --official-summary "$OFFICIAL_SUMMARY"
  fi
fi
wait "$DATA_UPLOAD_PID" || echo "[$(timestamp)] combined dataset upload failed" >&2
echo "[$(timestamp)] combined Sionic target-adaptation queue complete"
