#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
embedding_resolve_train_runtime
UTILITY_PYTHON="$EMBEDDING_TRAIN_PYTHON"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-$ROOT/outputs/reranker-kd-20260717-frontier}"
DATA_DIR="$ROOT/outputs/data/performance-v1/performance-1m/reranker-kd-pilot"
RAW_TRAIN="$ROOT/outputs/data/performance-v1/performance-1m/train.jsonl"
VAL_FILE="$ROOT/outputs/data/validation/legal-source-heldout-i-v2-text-strict-512/validation.jsonl"
REQUESTS="$DATA_DIR/teacher-requests.jsonl"
MINED_TRAIN="$DATA_DIR/train.faiss-1m-student-r095-n7.jsonl"
MINING_AUDIT="$DATA_DIR/train.faiss-1m-student-r095-n7.audit.jsonl"
MINING_MANIFEST="$DATA_DIR/train.faiss-1m-student-r095-n7.manifest.json"
SCORE_DIR="$DATA_DIR/qwen3-reranker-8b-scores"
KD_TRAIN="$DATA_DIR/train.reranker-quantile-kd15.jsonl"
KD_AUDIT="$DATA_DIR/train.reranker-quantile-kd15.audit.jsonl"
KD_MANIFEST="$DATA_DIR/train.reranker-quantile-kd15.manifest.json"
SELECTION="$LOG_DIR/clean-first-selection.json"
GENERAL_TRAINING_MANIFEST="${GENERAL_TRAINING_MANIFEST:-$ROOT/outputs/data/performance-v1/performance-1m/homogeneous-b16.manifest.json}"
GENERAL_BASE_UPLOAD_REPORT="${GENERAL_BASE_UPLOAD_REPORT:-$ROOT/outputs/scale-1m-20260717-frontier/private-clean-candidate-upload.json}"
CLEAN_OUT="$ROOT/outputs/evaluation/legal-source-heldout"
ROBUST_OUT="$ROOT/outputs/evaluation/conversational-noise-robustness"
REQUEST_ROWS="${KD_REQUEST_ROWS:-10000}"
CANDIDATES_PER_QUERY="${KD_CANDIDATES_PER_QUERY:-200}"
NEGATIVES_PER_QUERY="${KD_NEGATIVES_PER_QUERY:-15}"
mkdir -p "$LOG_DIR" "$DATA_DIR"
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1

unset HF_TOKEN HUGGINGFACE_HUB_TOKEN
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$ROOT/scripts:$ROOT/third_party/mteb${PYTHONPATH:+:$PYTHONPATH}"
OFFLINE_ENV=(env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1)

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
run_stage() {
  local name="$1"; shift
  echo "[$(timestamp)] START $name"
  "$@"
  local status=$?
  echo "[$(timestamp)] END $name status=$status"
  return "$status"
}

BASE_MODEL="${GENERAL_BASE_MODEL:-$ROOT/artifacts/models/qwen3-embedding-8b-ko-performance1m-lora-r64-best-merged}"
if [[ ! -s "$BASE_MODEL/merge_report.json" ]]; then
  BASE_MODEL="$ROOT/artifacts/models/qwen3-embedding-8b-ko-performance1m-lora-r64-b8-best-merged"
fi
if [[ ! -s "$BASE_MODEL/merge_report.json" || ! -s "$RAW_TRAIN" || ! -s "$VAL_FILE" ]]; then
  echo "[$(timestamp)] 1M base or KD input is unavailable" >&2
  exit 2
fi
if [[ ! "$REQUEST_ROWS" =~ ^[1-9][0-9]*$ \
    || ! "$CANDIDATES_PER_QUERY" =~ ^[1-9][0-9]*$ \
    || ! "$NEGATIVES_PER_QUERY" =~ ^[1-9][0-9]*$ ]]; then
  echo "[$(timestamp)] invalid KD row/candidate/negative count" >&2
  exit 2
fi
if (( CANDIDATES_PER_QUERY > 200 \
    || NEGATIVES_PER_QUERY > CANDIDATES_PER_QUERY )); then
  echo "[$(timestamp)] KD supports at most 200 candidates and negatives <= candidates" >&2
  exit 2
fi

embedding_require_storage_headroom "$ROOT" 500 1000000
embedding_require_storage_headroom /tmp 50 100000
if [[ ! -s "$REQUESTS" ]]; then
  run_stage mine-wide-current-student-candidates \
    "${OFFLINE_ENV[@]}" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/mine_faiss_hard_negatives.py" \
    --input "$RAW_TRAIN" --output "$MINED_TRAIN" \
    --audit-output "$MINING_AUDIT" --manifest-output "$MINING_MANIFEST" \
    --work-dir "$DATA_DIR/faiss-work-1m-student" --keep-work-dir \
    --model "$BASE_MODEL" --revision "" --encode-batch-size 128 \
    --candidate-pool-size 24 --num-negatives 7 --selection-strategy score_rank_quantiles \
    --positive-relative-ratio .95 --search-k 512 --nlist 1024 --nprobe 64 \
    --training-points 100000 --faiss-threads "$EFFECTIVE_CPU_COUNT" \
    --teacher-request-output "$REQUESTS" \
    --teacher-request-limit "$REQUEST_ROWS" \
    --teacher-candidate-count "$CANDIDATES_PER_QUERY" \
    --allow-target-adapted || exit 3
fi

embedding_require_storage_headroom "$ROOT" 500 1000000
embedding_require_storage_headroom /tmp 50 100000
if [[ ! -s "$SCORE_DIR/manifest.json" ]]; then
  run_stage score-wide-candidates-with-qwen3-reranker \
    "${OFFLINE_ENV[@]}" \
    "$UTILITY_PYTHON" "$ROOT/scripts/cache_qwen3_reranker_scores.py" \
    --input "$REQUESTS" --output-dir "$SCORE_DIR" \
    --device cuda --dtype bfloat16 --attention-implementation flash_attention_2 \
    --max-length 512 --model-batch-size "${KD_RERANKER_BATCH_SIZE:-8}" \
    --shard-size 32 || exit 4
else
  run_stage verify-reranker-score-cache \
    "${OFFLINE_ENV[@]}" \
    "$UTILITY_PYTHON" "$ROOT/scripts/cache_qwen3_reranker_scores.py" \
    --input "$REQUESTS" --output-dir "$SCORE_DIR" --verify-only || exit 4
fi

if [[ ! -s "$KD_MANIFEST" ]]; then
  run_stage compile-reranker-quantile-kd15 \
    "$UTILITY_PYTHON" "$ROOT/scripts/compile_reranker_kd_dataset.py" \
    --requests "$REQUESTS" --score-cache-dir "$SCORE_DIR" \
    --output "$KD_TRAIN" --audit "$KD_AUDIT" --manifest "$KD_MANIFEST" \
    --candidate-pool-size "$CANDIDATES_PER_QUERY" \
    --negatives-per-query "$NEGATIVES_PER_QUERY" \
    --positive-relative-ratio .95 --absolute-positive-margin .02 \
    --minimum-positive-score .5 --minimum-negative-score 0 || exit 5
fi

KD_ROWS="$(jq -r '.counters.output_rows // 0' "$KD_MANIFEST")"
if [[ ! "$KD_ROWS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[$(timestamp)] KD compiler emitted no admissible rows" >&2
  exit 5
fi
MAX_STEPS="$((KD_ROWS / 64))"
(( MAX_STEPS > 0 )) || exit 5
CHECKPOINT_INTERVAL=25
(( MAX_STEPS < CHECKPOINT_INTERVAL )) && CHECKPOINT_INTERVAL="$MAX_STEPS"

DATA_UPLOAD_PID=""
if [[ -f "$ROOT/.env" ]]; then
  (
    set -a
    source "$ROOT/.env"
    set +a
    for attempt in 1 2 3; do
      "$UTILITY_PYTHON" "$ROOT/scripts/publish_reranker_kd_dataset.py" \
        --train "$KD_TRAIN" --audit "$KD_AUDIT" --manifest "$KD_MANIFEST" \
        --requests "$REQUESTS" --score-cache-dir "$SCORE_DIR" \
        --repo-id LLM-OS-Models2/korean-embedding-qwen3-reranker-kd-pilot-v1 \
        --upload && exit 0
      (( attempt == 3 )) || sleep 15
    done
    exit 1
  ) >"$LOG_DIR/private-dataset-upload.log" 2>&1 &
  DATA_UPLOAD_PID=$!
  echo "[$(timestamp)] private KD dataset upload started pid=$DATA_UPLOAD_PID"
else
  echo "[$(timestamp)] .env unavailable; private KD dataset upload skipped" >&2
fi

variants=(
  "filter-only|1.0|0.0|kl|0"
  "listwise-kl07|0.3|0.7|kl|0"
  "listwise-kl07-queue4096|0.3|0.7|kl|4096"
)
if [[ "${ENABLE_MARGIN_MSE_KD:-0}" == 1 ]]; then
  variants+=("margin-mse07|0.3|0.7|margin_mse|0")
fi

candidate_args=()
base_rel="${BASE_MODEL#"$ROOT/"}"
base_weights_sha="$(jq -r '.model.weights_sha256' "$BASE_MODEL/merge_report.json")"
base_revision="model-${base_weights_sha:0:12}"
base_safe="${base_rel//\//__}"
base_clean="$CLEAN_OUT/$base_safe/$base_revision/summary.json"
if [[ ! -s "$base_clean" ]]; then
  for batch in 192 128 64 32 16 8 4 2; do
    run_stage "clean-1m-base-b$batch" "${OFFLINE_ENV[@]}" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_legal_source_holdout.py" \
      --model "$base_rel" --revision "$base_revision" --batch-size "$batch" \
      --max-length 8192 --attn-implementation flash_attention_2 \
      --output-dir "$CLEAN_OUT" \
      --embedding-cache-dir "$ROOT/outputs/embedding-cache/legal-source-heldout" && break
  done
fi
base_robust="$ROBUST_OUT/$base_safe/$base_revision/summary.json"
if [[ -s "$base_clean" && ! -s "$base_robust" ]]; then
  for batch in 192 128 64 32 16 8 4 2; do
    run_stage "robustness-1m-base-b$batch" "${OFFLINE_ENV[@]}" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_conversational_noise_robustness.py" \
      --model "$base_rel" --revision "$base_revision" --batch-size "$batch" \
      --max-length 8192 --attn-implementation flash_attention_2 \
      --output-dir "$ROBUST_OUT" \
      --embedding-cache-dir "$ROOT/outputs/embedding-cache/legal-source-heldout" && break
  done
fi
[[ -s "$base_clean" ]] && candidate_args+=(--candidate-model "$base_rel")

for variant in "${variants[@]}"; do
  IFS='|' read -r label hard_weight kd_weight kd_mode queue_size <<< "$variant"
  run_name="qwen3-embedding-8b-ko-performance1m-reranker-${label}-lora-r64"
  run_dir="$ROOT/outputs/$run_name"
  if ! find "$run_dir" -maxdepth 3 -type d -name "checkpoint-$MAX_STEPS" -print -quit 2>/dev/null | grep -q .; then
    embedding_require_storage_headroom "$ROOT" 500 1000000
    embedding_require_storage_headroom /tmp 50 100000
    run_stage "train-$label" env \
      EMBEDDING_OFFLINE=1 ENABLE_VALIDATED_CONTINUAL_BASE=0 AUTO_SELECT_FA2=0 \
      ENABLE_PRIVATE_CHECKPOINT_WATCHER=1 \
      CHECKPOINT_TRAINING_MANIFEST="$KD_MANIFEST" \
      CHECKPOINT_BASE_UPLOAD_REPORT="$GENERAL_BASE_UPLOAD_REPORT" \
      PRIVATE_CHECKPOINT_REPO_ID="LLM-OS-Models2/${run_name}-candidates" \
      TRAIN_ENV="$EMBEDDING_TRAIN_ENV" ATTN_IMPL=sdpa ENABLE_LISTWISE_KD=1 \
      EMBEDDING_KD_HARD_WEIGHT="$hard_weight" EMBEDDING_KD_WEIGHT="$kd_weight" \
      EMBEDDING_KD_MODE="$kd_mode" EMBEDDING_KD_QUEUE_SIZE="$queue_size" \
      EMBEDDING_KD_TEACHER_TEMPERATURE=1.0 EMBEDDING_KD_STUDENT_TEMPERATURE=.02 \
      RUN_NAME="$run_name" TRAIN_FILE="$KD_TRAIN" VAL_FILE="$VAL_FILE" \
      MAX_STEPS="$MAX_STEPS" EVAL_STEPS="$CHECKPOINT_INTERVAL" \
      SAVE_STEPS="$CHECKPOINT_INTERVAL" SAVE_TOTAL_LIMIT=5 \
      TRAIN_BATCH_SIZE=2 EVAL_BATCH_SIZE=4 GRAD_ACCUM_STEPS=32 MAX_LENGTH=512 \
      LORA_RANK=64 LORA_ALPHA=128 LORA_DROPOUT=.05 LEARNING_RATE=5e-6 \
      WARMUP_RATIO=.05 INFONCE_HARD_NEGATIVES="$NEGATIVES_PER_QUERY" \
      DATASET_SHUFFLE=false TRAIN_DATALOADER_SHUFFLE=false \
      BASE_MODEL="$BASE_MODEL" BASE_REVISION= \
      "$ROOT/experiments/020_hard_negative/train_pilot_lora_r64.sh" || continue
  fi
  checkpoint="$("$UTILITY_PYTHON" "$ROOT/scripts/select_best_checkpoint.py" \
    "$run_dir" --print-path 2>/dev/null)" || checkpoint=""
  [[ -n "$checkpoint" ]] || continue
  merged_rel="artifacts/models/${run_name}-best-merged"
  merged="$ROOT/$merged_rel"
  if [[ ! -s "$merged/merge_report.json" ]]; then
    run_stage "merge-$label" "${OFFLINE_ENV[@]}" \
      "$UTILITY_PYTHON" "$ROOT/scripts/merge_embedding_adapter.py" \
      --adapter "$checkpoint" --output-dir "$merged" \
      --base-model "$BASE_MODEL" --base-revision "" \
      --device cuda --dtype bfloat16 --local-files-only || continue
  fi
  weights_sha="$(jq -r '.model.weights_sha256' "$merged/merge_report.json")"
  revision="model-${weights_sha:0:12}"
  safe="${merged_rel//\//__}"
  clean_summary="$CLEAN_OUT/$safe/$revision/summary.json"
  if [[ ! -s "$clean_summary" ]]; then
    for batch in 192 128 64 32 16 8 4 2; do
      run_stage "clean-$label-b$batch" "${OFFLINE_ENV[@]}" \
        "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_legal_source_holdout.py" \
        --model "$merged_rel" --revision "$revision" --batch-size "$batch" \
        --max-length 8192 --attn-implementation flash_attention_2 \
        --output-dir "$CLEAN_OUT" \
        --embedding-cache-dir "$ROOT/outputs/embedding-cache/legal-source-heldout" && break
    done
  fi
  robustness_summary="$ROBUST_OUT/$safe/$revision/summary.json"
  if [[ -s "$clean_summary" && ! -s "$robustness_summary" ]]; then
    for batch in 192 128 64 32 16 8 4 2; do
      run_stage "robustness-$label-b$batch" "${OFFLINE_ENV[@]}" \
        "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_conversational_noise_robustness.py" \
        --model "$merged_rel" --revision "$revision" --batch-size "$batch" \
        --max-length 8192 --attn-implementation flash_attention_2 \
        --output-dir "$ROBUST_OUT" \
        --embedding-cache-dir "$ROOT/outputs/embedding-cache/legal-source-heldout" && break
    done
  fi
  [[ -s "$clean_summary" ]] && candidate_args+=(--candidate-model "$merged_rel")
done

if (( ${#candidate_args[@]} == 0 )); then
  echo "[$(timestamp)] no KD candidate passed clean evaluation" >&2
  exit 6
fi
run_stage select-clean-reranker-kd-winner \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/select_best_clean_model.py" \
  "$CLEAN_OUT" "$ROBUST_OUT" --workspace-root "$ROOT" \
  --output "$SELECTION" --disqualification-root "$ROOT/outputs" \
  "${candidate_args[@]}" || exit 6

MODEL_UPLOAD_REPORT="$LOG_DIR/private-clean-candidate-upload.json"
if [[ "$(jq -r '.visibility + ":" + (.remote_manifest_exact|tostring) + ":" + (.remote_file_set_exact|tostring)' \
    "$MODEL_UPLOAD_REPORT" 2>/dev/null)" != "private:true:true" ]]; then
  if [[ ! -f "$ROOT/.env" ]]; then
    echo "[$(timestamp)] .env unavailable for required KD winner backup" >&2
    exit 7
  fi
  selected_model_rel="$(jq -r '.best.model // empty' "$SELECTION")"
  selected_model="$ROOT/$selected_model_rel"
  if [[ -n "$selected_model_rel" && -s "$selected_model/merge_report.json" ]]; then
    selected_training_manifest="$KD_MANIFEST"
    if [[ "$selected_model_rel" == "$base_rel" ]]; then
      selected_training_manifest="$GENERAL_TRAINING_MANIFEST"
    fi
    if [[ ! -s "$selected_training_manifest" ]]; then
      echo "[$(timestamp)] selected model training manifest is unavailable" >&2
      exit 7
    else
      embedding_require_storage_headroom "$ROOT" 500 1000000
      run_stage "publish-private-clean-selected-kd-winner" \
        env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN \
        "$UTILITY_PYTHON" "$ROOT/scripts/publish_private_clean_candidate.py" \
        --model-dir "$selected_model" --selection "$SELECTION" \
        --training-manifest "$selected_training_manifest" \
        --repo-id LLM-OS-Models2/qwen3-embedding-8b-ko-reranker-kd-clean-winner-v1-private \
        --hf-token-file "$ROOT/.env" --report-output "$MODEL_UPLOAD_REPORT" \
        --upload >"$LOG_DIR/private-model-upload.log" 2>&1 || exit 7
    fi
  else
    echo "[$(timestamp)] selected KD winner is unavailable for private backup" >&2
    exit 7
  fi
fi
if [[ "$(jq -r '.visibility + ":" + (.remote_manifest_exact|tostring) + ":" + (.remote_file_set_exact|tostring)' \
    "$MODEL_UPLOAD_REPORT" 2>/dev/null)" != "private:true:true" ]]; then
  echo "[$(timestamp)] KD winner private remote verification is incomplete" >&2
  exit 7
fi

if [[ -n "$DATA_UPLOAD_PID" ]]; then
  if wait "$DATA_UPLOAD_PID"; then
    echo "[$(timestamp)] private KD dataset upload complete"
  else
    echo "[$(timestamp)] private KD dataset upload failed; see upload log" >&2
  fi
fi

echo "[$(timestamp)] reranker KD ablation queue complete selection=$SELECTION"
