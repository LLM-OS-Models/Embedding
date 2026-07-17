#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
source "$ROOT/scripts/backend_admission.sh"
embedding_resolve_train_runtime
TRAIN_ENV="${TRAIN_ENV:-$EMBEDDING_TRAIN_ENV}"
DATA_DIR="${DATA_DIR:-$ROOT/data/processed/ko_triplet_pilot_10k}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.hn-qwen3-r095-n4.jsonl}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/validation.hn-qwen3-r095-n4.jsonl}"
RUN_NAME="${RUN_NAME:-qwen3-embedding-8b-ko-hn10k-lora-r64}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/$RUN_NAME}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-Embedding-8B}"
BASE_REVISION="${BASE_REVISION-1d8ad4ca9b3dd8059ad90a75d4983776a23d44af}"

# Promote the 50K model into the 200K curriculum only when its held-out loss
# actually beats the 10K hard-negative pilot. Dataset scale alone is not a
# sufficient promotion signal.
if [[ "${ENABLE_VALIDATED_CONTINUAL_BASE:-1}" == 1 \
    && "$RUN_NAME" == *performance200k* \
    && "$BASE_MODEL" == Qwen/Qwen3-Embedding-8B ]]; then
  candidate_run="$ROOT/outputs/qwen3-embedding-8b-ko-performance50k-lora-r64"
  candidate_model="$ROOT/artifacts/models/qwen3-embedding-8b-ko-performance50k-lora-r64-best-merged"
  if [[ ! -s "$candidate_model/merge_report.json" ]]; then
    candidate_model="$ROOT/artifacts/models/qwen3-embedding-8b-ko-performance50k-lora-r64-b8-best-merged"
    candidate_run="$ROOT/outputs/qwen3-embedding-8b-ko-performance50k-lora-r64-b8"
  fi
  pilot_run="$ROOT/outputs/qwen3-embedding-8b-ko-hn10k-lora-r64"
  if [[ -s "$candidate_model/merge_report.json" ]] && \
      PROJECT_ROOT="$ROOT" CANDIDATE_RUN="$candidate_run" PILOT_RUN="$pilot_run" \
      "$TRAIN_ENV/bin/python" - <<'PY'
import json, os, subprocess, sys
selector = os.path.join(os.environ['PROJECT_ROOT'], 'scripts', 'select_best_checkpoint.py')
def loss(path):
    raw = subprocess.check_output([sys.executable, selector, path], text=True)
    return json.loads(raw).get('selected_eval_loss')
candidate = loss(os.environ['CANDIDATE_RUN'])
pilot = loss(os.environ['PILOT_RUN'])
raise SystemExit(0 if candidate is not None and pilot is not None and candidate < pilot else 1)
PY
  then
    BASE_MODEL="$candidate_model"
    BASE_REVISION=""
    LEARNING_RATE="${LEARNING_RATE:-1e-5}"
    echo "validated continual base promoted: $BASE_MODEL" >&2
  fi
fi
if [[ "$RUN_NAME" == *performance200k* ]]; then
  # The 174.6M-parameter r64 adapter is no longer a tiny perturbation. Use the
  # lower stable LR for the longer 200K run even when continual promotion did
  # not pass; explicit caller overrides still win.
  LEARNING_RATE="${LEARNING_RATE:-1e-5}"
fi

embedding_configure_hf_access
embedding_require_storage_headroom "$ROOT" \
  "${MIN_WORKSPACE_FREE_GIB:-500}" "${MIN_WORKSPACE_FREE_INODES:-1000000}"
embedding_require_storage_headroom /tmp \
  "${MIN_TMP_FREE_GIB:-50}" "${MIN_TMP_FREE_INODES:-100000}"

for path in "$TRAIN_FILE" "$VAL_FILE"; do
  if [[ ! -f "$path" ]]; then
    echo "missing mined dataset: $path" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export INFONCE_TEMPERATURE="${INFONCE_TEMPERATURE:-0.02}"
export INFONCE_USE_BATCH="true"
export INFONCE_HARD_NEGATIVES="${INFONCE_HARD_NEGATIVES:-4}"
export INFONCE_MASK_FAKE_NEGATIVE="true"
export INFONCE_FAKE_NEG_MARGIN="${INFONCE_FAKE_NEG_MARGIN:-0.1}"
export INFONCE_INCLUDE_QQ="${INFONCE_INCLUDE_QQ:-false}"
export INFONCE_INCLUDE_DD="${INFONCE_INCLUDE_DD:-false}"

# A real matched backward probe admits FA2 only for the exact base, data,
# batch/accumulation, length, LoRA, dtype, and runtime contract.  Admission is
# intentionally after continual-base selection so a Qwen-base probe cannot be
# reused for a promoted local base.  Dataset and DataLoader shuffle must both
# stay off or source-homogeneous microbatches are destroyed before training.
if [[ "$RUN_NAME" == *performance200k* && "${AUTO_SELECT_FA2:-1}" == 1 ]]; then
  if [[ "${DATASET_SHUFFLE:-${TRAIN_DATALOADER_SHUFFLE:-true}}" != false \
      || "${TRAIN_DATALOADER_SHUFFLE:-true}" != false ]]; then
    echo "performance200k requires dataset_shuffle=false and train_dataloader_shuffle=false" >&2
    exit 2
  fi
  admission_key="${BACKEND_ADMISSION_RUN_KEY:-performance200k-lora-r64}"
  if [[ "${TRAIN_BATCH_SIZE:-16}:${GRAD_ACCUM_STEPS:-4}:${MAX_LENGTH:-512}" \
      != "16:4:512" ]]; then
    admission_key="${admission_key}-b${TRAIN_BATCH_SIZE:-16}-a${GRAD_ACCUM_STEPS:-4}-m${MAX_LENGTH:-512}"
  fi
  if embedding_select_fa2_backend "$TRAIN_FILE" "$admission_key" \
      "${TRAIN_BATCH_SIZE:-16}" "${GRAD_ACCUM_STEPS:-4}" \
      "${MAX_LENGTH:-512}" "${LORA_RANK:-64}" "${LORA_ALPHA:-128}" \
      bfloat16 "$BASE_MODEL" "$BASE_REVISION" \
      "$INFONCE_HARD_NEGATIVES" "${LORA_DROPOUT:-0.05}"; then
    TRAIN_ENV="$BACKEND_ADMISSION_ENV"
    ATTN_IMPL="$BACKEND_ADMISSION_ATTN"
    echo "exact performance200k backend selected: $BACKEND_ADMISSION_SELECTED_KIND" >&2
  else
    TRAIN_ENV="$EMBEDDING_TRAIN_ENV"
    ATTN_IMPL=sdpa
    echo "FA2 backend rejected; using SDPA fallback" >&2
  fi
fi

if [[ "${ATTN_IMPL:-sdpa}" == flash_attention_2 ]]; then
  if [[ "${DATASET_SHUFFLE:-${TRAIN_DATALOADER_SHUFFLE:-true}}" != false \
      || "${TRAIN_DATALOADER_SHUFFLE:-true}" != false \
      || -z "${BACKEND_ADMISSION_VERIFIED_REPORT:-}" ]] \
      || ! embedding_check_fa2_admission "$BACKEND_ADMISSION_VERIFIED_REPORT" \
        "$TRAIN_FILE" "${TRAIN_BATCH_SIZE:-16}" "${GRAD_ACCUM_STEPS:-4}" \
        "${MAX_LENGTH:-512}" "${LORA_RANK:-64}" "${LORA_ALPHA:-128}" \
        bfloat16 "$BASE_MODEL" "$BASE_REVISION" "$INFONCE_HARD_NEGATIVES" \
        "${LORA_DROPOUT:-0.05}"; then
    echo "unverified or contract-mismatched FA2 workload; falling back to SDPA" >&2
    TRAIN_ENV="$EMBEDDING_TRAIN_ENV"
    ATTN_IMPL=sdpa
    BACKEND_ADMISSION_VERIFIED_REPORT=
    export BACKEND_ADMISSION_VERIFIED_REPORT
  fi
fi
if [[ "${ATTN_IMPL:-sdpa}" == sdpa \
    && "$TRAIN_ENV" == "$ROOT/.venv-train-fa2" ]]; then
  if [[ "${DATASET_SHUFFLE:-${TRAIN_DATALOADER_SHUFFLE:-true}}" != false \
      || "${TRAIN_DATALOADER_SHUFFLE:-true}" != false \
      || -z "${BACKEND_SDPA_VERIFIED_REPORT:-}" ]] \
      || ! embedding_check_matched_sdpa "$BACKEND_SDPA_VERIFIED_REPORT" \
        "$TRAIN_FILE" "${TRAIN_BATCH_SIZE:-16}" "${GRAD_ACCUM_STEPS:-4}" \
        "${MAX_LENGTH:-512}" "${LORA_RANK:-64}" "${LORA_ALPHA:-128}" \
        bfloat16 "$BASE_MODEL" "$BASE_REVISION" "$INFONCE_HARD_NEGATIVES" \
        "${LORA_DROPOUT:-0.05}"; then
    echo "unverified or contract-mismatched fast SDPA runtime; using stable SDPA" >&2
    TRAIN_ENV="$EMBEDDING_TRAIN_ENV"
    BACKEND_SDPA_VERIFIED_REPORT=
    export BACKEND_SDPA_VERIFIED_REPORT
  fi
fi
if [[ "$TRAIN_ENV" == "$ROOT/.venv-train-fa2" ]]; then
  embedding_enable_torch25_swift_compat
fi

mkdir -p "$OUTPUT_DIR"

MODEL_ARGS=(--model "$BASE_MODEL" --use_hf true)
if [[ -n "$BASE_REVISION" ]]; then
  MODEL_ARGS+=(--model_revision "$BASE_REVISION")
fi

LOSS_ARGS=(--loss_type infonce)
if [[ "${ENABLE_LISTWISE_KD:-0}" == 1 ]]; then
  "$TRAIN_ENV/bin/python" "$ROOT/scripts/validate_embedding_jsonl.py" \
    --require-teacher-scores "$TRAIN_FILE"
  "$TRAIN_ENV/bin/python" "$ROOT/scripts/validate_embedding_jsonl.py" "$VAL_FILE"
  LOSS_ARGS=(
    --external_plugins "$ROOT/experiments/030_teacher_distillation/listwise_kd_plugin.py"
    --remove_unused_columns false
    --loss_type listwise_embedding_kd
  )
else
  "$TRAIN_ENV/bin/python" "$ROOT/scripts/validate_embedding_jsonl.py" \
    "$TRAIN_FILE" "$VAL_FILE"
fi

"$TRAIN_ENV/bin/swift" sft \
  "${MODEL_ARGS[@]}" \
  --model_type qwen3_emb \
  --task_type embedding \
  --tuner_type lora \
  --lora_rank "${LORA_RANK:-64}" \
  --lora_alpha "${LORA_ALPHA:-128}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --target_modules all-linear \
  --dataset "$TRAIN_FILE" \
  --val_dataset "$VAL_FILE" \
  --dataset_shuffle "${DATASET_SHUFFLE:-${TRAIN_DATALOADER_SHUFFLE:-true}}" \
  --val_dataset_shuffle false \
  --load_from_cache_file false \
  --lazy_tokenize true \
  --strict true \
  --attn_impl "${ATTN_IMPL:-sdpa}" \
  --torch_dtype bfloat16 \
  --gradient_checkpointing true \
  --max_length "${MAX_LENGTH:-512}" \
  --truncation_strategy right \
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE:-16}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-4}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --lr_scheduler_type cosine \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --max_steps "${MAX_STEPS:-160}" \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS:-40}" \
  --save_steps "${SAVE_STEPS:-40}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --logging_steps 1 \
  --dataloader_drop_last true \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}" \
  --train_dataloader_shuffle "${TRAIN_DATALOADER_SHUFFLE:-true}" \
  --dataset_num_proc 1 \
  --seed "${SEED:-42}" \
  --report_to none \
  --output_dir "$OUTPUT_DIR" \
  "${LOSS_ARGS[@]}" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
