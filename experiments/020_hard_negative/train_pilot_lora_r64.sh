#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
source "$ROOT/scripts/backend_admission.sh"
embedding_resolve_train_runtime
TRAIN_ENV="${TRAIN_ENV:-$EMBEDDING_TRAIN_ENV}"
DATA_DIR="${DATA_DIR:-$ROOT/data/processed/ko_triplet_pilot_10k}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.hn-qwen3-r095-n4.jsonl}"
VAL_FILE="${VAL_FILE:-$ROOT/outputs/data/validation/legal-source-heldout-i-v2-text-strict-512/validation.jsonl}"
RUN_NAME="${RUN_NAME:-qwen3-embedding-8b-ko-hn10k-lora-r64}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/$RUN_NAME}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-Embedding-8B}"
BASE_REVISION="${BASE_REVISION-1d8ad4ca9b3dd8059ad90a75d4983776a23d44af}"
ENABLE_PRIVATE_CHECKPOINT_WATCHER="${ENABLE_PRIVATE_CHECKPOINT_WATCHER:-0}"
CHECKPOINT_TRAINING_MANIFEST="${CHECKPOINT_TRAINING_MANIFEST:-}"
CHECKPOINT_BASE_UPLOAD_REPORT="${CHECKPOINT_BASE_UPLOAD_REPORT:-}"
PRIVATE_CHECKPOINT_REPO_ID="${PRIVATE_CHECKPOINT_REPO_ID:-LLM-OS-Models2/${RUN_NAME}-candidates}"
AUTO_RESUME_FROM_LATEST_CHECKPOINT="${AUTO_RESUME_FROM_LATEST_CHECKPOINT:-1}"

if [[ "$ENABLE_PRIVATE_CHECKPOINT_WATCHER" != 0 \
    && "$ENABLE_PRIVATE_CHECKPOINT_WATCHER" != 1 ]]; then
  echo "ENABLE_PRIVATE_CHECKPOINT_WATCHER must be 0 or 1" >&2
  exit 2
fi
if [[ "$AUTO_RESUME_FROM_LATEST_CHECKPOINT" != 0 \
    && "$AUTO_RESUME_FROM_LATEST_CHECKPOINT" != 1 ]]; then
  echo "AUTO_RESUME_FROM_LATEST_CHECKPOINT must be 0 or 1" >&2
  exit 2
fi

# Legacy 50K/200K eval losses used a validation set later proven to overlap the
# 200K curriculum.  Continual-base promotion by that signal is permanently
# disabled; a caller must pass an already clean-selected BASE_MODEL explicitly.
if [[ "${ENABLE_VALIDATED_CONTINUAL_BASE:-0}" == 1 ]]; then
  echo "legacy eval-loss continual promotion is disabled; pass a clean-selected BASE_MODEL" >&2
  exit 2
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
embedding_require_clean_validation "$VAL_FILE"

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

RESUME_ARGS=()
if [[ "$AUTO_RESUME_FROM_LATEST_CHECKPOINT" == 1 ]] \
    && find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 -type d \
      -name 'checkpoint-*' -print -quit 2>/dev/null | grep -q .; then
  resume_checkpoint="$("$TRAIN_ENV/bin/python" \
    "$ROOT/scripts/select_best_checkpoint.py" "$OUTPUT_DIR" \
    --checkpoint-kind adapter --latest-resume --print-path 2>/dev/null)" || {
      echo "existing checkpoint history has no unambiguous complete resume point" >&2
      exit 2
    }
  resume_loss_type=infonce
  [[ "${ENABLE_LISTWISE_KD:-0}" == 1 ]] && resume_loss_type=listwise_embedding_kd
  resume_report_tmp="$OUTPUT_DIR/.resume-validation.json.tmp.$$"
  "$TRAIN_ENV/bin/python" "$ROOT/scripts/validate_resume_checkpoint.py" \
    --checkpoint "$resume_checkpoint" --run-dir "$OUTPUT_DIR" \
    --train-file "$TRAIN_FILE" --val-file "$VAL_FILE" \
    --base-model "$BASE_MODEL" --base-revision "$BASE_REVISION" \
    --max-steps "${MAX_STEPS:-160}" \
    --train-batch-size "${TRAIN_BATCH_SIZE:-16}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS:-4}" \
    --max-length "${MAX_LENGTH:-512}" \
    --lora-rank "${LORA_RANK:-64}" --lora-alpha "${LORA_ALPHA:-128}" \
    --lora-dropout "${LORA_DROPOUT:-0.05}" \
    --learning-rate "${LEARNING_RATE:-2e-5}" --loss-type "$resume_loss_type" \
    --dataset-shuffle "${DATASET_SHUFFLE:-${TRAIN_DATALOADER_SHUFFLE:-true}}" \
    --train-dataloader-shuffle "${TRAIN_DATALOADER_SHUFFLE:-true}" \
    --seed "${SEED:-42}" > "$resume_report_tmp"
  mv "$resume_report_tmp" "$OUTPUT_DIR/resume-validation.json"
  RESUME_ARGS+=(--resume_from_checkpoint "$resume_checkpoint")
  echo "resuming exact training contract from ${resume_checkpoint##*/}" >&2
fi

checkpoint_watcher_pid=""
checkpoint_watcher_args=()
stop_checkpoint_watcher() {
  if [[ -n "$checkpoint_watcher_pid" ]]; then
    kill "$checkpoint_watcher_pid" 2>/dev/null || true
    wait "$checkpoint_watcher_pid" 2>/dev/null || true
    checkpoint_watcher_pid=""
  fi
}

if [[ "$ENABLE_PRIVATE_CHECKPOINT_WATCHER" == 1 ]]; then
  if [[ ! -s "$CHECKPOINT_TRAINING_MANIFEST" ]]; then
    echo "checkpoint watcher requires an exact training manifest" >&2
    exit 2
  fi
  watcher_base_model="$BASE_MODEL"
  watcher_base_revision="$BASE_REVISION"
  if [[ "$BASE_MODEL" == /* ]]; then
    if [[ ! -s "$CHECKPOINT_BASE_UPLOAD_REPORT" ]]; then
      echo "local continual base requires a verified private upload report" >&2
      exit 2
    fi
    report_contract="$(jq -r \
      '.visibility + ":" + (.remote_manifest_exact|tostring)' \
      "$CHECKPOINT_BASE_UPLOAD_REPORT" 2>/dev/null || true)"
    report_model="$(jq -r '.model // empty' "$CHECKPOINT_BASE_UPLOAD_REPORT" 2>/dev/null || true)"
    report_weights_sha="$(jq -r '.weights_sha256 // empty' "$CHECKPOINT_BASE_UPLOAD_REPORT" 2>/dev/null || true)"
    watcher_base_model="$(jq -r '.repo_id // empty' "$CHECKPOINT_BASE_UPLOAD_REPORT" 2>/dev/null || true)"
    watcher_base_revision="$(jq -r '.commit_sha // empty' "$CHECKPOINT_BASE_UPLOAD_REPORT" 2>/dev/null || true)"
    base_evidence=""
    for name in merge_report.json full_tuning_report.json soup_report.json; do
      if [[ -s "$BASE_MODEL/$name" ]]; then
        [[ -z "$base_evidence" ]] || {
          echo "local continual base has ambiguous model evidence" >&2
          exit 2
        }
        base_evidence="$BASE_MODEL/$name"
      fi
    done
    if [[ -z "$base_evidence" ]]; then
      echo "local continual base has no model evidence" >&2
      exit 2
    fi
    expected_base_sha="$(jq -r '.model.weights_sha256 // empty' "$base_evidence" 2>/dev/null)"
    if [[ "$report_contract" != private:true \
        || "$watcher_base_model" != LLM-OS-Models2/* \
        || ! "$watcher_base_revision" =~ ^[0-9a-f]{40}$ \
        || ! "$report_weights_sha" =~ ^[0-9a-f]{64}$ \
        || "$report_weights_sha" != "$expected_base_sha" \
        || "$(readlink -f "$ROOT/$report_model" 2>/dev/null)" != "$(readlink -f "$BASE_MODEL")" ]]; then
      echo "local continual base private-upload lineage verification failed" >&2
      exit 2
    fi
  fi
  if [[ ! "$watcher_base_revision" =~ ^[0-9a-f]{40}$ ]]; then
    echo "checkpoint watcher base revision must be a pinned Hub commit" >&2
    exit 2
  fi
  training_data_sha="$(sha256sum "$TRAIN_FILE" | awk '{print $1}')"
  training_manifest_sha="$(sha256sum "$CHECKPOINT_TRAINING_MANIFEST" | awk '{print $1}')"
  checkpoint_watcher_args=(
    "$ROOT/scripts/watch_private_adapter_checkpoints.py"
    --watch-dir "$OUTPUT_DIR"
    --repo-id "$PRIVATE_CHECKPOINT_REPO_ID"
    --base-model "$watcher_base_model"
    --base-revision "$watcher_base_revision"
    --run-id "$RUN_NAME"
    --training-data-sha256 "$training_data_sha"
    --training-manifest-sha256 "$training_manifest_sha"
    --poll-seconds 5 --settle-seconds 10
    --remote-attempts 3 --remote-retry-seconds 15 --upload
  )
  admission_report="${BACKEND_ADMISSION_VERIFIED_REPORT:-${BACKEND_SDPA_VERIFIED_REPORT:-}}"
  if [[ -n "$admission_report" && -s "$admission_report" ]]; then
    checkpoint_watcher_args+=(
      --admission-report-sha256 "$(sha256sum "$admission_report" | awk '{print $1}')"
    )
  fi
  env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN \
    -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE -u HF_DATASETS_OFFLINE \
    "$TRAIN_ENV/bin/python" "${checkpoint_watcher_args[@]}" \
    >> "$OUTPUT_DIR/checkpoint-watcher.log" 2>&1 &
  checkpoint_watcher_pid=$!
  trap stop_checkpoint_watcher EXIT INT TERM
fi

training_status=0
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
  "${RESUME_ARGS[@]}" \
  "${LOSS_ARGS[@]}" \
  2>&1 | tee "$OUTPUT_DIR/train.log" || training_status=$?

if [[ "$ENABLE_PRIVATE_CHECKPOINT_WATCHER" == 1 ]]; then
  stop_checkpoint_watcher
  if ! env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN \
      -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE -u HF_DATASETS_OFFLINE \
      "$TRAIN_ENV/bin/python" "${checkpoint_watcher_args[@]}" \
      --once --settle-seconds 0 \
      >> "$OUTPUT_DIR/checkpoint-watcher.log" 2>&1; then
    echo "final private checkpoint reconciliation failed; local checkpoints retained" >&2
  fi
  trap - EXIT INT TERM
fi
exit "$training_status"
