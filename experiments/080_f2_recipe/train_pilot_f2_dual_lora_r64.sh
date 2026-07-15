#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
TRAIN_ENV="${TRAIN_ENV:-$ROOT/.venv-train}"

if [[ "$TRAIN_ENV" == "$ROOT/.venv-train-fa2" ]]; then
  embedding_enable_torch25_swift_compat
fi
DATA_DIR="${DATA_DIR:-$ROOT/data/processed/ko_triplet_pilot_10k}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.hn-qwen3-r095-n4.jsonl}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/validation.hn-qwen3-r095-n4.jsonl}"
RUN_NAME="${RUN_NAME:-qwen3-embedding-8b-ko-hn10k-f2dual-lora-r64}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/$RUN_NAME}"
PLUGIN="$ROOT/experiments/080_f2_recipe/f2_dual_loss_plugin.py"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-Embedding-8B}"
BASE_REVISION="${BASE_REVISION-1d8ad4ca9b3dd8059ad90a75d4983776a23d44af}"

embedding_configure_hf_access

for path in "$TRAIN_FILE" "$VAL_FILE" "$PLUGIN"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export F2_DUAL_TEMPERATURE="${F2_DUAL_TEMPERATURE:-0.05}"
export F2_DUAL_HARD_NEGATIVES="${F2_DUAL_HARD_NEGATIVES:-4}"
export F2_DUAL_INBATCH_WEIGHT="${F2_DUAL_INBATCH_WEIGHT:-1.0}"
export F2_DUAL_HARD_WEIGHT="${F2_DUAL_HARD_WEIGHT:-1.0}"

MRL_ARGS=()
if [[ "${USE_F2_MRL:-0}" == "1" ]]; then
  # Exact F2 code weighting: sqrt(dim / 4096) / 10 for 8..4096.
  MRL_ARGS+=(--mrl_dims '{"8":0.0044194174,"16":0.00625,"32":0.0088388348,"64":0.0125,"128":0.0176776695,"256":0.025,"512":0.0353553391,"1024":0.05,"2048":0.0707106781,"4096":0.1}')
fi

mkdir -p "$OUTPUT_DIR"
MODEL_ARGS=(--model "$BASE_MODEL" --use_hf true)
if [[ -n "$BASE_REVISION" ]]; then
  MODEL_ARGS+=(--model_revision "$BASE_REVISION")
fi
"$TRAIN_ENV/bin/python" "$ROOT/scripts/validate_embedding_jsonl.py" \
  "$TRAIN_FILE" "$VAL_FILE"

"$TRAIN_ENV/bin/swift" sft \
  --external_plugins "$PLUGIN" \
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
  --dataset_shuffle "${DATASET_SHUFFLE:-true}" \
  --val_dataset_shuffle false \
  --load_from_cache_file false \
  --lazy_tokenize true \
  --strict true \
  --attn_impl "${ATTN_IMPL:-sdpa}" \
  --torch_dtype bfloat16 \
  --gradient_checkpointing true \
  --max_length "${MAX_LENGTH:-512}" \
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
  --dataset_num_proc 1 \
  --seed "${SEED:-42}" \
  --report_to none \
  --output_dir "$OUTPUT_DIR" \
  --loss_type f2_dual_infonce \
  "${MRL_ARGS[@]}" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
