#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_ENV="${TRAIN_ENV:-$ROOT/.venv-train}"
DATA_DIR="${DATA_DIR:-$ROOT/data/processed/ko_triplet_smoke}"
RUN_NAME="${RUN_NAME:-qwen3-embedding-8b-ko-smoke-r32}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/$RUN_NAME}"
BASE_REVISION="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"

if [[ -f "$ROOT/.env" ]]; then
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "$ROOT/.env" | tail -n 1)"
  export HF_TOKEN
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export INFONCE_TEMPERATURE="${INFONCE_TEMPERATURE:-0.02}"
export INFONCE_USE_BATCH="${INFONCE_USE_BATCH:-true}"
export INFONCE_HARD_NEGATIVES="${INFONCE_HARD_NEGATIVES:-1}"
export INFONCE_MASK_FAKE_NEGATIVE="${INFONCE_MASK_FAKE_NEGATIVE:-true}"
export INFONCE_FAKE_NEG_MARGIN="${INFONCE_FAKE_NEG_MARGIN:-0.1}"
export INFONCE_INCLUDE_QQ="${INFONCE_INCLUDE_QQ:-false}"
export INFONCE_INCLUDE_DD="${INFONCE_INCLUDE_DD:-false}"

mkdir -p "$OUTPUT_DIR"

"$TRAIN_ENV/bin/python" "$ROOT/scripts/validate_embedding_jsonl.py" \
  "$DATA_DIR/train.jsonl" \
  "$DATA_DIR/validation.jsonl"

"$TRAIN_ENV/bin/swift" sft \
  --model Qwen/Qwen3-Embedding-8B \
  --use_hf true \
  --model_revision "$BASE_REVISION" \
  --model_type qwen3_emb \
  --task_type embedding \
  --tuner_type lora \
  --lora_rank "${LORA_RANK:-32}" \
  --lora_alpha "${LORA_ALPHA:-64}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --target_modules all-linear \
  --dataset "$DATA_DIR/train.jsonl" \
  --val_dataset "$DATA_DIR/validation.jsonl" \
  --load_from_cache_file false \
  --attn_impl "${ATTN_IMPL:-sdpa}" \
  --torch_dtype bfloat16 \
  --gradient_checkpointing true \
  --max_length "${MAX_LENGTH:-512}" \
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE:-2}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-4}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --lr_scheduler_type cosine \
  --warmup_ratio "${WARMUP_RATIO:-0.1}" \
  --max_steps "${MAX_STEPS:-20}" \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS:-10}" \
  --save_steps "${SAVE_STEPS:-10}" \
  --save_total_limit 2 \
  --logging_steps 1 \
  --dataloader_drop_last true \
  --dataloader_num_workers 2 \
  --dataset_num_proc 1 \
  --seed "${SEED:-42}" \
  --report_to none \
  --output_dir "$OUTPUT_DIR" \
  --loss_type infonce \
  2>&1 | tee "$OUTPUT_DIR/train.log"
