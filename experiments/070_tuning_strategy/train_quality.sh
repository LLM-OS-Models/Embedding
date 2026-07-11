#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-}"
SWIFT="${TRAIN_ENV:-$ROOT/.venv-train}/bin/swift"
TRAIN_FILE="${TRAIN_FILE:-$ROOT/outputs/data/performance-v1/ablation-200k/train.homogeneous-b16.jsonl}"
VAL_FILE="${VAL_FILE:-$ROOT/data/processed/ko_triplet_pilot_10k/validation.hn-qwen3-r095-n4.jsonl}"
REVISION="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"

if [[ ! "$MODE" =~ ^(last4|galore|lisa8|standard_full)$ ]]; then
  echo "usage: $0 {last4|galore|lisa8|standard_full}" >&2
  exit 2
fi
if [[ "$MODE" == standard_full && "${ALLOW_STANDARD_FULL:-0}" != 1 ]]; then
  echo "standard_full requires ALLOW_STANDARD_FULL=1 after a successful memory probe" >&2
  exit 3
fi
for path in "$TRAIN_FILE" "$VAL_FILE"; do
  [[ -s "$path" ]] || { echo "missing dataset: $path" >&2; exit 4; }
done

if [[ -f "$ROOT/.env" ]]; then
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "$ROOT/.env" | tail -n 1)"
  export HF_TOKEN
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export INFONCE_TEMPERATURE="${INFONCE_TEMPERATURE:-0.02}"
export INFONCE_USE_BATCH=true
export INFONCE_HARD_NEGATIVES="${INFONCE_HARD_NEGATIVES:-4}"
export INFONCE_MASK_FAKE_NEGATIVE=true
export INFONCE_FAKE_NEG_MARGIN="${INFONCE_FAKE_NEG_MARGIN:-0.1}"
export INFONCE_INCLUDE_QQ=false
export INFONCE_INCLUDE_DD=false

RUN_NAME="${RUN_NAME:-qwen3-embedding-8b-ko-performance200k-$MODE}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/$RUN_NAME}"
MAX_STEPS="${MAX_STEPS:-3123}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-6e-6}"

COMMON=(
  sft
  --model Qwen/Qwen3-Embedding-8B
  --use_hf true
  --model_revision "$REVISION"
  --model_type qwen3_emb
  --task_type embedding
  --tuner_type full
  --freeze_parameters lm_head
  --dataset "$TRAIN_FILE"
  --val_dataset "$VAL_FILE"
  --load_from_cache_file false
  --attn_impl "${ATTN_IMPL:-sdpa}"
  --torch_dtype bfloat16
  --max_length 512
  --per_device_train_batch_size "$TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size 2
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS"
  --learning_rate "$LEARNING_RATE"
  --weight_decay .01
  --lr_scheduler_type cosine
  --warmup_ratio .05
  --max_steps "$MAX_STEPS"
  --eval_strategy steps
  --eval_steps 250
  --save_steps 250
  --save_total_limit 2
  --logging_steps 1
  --dataloader_drop_last true
  --dataloader_num_workers 2
  --train_dataloader_shuffle false
  --dataset_num_proc 1
  --seed 42
  --report_to none
  --output_dir "$OUTPUT_DIR"
  --loss_type infonce
)

case "$MODE" in
  last4)
    EXTRA=(
      --freeze_parameters_ratio 1
      --trainable_parameters_regex '^(model\.layers\.(3[2-5])\.|model\.norm\.)'
      --gradient_checkpointing false
      --optim adamw_torch_fused
    )
    ;;
  galore)
    EXTRA=(
      --gradient_checkpointing true
      --optim adamw_torch_fused
      --use_galore true
      --galore_rank 128
      --galore_update_proj_gap 50
      --galore_optim_per_parameter true
    )
    ;;
  lisa8)
    EXTRA=(
      --gradient_checkpointing true
      --optim adamw_torch_fused
      --lisa_activated_layers 8
      --lisa_step_interval 20
    )
    ;;
  standard_full)
    EXTRA=(--gradient_checkpointing true --optim adamw_torch_fused)
    ;;
esac

mkdir -p "$OUTPUT_DIR"
"$ROOT/.venv-train/bin/python" "$ROOT/scripts/validate_embedding_jsonl.py" \
  "$TRAIN_FILE" "$VAL_FILE"
"$SWIFT" "${COMMON[@]}" "${EXTRA[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
