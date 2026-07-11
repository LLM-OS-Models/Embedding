#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-}"
SWIFT="${TRAIN_ENV:-$ROOT/.venv-train}/bin/swift"
DATA="${DATA:-$ROOT/data/processed/ko_triplet_smoke/train.jsonl}"
REVISION="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"

if [[ -z "$MODE" ]]; then
  echo "usage: $0 {lora_r64|dora_r32|last4|galore|standard_full}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export INFONCE_TEMPERATURE="0.02"
export INFONCE_USE_BATCH="true"
export INFONCE_HARD_NEGATIVES="1"
export INFONCE_MASK_FAKE_NEGATIVE="true"
export INFONCE_FAKE_NEG_MARGIN="0.1"
export INFONCE_INCLUDE_QQ="false"
export INFONCE_INCLUDE_DD="false"

COMMON=(
  sft
  --model Qwen/Qwen3-Embedding-8B
  --use_hf true
  --model_revision "$REVISION"
  --model_type qwen3_emb
  --task_type embedding
  --dataset "$DATA"
  --load_from_cache_file false
  --split_dataset_ratio 0
  --attn_impl "${ATTN_IMPL:-flash_attention_2}"
  --torch_dtype bfloat16
  --max_length "${MAX_LENGTH:-512}"
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 1
  --learning_rate 6e-6
  --weight_decay 0.01
  --lr_scheduler_type constant
  --warmup_steps 0
  --max_steps 1
  --eval_strategy no
  --save_strategy no
  --logging_steps 1
  --dataloader_drop_last true
  --dataloader_num_workers 0
  --dataset_num_proc 1
  --seed 42
  --report_to none
  --loss_type infonce
)

suffix="${PROBE_SUFFIX:-}"
[[ -z "$suffix" ]] || suffix="-$suffix"
OUT="$ROOT/outputs/memory_probes/$MODE$suffix"
mkdir -p "$OUT"

case "$MODE" in
  lora_r64)
    EXTRA=(
      --tuner_type lora
      --lora_rank 64
      --lora_alpha 128
      --lora_dropout 0.05
      --target_modules all-linear
      --gradient_checkpointing true
    )
    ;;
  dora_r32)
    EXTRA=(
      --tuner_type lora
      --lora_rank 32
      --lora_alpha 64
      --lora_dropout 0.05
      --target_modules all-linear
      --use_dora true
      --gradient_checkpointing true
    )
    ;;
  last4)
    EXTRA=(
      --tuner_type full
      --freeze_parameters lm_head
      --freeze_parameters_ratio 1
      --trainable_parameters_regex '^(model\.layers\.(3[2-5])\.|model\.norm\.)'
      --gradient_checkpointing false
      --optim adamw_torch_fused
    )
    ;;
  galore)
    EXTRA=(
      --tuner_type full
      --freeze_parameters lm_head
      --gradient_checkpointing true
      --optim adamw_torch_fused
      --use_galore true
      --galore_rank 128
      --galore_update_proj_gap 50
      --galore_optim_per_parameter true
    )
    ;;
  standard_full)
    EXTRA=(
      --tuner_type full
      --freeze_parameters lm_head
      --gradient_checkpointing true
      --optim adamw_torch_fused
    )
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac

"$SWIFT" "${COMMON[@]}" "${EXTRA[@]}" --output_dir "$OUT" 2>&1 | tee "$OUT/probe.log"
