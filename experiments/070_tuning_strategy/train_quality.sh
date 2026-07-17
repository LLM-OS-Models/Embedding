#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
embedding_resolve_train_runtime
MODE="${1:-}"
TRAIN_ENV="${TRAIN_ENV:-$EMBEDDING_TRAIN_ENV}"
SWIFT="$TRAIN_ENV/bin/swift"
TRAIN_FILE="${TRAIN_FILE:-$ROOT/outputs/data/performance-v1/ablation-200k/train.homogeneous-b16.jsonl}"
VAL_FILE="${VAL_FILE:-$ROOT/outputs/data/validation/legal-source-heldout-i-v2-text-strict-512/validation.jsonl}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-Embedding-8B}"
BASE_REVISION="${BASE_REVISION-1d8ad4ca9b3dd8059ad90a75d4983776a23d44af}"

if [[ "$TRAIN_ENV" == "$ROOT/.venv-train-fa2" ]]; then
  embedding_enable_torch25_swift_compat
fi

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
embedding_require_clean_validation "$VAL_FILE"

embedding_configure_hf_access
embedding_require_storage_headroom "$ROOT" \
  "${MIN_WORKSPACE_FREE_GIB:-500}" "${MIN_WORKSPACE_FREE_INODES:-1000000}"
embedding_require_storage_headroom /tmp \
  "${MIN_TMP_FREE_GIB:-50}" "${MIN_TMP_FREE_INODES:-100000}"
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
if [[ "$MODE" == last4 ]]; then
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
else
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
fi
LEARNING_RATE="${LEARNING_RATE:-6e-6}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"

MODEL_ARGS=(--model "$BASE_MODEL" --use_hf true)
if [[ -n "$BASE_REVISION" ]]; then
  MODEL_ARGS+=(--model_revision "$BASE_REVISION")
fi

COMMON=(
  sft
  "${MODEL_ARGS[@]}"
  --model_type qwen3_emb
  --task_type embedding
  --tuner_type full
  --freeze_parameters lm_head
  --dataset "$TRAIN_FILE"
  --val_dataset "$VAL_FILE"
  --dataset_shuffle false
  --val_dataset_shuffle false
  --load_from_cache_file false
  --lazy_tokenize true
  --strict true
  --attn_impl "${ATTN_IMPL:-sdpa}"
  --torch_dtype bfloat16
  --max_length 512
  --truncation_strategy right
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
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --load_best_model_at_end true
  --metric_for_best_model eval_loss
  --greater_is_better false
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
"$TRAIN_ENV/bin/python" "$ROOT/scripts/validate_embedding_jsonl.py" \
  "$TRAIN_FILE" "$VAL_FILE"

manifest_tmp="$OUTPUT_DIR/.capacity_run_manifest.json.tmp.$$"
"$TRAIN_ENV/bin/python" - "$manifest_tmp" "$MODE" "$RUN_NAME" \
  "$BASE_MODEL" "$BASE_REVISION" "$TRAIN_FILE" "$VAL_FILE" \
  "$MAX_STEPS" "$TRAIN_BATCH_SIZE" "$GRAD_ACCUM_STEPS" \
  "$LEARNING_RATE" "$SAVE_TOTAL_LIMIT" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


(
    output,
    mode,
    run_name,
    base_model,
    base_revision,
    train_file,
    validation_file,
    max_steps,
    batch_size,
    accumulation_steps,
    learning_rate,
    save_total_limit,
) = sys.argv[1:]
train_path = Path(train_file).resolve()
validation_path = Path(validation_file).resolve()
payload = {
    "schema_version": 1,
    "artifact_type": "embedding-capacity-training-contract",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "armed",
    "mode": mode,
    "run_name": run_name,
    "base_model": base_model,
    "base_revision": base_revision,
    "train": {
        "path": str(train_path),
        "sha256": sha256(train_path),
        "size_bytes": train_path.stat().st_size,
    },
    "validation": {
        "path": str(validation_path),
        "sha256": sha256(validation_path),
        "size_bytes": validation_path.stat().st_size,
    },
    "optimization": {
        "max_steps": int(max_steps),
        "per_device_train_batch_size": int(batch_size),
        "gradient_accumulation_steps": int(accumulation_steps),
        "global_batch_size": int(batch_size) * int(accumulation_steps),
        "learning_rate": float(learning_rate),
        "save_total_limit": int(save_total_limit),
        "dataset_shuffle": False,
        "train_dataloader_shuffle": False,
    },
}
path = Path(output)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(path, path.parent / "capacity_run_manifest.json")
PY
"$SWIFT" "${COMMON[@]}" "${EXTRA[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
