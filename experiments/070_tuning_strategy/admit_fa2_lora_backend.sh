#!/usr/bin/env bash
set -euo pipefail

# Admit FlashAttention 2 for the long 200K LoRA run only after a real 8B
# forward/backward throughput probe. Import-only checks are insufficient.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FA2_ENV="${FA2_ENV:-$ROOT/.venv-train-fa2}"
TRAIN_FILE="${TRAIN_FILE:?TRAIN_FILE is required}"
RUN_KEY="${RUN_KEY:-performance200k-lora-r64}"
BASELINE_SECONDS_PER_STEP="${BASELINE_SECONDS_PER_STEP:-23.2}"
REQUIRED_SPEEDUP="${REQUIRED_SPEEDUP:-1.05}"
PROBE_STEPS="${PROBE_STEPS:-5}"
OUT="$ROOT/outputs/backend-probes/$RUN_KEY"
REPORT="$OUT/admission.json"
LOG="$OUT/train.log"

mkdir -p "$OUT"
if [[ -s "$REPORT" ]]; then
  jq -e '.admitted == true' "$REPORT" >/dev/null
  exit $?
fi

[[ -x "$FA2_ENV/bin/swift" ]] || exit 10
"$FA2_ENV/bin/python" -c 'import flash_attn, swift, torch' >/dev/null

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export INFONCE_TEMPERATURE="${INFONCE_TEMPERATURE:-0.02}"
export INFONCE_USE_BATCH=true
export INFONCE_HARD_NEGATIVES="${INFONCE_HARD_NEGATIVES:-4}"
export INFONCE_MASK_FAKE_NEGATIVE=true
export INFONCE_FAKE_NEG_MARGIN="${INFONCE_FAKE_NEG_MARGIN:-0.1}"
export INFONCE_INCLUDE_QQ=false
export INFONCE_INCLUDE_DD=false

set +e
"$FA2_ENV/bin/swift" sft \
  --model Qwen/Qwen3-Embedding-8B \
  --use_hf true \
  --model_revision 1d8ad4ca9b3dd8059ad90a75d4983776a23d44af \
  --model_type qwen3_emb \
  --task_type embedding \
  --tuner_type lora \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout .05 \
  --target_modules all-linear \
  --dataset "$TRAIN_FILE" \
  --load_from_cache_file false \
  --split_dataset_ratio 0 \
  --attn_impl flash_attention_2 \
  --torch_dtype bfloat16 \
  --gradient_checkpointing true \
  --max_length 512 \
  --truncation_strategy right \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --weight_decay .01 \
  --lr_scheduler_type constant \
  --warmup_steps 0 \
  --max_steps "$PROBE_STEPS" \
  --eval_strategy no \
  --save_strategy no \
  --logging_steps 1 \
  --dataloader_drop_last true \
  --dataloader_num_workers 2 \
  --train_dataloader_shuffle false \
  --dataset_num_proc 1 \
  --seed 42 \
  --report_to none \
  --output_dir "$OUT/swift-output" \
  --loss_type infonce 2>&1 | tee "$LOG"
swift_status=${PIPESTATUS[0]}
set -e

"$FA2_ENV/bin/python" - "$LOG" "$REPORT" "$swift_status" \
  "$BASELINE_SECONDS_PER_STEP" "$REQUIRED_SPEEDUP" "$PROBE_STEPS" <<'PY'
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

log_path, report_path = map(Path, sys.argv[1:3])
status = int(sys.argv[3])
baseline = float(sys.argv[4])
required = float(sys.argv[5])
steps = int(sys.argv[6])
text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
speeds = [float(value) for value in re.findall(r"train_speed\(s/it\).*?([0-9]+(?:\.[0-9]+)?)", text)]
measured = speeds[-1] if speeds else None
threshold = baseline / required
admitted = status == 0 and measured is not None and measured <= threshold
report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "backend": "flash_attention_2",
    "environment": ".venv-train-fa2",
    "real_8b_backward_probe": True,
    "probe_steps": steps,
    "process_status": status,
    "baseline_sdpa_seconds_per_step": baseline,
    "required_speedup": required,
    "admission_threshold_seconds_per_step": threshold,
    "measured_seconds_per_step": measured,
    "admitted": admitted,
    "fallback": None if admitted else ".venv-train + sdpa",
}
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
raise SystemExit(0 if admitted else 1)
PY

