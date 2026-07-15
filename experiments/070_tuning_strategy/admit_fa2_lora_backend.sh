#!/usr/bin/env bash
set -euo pipefail

# Admit FlashAttention 2 for the long 200K LoRA run only after a real 8B
# forward/backward throughput probe. Import-only checks are insufficient.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
FA2_ENV="${FA2_ENV:-$ROOT/.venv-train-fa2}"
TRAIN_FILE="${TRAIN_FILE:?TRAIN_FILE is required}"
if [[ -z "${TRAIN_PROVENANCE:-}" ]]; then
  train_name="$(basename "$TRAIN_FILE")"
  case "$train_name" in
    train.jsonl) provenance_name=provenance.jsonl ;;
    train.*) provenance_name="provenance.${train_name#train.}" ;;
    *) provenance_name= ;;
  esac
  if [[ -n "$provenance_name" ]]; then
    TRAIN_PROVENANCE="$(dirname "$TRAIN_FILE")/$provenance_name"
  fi
fi
TRAIN_PROVENANCE="${TRAIN_PROVENANCE:?TRAIN_PROVENANCE is required (or use a train*.jsonl file with an aligned provenance*.jsonl sibling)}"
RUN_KEY="${RUN_KEY:-performance200k-lora-r64}"
BASELINE_SECONDS_PER_STEP="${BASELINE_SECONDS_PER_STEP:-23.2}"
REQUIRED_SPEEDUP="${REQUIRED_SPEEDUP:-1.05}"
PROBE_STEPS="${PROBE_STEPS:-5}"
MEASURE_MATCHED_SDPA="${MEASURE_MATCHED_SDPA:-1}"
if [[ "$MEASURE_MATCHED_SDPA" != 0 && "$MEASURE_MATCHED_SDPA" != 1 ]]; then
  echo "MEASURE_MATCHED_SDPA must be 0 or 1" >&2
  exit 2
fi
OUT="$ROOT/outputs/backend-probes/$RUN_KEY"
REPORT="$OUT/admission.json"
LOG="$OUT/train.log"
SDPA_LOG="$OUT/sdpa.log"
PROBE_SUBSET_DIR="${PROBE_SUBSET_DIR:-$OUT/probe-subset}"
PROBE_SUBSET_MANIFEST="$PROBE_SUBSET_DIR/manifest.json"
PROBE_TRAIN="$PROBE_SUBSET_DIR/train.jsonl"

mkdir -p "$OUT"
if [[ -s "$REPORT" && "${FORCE_PROBE:-0}" != 1 ]]; then
  jq -e '.admitted == true' "$REPORT" >/dev/null
  exit $?
fi
if [[ -s "$REPORT" && "${FORCE_PROBE:-0}" == 1 ]]; then
  attempt_dir="$OUT/attempts/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$attempt_dir"
  cp "$REPORT" "$attempt_dir/admission.json"
  [[ ! -s "$LOG" ]] || cp "$LOG" "$attempt_dir/train.log"
  [[ ! -s "$SDPA_LOG" ]] || cp "$SDPA_LOG" "$attempt_dir/sdpa.log"
fi

embedding_enable_torch25_swift_compat
[[ -x "$FA2_ENV/bin/swift" ]] || exit 10
"$FA2_ENV/bin/python" -c 'import flash_attn, swift, torch' >/dev/null
[[ -s "$TRAIN_FILE" && -s "$TRAIN_PROVENANCE" ]] || exit 11
"$FA2_ENV/bin/python" "$ROOT/scripts/build_fa2_probe_subset.py" \
  --train "$TRAIN_FILE" \
  --provenance "$TRAIN_PROVENANCE" \
  --output-dir "$PROBE_SUBSET_DIR" \
  --batch-size 16 \
  --probe-steps "$PROBE_STEPS" \
  --gradient-accumulation-steps 4 \
  --training-max-length 512 \
  --seed 42 > "$OUT/probe-subset-build.log"
jq -e \
  --argjson expected_rows "$((PROBE_STEPS * 4 * 16))" \
  '.parameters.selected_rows == $expected_rows
   and .semantics.complete_source_homogeneous_batches == true
   and .semantics.train_rows_copied_byte_for_byte == true' \
  "$PROBE_SUBSET_MANIFEST" >/dev/null

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export INFONCE_TEMPERATURE="${INFONCE_TEMPERATURE:-0.02}"
export INFONCE_USE_BATCH=true
export INFONCE_HARD_NEGATIVES="${INFONCE_HARD_NEGATIVES:-4}"
export INFONCE_MASK_FAKE_NEGATIVE=true
export INFONCE_FAKE_NEG_MARGIN="${INFONCE_FAKE_NEG_MARGIN:-0.1}"
export INFONCE_INCLUDE_QQ=false
export INFONCE_INCLUDE_DD=false

run_probe() {
  local backend="$1" log="$2" output_dir="$3"
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
    --dataset "$PROBE_TRAIN" \
    --load_from_cache_file false \
    --lazy_tokenize false \
    --split_dataset_ratio 0 \
    --attn_impl "$backend" \
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
    --output_dir "$output_dir" \
    --loss_type infonce 2>&1 | tee "$log"
  return "${PIPESTATUS[0]}"
}

sdpa_status=-1
set +e
if [[ "$MEASURE_MATCHED_SDPA" == 1 ]]; then
  run_probe sdpa "$SDPA_LOG" "$OUT/swift-output-sdpa"
  sdpa_status=$?
fi
run_probe flash_attention_2 "$LOG" "$OUT/swift-output-fa2"
swift_status=$?
set -e

"$FA2_ENV/bin/python" - "$LOG" "$REPORT" "$swift_status" \
  "$BASELINE_SECONDS_PER_STEP" "$REQUIRED_SPEEDUP" "$PROBE_STEPS" \
  "$TRAIN_FILE" "$PROBE_SUBSET_MANIFEST" "$MEASURE_MATCHED_SDPA" \
  "$SDPA_LOG" "$sdpa_status" <<'PY'
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
source_train_path = Path(sys.argv[7])
subset_manifest_path = Path(sys.argv[8])
measure_matched_sdpa = bool(int(sys.argv[9]))
sdpa_log_path = Path(sys.argv[10])
sdpa_status = int(sys.argv[11])
text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
speeds = [float(value) for value in re.findall(r"train_speed\(s/it\).*?([0-9]+(?:\.[0-9]+)?)", text)]
measured = speeds[-1] if speeds else None
memories = [float(value) for value in re.findall(r"memory\(GiB\).*?([0-9]+(?:\.[0-9]+)?)", text)]
measured_peak_memory = max(memories) if memories else None
if measure_matched_sdpa:
    sdpa_text = sdpa_log_path.read_text(encoding="utf-8", errors="replace") if sdpa_log_path.exists() else ""
    sdpa_speeds = [float(value) for value in re.findall(r"train_speed\(s/it\).*?([0-9]+(?:\.[0-9]+)?)", sdpa_text)]
    sdpa_memories = [float(value) for value in re.findall(r"memory\(GiB\).*?([0-9]+(?:\.[0-9]+)?)", sdpa_text)]
    baseline = sdpa_speeds[-1] if sdpa_speeds else None
    baseline_peak_memory = max(sdpa_memories) if sdpa_memories else None
    baseline_source = "matched_subset_same_environment"
else:
    sdpa_speeds = []
    baseline_peak_memory = None
    baseline_source = "configured_historical_long_run"
threshold = baseline / required if baseline is not None else None
admitted = (
    status == 0
    and measured is not None
    and baseline is not None
    and (not measure_matched_sdpa or sdpa_status == 0)
    and threshold is not None
    and measured <= threshold
)
subset_manifest = json.loads(subset_manifest_path.read_text(encoding="utf-8"))
subset_train = subset_manifest["files"]["train.jsonl"]
report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "backend": "flash_attention_2",
    "environment": ".venv-train-fa2",
    "real_8b_backward_probe": True,
    "probe_steps": steps,
    "process_status": status,
    "matched_sdpa_process_status": sdpa_status if measure_matched_sdpa else None,
    "baseline_source": baseline_source,
    "configured_historical_sdpa_seconds_per_step": float(sys.argv[4]),
    "baseline_sdpa_seconds_per_step": baseline,
    "required_speedup": required,
    "admission_threshold_seconds_per_step": threshold,
    "measured_seconds_per_step": measured,
    "baseline_sdpa_peak_memory_gib": baseline_peak_memory,
    "measured_peak_memory_gib": measured_peak_memory,
    "measured_speedup_vs_sdpa": (
        baseline / measured if baseline is not None and measured else None
    ),
    "admitted": admitted,
    "fallback": None if admitted else ".venv-train + sdpa",
    "workload": {
        "source_train_path": str(source_train_path.resolve()),
        "source_train_sha256": subset_manifest["inputs"]["train"]["sha256"],
        "probe_subset_manifest": str(subset_manifest_path.resolve()),
        "probe_train_path": subset_train["path"],
        "probe_train_sha256": subset_train["sha256"],
        "probe_rows": subset_train["rows"],
        "selection_contract": subset_manifest["selection_contract"],
    },
}
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
raise SystemExit(0 if admitted else 1)
PY
