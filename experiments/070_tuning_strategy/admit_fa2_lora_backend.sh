#!/usr/bin/env bash
set -euo pipefail

# Admit FlashAttention 2 only for the exact 8B LoRA workload exercised by a
# matched SDPA/FA2 forward-backward probe. Import-only and boolean-only checks
# are insufficient.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
source "$ROOT/scripts/backend_admission.sh"
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
    train_dir="$(dirname "$TRAIN_FILE")"
    sibling_provenance="$train_dir/$provenance_name"
    metadata_provenance="$(dirname "$train_dir")/metadata/$provenance_name"
    if [[ -s "$sibling_provenance" ]]; then
      TRAIN_PROVENANCE="$sibling_provenance"
    elif [[ -s "$metadata_provenance" ]]; then
      TRAIN_PROVENANCE="$metadata_provenance"
    else
      TRAIN_PROVENANCE="$sibling_provenance"
    fi
  fi
fi
TRAIN_PROVENANCE="${TRAIN_PROVENANCE:?TRAIN_PROVENANCE is required (or use a train*.jsonl file with an aligned provenance*.jsonl sibling)}"
RUN_KEY="${RUN_KEY:-performance200k-lora-r64}"
PROBE_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PROBE_GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
PROBE_MAX_LENGTH="${MAX_LENGTH:-512}"
PROBE_LORA_RANK="${LORA_RANK:-64}"
PROBE_LORA_ALPHA="${LORA_ALPHA:-128}"
PROBE_LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
PROBE_DTYPE="${TRAIN_DTYPE:-bfloat16}"
PROBE_BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-Embedding-8B}"
PROBE_BASE_REVISION="${BASE_REVISION-1d8ad4ca9b3dd8059ad90a75d4983776a23d44af}"
PROBE_HARD_NEGATIVES="${INFONCE_HARD_NEGATIVES:-4}"
BASELINE_SECONDS_PER_STEP="${BASELINE_SECONDS_PER_STEP:-23.2}"
REQUIRED_SPEEDUP="${REQUIRED_SPEEDUP:-1.05}"
PROBE_STEPS="${PROBE_STEPS:-5}"
MEASURE_MATCHED_SDPA="${MEASURE_MATCHED_SDPA:-1}"
if [[ "$MEASURE_MATCHED_SDPA" != 1 ]]; then
  echo "MEASURE_MATCHED_SDPA must be 1 for fail-closed admission" >&2
  exit 2
fi
for value in "$PROBE_BATCH_SIZE" "$PROBE_GRAD_ACCUM_STEPS" \
    "$PROBE_MAX_LENGTH" "$PROBE_LORA_RANK" "$PROBE_LORA_ALPHA" \
    "$PROBE_HARD_NEGATIVES" "$PROBE_STEPS"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "backend probe integer fields must be positive: $value" >&2
    exit 2
  fi
done
if [[ "$PROBE_DTYPE" != bfloat16 && "$PROBE_DTYPE" != float16 ]]; then
  echo "TRAIN_DTYPE must be bfloat16 or float16" >&2
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -s "$REPORT" && "${FORCE_PROBE:-0}" != 1 ]]; then
  if embedding_check_fa2_admission "$REPORT" "$TRAIN_FILE" \
      "$PROBE_BATCH_SIZE" "$PROBE_GRAD_ACCUM_STEPS" "$PROBE_MAX_LENGTH" \
      "$PROBE_LORA_RANK" "$PROBE_LORA_ALPHA" "$PROBE_DTYPE" \
      "$PROBE_BASE_MODEL" "$PROBE_BASE_REVISION" "$PROBE_HARD_NEGATIVES" \
      "$PROBE_LORA_DROPOUT"; then
    exit 0
  fi
  echo "cached FA2 admission contract/runtime mismatch; probing exact workload" >&2
fi
if [[ -s "$REPORT" ]]; then
  attempt_dir="$OUT/attempts/$(date -u +%Y%m%dT%H%M%S)-$$"
  mkdir -p "$attempt_dir"
  cp "$REPORT" "$attempt_dir/admission.json"
  [[ ! -s "$LOG" ]] || cp "$LOG" "$attempt_dir/train.log"
  [[ ! -s "$SDPA_LOG" ]] || cp "$SDPA_LOG" "$attempt_dir/sdpa.log"
  rm -f "$REPORT"
fi

embedding_enable_torch25_swift_compat
[[ -x "$FA2_ENV/bin/swift" ]] || exit 10
"$FA2_ENV/bin/python" -c 'import flash_attn, swift, torch' >/dev/null
[[ -s "$TRAIN_FILE" && -s "$TRAIN_PROVENANCE" ]] || exit 11
INPUT_BATCH_SIZE="$(head -n 1 "$TRAIN_PROVENANCE" | jq -r \
  'select(.homogeneous_batch != null) | .homogeneous_batch.batch_size')"
if [[ ! "$INPUT_BATCH_SIZE" =~ ^[1-9][0-9]*$ \
    || $((INPUT_BATCH_SIZE % PROBE_BATCH_SIZE)) -ne 0 \
    || $(((PROBE_BATCH_SIZE * PROBE_GRAD_ACCUM_STEPS) % INPUT_BATCH_SIZE)) -ne 0 ]]; then
  echo "probe batch=$PROBE_BATCH_SIZE/accum=$PROBE_GRAD_ACCUM_STEPS cannot preserve input homogeneous batch=$INPUT_BATCH_SIZE" >&2
  exit 12
fi
SELECTION_ACCUM_STEPS="$((PROBE_BATCH_SIZE * PROBE_GRAD_ACCUM_STEPS / INPUT_BATCH_SIZE))"
"$FA2_ENV/bin/python" "$ROOT/scripts/build_fa2_probe_subset.py" \
  --train "$TRAIN_FILE" \
  --provenance "$TRAIN_PROVENANCE" \
  --output-dir "$PROBE_SUBSET_DIR" \
  --batch-size "$INPUT_BATCH_SIZE" \
  --probe-steps "$PROBE_STEPS" \
  --gradient-accumulation-steps "$SELECTION_ACCUM_STEPS" \
  --training-max-length "$PROBE_MAX_LENGTH" \
  --seed 42 > "$OUT/probe-subset-build.log"
jq -e \
  --argjson expected_rows "$((PROBE_STEPS * PROBE_GRAD_ACCUM_STEPS * PROBE_BATCH_SIZE))" \
  --argjson input_batch "$INPUT_BATCH_SIZE" \
  --argjson selection_accum "$SELECTION_ACCUM_STEPS" \
  --argjson max_length "$PROBE_MAX_LENGTH" \
  '.parameters.selected_rows == $expected_rows
   and .parameters.batch_size == $input_batch
   and .parameters.gradient_accumulation_steps == $selection_accum
   and .parameters.training_max_length_tokens == $max_length
   and .semantics.complete_source_homogeneous_batches == true
   and .semantics.train_rows_copied_byte_for_byte == true' \
  "$PROBE_SUBSET_MANIFEST" >/dev/null

export INFONCE_TEMPERATURE="${INFONCE_TEMPERATURE:-0.02}"
export INFONCE_USE_BATCH=true
export INFONCE_HARD_NEGATIVES="$PROBE_HARD_NEGATIVES"
export INFONCE_MASK_FAKE_NEGATIVE=true
export INFONCE_FAKE_NEG_MARGIN="${INFONCE_FAKE_NEG_MARGIN:-0.1}"
export INFONCE_INCLUDE_QQ=false
export INFONCE_INCLUDE_DD=false

run_probe() {
  local backend="$1" log="$2" output_dir="$3"
  local model_args=(--model "$PROBE_BASE_MODEL" --use_hf true)
  if [[ -n "$PROBE_BASE_REVISION" ]]; then
    model_args+=(--model_revision "$PROBE_BASE_REVISION")
  fi
  "$FA2_ENV/bin/swift" sft \
    "${model_args[@]}" \
    --model_type qwen3_emb \
    --task_type embedding \
    --tuner_type lora \
    --lora_rank "$PROBE_LORA_RANK" \
    --lora_alpha "$PROBE_LORA_ALPHA" \
    --lora_dropout "$PROBE_LORA_DROPOUT" \
    --target_modules all-linear \
    --dataset "$PROBE_TRAIN" \
    --load_from_cache_file false \
    --lazy_tokenize true \
    --dataset_shuffle false \
    --strict true \
    --split_dataset_ratio 0 \
    --attn_impl "$backend" \
    --torch_dtype "$PROBE_DTYPE" \
    --gradient_checkpointing true \
    --max_length "$PROBE_MAX_LENGTH" \
    --truncation_strategy right \
    --per_device_train_batch_size "$PROBE_BATCH_SIZE" \
    --gradient_accumulation_steps "$PROBE_GRAD_ACCUM_STEPS" \
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

BACKEND_CONTRACT_BATCH="$PROBE_BATCH_SIZE" \
BACKEND_CONTRACT_ACCUM="$PROBE_GRAD_ACCUM_STEPS" \
BACKEND_CONTRACT_MAX_LENGTH="$PROBE_MAX_LENGTH" \
BACKEND_CONTRACT_LORA_RANK="$PROBE_LORA_RANK" \
BACKEND_CONTRACT_LORA_ALPHA="$PROBE_LORA_ALPHA" \
BACKEND_CONTRACT_LORA_DROPOUT="$PROBE_LORA_DROPOUT" \
BACKEND_CONTRACT_DTYPE="$PROBE_DTYPE" \
BACKEND_CONTRACT_BASE_MODEL="$PROBE_BASE_MODEL" \
BACKEND_CONTRACT_BASE_REVISION="$PROBE_BASE_REVISION" \
BACKEND_CONTRACT_HARD_NEGATIVES="$PROBE_HARD_NEGATIVES" \
BACKEND_CONTRACT_INPUT_BATCH="$INPUT_BATCH_SIZE" \
BACKEND_CONTRACT_SELECTION_ACCUM="$SELECTION_ACCUM_STEPS" \
PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}" \
"$FA2_ENV/bin/python" - "$LOG" "$REPORT" "$swift_status" \
  "$BASELINE_SECONDS_PER_STEP" "$REQUIRED_SPEEDUP" "$PROBE_STEPS" \
  "$TRAIN_FILE" "$PROBE_SUBSET_MANIFEST" "$MEASURE_MATCHED_SDPA" \
  "$SDPA_LOG" "$sdpa_status" <<'PY'
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from backend_admission import (
    SCHEMA_VERSION,
    build_workload_contract,
    canonical_sha256,
    collect_runtime_fingerprint,
)

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
contract = build_workload_contract(
    train_file=source_train_path,
    train_sha256=subset_manifest["inputs"]["train"]["sha256"],
    backend="flash_attention_2",
    batch_size=int(os.environ["BACKEND_CONTRACT_BATCH"]),
    gradient_accumulation_steps=int(os.environ["BACKEND_CONTRACT_ACCUM"]),
    max_length=int(os.environ["BACKEND_CONTRACT_MAX_LENGTH"]),
    lora_rank=int(os.environ["BACKEND_CONTRACT_LORA_RANK"]),
    lora_alpha=int(os.environ["BACKEND_CONTRACT_LORA_ALPHA"]),
    lora_dropout=float(os.environ["BACKEND_CONTRACT_LORA_DROPOUT"]),
    dtype=os.environ["BACKEND_CONTRACT_DTYPE"],
    base_model=os.environ["BACKEND_CONTRACT_BASE_MODEL"],
    base_revision=os.environ["BACKEND_CONTRACT_BASE_REVISION"],
    hard_negatives=int(os.environ["BACKEND_CONTRACT_HARD_NEGATIVES"]),
)
runtime = collect_runtime_fingerprint()
report = {
    "schema_version": SCHEMA_VERSION,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "backend": "flash_attention_2",
    "environment": runtime["python_prefix"],
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
    "workload_contract": contract,
    "workload_contract_sha256": canonical_sha256(contract),
    "runtime_fingerprint": runtime,
    "runtime_fingerprint_sha256": canonical_sha256(runtime),
    "workload": {
        "source_train_path": str(source_train_path.resolve()),
        "source_train_sha256": subset_manifest["inputs"]["train"]["sha256"],
        "probe_subset_manifest": str(subset_manifest_path.resolve()),
        "probe_train_path": subset_train["path"],
        "probe_train_sha256": subset_train["sha256"],
        "probe_rows": subset_train["rows"],
        "selection_contract": subset_manifest["selection_contract"],
        "execution_projection": {
            "input_homogeneous_batch_size": int(
                os.environ["BACKEND_CONTRACT_INPUT_BATCH"]
            ),
            "actual_microbatch_size": int(os.environ["BACKEND_CONTRACT_BATCH"]),
            "actual_gradient_accumulation_steps": int(
                os.environ["BACKEND_CONTRACT_ACCUM"]
            ),
            "selected_input_batches_per_optimizer_step": int(
                os.environ["BACKEND_CONTRACT_SELECTION_ACCUM"]
            ),
            "source_homogeneity_preserved_by_exact_divisor": True,
        },
    },
}
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
raise SystemExit(0 if admitted else 1)
PY
