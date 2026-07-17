#!/usr/bin/env bash
set -euo pipefail

# Run one high-value tuning-capacity challenger after Qwen/Comsat lineage
# selection.  The clean selector decides the lineage without public benchmark
# input; this queue then trains the same raw base with the same 200K rows and
# token budget while updating the final four transformer blocks plus norm.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
embedding_resolve_train_runtime
cd "$ROOT"

LINEAGE_SELECTION="${LINEAGE_SELECTION:-$ROOT/outputs/post-training-eval-20260717-frontier/clean-first-selection.json}"
TRAIN_FILE="$ROOT/outputs/data/performance-v1/ablation-200k/train.homogeneous-b16.jsonl"
VAL_FILE="$ROOT/data/processed/ko_triplet_pilot_10k/validation.hn-qwen3-r095-n4.jsonl"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/capacity-ablation-20260717-frontier}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1

unset HF_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }

for path in "$LINEAGE_SELECTION" "$TRAIN_FILE" "$VAL_FILE"; do
  if [[ ! -s "$path" ]]; then
    echo "[$(timestamp)] missing capacity input: $path" >&2
    exit 2
  fi
done

selected_rel="$(jq -r '.best.model // empty' "$LINEAGE_SELECTION")"
if [[ -z "$selected_rel" || "$selected_rel" == /* \
    || "$selected_rel" == ../* || "$selected_rel" == *../* ]]; then
  echo "[$(timestamp)] unsafe or empty lineage selection" >&2
  exit 2
fi
selected="$ROOT/$selected_rel"
evidence="$selected/merge_report.json"
if [[ ! -s "$evidence" ]]; then
  echo "[$(timestamp)] lineage winner is not a safe-merged LoRA model" >&2
  exit 2
fi
base_model="$(jq -r '.base_model // empty' "$evidence")"
base_revision="$(jq -r '.base_revision // empty' "$evidence")"
case "$base_model@$base_revision" in
  Qwen/Qwen3-Embedding-8B@1d8ad4ca9b3dd8059ad90a75d4983776a23d44af)
    lineage=qwen
    run_name=qwen3-embedding-8b-ko-performance200k-last4
    ;;
  sionic-ai/comsat-embed-ko-8b-preview@a5cc22b651c1b2e51cdd8bf671774ae93584f0ab)
    lineage=comsat
    run_name=comsat-embed-ko-8b-performance200k-last4
    ;;
  *)
    echo "[$(timestamp)] unsupported selected base contract: $base_model@$base_revision" >&2
    exit 2
    ;;
esac

run_dir="$ROOT/outputs/$run_name"
completion_evidence() {
  local logging
  rg -q '\[INFO:swift\] End time of running main:' "$run_dir/train.log" 2>/dev/null || return 1
  logging="$(find "$run_dir" -mindepth 2 -maxdepth 2 -type f \
    -name logging.jsonl -print 2>/dev/null | sort | tail -n 1)"
  [[ -n "$logging" ]] && rg -q '"3123/3123"' "$logging"
}

finalize_training_contract() {
  local contract="$run_dir/capacity_run_manifest.json" logging
  [[ -s "$contract" && -s "$run_dir/train.log" ]] || return 1
  logging="$(find "$run_dir" -mindepth 2 -maxdepth 2 -type f \
    -name logging.jsonl -print 2>/dev/null | sort | tail -n 1)"
  [[ -n "$logging" && -s "$logging" ]] || return 1
  "$EMBEDDING_TRAIN_PYTHON" - "$contract" "$run_dir/train.log" "$logging" <<'PY'
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


contract, train_log, logging = map(lambda value: Path(value).resolve(), sys.argv[1:])
payload = json.loads(contract.read_text(encoding="utf-8"))
if payload.get("status") == "complete":
    raise SystemExit(0)
if payload.get("status") != "armed":
    raise ValueError("capacity contract is neither armed nor complete")
payload["status"] = "complete"
payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
payload["completion"] = {
    "expected_steps": 3123,
    "train_log": {"path": str(train_log), "sha256": sha256(train_log)},
    "logging_jsonl": {"path": str(logging), "sha256": sha256(logging)},
}
temporary = contract.with_name(f".{contract.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, contract)
PY
}

if completion_evidence; then
  finalize_training_contract
  echo "[$(timestamp)] reuse completed capacity run: $run_name"
  exit 0
fi

embedding_require_storage_headroom "$ROOT" 500 1000000
embedding_require_storage_headroom /tmp 50 100000
echo "[$(timestamp)] probing last4 partial-full memory for $lineage lineage"
if ! env EMBEDDING_OFFLINE=1 TRAIN_ENV="$ROOT/.venv-train-fa2" \
    BASE_MODEL="$base_model" BASE_REVISION="$base_revision" \
    DATA="$TRAIN_FILE" MAX_LENGTH=512 ATTN_IMPL=sdpa \
    TRAIN_BATCH_SIZE=8 GRAD_ACCUM_STEPS=8 INFONCE_HARD_NEGATIVES=4 \
    PROBE_SUFFIX="${lineage}-performance200k-b8" \
    "$ROOT/experiments/070_tuning_strategy/probe_memory.sh" last4; then
  "$EMBEDDING_TRAIN_PYTHON" - "$LOG_DIR/capacity-skipped.json" \
    "$lineage" "$base_model" "$base_revision" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "status": "skipped",
    "reason": "production_microbatch_memory_probe_failed",
    "lineage": sys.argv[2],
    "base_model": sys.argv[3],
    "base_revision": sys.argv[4],
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  echo "[$(timestamp)] last4 candidate skipped after failed memory probe"
  exit 0
fi

embedding_require_storage_headroom "$ROOT" 500 1000000
embedding_require_storage_headroom /tmp 50 100000
echo "[$(timestamp)] starting capacity challenger: $run_name"
env EMBEDDING_OFFLINE=1 TRAIN_ENV="$ROOT/.venv-train-fa2" \
  RUN_NAME="$run_name" BASE_MODEL="$base_model" BASE_REVISION="$base_revision" \
  TRAIN_FILE="$TRAIN_FILE" VAL_FILE="$VAL_FILE" \
  MAX_STEPS=3123 SAVE_TOTAL_LIMIT=5 TRAIN_BATCH_SIZE=8 GRAD_ACCUM_STEPS=8 \
  ATTN_IMPL=sdpa \
  "$ROOT/experiments/070_tuning_strategy/train_quality.sh" last4

if ! completion_evidence; then
  echo "[$(timestamp)] capacity training exited without exact 3123-step evidence" >&2
  exit 10
fi
finalize_training_contract
echo "[$(timestamp)] capacity challenger completed: $run_name"
