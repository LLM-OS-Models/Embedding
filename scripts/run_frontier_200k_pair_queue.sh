#!/usr/bin/env bash
set -euo pipefail

# Keep the single H100 occupied without overlapping two 8B jobs.  The active
# Qwen clean-lineage run was launched before this queue; after it finishes
# successfully, probe and train the Comsat warm-start under the same 200K
# contract.  Each lineage has a distinct backend report and private repo.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
cd "$ROOT"

QWEN_RUN="$ROOT/outputs/qwen3-embedding-8b-ko-performance200k-lora-r64"
QWEN_LOG="$QWEN_RUN/train.log"
COMSAT_RUN_ID="comsat-embed-ko-8b-performance200k-lora-r64"
COMSAT_RUN="$ROOT/outputs/$COMSAT_RUN_ID"
COMSAT_MODEL="sionic-ai/comsat-embed-ko-8b-preview"
COMSAT_REVISION="a5cc22b651c1b2e51cdd8bf671774ae93584f0ab"
COMSAT_ADMISSION_KEY="comsat-performance200k-lora-r64"
TRAIN_FILE="$ROOT/outputs/data/performance-v1/ablation-200k/train.homogeneous-b16.jsonl"
TRAIN_MANIFEST="$ROOT/outputs/data/performance-v1/ablation-200k/homogeneous-b16.manifest.json"
VAL_FILE="$ROOT/data/processed/ko_triplet_pilot_10k/validation.hn-qwen3-r095-n4.jsonl"
QUEUE_LOG="$ROOT/outputs/frontier-200k-pair-queue.log"
POST_EVAL_LOG="$ROOT/outputs/post-training-eval-20260717-frontier"
POST_EVAL_SELECTION="$POST_EVAL_LOG/clean-first-selection.json"
SCALE_LOG="$ROOT/outputs/scale-1m-20260717-frontier"
LEGAL_LOG="$ROOT/outputs/legal-adaptation-20260717-frontier"

mkdir -p "$COMSAT_RUN"
exec > >(tee -a "$QUEUE_LOG") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

echo "[$(timestamp)] waiting for Qwen 200K completion"
heartbeat=0
while ! rg -q '\[INFO:swift\] End time of running main:' "$QWEN_LOG" 2>/dev/null; do
  if tail -n 300 "$QWEN_LOG" 2>/dev/null | \
      rg -q '^Traceback \(most recent call last\)|CUDA out of memory'; then
    echo "[$(timestamp)] Qwen run failed before successful completion" >&2
    exit 10
  fi
  sleep 30
  heartbeat=$((heartbeat + 1))
  if (( heartbeat % 20 == 0 )); then
    echo "[$(timestamp)] Qwen 200K still active"
  fi
done

qwen_logging="$(find "$QWEN_RUN" -mindepth 2 -maxdepth 2 -type f \
  -name logging.jsonl -print | sort | tail -n 1)"
if [[ -z "$qwen_logging" ]] || ! rg -q '"3123/3123"' "$qwen_logging"; then
  echo "[$(timestamp)] Qwen end marker found without step 3123 evidence" >&2
  exit 11
fi
echo "[$(timestamp)] Qwen 200K completed; starting Comsat exact probe"

env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TRAIN_FILE="$TRAIN_FILE" RUN_KEY="$COMSAT_ADMISSION_KEY" \
  BASE_MODEL="$COMSAT_MODEL" BASE_REVISION="$COMSAT_REVISION" \
  TRAIN_BATCH_SIZE=16 GRAD_ACCUM_STEPS=4 MAX_LENGTH=512 \
  LORA_RANK=64 LORA_ALPHA=128 LORA_DROPOUT=0.05 \
  INFONCE_HARD_NEGATIVES=4 PROBE_STEPS=5 FORCE_PROBE=1 \
  "$ROOT/experiments/070_tuning_strategy/admit_fa2_lora_backend.sh" || true

COMSAT_ADMISSION="$ROOT/outputs/backend-probes/$COMSAT_ADMISSION_KEY/admission.json"
if ! ROOT="$ROOT" FA2_ENV="$ROOT/.venv-train-fa2" \
    bash -c 'source "$ROOT/scripts/common_runtime.sh"; source "$ROOT/scripts/backend_admission.sh"; embedding_check_matched_sdpa "$1" "$2" 16 4 512 64 128 bfloat16 "$3" "$4" 4 0.05' \
    bash "$COMSAT_ADMISSION" "$TRAIN_FILE" "$COMSAT_MODEL" "$COMSAT_REVISION"; then
  echo "[$(timestamp)] Comsat matched-SDPA contract was not admitted" >&2
  exit 12
fi

training_manifest_sha="$(sha256sum "$TRAIN_MANIFEST" | awk '{print $1}')"
admission_sha="$(sha256sum "$COMSAT_ADMISSION" | awk '{print $1}')"

"$ROOT/.venv-train-fa2/bin/python" \
  "$ROOT/scripts/watch_private_adapter_checkpoints.py" \
  --watch-dir "$COMSAT_RUN" \
  --repo-id LLM-OS-Models2/comsat-embed-ko-8b-performance200k-lora-r64-candidates \
  --base-model "$COMSAT_MODEL" \
  --base-revision "$COMSAT_REVISION" \
  --run-id "$COMSAT_RUN_ID" \
  --training-manifest-sha256 "$training_manifest_sha" \
  --admission-report-sha256 "$admission_sha" \
  --poll-seconds 5 --settle-seconds 10 --upload \
  >> "$COMSAT_RUN/checkpoint-watcher.log" 2>&1 &
watcher_pid=$!
cleanup() {
  kill "$watcher_pid" 2>/dev/null || true
  wait "$watcher_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[$(timestamp)] starting Comsat 200K production"
env EMBEDDING_OFFLINE=1 ENABLE_VALIDATED_CONTINUAL_BASE=0 \
  RUN_NAME="$COMSAT_RUN_ID" BASE_MODEL="$COMSAT_MODEL" \
  BASE_REVISION="$COMSAT_REVISION" \
  BACKEND_ADMISSION_RUN_KEY="$COMSAT_ADMISSION_KEY" \
  TRAIN_FILE="$TRAIN_FILE" VAL_FILE="$VAL_FILE" \
  MAX_STEPS=3123 EVAL_STEPS=250 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=5 \
  TRAIN_BATCH_SIZE=16 EVAL_BATCH_SIZE=4 GRAD_ACCUM_STEPS=4 \
  DATASET_SHUFFLE=false TRAIN_DATALOADER_SHUFFLE=false AUTO_SELECT_FA2=1 \
  "$ROOT/experiments/020_hard_negative/train_pilot_lora_r64.sh"

echo "[$(timestamp)] Comsat 200K production completed"

# The watcher has finished its production responsibility.  Stop it explicitly
# before moving into evaluation so the EXIT trap cannot linger for the rest of
# the multi-day campaign.
cleanup
trap - EXIT INT TERM

embedding_require_storage_headroom "$ROOT" 500 1000000
embedding_require_storage_headroom /tmp 50 100000
echo "[$(timestamp)] starting clean-first Qwen/Comsat comparison"
env WAIT_PID= LOG_DIR="$POST_EVAL_LOG" \
  CAMPAIGN_EVAL_BATCH_SIZES="192 128 64 32 16 8 4 2" \
  bash "$ROOT/scripts/run_post_training_eval_queue.sh"
if [[ ! -s "$POST_EVAL_SELECTION" ]]; then
  echo "[$(timestamp)] clean-first selection was not produced" >&2
  exit 20
fi

embedding_require_storage_headroom "$ROOT" 500 1000000
embedding_require_storage_headroom /tmp 50 100000
echo "[$(timestamp)] starting 1M scale from the clean-selected lineage"
env WAIT_PID= LOG_DIR="$SCALE_LOG" \
  POSTTRAIN_SELECTION="$POST_EVAL_SELECTION" \
  bash "$ROOT/scripts/run_scale_1m_queue.sh"

embedding_require_storage_headroom "$ROOT" 500 1000000
embedding_require_storage_headroom /tmp 50 100000
echo "[$(timestamp)] starting legal replay and combined target adaptation"
env WAIT_PID= LOG_DIR="$LEGAL_LOG" \
  GENERAL_SELECTION="$ROOT/outputs/reranker-kd-20260717-frontier/clean-first-selection.json" \
  ENABLE_SIONIC_COMBINED_ADAPTATION=1 \
  bash "$ROOT/scripts/run_legal_adaptation_queue.sh"

echo "[$(timestamp)] frontier campaign queue completed"
