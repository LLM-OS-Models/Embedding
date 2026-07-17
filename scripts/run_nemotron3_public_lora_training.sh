#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NEMOTRON_REVISION="2b29550c4ab0646bb6bb47032dda54ea11f6dfe2"
MODEL="${MODEL:-$ROOT/.cache/huggingface/hub/models--nvidia--Nemotron-3-Embed-8B-BF16/snapshots/$NEMOTRON_REVISION}"
DATA_DIR="${DATA_DIR:-$ROOT/outputs/data/public-legal-source-training-v1/mined-nemotron3}"
TRAIN="${TRAIN:-$DATA_DIR/train.homogeneous-b32.jsonl}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$DATA_DIR/final-public-manifest.json}"
EVAL="${EVAL:-$ROOT/outputs/data/validation/legal-source-heldout-i-v2-text-strict-512/validation.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-$ROOT/outputs/data/validation/legal-source-heldout-i-v2-text-strict-512/manifest.json}"
DECISION="${DECISION:-$ROOT/outputs/evaluation/nemotron3-base-decision.json}"
OUT="${OUT:-$ROOT/outputs/nemotron3-ko-public-lora-r16}"
REPO_ID="${REPO_ID:-LLM-OS-Models2/nemotron3-ko-public-lora-r16-checkpoints}"
RUN_ID="${RUN_ID:-nemotron3-ko-public-lora-r16}"
MAX_STEPS="${MAX_STEPS:-300}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-2}"
SAVE_STEPS="${SAVE_STEPS:-50}"
LOG_DIR="${LOG_DIR:-$OUT/logs}"

mkdir -p "$OUT" "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/runner.log") 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }

for path in "$MODEL/config.json" "$TRAIN" "$TRAIN_MANIFEST" "$EVAL" "$EVAL_MANIFEST" "$DECISION"; do
  [[ -s "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done
action="$(jq -er '.decision' "$DECISION")"
case "$action" in
  adopt_nemotron3_raw_and_run_short_public_lora|short_public_nemotron3_lora_then_retest) ;;
  *) echo "base decision does not authorize Nemotron training: $action" >&2; exit 3 ;;
esac

train_sha="$(sha256sum "$TRAIN" | awk '{print $1}')"
manifest_sha="$(sha256sum "$TRAIN_MANIFEST" | awk '{print $1}')"
[[ "$(jq -r '.release_eligible' "$TRAIN_MANIFEST")" == true ]]
[[ "$(jq -r '.visibility' "$TRAIN_MANIFEST")" == public ]]
[[ "$(jq -r '.release_blockers | length' "$TRAIN_MANIFEST")" == 0 ]]
[[ "$(jq -r '.outputs.train.sha256' "$TRAIN_MANIFEST")" == "$train_sha" ]]

watcher_args=(
  "$ROOT/scripts/watch_private_adapter_checkpoints.py"
  --watch-dir "$OUT"
  --repo-id "$REPO_ID"
  --base-model nvidia/Nemotron-3-Embed-8B-BF16
  --base-revision "$NEMOTRON_REVISION"
  --run-id "$RUN_ID"
  --training-data-sha256 "$train_sha"
  --training-manifest-sha256 "$manifest_sha"
  --training-manifest-path "$TRAIN_MANIFEST"
  --base-license OpenMDW-1.1
  --poll-seconds 5 --settle-seconds 10
  --remote-attempts 3 --remote-retry-seconds 15
  --upload --public
)

env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN -u HUGGING_FACE_HUB_TOKEN \
  "$ROOT/.venv-train-fa2/bin/python" "${watcher_args[@]}" \
  >> "$LOG_DIR/checkpoint-watcher.log" 2>&1 &
watcher_pid=$!
cleanup() {
  kill -TERM "$watcher_pid" 2>/dev/null || true
  wait "$watcher_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[$(timestamp)] starting Nemotron public LoRA training"
env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN -u HUGGING_FACE_HUB_TOKEN \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  "$ROOT/.venv-train-fa2/bin/python" "$ROOT/scripts/train_nemotron3_public_lora.py" \
  --model "$MODEL" --revision "$NEMOTRON_REVISION" \
  --train "$TRAIN" --training-manifest "$TRAIN_MANIFEST" \
  --eval "$EVAL" --eval-manifest "$EVAL_MANIFEST" \
  --output-dir "$OUT" --max-steps "$MAX_STEPS" \
  --batch-size "$BATCH_SIZE" --mini-batch-size "$MINI_BATCH_SIZE" \
  --save-steps "$SAVE_STEPS"

cleanup
trap - EXIT INT TERM
"$ROOT/.venv-train-fa2/bin/python" "${watcher_args[@]}" --once \
  >> "$LOG_DIR/checkpoint-watcher.log" 2>&1
test -s "$OUT/final-adapter/adapter_model.safetensors"
test -s "$OUT/final-adapter/adapter_config.json"
jq -n \
  --arg completed_at "$(date --iso-8601=seconds)" \
  --arg decision "$action" \
  --arg repo_id "$REPO_ID" \
  --arg final_adapter "$OUT/final-adapter" \
  '{status:"complete",completed_at:$completed_at,decision:$decision,repo_id:$repo_id,final_adapter:$final_adapter}' \
  > "$OUT/training-complete.json"
echo "[$(timestamp)] Nemotron public LoRA training and checkpoint publication complete"
