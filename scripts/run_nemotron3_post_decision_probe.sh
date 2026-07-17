#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WAIT_PID="${WAIT_PID:-}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/nemotron3-post-decision-probe}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/runner.log") 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }

if [[ -n "$WAIT_PID" ]]; then
  [[ "$WAIT_PID" =~ ^[0-9]+$ ]] || { echo "WAIT_PID must be numeric" >&2; exit 2; }
  echo "[$(timestamp)] waiting for base-decision pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi

DECISION="$ROOT/outputs/evaluation/nemotron3-base-decision.json"
"$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/decide_nemotron3_base.py" \
  --sionic-dir "$ROOT/outputs/evaluation/sionic9-nemotron3-full-fixed-prompt" \
  --legal-dir "$ROOT/outputs/evaluation/legal-source-heldout-base-decision" \
  --multidomain-dir "$ROOT/outputs/evaluation/multidomain-selection-base-decision" \
  --output "$DECISION"

action="$(jq -er '.decision' "$DECISION")"
case "$action" in
  adopt_nemotron3_raw_and_run_short_public_lora|short_public_nemotron3_lora_then_retest)
    ;;
  *)
    printf '%s\n' "$action" > "$LOG_DIR/probe-skipped.decision"
    echo "[$(timestamp)] probe skipped decision=$action"
    exit 0
    ;;
esac

NEMOTRON_REVISION="2b29550c4ab0646bb6bb47032dda54ea11f6dfe2"
NEMOTRON_PATH="$ROOT/.cache/huggingface/hub/models--nvidia--Nemotron-3-Embed-8B-BF16/snapshots/$NEMOTRON_REVISION"
TRAIN="$ROOT/outputs/data/public-legal-source-training-v1/data/train.jsonl"
MANIFEST="$ROOT/outputs/data/public-legal-source-training-v1/metadata/manifest.json"
OUT="$ROOT/outputs/nemotron3-public-legal-lora-probe"

env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN -u HUGGING_FACE_HUB_TOKEN \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  "$ROOT/.venv-train-fa2/bin/python" "$ROOT/scripts/train_nemotron3_public_lora.py" \
  --model "$NEMOTRON_PATH" --revision "$NEMOTRON_REVISION" \
  --train "$TRAIN" --training-manifest "$MANIFEST" --output-dir "$OUT" \
  --max-steps 1 --batch-size 2 --mini-batch-size 1 --save-steps 1

test -s "$OUT/checkpoint-1/trainer_state.json"
test -s "$OUT/checkpoint-1/optimizer.pt"
test -s "$OUT/checkpoint-1/scheduler.pt"
jq -n \
  --arg completed_at "$(date --iso-8601=seconds)" \
  --arg decision "$action" \
  --arg checkpoint "$OUT/checkpoint-1" \
  '{status:"pass",completed_at:$completed_at,decision:$decision,checkpoint:$checkpoint}' \
  > "$LOG_DIR/probe-complete.json"
echo "[$(timestamp)] Nemotron public LoRA backward probe passed"
