#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
cd "$ROOT"
WAIT_PID="${WAIT_PID:-}"
CONFIG="${CONFIG:-$ROOT/configs/models_to_evaluate.json}"
OUT="${OUT:-$ROOT/outputs/evaluation/sionic9-top-models}"
OFFICIAL_OUT="$ROOT/outputs/evaluation/mteb-korean-v1-qwen-base"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/top-model-eval-20260711}"
mkdir -p "$OUT" "$OFFICIAL_OUT" "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1

if [[ -f "$ROOT/.env" ]]; then
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "$ROOT/.env" | tail -n 1)"
  export HF_TOKEN
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$ROOT/third_party/mteb${PYTHONPATH:+:$PYTHONPATH}"

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }

if [[ -n "$WAIT_PID" ]]; then
  echo "[$(timestamp)] waiting for post-training queue pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi

# Qwen has no live Korean-v1 row. Use its pinned MTEB registry wrapper so each
# task receives the official model-specific instruction contract.
QWEN_REVISION="4e423935c619ae4df87b646a3ce949610c66241c"
QWEN_OFFICIAL_SUMMARY="$OFFICIAL_OUT/Qwen__Qwen3-Embedding-8B/$QWEN_REVISION/summary.json"
if [[ ! -s "$QWEN_OFFICIAL_SUMMARY" ]]; then
  for candidate_batch in 192 96 48; do
    echo "[$(timestamp)] START Qwen official Korean v1 batch=$candidate_batch"
    if "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_mteb_korean_v1.py" \
        --model Qwen/Qwen3-Embedding-8B --revision "$QWEN_REVISION" \
        --registered-loader --batch-size "$candidate_batch" --max-length 32768 \
        --attn-implementation flash_attention_2 --output-dir "$OFFICIAL_OUT" \
        --embedding-cache-dir "$ROOT/outputs/embedding-cache/official-qwen-base"; then
      status=0
    else
      status=$?
    fi
    echo "[$(timestamp)] END Qwen official Korean v1 batch=$candidate_batch status=$status"
    [[ -s "$QWEN_OFFICIAL_SUMMARY" ]] && break
  done
fi
if [[ -s "$QWEN_OFFICIAL_SUMMARY" ]]; then
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/compare_local_mteb_korean.py" \
    --summary "$QWEN_OFFICIAL_SUMMARY" \
    --output "$OFFICIAL_OUT/qwen-live-comparison.json" && \
    "$ROOT/scripts/commit_qwen_official_result.sh" || true
fi

mapfile -t models < <(jq -r '.models | sort_by(.queue_order)[] | select(.execution.sionic9.supported == true) | .id' "$CONFIG")
for model in "${models[@]}"; do
  revision="$(jq -r --arg model "$model" '.models[] | select(.id == $model) | .snapshots.sionic9_local_revision' "$CONFIG")"
  batch="$(jq -r --arg model "$model" '.models[] | select(.id == $model) | .execution.sionic9.batch_size' "$CONFIG")"
  max_length="$(jq -r --arg model "$model" '.models[] | select(.id == $model) | .lengths.sionic9_effective_max_tokens' "$CONFIG")"
  trust="$(jq -r --arg model "$model" '.models[] | select(.id == $model) | .encoder_contract.trust_remote_code' "$CONFIG")"
  safe="${model//\//__}"
  cache="$ROOT/outputs/embedding-cache/sionic9-top-models/$safe"
  success=0
  for candidate_batch in "$batch" "$((batch / 2))" "$((batch / 4))"; do
    (( candidate_batch < 1 )) && candidate_batch=1
    args=(
      --model "$model"
      --revision "$revision"
      --batch-size "$candidate_batch"
      --max-length "$max_length"
      --attn-implementation flash_attention_2
      --output-dir "$OUT"
      --embedding-cache-dir "$cache"
    )
    [[ "$trust" == true ]] && args+=(--trust-remote-code)
    echo "[$(timestamp)] START model=$model batch=$candidate_batch"
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_sionic9.py" "${args[@]}"
    status=$?
    echo "[$(timestamp)] END model=$model batch=$candidate_batch status=$status"
    if (( status == 0 )); then
      success=1
      break
    fi
  done
  (( success == 1 )) || echo "[$(timestamp)] FAILED all batches model=$model"
done

echo "[$(timestamp)] top-model Sionic queue complete"
