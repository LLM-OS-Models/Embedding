#!/usr/bin/env bash
set -uo pipefail

# After the training queue exits, merge every completed experiment, run the
# complete Sionic 9 suite, select by its exact primary metric, then evaluate the
# winner on official Korean MTEB v1. Failures are isolated per candidate.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WAIT_PID="${WAIT_PID:-}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/post-training-eval-20260711}"
SIONIC_OUT="$ROOT/outputs/evaluation/sionic9-posttrain"
OFFICIAL_OUT="$ROOT/outputs/evaluation/mteb-korean-v1-posttrain"
MODEL_ROOT="$ROOT/artifacts/models"

RUNS=(
  qwen3-embedding-8b-ko-hn10k-lora-r64
  qwen3-embedding-8b-ko-hn10k-lora-r64-b8
  qwen3-embedding-8b-ko-performance50k-lora-r64
  qwen3-embedding-8b-ko-performance50k-lora-r64-b8
  qwen3-embedding-8b-ko-performance200k-lora-r64
  qwen3-embedding-8b-ko-performance200k-lora-r64-b8
  qwen3-embedding-8b-ko-hn10k-f2dual-lora-r64
  qwen3-embedding-8b-ko-hn10k-f2dual-t002-lora-r64
  qwen3-embedding-8b-ko-hn10k-f2dual-mrl-lora-r64
)

mkdir -p "$LOG_DIR" "$SIONIC_OUT" "$OFFICIAL_OUT" "$MODEL_ROOT"
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1

if [[ -f "$ROOT/.env" ]]; then
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "$ROOT/.env" | tail -n 1)"
  export HF_TOKEN
fi

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }

run_stage() {
  local name="$1"
  shift
  echo "[$(timestamp)] START $name"
  "$@"
  local status=$?
  echo "[$(timestamp)] END $name status=$status"
  return "$status"
}

if [[ -n "$WAIT_PID" ]]; then
  echo "[$(timestamp)] waiting for training queue pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 15; done
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$ROOT/third_party/mteb${PYTHONPATH:+:$PYTHONPATH}"

for run_name in "${RUNS[@]}"; do
  run_dir="$ROOT/outputs/$run_name"
  [[ -d "$run_dir" ]] || continue
  checkpoint="$($ROOT/.venv-train/bin/python "$ROOT/scripts/select_best_checkpoint.py" \
    "$run_dir" --print-path 2>/dev/null)" || continue
  [[ -n "$checkpoint" ]] || continue
  merged_rel="artifacts/models/${run_name}-best-merged"
  merged="$ROOT/$merged_rel"
  if [[ ! -s "$merged/merge_report.json" ]]; then
    run_stage "merge-$run_name" \
      "$ROOT/.venv-train/bin/python" "$ROOT/scripts/merge_embedding_adapter.py" \
      --adapter "$checkpoint" --output-dir "$merged" \
      --device cuda --dtype bfloat16 --local-files-only || continue
  fi
  run_stage "sionic9-$run_name" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_sionic9.py" \
    --model "$merged_rel" --batch-size 192 --max-length 8192 \
    --attn-implementation flash_attention_2 \
    --output-dir "$SIONIC_OUT" \
    --embedding-cache-dir "$ROOT/outputs/embedding-cache/sionic9/$run_name"
done

SELECTION="$LOG_DIR/sionic9-selection.json"
run_stage "select-best-sionic9" \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/select_best_sionic_model.py" \
  "$SIONIC_OUT" --output "$SELECTION"

if [[ -s "$SELECTION" ]]; then
  best_model="$(jq -r '.best.model' "$SELECTION")"
  best_abs="$ROOT/$best_model"
  adapter_sha="$(jq -r '.adapter.weights_sha256' "$best_abs/merge_report.json")"
  local_revision="adapter-${adapter_sha:0:12}"
  run_stage "official-korean-v1-best" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_mteb_korean_v1.py" \
    --model "$best_model" --revision "$local_revision" --max-length 8192 \
    --batch-size 192 --attn-implementation flash_attention_2 \
    --output-dir "$OFFICIAL_OUT" \
    --embedding-cache-dir "$ROOT/outputs/embedding-cache/official-best"

  safe_model_name="${best_model//\//__}"
  official_summary="$OFFICIAL_OUT/$safe_model_name/$local_revision/summary.json"
  sionic_summary="$(jq -r '.best.summary' "$SELECTION")"
  if [[ "$best_model" == *performance200k* ]]; then
    training_manifest="$ROOT/outputs/data/performance-v1/ablation-200k/manifest.json"
  elif [[ "$best_model" == *performance50k* ]]; then
    training_manifest="$ROOT/outputs/data/performance-v1/pilot-50k/manifest.json"
  else
    training_manifest="$ROOT/data/processed/ko_triplet_pilot_10k/train.hn-qwen3-r095-n4.jsonl.manifest.json"
    [[ -s "$training_manifest" ]] || training_manifest="$ROOT/data/processed/ko_triplet_pilot_10k/manifest.json"
  fi
  if [[ -s "$official_summary" && -s "$sionic_summary" && -s "$training_manifest" ]]; then
    run_stage "publish-best-public-model" env HF_TOKEN="${HF_TOKEN:-}" \
      "$ROOT/.venv-train/bin/python" "$ROOT/scripts/publish_best_embedding_model.py" \
      --model-dir "$best_abs" \
      --sionic-summary "$sionic_summary" \
      --official-summary "$official_summary" \
      --training-manifest "$training_manifest" \
      --repo-id LLM-OS-Models/qwen3-embedding-8b-ko-performance-v1 \
      --upload --public
  fi
fi

echo "[$(timestamp)] post-training evaluation queue complete"
