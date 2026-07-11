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
CLEAN_OUT="$ROOT/outputs/evaluation/legal-source-heldout"
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
FULL_RUNS=(
  qwen3-embedding-8b-ko-performance200k-last4
)

mkdir -p "$LOG_DIR" "$SIONIC_OUT" "$OFFICIAL_OUT" "$CLEAN_OUT" "$MODEL_ROOT"
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
  weights_sha="$(jq -r '.model.weights_sha256' "$merged/merge_report.json")"
  revision="model-${weights_sha:0:12}"
  run_stage "sionic9-$run_name" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_sionic9.py" \
    --model "$merged_rel" --revision "$revision" --batch-size 192 --max-length 8192 \
    --attn-implementation flash_attention_2 \
    --output-dir "$SIONIC_OUT" \
    --embedding-cache-dir "$ROOT/outputs/embedding-cache/sionic9/$run_name"
done

for run_name in "${FULL_RUNS[@]}"; do
  run_dir="$ROOT/outputs/$run_name"
  [[ -d "$run_dir" ]] || continue
  checkpoint="$($ROOT/.venv-train/bin/python "$ROOT/scripts/select_best_checkpoint.py" \
    "$run_dir" --checkpoint-kind full --print-path 2>/dev/null)" || continue
  [[ -n "$checkpoint" ]] || continue
  packaged_rel="artifacts/models/${run_name}-best-full"
  packaged="$ROOT/$packaged_rel"
  if [[ ! -s "$packaged/full_tuning_report.json" ]]; then
    run_stage "package-$run_name" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/package_full_embedding_checkpoint.py" \
      --checkpoint "$checkpoint" --output-dir "$packaged" \
      --device cuda --dtype bfloat16 --attn-implementation flash_attention_2 || continue
  fi
  weights_sha="$(jq -r '.model.weights_sha256' "$packaged/full_tuning_report.json")"
  revision="model-${weights_sha:0:12}"
  run_stage "sionic9-$run_name" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_sionic9.py" \
    --model "$packaged_rel" --revision "$revision" --batch-size 192 --max-length 8192 \
    --attn-implementation flash_attention_2 \
    --output-dir "$SIONIC_OUT" \
    --embedding-cache-dir "$ROOT/outputs/embedding-cache/sionic9/$run_name"
done

SELECTION="$LOG_DIR/sionic9-selection.json"
run_stage "select-best-sionic9" \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/select_best_sionic_model.py" \
  "$SIONIC_OUT" --output "$SELECTION"

best_model=""
local_revision=""
if [[ -s "$SELECTION" ]]; then
  best_model="$(jq -r '.best.model' "$SELECTION")"
  best_abs="$ROOT/$best_model"
  if [[ -s "$best_abs/merge_report.json" ]]; then
    weights_sha="$(jq -r '.model.weights_sha256' "$best_abs/merge_report.json")"
    local_revision="model-${weights_sha:0:12}"
  else
    weights_sha="$(jq -r '.model.weights_sha256' "$best_abs/full_tuning_report.json")"
    local_revision="model-${weights_sha:0:12}"
  fi
  run_stage "official-korean-v1-best" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_mteb_korean_v1.py" \
    --model "$best_model" --revision "$local_revision" --max-length 8192 \
    --batch-size 192 --attn-implementation flash_attention_2 \
    --output-dir "$OFFICIAL_OUT" \
    --embedding-cache-dir "$ROOT/outputs/embedding-cache/official-best"

  safe_model_name="${best_model//\//__}"
  official_summary="$OFFICIAL_OUT/$safe_model_name/$local_revision/summary.json"
  sionic_summary="$(jq -r '.best.summary' "$SELECTION")"
  clean_summary="$CLEAN_OUT/$safe_model_name/$local_revision/summary.json"
  clean_success=0
  for batch in 192 96 48; do
    if run_stage "clean-legal-selected-before-publish-b$batch" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_legal_source_holdout.py" \
      --model "$best_model" --revision "$local_revision" --batch-size "$batch" \
      --max-length 8192 --attn-implementation flash_attention_2 \
      --output-dir "$CLEAN_OUT" \
      --embedding-cache-dir "$ROOT/outputs/embedding-cache/legal-source-heldout"; then
      clean_success=1
      break
    fi
  done
  (( clean_success == 1 )) || echo "[$(timestamp)] selected-model clean legal evaluation failed"
  if [[ "$best_model" == *performance200k* ]]; then
    training_manifest="$ROOT/outputs/data/performance-v1/ablation-200k/homogeneous-b16.manifest.json"
  elif [[ "$best_model" == *performance50k* ]]; then
    training_manifest="$ROOT/outputs/data/performance-v1/pilot-50k/homogeneous-b16.manifest.json"
  else
    training_manifest="$ROOT/data/processed/ko_triplet_pilot_10k/train.hn-qwen3-r095-n4.jsonl.manifest.json"
    [[ -s "$training_manifest" ]] || training_manifest="$ROOT/data/processed/ko_triplet_pilot_10k/manifest.json"
  fi
  if [[ -s "$official_summary" && -s "$sionic_summary" && -s "$training_manifest" ]]; then
    clean_args=()
    [[ -s "$clean_summary" ]] && clean_args+=(--clean-summary "$clean_summary")
    if run_stage "publish-best-public-model" env HF_TOKEN="${HF_TOKEN:-}" \
      "$ROOT/.venv-train/bin/python" "$ROOT/scripts/publish_best_embedding_model.py" \
      --model-dir "$best_abs" \
      --sionic-summary "$sionic_summary" \
      --official-summary "$official_summary" \
      --training-manifest "$training_manifest" \
      "${clean_args[@]}" \
      --repo-id LLM-OS-Models/qwen3-embedding-8b-ko-performance-v1 \
      --upload --public; then
      run_stage "record-pilot-best-result" \
        "$ROOT/scripts/commit_campaign_result.sh" \
        --stage pilot-best --model "$best_model" \
        --repo-id LLM-OS-Models/qwen3-embedding-8b-ko-performance-v1 \
        --sionic-summary "$sionic_summary" --official-summary "$official_summary"
    fi
  fi
fi

# Third board: fixed 10K same-repository source-document-held-out legal set.
# Run trusted baselines and the selected local candidate after the two primary
# public boards, without feeding these scores back into checkpoint selection.
clean_models=(
  "Qwen/Qwen3-Embedding-8B|1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
  "sionic-ai/comsat-embed-ko-8b-preview|a5cc22b651c1b2e51cdd8bf671774ae93584f0ab"
)
if [[ -n "$best_model" && -n "$local_revision" \
    && ! -s "$CLEAN_OUT/${best_model//\//__}/$local_revision/summary.json" ]]; then
  clean_models+=("$best_model|$local_revision")
fi
for spec in "${clean_models[@]}"; do
  model="${spec%%|*}"
  revision="${spec#*|}"
  success=0
  for batch in 192 96 48; do
    if run_stage "clean-legal-${model//\//__}-b$batch" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_legal_source_holdout.py" \
      --model "$model" --revision "$revision" --batch-size "$batch" \
      --max-length 8192 --attn-implementation flash_attention_2 \
      --output-dir "$CLEAN_OUT" \
      --embedding-cache-dir "$ROOT/outputs/embedding-cache/legal-source-heldout"; then
      success=1
      break
    fi
  done
  (( success == 1 )) || echo "[$(timestamp)] clean legal evaluation failed: $model"
done
run_stage "record-clean-legal-results" "$ROOT/scripts/commit_clean_legal_results.sh" || true

echo "[$(timestamp)] post-training evaluation queue complete"
