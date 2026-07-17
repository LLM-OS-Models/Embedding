#!/usr/bin/env bash
set -uo pipefail

# After training exits, merge each internally selected checkpoint and evaluate
# every local candidate on the Grade-I legal holdout plus its noise robustness
# companion.  Select within a clean-score near-tie without public benchmark
# input.  Only then run Sionic9 and official Korean MTEB once on the winner.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
embedding_resolve_train_runtime
UTILITY_PYTHON="$EMBEDDING_TRAIN_PYTHON"
cd "$ROOT"
WAIT_PID="${WAIT_PID:-}"
SELECTION_ONLY="${SELECTION_ONLY:-0}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/post-training-eval-20260711}"
SIONIC_OUT="$ROOT/outputs/evaluation/sionic9-posttrain-contract-v1"
OFFICIAL_OUT="$ROOT/outputs/evaluation/mteb-korean-v1-posttrain-contract-v1"
COMPREHENSIVE_OUT="$ROOT/outputs/evaluation/comprehensive-text-v1-posttrain"
CLEAN_OUT="$ROOT/outputs/evaluation/legal-source-heldout"
ROBUST_OUT="$ROOT/outputs/evaluation/conversational-noise-robustness"
MODEL_ROOT="$ROOT/artifacts/models"

RUNS=(
  qwen3-embedding-8b-ko-hn10k-lora-r64
  qwen3-embedding-8b-ko-hn10k-lora-r64-b8
  qwen3-embedding-8b-ko-performance50k-lora-r64
  qwen3-embedding-8b-ko-performance50k-lora-r64-b8
  qwen3-embedding-8b-ko-performance200k-lora-r64
  qwen3-embedding-8b-ko-performance200k-lora-r64-b8
  comsat-embed-ko-8b-performance200k-lora-r64
  comsat-embed-ko-8b-performance200k-lora-r64-b8
  qwen3-embedding-8b-ko-performance1m-lora-r64
  qwen3-embedding-8b-ko-performance1m-lora-r64-b8
  qwen3-embedding-8b-ko-performance1m-reranker-filter-only-lora-r64
  qwen3-embedding-8b-ko-performance1m-reranker-listwise-kl07-lora-r64
  qwen3-embedding-8b-ko-performance1m-reranker-listwise-kl07-queue4096-lora-r64
  qwen3-embedding-8b-ko-performance1m-reranker-margin-mse07-lora-r64
  qwen3-embedding-8b-ko-sionic-retrieval-family50-replay50-lora-r64
  qwen3-embedding-8b-ko-sionic-retrieval-family50-replay50-lora-r64-b2
  qwen3-embedding-8b-ko-sionic-squad50-replay50-lora-r64
  qwen3-embedding-8b-ko-sionic-squad50-replay50-lora-r64-b4
  qwen3-embedding-8b-ko-sionic-health50-replay50-lora-r64
  qwen3-embedding-8b-ko-sionic-health50-replay50-lora-r64-b4
  qwen3-embedding-8b-ko-sionic-autorag50-replay50-lora-r64
  qwen3-embedding-8b-ko-sionic-autorag50-replay50-lora-r64-b4
  qwen3-embedding-8b-ko-legal25-replay75-lora-r64
  qwen3-embedding-8b-ko-legal25-replay75-lora-r64-b4
  qwen3-embedding-8b-ko-sionic-combined-target-lora-r64
  qwen3-embedding-8b-ko-sionic-combined-target-lora-r64-b4
  qwen3-embedding-8b-ko-hn10k-f2dual-lora-r64
  qwen3-embedding-8b-ko-hn10k-f2dual-t002-lora-r64
  qwen3-embedding-8b-ko-hn10k-f2dual-mrl-lora-r64
)
FULL_RUNS=(
  qwen3-embedding-8b-ko-performance200k-last4
  comsat-embed-ko-8b-performance200k-last4
)
SOUP_MODELS=(
  qwen3-embedding-8b-ko-soup-general50-combined50
  qwen3-embedding-8b-ko-soup-general50-specialists10x5
  qwen3-embedding-8b-ko-soup-general25-combined25-specialists10x5
  qwen3-embedding-8b-ko-soup-combined50-specialists10x5
)

mkdir -p \
  "$LOG_DIR" "$SIONIC_OUT" "$OFFICIAL_OUT" "$COMPREHENSIVE_OUT" \
  "$CLEAN_OUT" "$ROBUST_OUT" "$MODEL_ROOT"
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1

unset HF_TOKEN HUGGINGFACE_HUB_TOKEN
PUBLISH_HF_TOKEN_FILE="$ROOT/.env"
if [[ "$SELECTION_ONLY" != 0 && "$SELECTION_ONLY" != 1 ]]; then
  echo "SELECTION_ONLY must be 0 or 1" >&2
  exit 2
fi
read -r -a EVAL_BATCHES <<< "${CAMPAIGN_EVAL_BATCH_SIZES:-192 128 64 32 16 8 4 2}"
for batch in "${EVAL_BATCHES[@]}"; do
  [[ "$batch" =~ ^[1-9][0-9]*$ ]] || {
    echo "Invalid evaluation batch size: $batch" >&2
    exit 2
  }
done
OFFLINE_ENV=(
  env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN
  EMBEDDING_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
)
candidate_args=()
LAST_SIONIC_SUMMARY=""
LAST_OFFICIAL_SUMMARY=""
LAST_COMPREHENSIVE_SUMMARY=""

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

retry_stage() {
  local name="$1" attempts="$2" attempt status=1
  shift 2
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    run_stage "$name-attempt-$attempt" "$@" && return 0
    status=$?
    (( attempt == attempts )) || sleep 15
  done
  return "$status"
}

run_sionic_with_fallback() {
  local label="$1" model="$2" revision="$3" cache="$4"
  local batch output
  for batch in "${EVAL_BATCHES[@]}"; do
    output="$SIONIC_OUT/b$batch"
    if run_stage "sionic9-$label-b$batch" \
      "${OFFLINE_ENV[@]}" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_sionic9.py" \
      --model "$model" --revision "$revision" --batch-size "$batch" --max-length 8192 \
      --attn-implementation flash_attention_2 --output-dir "$output" \
      --embedding-cache-dir "$cache"; then
      LAST_SIONIC_SUMMARY="$output/${model//\//__}/summary.json"
      return 0
    fi
  done
  return 1
}

run_official_with_fallback() {
  local label="$1" model="$2" revision="$3" cache="$4"
  local batch output
  for batch in "${EVAL_BATCHES[@]}"; do
    output="$OFFICIAL_OUT/b$batch"
    if run_stage "official-korean-$label-b$batch" \
      "${OFFLINE_ENV[@]}" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_mteb_korean_v1.py" \
      --model "$model" --revision "$revision" --max-length 8192 \
      --qwen3-instruction-loader --batch-size "$batch" \
      --attn-implementation flash_attention_2 --output-dir "$output" \
      --embedding-cache-dir "$cache"; then
      LAST_OFFICIAL_SUMMARY="$output/${model//\//__}/$revision/summary.json"
      return 0
    fi
  done
  return 1
}

run_comprehensive_with_fallback() {
  local label="$1" model="$2" revision="$3" cache="$4"
  local batch output safe_name
  safe_name="${model##*/}"
  for batch in "${EVAL_BATCHES[@]}"; do
    output="$COMPREHENSIVE_OUT/b$batch"
    if run_stage "comprehensive-text-$label-b$batch" \
      "${OFFLINE_ENV[@]}" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_comprehensive_text_v1.py" \
      --model "$model" --revision "$revision" --max-length 8192 \
      --qwen3-instruction-loader --batch-size "$batch" \
      --attn-implementation flash_attention_2 --output-dir "$output" \
      --embedding-cache-dir "$cache"; then
      LAST_COMPREHENSIVE_SUMMARY="$output/$safe_name/$revision/summary.json"
      return 0
    fi
  done
  return 1
}

run_clean_with_fallback() {
  local label="$1" model="$2" revision="$3"
  local batch
  for batch in "${EVAL_BATCHES[@]}"; do
    if run_stage "clean-legal-$label-b$batch" \
      "${OFFLINE_ENV[@]}" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_legal_source_holdout.py" \
      --model "$model" --revision "$revision" --batch-size "$batch" \
      --max-length 8192 --attn-implementation flash_attention_2 \
      --output-dir "$CLEAN_OUT" \
      --embedding-cache-dir "$ROOT/outputs/embedding-cache/legal-source-heldout"; then
      return 0
    fi
  done
  return 1
}

run_robustness_with_fallback() {
  local label="$1" model="$2" revision="$3"
  local batch
  for batch in "${EVAL_BATCHES[@]}"; do
    if run_stage "robustness-$label-b$batch" \
      "${OFFLINE_ENV[@]}" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_conversational_noise_robustness.py" \
      --model "$model" --revision "$revision" --batch-size "$batch" \
      --max-length 8192 --attn-implementation flash_attention_2 \
      --output-dir "$ROBUST_OUT" \
      --embedding-cache-dir "$ROOT/outputs/embedding-cache/legal-source-heldout"; then
      return 0
    fi
  done
  return 1
}

resolve_training_manifest() {
  local model="$1" candidate
  local candidates=()
  case "$model" in
    *-soup-*)
      candidates+=("$ROOT/$model/soup_report.json")
      ;;
    *reranker-*)
      candidates+=("$ROOT/outputs/data/performance-v1/performance-1m/reranker-kd-pilot/train.reranker-quantile-kd15.manifest.json")
      ;;
    *sionic-combined-target*)
      candidates+=("$ROOT/outputs/data/sionic-combined-target-v1/multidomain.manifest.json")
      ;;
    *sionic-retrieval-family50-replay50*)
      candidates+=("$ROOT/outputs/data/performance-v1/sionic-retrieval-train-family-4146/faiss-current-r095-n7.retrieval_family50-replay50.manifest.json")
      ;;
    *sionic-squad50-replay50*)
      candidates+=("$ROOT/outputs/data/performance-v1/sionic-squad-train-60k/faiss-current-r095-n7.squad50-replay50.manifest.json")
      ;;
    *sionic-health50-replay50*)
      candidates+=("$ROOT/outputs/data/performance-v1/sionic-health-multilingual-100k/faiss-current-r095-n7.health50-replay50.manifest.json")
      ;;
    *sionic-autorag50-replay50*)
      candidates+=("$ROOT/outputs/data/performance-v1/sionic-autorag-domain-100k/faiss-current-r095-n7.autorag50-replay50.manifest.json")
      ;;
    *legal25-replay75*)
      candidates+=("$ROOT/outputs/data/legal-performance-v1/faiss-r095-n7.legal25-replay75.manifest.json")
      ;;
    *performance1m*)
      candidates+=(
        "$ROOT/outputs/data/performance-v1/performance-1m/faiss-current-r095-n7.homogeneous-b16.manifest.json"
        "$ROOT/outputs/data/performance-v1/performance-1m/homogeneous-b16.manifest.json"
        "$ROOT/outputs/data/performance-v1/performance-1m/manifest.json"
      )
      ;;
    *performance200k*)
      candidates+=("$ROOT/outputs/data/performance-v1/ablation-200k/homogeneous-b16.manifest.json")
      ;;
    *performance50k*)
      candidates+=("$ROOT/outputs/data/performance-v1/pilot-50k/homogeneous-b16.manifest.json")
      ;;
    *)
      candidates+=(
        "$ROOT/data/processed/ko_triplet_pilot_10k/train.hn-qwen3-r095-n4.jsonl.manifest.json"
        "$ROOT/data/processed/ko_triplet_pilot_10k/manifest.json"
      )
      ;;
  esac
  for candidate in "${candidates[@]}"; do
    if [[ -s "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
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
  if [[ -s "$run_dir/DISQUALIFIED.json" ]]; then
    echo "[$(timestamp)] skip disqualified candidate: $run_name"
    continue
  fi
  checkpoint="$("$UTILITY_PYTHON" "$ROOT/scripts/select_best_checkpoint.py" \
    "$run_dir" --print-path 2>/dev/null)" || continue
  [[ -n "$checkpoint" ]] || continue
  merge_base_args=()
  case "$run_name" in
    comsat-embed-ko-8b-performance200k-*)
      merge_base_args=(
        --base-model sionic-ai/comsat-embed-ko-8b-preview
        --base-revision a5cc22b651c1b2e51cdd8bf671774ae93584f0ab
      )
      ;;
  esac
  merged_rel="artifacts/models/${run_name}-best-merged"
  merged="$ROOT/$merged_rel"
  if [[ ! -s "$merged/merge_report.json" ]]; then
    run_stage "merge-$run_name" \
      "${OFFLINE_ENV[@]}" \
      "$UTILITY_PYTHON" "$ROOT/scripts/merge_embedding_adapter.py" \
      --adapter "$checkpoint" --output-dir "$merged" \
      "${merge_base_args[@]}" \
      --device cuda --dtype bfloat16 --local-files-only || continue
  fi
  weights_sha="$(jq -r '.model.weights_sha256' "$merged/merge_report.json")"
  revision="model-${weights_sha:0:12}"
  candidate_args+=(--candidate-model "$merged_rel")
  if run_clean_with_fallback "$run_name" "$merged_rel" "$revision"; then
    run_robustness_with_fallback "$run_name" "$merged_rel" "$revision" || true
  fi

  # Compare the internally best checkpoint with an FP32 arithmetic mean of up
  # to the latest five checkpoints from the same exact Trainer version.  Older
  # runs may only retain 2–3 checkpoints; the manifest records the actual set.
  # The averaged model is merely another clean-evaluation candidate and cannot
  # bypass the same near-tie/robustness gate as the single best checkpoint.
  average_adapter_rel="artifacts/adapters/${run_name}-last-available5-fp32-average"
  average_adapter="$ROOT/$average_adapter_rel"
  if [[ ! -s "$average_adapter/average_report.json" ]]; then
    embedding_require_storage_headroom "$ROOT" 500 1000000
    embedding_require_storage_headroom /tmp 50 100000
    run_stage "average-last-checkpoints-$run_name" \
      "${OFFLINE_ENV[@]}" \
      "$UTILITY_PYTHON" "$ROOT/scripts/average_lora_checkpoints.py" \
      --run-dir "$run_dir" --anchor-checkpoint "$checkpoint" \
      --output-dir "$average_adapter" --last-n 5 --minimum-checkpoints 2 || true
  fi
  [[ -s "$average_adapter/average_report.json" ]] || continue
  average_merged_rel="artifacts/models/${run_name}-last-available5-fp32-average-merged"
  average_merged="$ROOT/$average_merged_rel"
  if [[ ! -s "$average_merged/merge_report.json" ]]; then
    embedding_require_storage_headroom "$ROOT" 500 1000000
    embedding_require_storage_headroom /tmp 50 100000
    run_stage "merge-last-checkpoint-average-$run_name" \
      "${OFFLINE_ENV[@]}" \
      "$UTILITY_PYTHON" "$ROOT/scripts/merge_embedding_adapter.py" \
      --adapter "$average_adapter" --output-dir "$average_merged" \
      "${merge_base_args[@]}" \
      --device cuda --dtype bfloat16 --local-files-only || continue
  fi
  average_weights_sha="$(jq -r '.model.weights_sha256' "$average_merged/merge_report.json")"
  average_revision="model-${average_weights_sha:0:12}"
  candidate_args+=(--candidate-model "$average_merged_rel")
  if run_clean_with_fallback \
      "$run_name-last-available5-fp32-average" \
      "$average_merged_rel" "$average_revision"; then
    run_robustness_with_fallback \
      "$run_name-last-available5-fp32-average" \
      "$average_merged_rel" "$average_revision" || true
  fi
done

for run_name in "${FULL_RUNS[@]}"; do
  run_dir="$ROOT/outputs/$run_name"
  [[ -d "$run_dir" ]] || continue
  if [[ -s "$run_dir/DISQUALIFIED.json" ]]; then
    echo "[$(timestamp)] skip disqualified candidate: $run_name"
    continue
  fi
  checkpoint="$("$UTILITY_PYTHON" "$ROOT/scripts/select_best_checkpoint.py" \
    "$run_dir" --checkpoint-kind full --print-path 2>/dev/null)" || continue
  [[ -n "$checkpoint" ]] || continue
  packaged_rel="artifacts/models/${run_name}-best-full"
  packaged="$ROOT/$packaged_rel"
  training_contract="$run_dir/capacity_run_manifest.json"
  [[ -s "$training_contract" ]] || {
    echo "[$(timestamp)] skip full candidate without capacity contract: $run_name" >&2
    continue
  }
  base_model="$(jq -r '.base_model // empty' "$training_contract")"
  base_revision="$(jq -r '.base_revision // empty' "$training_contract")"
  [[ -n "$base_model" && -n "$base_revision" ]] || {
    echo "[$(timestamp)] invalid full candidate base contract: $run_name" >&2
    continue
  }
  if [[ ! -s "$packaged/full_tuning_report.json" ]]; then
    run_stage "package-$run_name" \
      "${OFFLINE_ENV[@]}" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/package_full_embedding_checkpoint.py" \
      --checkpoint "$checkpoint" --output-dir "$packaged" \
      --base-model "$base_model" --base-revision "$base_revision" \
      --training-contract "$training_contract" \
      --device cuda --dtype bfloat16 --attn-implementation flash_attention_2 || continue
  fi
  weights_sha="$(jq -r '.model.weights_sha256' "$packaged/full_tuning_report.json")"
  revision="model-${weights_sha:0:12}"
  candidate_args+=(--candidate-model "$packaged_rel")
  if run_clean_with_fallback "$run_name" "$packaged_rel" "$revision"; then
    run_robustness_with_fallback "$run_name" "$packaged_rel" "$revision" || true
  fi
done

for soup_name in "${SOUP_MODELS[@]}"; do
  soup_rel="artifacts/models/$soup_name"
  soup="$ROOT/$soup_rel"
  [[ -s "$soup/soup_report.json" ]] || continue
  weights_sha="$(jq -r '.model.weights_sha256 // empty' "$soup/soup_report.json")"
  [[ "$weights_sha" =~ ^[0-9a-f]{64}$ ]] || continue
  revision="model-${weights_sha:0:12}"
  candidate_args+=(--candidate-model "$soup_rel")
  if run_clean_with_fallback "$soup_name" "$soup_rel" "$revision"; then
    run_robustness_with_fallback "$soup_name" "$soup_rel" "$revision" || true
  fi
done

SELECTION="$LOG_DIR/clean-first-selection.json"
rm -f "$SELECTION"
if (( ${#candidate_args[@]} > 0 )); then
  run_stage "select-best-clean-near-tie-robustness" \
    "${OFFLINE_ENV[@]}" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/select_best_clean_model.py" \
    "$CLEAN_OUT" "$ROBUST_OUT" --workspace-root "$ROOT" \
    --output "$SELECTION" --disqualification-root "$ROOT/outputs" \
    "${candidate_args[@]}" || true
else
  echo "[$(timestamp)] no packaged candidates are eligible for clean selection"
fi

if [[ "$SELECTION_ONLY" == 1 ]]; then
  if [[ ! -s "$SELECTION" ]]; then
    echo "[$(timestamp)] selection-only run produced no clean selection" >&2
    exit 20
  fi
  echo "[$(timestamp)] selection-only run complete; public evaluation and publication skipped"
  exit 0
fi

best_model=""
local_revision=""
if [[ -s "$SELECTION" ]]; then
  best_model="$(jq -r '.best.model' "$SELECTION")"
  best_abs="$ROOT/$best_model"
  if [[ -s "$best_abs/merge_report.json" ]]; then
    weights_sha="$(jq -r '.model.weights_sha256' "$best_abs/merge_report.json")"
    local_revision="model-${weights_sha:0:12}"
  elif [[ -s "$best_abs/full_tuning_report.json" ]]; then
    weights_sha="$(jq -r '.model.weights_sha256' "$best_abs/full_tuning_report.json")"
    local_revision="model-${weights_sha:0:12}"
  else
    weights_sha="$(jq -r '.model.weights_sha256' "$best_abs/soup_report.json")"
    local_revision="model-${weights_sha:0:12}"
  fi
  clean_summary="$(jq -r '.best.clean_summary' "$SELECTION")"
  robustness_summary="$(jq -r '.best.robustness_summary' "$SELECTION")"
  sionic_summary=""
  official_summary=""
  comprehensive_summary=""
  if run_sionic_with_fallback "final-selected" "$best_model" "$local_revision" \
    "$ROOT/outputs/embedding-cache/sionic9-final-selected"; then
    sionic_summary="$LAST_SIONIC_SUMMARY"
  else
    echo "[$(timestamp)] final selected model Sionic9 evaluation failed"
  fi
  if run_official_with_fallback "v1-final-selected" "$best_model" "$local_revision" \
    "$ROOT/outputs/embedding-cache/official-final-selected"; then
    official_summary="$LAST_OFFICIAL_SUMMARY"
  else
    echo "[$(timestamp)] final selected model official evaluation failed"
  fi
  if run_comprehensive_with_fallback \
    "v1-final-selected" "$best_model" "$local_revision" \
    "$ROOT/outputs/embedding-cache/comprehensive-text-final-selected"; then
    comprehensive_summary="$LAST_COMPREHENSIVE_SUMMARY"
  else
    echo "[$(timestamp)] final selected model comprehensive text evaluation failed"
  fi
  training_manifest="$(resolve_training_manifest "$best_model" 2>/dev/null)" || training_manifest=""
  if [[ -s "$official_summary" && -s "$sionic_summary" \
      && -s "$comprehensive_summary" && -s "$training_manifest" ]]; then
    clean_args=()
    [[ -s "$clean_summary" ]] && clean_args+=(--clean-summary "$clean_summary")
    robustness_args=()
    [[ -s "$robustness_summary" ]] && \
      robustness_args+=(--robustness-summary "$robustness_summary")
    if [[ ! -f "$PUBLISH_HF_TOKEN_FILE" ]]; then
      echo "[$(timestamp)] no Hugging Face token file available; skip private publication"
    elif retry_stage "publish-best-private-candidate" 3 \
      "$UTILITY_PYTHON" "$ROOT/scripts/publish_best_embedding_model.py" \
      --model-dir "$best_abs" \
      --sionic-summary "$sionic_summary" \
      --official-summary "$official_summary" \
      --comprehensive-summary "$comprehensive_summary" \
      --training-manifest "$training_manifest" \
      "${clean_args[@]}" \
      "${robustness_args[@]}" \
      --repo-id LLM-OS-Models2/qwen3-embedding-8b-ko-performance-v1-private-candidate \
      --hf-token-file "$PUBLISH_HF_TOKEN_FILE" --upload; then
      run_stage "record-pilot-best-result" \
        "$ROOT/scripts/commit_campaign_result.sh" \
        --stage pilot-best --model "$best_model" \
        --repo-id LLM-OS-Models2/qwen3-embedding-8b-ko-performance-v1-private-candidate \
        --sionic-summary "$sionic_summary" --official-summary "$official_summary"
    fi
  fi
fi

# Add trusted baselines to the same disclosed Grade-I comparison after local
# selection.  These baseline results do not retroactively change the winner.
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
  run_clean_with_fallback "${model//\//__}" "$model" "$revision" || \
    echo "[$(timestamp)] clean legal evaluation failed: $model"
done

robustness_models=(
  "Qwen/Qwen3-Embedding-8B|1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
  "sionic-ai/comsat-embed-ko-8b-preview|a5cc22b651c1b2e51cdd8bf671774ae93584f0ab"
)
if [[ -n "$best_model" && -n "$local_revision" \
    && ! -s "$ROBUST_OUT/${best_model//\//__}/$local_revision/summary.json" ]]; then
  robustness_models+=("$best_model|$local_revision")
fi
for spec in "${robustness_models[@]}"; do
  model="${spec%%|*}"
  revision="${spec#*|}"
  run_robustness_with_fallback "${model//\//__}" "$model" "$revision" || \
    echo "[$(timestamp)] robustness evaluation failed: $model"
done
run_stage "record-clean-legal-results" "$ROOT/scripts/commit_clean_legal_results.sh" || true

echo "[$(timestamp)] post-training evaluation queue complete"
