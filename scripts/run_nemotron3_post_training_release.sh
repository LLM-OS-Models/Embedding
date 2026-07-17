#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WAIT_PID="${WAIT_PID:-}"
BASE_REVISION="2b29550c4ab0646bb6bb47032dda54ea11f6dfe2"
BASE="$ROOT/.cache/huggingface/hub/models--nvidia--Nemotron-3-Embed-8B-BF16/snapshots/$BASE_REVISION"
PIPELINE_DIR="$ROOT/outputs/nemotron3-public-pipeline"
RUN_DIR="$ROOT/outputs/nemotron3-ko-public-lora-r16"
TRAIN_MANIFEST="$ROOT/outputs/data/public-legal-source-training-v1/mined-nemotron3/final-public-manifest.json"
SELECTION="$RUN_DIR/checkpoint-selection.json"
MERGED="$ROOT/artifacts/models/nemotron3-embed-8b-ko-public-r16"
DECISION="$ROOT/outputs/evaluation/nemotron3-base-decision.json"
EVAL_ROOT="$ROOT/outputs/evaluation/nemotron3-final-release"
CACHE_ROOT="$ROOT/outputs/embedding-cache/nemotron3-final-release"
LEGAL_OUT="$EVAL_ROOT/legal"
MULTI_OUT="$EVAL_ROOT/multidomain"
SIONIC_OUT="$EVAL_ROOT/sionic9"
OFFICIAL_OUT="$EVAL_ROOT/official-korean-v1"
COMPREHENSIVE_OUT="$EVAL_ROOT/comprehensive-text-v1"
ROBUST_OUT="$EVAL_ROOT/robustness"
FINAL_GATE="$EVAL_ROOT/final-gate.json"
APPROVAL="$EVAL_ROOT/public-release-approval.json"
PUBLICATION_REPORT="$EVAL_ROOT/final-publication-report.json"
REPO_ID="${REPO_ID:-LLM-OS-Models2/nemotron3-embed-8b-ko-public-v1}"
LOG_DIR="$ROOT/outputs/nemotron3-post-training-release"

mkdir -p "$LOG_DIR" "$EVAL_ROOT" "$CACHE_ROOT"
exec > >(tee -a "$LOG_DIR/runner.log") 2>&1
timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
run_stage() {
  local name="$1"; shift
  echo "[$(timestamp)] START $name"
  "$@"
  echo "[$(timestamp)] END $name"
}
OFFLINE_ENV=(
  env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN -u HUGGING_FACE_HUB_TOKEN
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  PYTHONPATH="$ROOT/third_party/mteb"
)

if [[ -n "$WAIT_PID" ]]; then
  [[ "$WAIT_PID" =~ ^[0-9]+$ ]] || { echo "WAIT_PID must be numeric" >&2; exit 2; }
  echo "[$(timestamp)] waiting for public training pipeline pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi
if [[ ! -s "$PIPELINE_DIR/pipeline-complete.json" ]]; then
  if [[ -s "$PIPELINE_DIR/pipeline-skipped.decision" ]]; then
    echo "[$(timestamp)] Nemotron release skipped decision=$(cat "$PIPELINE_DIR/pipeline-skipped.decision")"
    exit 0
  fi
  echo "Public Nemotron training pipeline did not complete" >&2
  exit 3
fi

run_stage select-public-checkpoint \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/select_nemotron3_public_checkpoint.py" \
  --run-dir "$RUN_DIR" --training-manifest "$TRAIN_MANIFEST" --output "$SELECTION"
ADAPTER="$(jq -er '.selected.checkpoint' "$SELECTION")"

merge_args=(
  "$ROOT/.venv-train-fa2/bin/python" "$ROOT/scripts/merge_nemotron3_adapter.py"
  --adapter "$ADAPTER" --selection "$SELECTION" --training-manifest "$TRAIN_MANIFEST"
  --base-model "$BASE" --base-revision "$BASE_REVISION" --output-dir "$MERGED"
  --device cuda --max-length 512
)
if [[ -s "$MERGED/merge_report.json" ]]; then
  run_stage validate-existing-merge "${merge_args[@]}" --validate-existing
else
  run_stage merge-selected-adapter "${merge_args[@]}"
fi
WEIGHTS_SHA="$(jq -er '.model.weights_sha256' "$MERGED/merge_report.json")"
REVISION="model-${WEIGHTS_SHA:0:12}"
SAFE_MODEL="${MERGED//\//__}"

run_stage evaluate-clean-legal \
  "${OFFLINE_ENV[@]}" "$ROOT/.venv-mteb/bin/python" \
  "$ROOT/scripts/evaluate_legal_source_holdout.py" \
  --model "$MERGED" --revision "$REVISION" --batch-size 64 --max-length 8192 \
  --attn-implementation flash_attention_2 --output-dir "$LEGAL_OUT" \
  --embedding-cache-dir "$CACHE_ROOT/legal"
LEGAL_SUMMARY="$LEGAL_OUT/$SAFE_MODEL/$REVISION/summary.json"

run_stage evaluate-multidomain \
  "${OFFLINE_ENV[@]}" "$ROOT/.venv-mteb/bin/python" \
  "$ROOT/scripts/evaluate_multidomain_selection.py" \
  --model "$MERGED" --revision "$REVISION" --batch-size 64 --max-length 8192 \
  --attn-implementation flash_attention_2 \
  --dataset-dir "$ROOT/outputs/evaluation/multidomain-selection-heldout-v1" \
  --output-dir "$MULTI_OUT" --embedding-cache-dir "$CACHE_ROOT/multidomain"
MULTI_SUMMARY="$MULTI_OUT/$SAFE_MODEL/$REVISION/summary.json"

run_stage evaluate-sionic9 \
  "${OFFLINE_ENV[@]}" "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_sionic9.py" \
  --model "$MERGED" --revision "$REVISION" --batch-size 64 --max-length 8192 \
  --attn-implementation flash_attention_2 --output-dir "$SIONIC_OUT" \
  --embedding-cache-dir "$CACHE_ROOT/sionic9"
SIONIC_SUMMARY="$SIONIC_OUT/$SAFE_MODEL/summary.json"

run_stage gate-final-candidate \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/gate_nemotron3_final_candidate.py" \
  --base-decision "$DECISION" --legal-summary "$LEGAL_SUMMARY" \
  --multidomain-summary "$MULTI_SUMMARY" --sionic-summary "$SIONIC_SUMMARY" \
  --model-dir "$MERGED" --output "$FINAL_GATE"

run_stage evaluate-official-korean-v1 \
  "${OFFLINE_ENV[@]}" "$ROOT/.venv-mteb/bin/python" \
  "$ROOT/scripts/evaluate_mteb_korean_v1.py" \
  --model "$MERGED" --revision "$REVISION" --batch-size 64 --max-length 8192 \
  --qwen3-instruction-loader --attn-implementation flash_attention_2 \
  --output-dir "$OFFICIAL_OUT" --embedding-cache-dir "$CACHE_ROOT/official"
OFFICIAL_SUMMARY="$OFFICIAL_OUT/$SAFE_MODEL/$REVISION/summary.json"

run_stage evaluate-comprehensive-text-v1 \
  "${OFFLINE_ENV[@]}" "$ROOT/.venv-mteb/bin/python" \
  "$ROOT/scripts/evaluate_comprehensive_text_v1.py" \
  --model "$MERGED" --revision "$REVISION" --batch-size 64 --max-length 8192 \
  --qwen3-instruction-loader --attn-implementation flash_attention_2 \
  --output-dir "$COMPREHENSIVE_OUT" --embedding-cache-dir "$CACHE_ROOT/comprehensive"
COMPREHENSIVE_SUMMARY="$COMPREHENSIVE_OUT/$(basename "$MERGED")/$REVISION/summary.json"

run_stage evaluate-conversational-robustness \
  "${OFFLINE_ENV[@]}" "$ROOT/.venv-mteb/bin/python" \
  "$ROOT/scripts/evaluate_conversational_noise_robustness.py" \
  --model "$MERGED" --revision "$REVISION" --batch-size 64 --max-length 8192 \
  --attn-implementation flash_attention_2 --output-dir "$ROBUST_OUT" \
  --embedding-cache-dir "$CACHE_ROOT/robustness"
ROBUST_SUMMARY="$ROBUST_OUT/$SAFE_MODEL/$REVISION/summary.json"

run_stage approve-exact-public-release \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/approve_nemotron3_public_release.py" \
  --model-dir "$MERGED" --repo-id "$REPO_ID" --training-manifest "$TRAIN_MANIFEST" \
  --final-gate "$FINAL_GATE" --sionic-summary "$SIONIC_SUMMARY" \
  --official-summary "$OFFICIAL_SUMMARY" --comprehensive-summary "$COMPREHENSIVE_SUMMARY" \
  --clean-summary "$LEGAL_SUMMARY" --robustness-summary "$ROBUST_SUMMARY" \
  --output "$APPROVAL"

run_stage publish-final-public-model \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/publish_best_embedding_model.py" \
  --model-dir "$MERGED" --sionic-summary "$SIONIC_SUMMARY" \
  --official-summary "$OFFICIAL_SUMMARY" --comprehensive-summary "$COMPREHENSIVE_SUMMARY" \
  --training-manifest "$TRAIN_MANIFEST" --clean-summary "$LEGAL_SUMMARY" \
  --robustness-summary "$ROBUST_SUMMARY" --multidomain-summary "$MULTI_SUMMARY" \
  --release-approval "$APPROVAL" --repo-id "$REPO_ID" --hf-token-file "$ROOT/.env" \
  --report-output "$PUBLICATION_REPORT" --upload --public

[[ "$(jq -r '.visibility' "$PUBLICATION_REPORT")" == public ]]
[[ "$(jq -r '.remote_manifest_exact' "$PUBLICATION_REPORT")" == true ]]
[[ "$(jq -r '.remote_file_set_exact' "$PUBLICATION_REPORT")" == true ]]
jq -n --arg completed_at "$(date --iso-8601=seconds)" --arg repo_id "$REPO_ID" \
  --arg weights_sha256 "$WEIGHTS_SHA" \
  '{status:"complete",completed_at:$completed_at,repo_id:$repo_id,weights_sha256:$weights_sha256}' \
  > "$LOG_DIR/release-complete.json"
echo "[$(timestamp)] final public Nemotron release complete repo=$REPO_ID"
