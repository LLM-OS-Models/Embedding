#!/usr/bin/env bash
set -euo pipefail

# Materialize fixed, pre-registered full-weight soups after every specialist
# run has finished.  No public benchmark score or adaptive coefficient enters
# this stage; all variants proceed to the same final clean-first selector.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
embedding_resolve_train_runtime
PYTHON="$EMBEDDING_TRAIN_PYTHON"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-$ROOT/outputs/model-soup-20260717-frontier}"
GENERAL_SELECTION="${GENERAL_SELECTION:-$ROOT/outputs/reranker-kd-20260717-frontier/clean-first-selection.json}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1

unset HF_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
run_stage() {
  local name="$1"; shift
  echo "[$(timestamp)] START $name"
  "$@"
  local status=$?
  echo "[$(timestamp)] END $name status=$status"
  return "$status"
}

resolve_merged_model() {
  local name candidate
  for name in "$@"; do
    candidate="$ROOT/artifacts/models/${name}-best-merged"
    if [[ -s "$candidate/merge_report.json" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

resolve_local_parent_model() {
  local model="$1" raw candidate evidence_count=0
  [[ -s "$model/merge_report.json" ]] || return 1
  raw="$(jq -r '.base_model // empty' "$model/merge_report.json")"
  [[ -n "$raw" && "$raw" == /* ]] || return 1
  candidate="$(realpath -e "$raw" 2>/dev/null)" || return 1
  case "$candidate" in
    "$ROOT"/artifacts/models/*) ;;
    *) return 1 ;;
  esac
  [[ -s "$candidate/merge_report.json" ]] && evidence_count=$((evidence_count + 1))
  [[ -s "$candidate/full_tuning_report.json" ]] && evidence_count=$((evidence_count + 1))
  [[ -s "$candidate/soup_report.json" ]] && evidence_count=$((evidence_count + 1))
  (( evidence_count == 1 )) || return 1
  printf '%s\n' "$candidate"
}

if [[ ! -s "$GENERAL_SELECTION" ]]; then
  echo "[$(timestamp)] general clean selection is unavailable" >&2
  exit 2
fi
general_rel="$(jq -r '.best.model // empty' "$GENERAL_SELECTION")"
general="$ROOT/$general_rel"
if [[ -z "$general_rel" || ! -s "$general/merge_report.json" ]]; then
  echo "[$(timestamp)] general winner is not a safe-merged model" >&2
  exit 2
fi

retrieval="$(resolve_merged_model \
  qwen3-embedding-8b-ko-sionic-retrieval-family50-replay50-lora-r64 \
  qwen3-embedding-8b-ko-sionic-retrieval-family50-replay50-lora-r64-b2)" || retrieval=""
squad="$(resolve_merged_model \
  qwen3-embedding-8b-ko-sionic-squad50-replay50-lora-r64 \
  qwen3-embedding-8b-ko-sionic-squad50-replay50-lora-r64-b4)" || squad=""
health="$(resolve_merged_model \
  qwen3-embedding-8b-ko-sionic-health50-replay50-lora-r64 \
  qwen3-embedding-8b-ko-sionic-health50-replay50-lora-r64-b4)" || health=""
autorag="$(resolve_merged_model \
  qwen3-embedding-8b-ko-sionic-autorag50-replay50-lora-r64 \
  qwen3-embedding-8b-ko-sionic-autorag50-replay50-lora-r64-b4)" || autorag=""
legal="$(resolve_merged_model \
  qwen3-embedding-8b-ko-legal25-replay75-lora-r64 \
  qwen3-embedding-8b-ko-legal25-replay75-lora-r64-b4)" || legal=""
combined="$(resolve_merged_model \
  qwen3-embedding-8b-ko-sionic-combined-target-lora-r64 \
  qwen3-embedding-8b-ko-sionic-combined-target-lora-r64-b4)" || combined=""

build_soup() {
  local label="$1"; shift
  local output="$ROOT/artifacts/models/$label"
  if [[ -s "$output/soup_report.json" ]]; then
    echo "[$(timestamp)] reuse completed soup: $label"
    return 0
  fi
  embedding_require_storage_headroom "$ROOT" 500 1000000
  embedding_require_storage_headroom /tmp 50 100000
  run_stage "build-$label" \
    "$PYTHON" "$ROOT/scripts/merge_full_model_soup.py" \
    "$@" --output-dir "$output" --output-dtype bfloat16 \
    --torch-threads "${SOUP_TORCH_THREADS:-4}"
}

general_parent="$(resolve_local_parent_model "$general")" || general_parent=""
if [[ -n "$general_parent" ]]; then
  build_soup qwen3-embedding-8b-ko-soup-general75-parent25 \
    --model "$general" --weight .75 \
    --model "$general_parent" --weight .25
  build_soup qwen3-embedding-8b-ko-soup-general50-parent50 \
    --model "$general" --weight .5 \
    --model "$general_parent" --weight .5
else
  echo "[$(timestamp)] general winner has no eligible local parent; parent interpolation skipped" >&2
fi

if [[ -n "$combined" ]]; then
  build_soup qwen3-embedding-8b-ko-soup-general50-combined50 \
    --model "$general" --weight .5 \
    --model "$combined" --weight .5
  build_soup qwen3-embedding-8b-ko-soup-general25-combined75 \
    --model "$general" --weight .25 \
    --model "$combined" --weight .75
else
  echo "[$(timestamp)] combined model unavailable; combined soups skipped" >&2
fi

specialists=("$retrieval" "$squad" "$health" "$autorag" "$legal")
all_specialists=1
for specialist in "${specialists[@]}"; do
  [[ -n "$specialist" ]] || all_specialists=0
done
if (( all_specialists == 1 )); then
  build_soup qwen3-embedding-8b-ko-soup-general50-specialists10x5 \
    --model "$general" --weight .5 \
    --model "$retrieval" --weight .1 \
    --model "$squad" --weight .1 \
    --model "$health" --weight .1 \
    --model "$autorag" --weight .1 \
    --model "$legal" --weight .1
  if [[ -n "$combined" ]]; then
    build_soup qwen3-embedding-8b-ko-soup-general25-combined25-specialists10x5 \
      --model "$general" --weight .25 \
      --model "$combined" --weight .25 \
      --model "$retrieval" --weight .1 \
      --model "$squad" --weight .1 \
      --model "$health" --weight .1 \
      --model "$autorag" --weight .1 \
      --model "$legal" --weight .1
    build_soup qwen3-embedding-8b-ko-soup-combined50-specialists10x5 \
      --model "$combined" --weight .5 \
      --model "$retrieval" --weight .1 \
      --model "$squad" --weight .1 \
      --model "$health" --weight .1 \
      --model "$autorag" --weight .1 \
      --model "$legal" --weight .1
  fi
else
  echo "[$(timestamp)] one or more specialist models unavailable; balanced soups skipped" >&2
fi

echo "[$(timestamp)] fixed model soup queue complete"
