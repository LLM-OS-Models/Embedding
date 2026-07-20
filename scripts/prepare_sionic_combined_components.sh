#!/usr/bin/env bash
set -uo pipefail

# Prepare only the mined/ordered component data that
# run_sionic_combined_adaptation_queue.sh consumes.  The per-target queues also
# train a specialist model each; the combined 400K candidate does not need those
# specialists, so this runs their data stages alone with identical parameters.
#
# Each component reuses the exact mining contract of its own queue: current
# student as miner, candidate pool 24, HN 7, score-rank quantiles,
# positive-relative .95, then length-bucketed homogeneous batch-16 ordering.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
cd "$ROOT"
UTILITY_PYTHON="$ROOT/.venv-train-fa2/bin/python"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/sionic-combined-components-20260720}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/prepare.log") 2>&1

GENERAL_SELECTION="${GENERAL_SELECTION:-$ROOT/outputs/post-capacity-eval-20260717-frontier/clean-first-selection.json}"
export GENERAL_SELECTION
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$ROOT/scripts:$ROOT/third_party/mteb${PYTHONPATH:+:$PYTHONPATH}"
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# EFFECTIVE_CPU_COUNT is exported by common_runtime.sh
FAISS_THREADS="${FAISS_THREADS:-$EFFECTIVE_CPU_COUNT}"

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
run_stage() {
  local name="$1"; shift
  echo "[$(timestamp)] START $name"
  "$@"
  local status=$?
  echo "[$(timestamp)] END $name status=$status"
  return "$status"
}

embedding_resolve_general_base || {
  echo "clean-selected student model is unavailable" >&2
  exit 4
}
MINING_MODEL="$EMBEDDING_GENERAL_BASE"
[[ -s "$MINING_MODEL/merge_report.json" ]] || {
  echo "student model has no merge evidence: $MINING_MODEL" >&2
  exit 4
}
echo "[$(timestamp)] current student for mining: $MINING_MODEL"

# name | data dir | adaptation label | nlist | training points | max length | encode batch
COMPONENTS=(
  "squad|outputs/data/performance-v1/sionic-squad-train-60k|target-adapted-squad-train-family|128|9606|512|128"
  "health|outputs/data/performance-v1/sionic-health-multilingual-100k|target-adapted-health-domain|512|50000|512|128"
  "autorag|outputs/data/performance-v1/sionic-autorag-domain-100k|target-adapted-autorag-domain|512|50000|512|128"
  "retrieval_family|outputs/data/performance-v1/sionic-retrieval-train-family-4146|target-adapted-retrieval-family|128|4000|2048|32"
)

status=0
for spec in "${COMPONENTS[@]}"; do
  IFS='|' read -r kind rel label nlist points max_length encode_batch <<< "$spec"
  data_dir="$ROOT/$rel"
  bootstrap="$data_dir/train.jsonl"
  provenance="$data_dir/provenance.jsonl"
  mined="$data_dir/train.faiss-current-r095-n7.jsonl"
  audit="$data_dir/train.faiss-current-r095-n7.audit.jsonl"
  mining_manifest="$data_dir/train.faiss-current-r095-n7.manifest.json"
  mined_provenance="$data_dir/provenance.faiss-current-r095-n7.jsonl"
  ordered="$data_dir/train.faiss-current-r095-n7.homogeneous-b16.jsonl"
  ordered_provenance="$data_dir/provenance.faiss-current-r095-n7.homogeneous-b16.jsonl"
  ordered_manifest="$data_dir/faiss-current-r095-n7.homogeneous-b16.manifest.json"

  if [[ ! -s "$bootstrap" || ! -s "$provenance" ]]; then
    echo "[$(timestamp)] missing source data for $kind: $data_dir" >&2
    status=1
    continue
  fi

  if [[ ! -s "$mining_manifest" ]]; then
    rm -f "$mined" "$audit" "$mining_manifest" "$mined_provenance" \
      "$ordered" "$ordered_provenance" "$ordered_manifest"
    run_stage "mine-sionic-${kind}-current-student" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/mine_faiss_hard_negatives.py" \
      --input "$bootstrap" --output "$mined" --audit-output "$audit" \
      --manifest-output "$mining_manifest" \
      --work-dir "$data_dir/faiss-work-current-student" --keep-work-dir \
      --model "$MINING_MODEL" --revision "" \
      --encode-batch-size "$encode_batch" --max-seq-length "$max_length" \
      --candidate-pool-size 24 --search-k 256 \
      --num-negatives 7 --selection-strategy score_rank_quantiles \
      --positive-relative-ratio .95 --nlist "$nlist" --nprobe 32 \
      --training-points "$points" --faiss-threads "$FAISS_THREADS" \
      --allow-target-adapted || { status=1; continue; }
  fi

  if [[ ! -s "$mined_provenance" ]]; then
    run_stage "project-sionic-${kind}-mined-provenance" \
      "$UTILITY_PYTHON" "$ROOT/scripts/project_mined_provenance.py" \
      --input-provenance "$provenance" --mining-audit "$audit" \
      --mined-train "$mined" --output "$mined_provenance" \
      --manifest-output "$data_dir/provenance.faiss-current-r095-n7.manifest.json" \
      || { status=1; continue; }
  fi

  if [[ ! -s "$ordered_manifest" ]]; then
    run_stage "order-sionic-${kind}-homogeneous" \
      "$UTILITY_PYTHON" "$ROOT/scripts/build_homogeneous_batches.py" \
      --train "$mined" --provenance "$mined_provenance" \
      --output "$ordered" --provenance-output "$ordered_provenance" \
      --manifest-output "$ordered_manifest" --batch-size 16 --seed 42 \
      --length-bucketed --benchmark-adaptation "${label}-source" \
      || { status=1; continue; }
  fi
  echo "[$(timestamp)] component ready: $kind rows=$(jq -r '.output_rows' "$ordered_manifest")"
done

echo "[$(timestamp)] component preparation finished status=$status"
exit "$status"
