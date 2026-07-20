#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
cd "$ROOT"

WAIT_PID="${WAIT_PID:-}"
NEMOTRON_REVISION="2b29550c4ab0646bb6bb47032dda54ea11f6dfe2"
MODEL="$ROOT/.cache/huggingface/hub/models--nvidia--Nemotron-3-Embed-8B-BF16/snapshots/$NEMOTRON_REVISION"
SOURCE_DIR="$ROOT/outputs/data/public-legal-source-training-v1"
SOURCE_TRAIN="$SOURCE_DIR/data/train.jsonl"
SOURCE_PROVENANCE="$SOURCE_DIR/metadata/provenance.jsonl"
SOURCE_MANIFEST="$SOURCE_DIR/metadata/manifest.json"
OUT="$SOURCE_DIR/mined-nemotron3"
MINED="$OUT/train.faiss-r095-n7.jsonl"
MINING_AUDIT="$OUT/train.faiss-r095-n7.audit.jsonl"
MINING_MANIFEST="$OUT/train.faiss-r095-n7.manifest.json"
MINED_PROVENANCE="$OUT/provenance.faiss-r095-n7.jsonl"
PROJECTION_MANIFEST="$OUT/provenance.faiss-r095-n7.manifest.json"
ORDERED="$OUT/train.homogeneous-b32.jsonl"
ORDERED_PROVENANCE="$OUT/provenance.homogeneous-b32.jsonl"
ORDERED_MANIFEST="$OUT/homogeneous-b32.manifest.json"
OVERLAP="$OUT/benchmark-overlap-audit.json"
FINAL_MANIFEST="$OUT/final-public-manifest.json"
PUBLICATION_REPORT="$OUT/publication-report.json"
DECISION="$ROOT/outputs/evaluation/nemotron3-base-decision.json"
PROBE_MARKER="$ROOT/outputs/nemotron3-post-decision-probe/probe-complete.json"
DATASET_REPO="LLM-OS-Models2/ko-legal-embedding-training-nemotron3-hn-v1"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/nemotron3-public-pipeline}"
FAISS_THREADS="${FAISS_THREADS:-32}"
QUERY_PROMPT=$'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'

mkdir -p "$OUT" "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/runner.log") 2>&1
timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
run_stage() {
  local name="$1"; shift
  echo "[$(timestamp)] START $name"
  "$@"
  echo "[$(timestamp)] END $name"
}

if [[ -n "$WAIT_PID" ]]; then
  [[ "$WAIT_PID" =~ ^[0-9]+$ ]] || { echo "WAIT_PID must be numeric" >&2; exit 2; }
  echo "[$(timestamp)] waiting for decision/probe pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi

[[ -s "$DECISION" ]] || { echo "base decision is unavailable" >&2; exit 3; }
action="$(jq -er '.decision' "$DECISION")"
case "$action" in
  adopt_nemotron3_raw_and_run_short_public_lora|short_public_nemotron3_lora_then_retest) ;;
  *)
    printf '%s\n' "$action" > "$LOG_DIR/pipeline-skipped.decision"
    echo "[$(timestamp)] public Nemotron pipeline skipped decision=$action"
    exit 0
    ;;
esac
[[ "$(jq -r '.status' "$PROBE_MARKER")" == pass ]] || {
  echo "Nemotron backward probe did not pass" >&2
  exit 4
}
for path in "$MODEL/config.json" "$SOURCE_TRAIN" "$SOURCE_PROVENANCE" "$SOURCE_MANIFEST"; do
  [[ -s "$path" ]] || { echo "missing required input: $path" >&2; exit 5; }
done

if [[ ! -s "$MINING_MANIFEST" \
    || "$(jq -r '.revision // empty' "$MINING_MANIFEST")" != "$NEMOTRON_REVISION" \
    || "$(jq -r '.selection_strategy // empty' "$MINING_MANIFEST")" != score_rank_quantiles ]]; then
  run_stage mine-public-legal-nemotron3 \
    env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN -u HUGGING_FACE_HUB_TOKEN \
      HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      HF_DATASETS_OFFLINE=1 PYTHONPATH="$ROOT/third_party/mteb" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/mine_faiss_hard_negatives.py" \
      --input "$SOURCE_TRAIN" --output "$MINED" \
      --audit-output "$MINING_AUDIT" --manifest-output "$MINING_MANIFEST" \
      --work-dir "$OUT/faiss-work" --keep-work-dir \
      --model "$MODEL" --revision "$NEMOTRON_REVISION" \
      --encode-batch-size 128 --max-seq-length 512 \
      --strip-stored-query-instruction \
      --query-prefix "$QUERY_PROMPT" --document-prefix "" \
      --candidate-pool-size 24 --search-k 256 --num-negatives 7 \
      --selection-strategy score_rank_quantiles --positive-relative-ratio .95 \
      --nlist 512 --nprobe 32 --training-points 50000 \
      --faiss-threads "$FAISS_THREADS" --assert-no-benchmark-data --allow-target-adapted
fi

if [[ ! -s "$PROJECTION_MANIFEST" ]]; then
  run_stage project-public-mined-provenance \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/project_mined_provenance.py" \
    --input-provenance "$SOURCE_PROVENANCE" --mining-audit "$MINING_AUDIT" \
    --mined-train "$MINED" \
    --output "$MINED_PROVENANCE" --manifest-output "$PROJECTION_MANIFEST"
fi

if [[ ! -s "$ORDERED_MANIFEST" ]]; then
  run_stage order-public-homogeneous-batches \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/build_homogeneous_batches.py" \
    --train "$MINED" --provenance "$MINED_PROVENANCE" \
    --output "$ORDERED" --provenance-output "$ORDERED_PROVENANCE" \
    --manifest-output "$ORDERED_MANIFEST" --batch-size 32 --seed 42 \
    --length-bucketed --benchmark-adaptation target-adapted-legal-public-source
fi

run_stage audit-final-public-curriculum \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/audit_training_benchmark_overlap.py" \
  --train "$ORDERED" --provenance "$ORDERED_PROVENANCE" \
  --blocklist-root "$ROOT/outputs/decontamination/benchmark_blocklist" \
  --output "$OVERLAP" --fail-on-any-text

run_stage finalize-public-curriculum \
  "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/finalize_public_training_manifest.py" \
  --train "$ORDERED" --provenance "$ORDERED_PROVENANCE" \
  --source-manifest "$SOURCE_MANIFEST" \
  --transform-manifest "$MINING_MANIFEST" \
  --transform-manifest "$PROJECTION_MANIFEST" \
  --transform-manifest "$ORDERED_MANIFEST" \
  --benchmark-overlap-audit "$OVERLAP" --output "$FINAL_MANIFEST" \
  --artifact-id public-legal-source-training-nemotron3-hn-v1

if [[ ! -s "$PUBLICATION_REPORT" \
    || "$(jq -r '.remote_payload_hashes_exact // false' "$PUBLICATION_REPORT")" != true ]]; then
  echo "[$(timestamp)] START publish-final-public-curriculum"
  publication_tmp="$PUBLICATION_REPORT.tmp"
  (
    embedding_load_hf_credential "$ROOT/.env"
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/publish_derived_training_dataset.py" \
      --train "$ORDERED" --provenance "$ORDERED_PROVENANCE" \
      --manifest "$FINAL_MANIFEST" --mining-manifest "$MINING_MANIFEST" \
      --mining-audit "$MINING_AUDIT" --benchmark-overlap-audit "$OVERLAP" \
      --repo-id "$DATASET_REPO" --title "Korean Public Legal Embedding Training Nemotron-3 HN v1" \
      --source-dataset LLM-OS-Models2/ko-legal-embedding-training-v1 \
      --public --upload
  ) > "$publication_tmp"
  [[ "$(jq -r '.remote_payload_hashes_exact' "$publication_tmp")" == true ]]
  mv "$publication_tmp" "$PUBLICATION_REPORT"
  echo "[$(timestamp)] END publish-final-public-curriculum"
fi

run_stage train-public-nemotron3 \
  env DATA_DIR="$OUT" "$ROOT/scripts/run_nemotron3_public_lora_training.sh"

jq -n \
  --arg completed_at "$(date --iso-8601=seconds)" \
  --arg decision "$action" \
  --arg dataset_repo "$DATASET_REPO" \
  '{status:"complete",completed_at:$completed_at,decision:$decision,dataset_repo:$dataset_repo}' \
  > "$LOG_DIR/pipeline-complete.json"
echo "[$(timestamp)] public Nemotron mining, publication, and training pipeline complete"
