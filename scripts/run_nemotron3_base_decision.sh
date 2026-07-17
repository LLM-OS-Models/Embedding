#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NEMOTRON_REVISION="2b29550c4ab0646bb6bb47032dda54ea11f6dfe2"
QWEN_REVISION="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
COMSAT_REVISION="a5cc22b651c1b2e51cdd8bf671774ae93584f0ab"
NEMOTRON_PATH="$ROOT/.cache/huggingface/hub/models--nvidia--Nemotron-3-Embed-8B-BF16/snapshots/$NEMOTRON_REVISION"

[[ -s "$NEMOTRON_PATH/model.safetensors.index.json" ]] || {
  echo "Pinned Nemotron-3 snapshot is incomplete" >&2
  exit 2
}

OFFLINE_ENV=(
  env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  PYTHONPATH="$ROOT/third_party/mteb"
)
PYTHON="$ROOT/.venv-mteb/bin/python"

"${OFFLINE_ENV[@]}" "$PYTHON" "$ROOT/scripts/evaluate_sionic9.py" \
  --model "$NEMOTRON_PATH" --revision "$NEMOTRON_REVISION" \
  --batch-size 64 --max-length 8192 \
  --attn-implementation flash_attention_2 \
  --output-dir "$ROOT/outputs/evaluation/sionic9-nemotron3-full-fixed-prompt" \
  --embedding-cache-dir "$ROOT/outputs/embedding-cache/sionic9-nemotron3/full-fixed-prompt"

models=(
  "$NEMOTRON_PATH|$NEMOTRON_REVISION|nemotron3"
  "Qwen/Qwen3-Embedding-8B|$QWEN_REVISION|qwen3"
  "sionic-ai/comsat-embed-ko-8b-preview|$COMSAT_REVISION|comsat"
)

for spec in "${models[@]}"; do
  model="${spec%%|*}"
  remainder="${spec#*|}"
  revision="${remainder%%|*}"
  label="${remainder##*|}"

  "${OFFLINE_ENV[@]}" "$PYTHON" "$ROOT/scripts/evaluate_legal_source_holdout.py" \
    --model "$model" --revision "$revision" --batch-size 64 --max-length 8192 \
    --attn-implementation flash_attention_2 \
    --output-dir "$ROOT/outputs/evaluation/legal-source-heldout-base-decision" \
    --embedding-cache-dir "$ROOT/outputs/embedding-cache/legal-source-heldout-base-decision/$label"

  "${OFFLINE_ENV[@]}" "$PYTHON" "$ROOT/scripts/evaluate_multidomain_selection.py" \
    --model "$model" --revision "$revision" --batch-size 64 --max-length 8192 \
    --attn-implementation flash_attention_2 \
    --dataset-dir "$ROOT/outputs/evaluation/multidomain-selection-heldout-v1" \
    --output-dir "$ROOT/outputs/evaluation/multidomain-selection-base-decision" \
    --embedding-cache-dir "$ROOT/outputs/embedding-cache/multidomain-selection-base-decision/$label"
done

echo "Nemotron/Qwen/Comsat base-decision evaluation complete"
