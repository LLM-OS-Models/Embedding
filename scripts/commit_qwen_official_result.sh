#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
embedding_resolve_train_runtime
cd "$ROOT"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
[[ -n "${GITHUB:-}" ]] || { echo "GITHUB token is unavailable" >&2; exit 2; }

REVISION=4e423935c619ae4df87b646a3ce949610c66241c
SUMMARY="$ROOT/outputs/evaluation/mteb-korean-v1-qwen-base/Qwen__Qwen3-Embedding-8B/$REVISION/summary.json"
COMPARISON="$ROOT/outputs/evaluation/mteb-korean-v1-qwen-base/qwen-live-comparison.json"
"$EMBEDDING_TRAIN_PYTHON" "$ROOT/scripts/update_qwen_official_readme.py" \
  --summary "$SUMMARY" --comparison "$COMPARISON" --readme "$ROOT/README.md" \
  --output "$ROOT/reports/qwen-mteb-korean-v1-local.json"
git add README.md reports/qwen-mteb-korean-v1-local.json
if ! git diff --cached --quiet; then
  git commit -m "Record Qwen official Korean v1 local result"
fi
git -c 'credential.helper=!f() { if [ "$1" = get ]; then echo username=x-access-token; echo "password=$GITHUB"; fi; }; f' push origin main
