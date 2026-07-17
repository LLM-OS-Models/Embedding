#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
embedding_resolve_train_runtime
cd "$ROOT"
embedding_load_github_credential "$ROOT/.env" || {
  echo "GITHUB token is unavailable" >&2
  exit 2
}

"$EMBEDDING_TRAIN_PYTHON" "$ROOT/scripts/summarize_legal_holdout_results.py" \
  "$ROOT/outputs/evaluation/legal-source-heldout" \
  --output "$ROOT/reports/legal-source-heldout-results.json" \
  --robustness-root "$ROOT/outputs/evaluation/conversational-noise-robustness" \
  --readme "$ROOT/README.md"
git add README.md reports/legal-source-heldout-results.json
if ! git diff --cached --quiet; then
  git commit -m "Record clean legal holdout results"
fi
git -c 'credential.helper=!f() { if [ "$1" = get ]; then echo username=x-access-token; echo "password=$GITHUB"; fi; }; f' push origin main
