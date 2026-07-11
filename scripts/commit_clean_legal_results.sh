#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
[[ -n "${GITHUB:-}" ]] || { echo "GITHUB token is unavailable" >&2; exit 2; }

"$ROOT/.venv-train/bin/python" "$ROOT/scripts/summarize_legal_holdout_results.py" \
  "$ROOT/outputs/evaluation/legal-source-heldout" \
  --output "$ROOT/reports/legal-source-heldout-results.json" \
  --robustness-root "$ROOT/outputs/evaluation/conversational-noise-robustness" \
  --readme "$ROOT/README.md"
git add README.md reports/legal-source-heldout-results.json
if ! git diff --cached --quiet; then
  git commit -m "Record clean legal holdout results"
fi
git -c 'credential.helper=!f() { if [ "$1" = get ]; then echo username=x-access-token; echo "password=$GITHUB"; fi; }; f' push origin main
