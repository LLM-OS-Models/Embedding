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

"$EMBEDDING_TRAIN_PYTHON" "$ROOT/scripts/record_campaign_result.py" "$@"
git add README.md reports/campaign-results.json reports/evidence
if ! git diff --cached --quiet; then
  stage="unknown"
  for ((index=1; index<=$#; index++)); do
    if [[ "${!index}" == --stage ]]; then
      next=$((index + 1))
      stage="${!next}"
      break
    fi
  done
  git commit -m "Record completed embedding campaign result: $stage"
fi
git -c 'credential.helper=!f() { if [ "$1" = get ]; then echo username=x-access-token; echo "password=$GITHUB"; fi; }; f' push origin main
