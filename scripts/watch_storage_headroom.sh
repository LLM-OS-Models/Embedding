#!/usr/bin/env bash
set -euo pipefail

# Protect the shared machine during a multi-day campaign. Only explicitly
# supplied process-group leaders are eligible for termination, and their live
# PGID identity is revalidated immediately before every signal.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"
cd "$ROOT"

WATCH_PGIDS="${WATCH_PGIDS:?WATCH_PGIDS must list explicit process-group leaders}"
POLL_SECONDS="${POLL_SECONDS:-30}"
FAILURES_REQUIRED="${FAILURES_REQUIRED:-2}"
TERM_GRACE_SECONDS="${TERM_GRACE_SECONDS:-30}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/storage-watchdog-20260717}"

for value in "$POLL_SECONDS" "$FAILURES_REQUIRED" "$TERM_GRACE_SECONDS"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "watchdog timing values must be positive integers" >&2
    exit 2
  fi
done
read -r -a TARGETS <<< "$WATCH_PGIDS"
if (( ${#TARGETS[@]} == 0 )); then
  echo "WATCH_PGIDS is empty" >&2
  exit 2
fi
for leader in "${TARGETS[@]}"; do
  if [[ ! "$leader" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid process-group leader: $leader" >&2
    exit 2
  fi
  current="$(ps -o pgid= -p "$leader" 2>/dev/null | tr -d ' ')"
  if [[ "$current" != "$leader" ]]; then
    echo "PID is not its own live process-group leader: $leader" >&2
    exit 2
  fi
done

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/watchdog.log") 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
live_targets() {
  local leader current
  LIVE_TARGETS=()
  for leader in "${TARGETS[@]}"; do
    current="$(ps -o pgid= -p "$leader" 2>/dev/null | tr -d ' ' || true)"
    [[ "$current" == "$leader" ]] && LIVE_TARGETS+=("$leader")
  done
  return 0
}

check_headroom() {
  embedding_require_storage_headroom "$ROOT" \
    "${MIN_WORKSPACE_FREE_GIB:-500}" "${MIN_WORKSPACE_FREE_INODES:-1000000}" \
    && embedding_require_storage_headroom / \
      "${MIN_ROOT_FREE_GIB:-100}" "${MIN_ROOT_FREE_INODES:-200000}" \
    && embedding_require_storage_headroom /tmp \
      "${MIN_TMP_FREE_GIB:-50}" "${MIN_TMP_FREE_INODES:-100000}"
}

echo "[$(timestamp)] storage watchdog armed for PGIDs: ${TARGETS[*]}"
consecutive_failures=0
poll_count=0
while true; do
  live_targets
  if (( ${#LIVE_TARGETS[@]} == 0 )); then
    echo "[$(timestamp)] all watched process groups exited; watchdog complete"
    exit 0
  fi
  if check_headroom; then
    consecutive_failures=0
  else
    consecutive_failures=$((consecutive_failures + 1))
    echo "[$(timestamp)] storage headroom failure $consecutive_failures/$FAILURES_REQUIRED"
    df -h "$ROOT" / /tmp || true
    df -ih "$ROOT" / /tmp || true
  fi
  if (( consecutive_failures >= FAILURES_REQUIRED )); then
    echo "[$(timestamp)] storage emergency; terminating only verified campaign PGIDs: ${LIVE_TARGETS[*]}"
    for leader in "${LIVE_TARGETS[@]}"; do
      current="$(ps -o pgid= -p "$leader" 2>/dev/null | tr -d ' ' || true)"
      [[ "$current" == "$leader" ]] && kill -TERM -- "-$leader" 2>/dev/null || true
    done
    sleep "$TERM_GRACE_SECONDS"
    live_targets
    for leader in "${LIVE_TARGETS[@]}"; do
      current="$(ps -o pgid= -p "$leader" 2>/dev/null | tr -d ' ' || true)"
      [[ "$current" == "$leader" ]] && kill -KILL -- "-$leader" 2>/dev/null || true
    done
    echo "[$(timestamp)] storage emergency shutdown complete"
    exit 20
  fi
  poll_count=$((poll_count + 1))
  if (( poll_count % 20 == 0 )); then
    echo "[$(timestamp)] storage healthy; watched PGIDs alive: ${LIVE_TARGETS[*]}"
  fi
  sleep "$POLL_SECONDS"
done
