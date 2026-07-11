#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BASELINE_PID="${BASELINE_PID:-}"
CAMPAIGN_LOG="${CAMPAIGN_LOG:-$ROOT/outputs/full-campaign-20260711.log}"
mkdir -p "$(dirname "$CAMPAIGN_LOG")"
exec > >(tee -a "$CAMPAIGN_LOG") 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
stage() {
  local name="$1"; shift
  echo "[$(timestamp)] CAMPAIGN START $name"
  "$@"
  local status=$?
  echo "[$(timestamp)] CAMPAIGN END $name status=$status"
  return "$status"
}

stage training-and-tuning env WAIT_PID="$BASELINE_PID" \
  LOG_DIR="$ROOT/outputs/night-queue-20260711-resumed" \
  bash "$ROOT/scripts/run_night_gpu_queue.sh"

stage post-training-evaluation env WAIT_PID= \
  LOG_DIR="$ROOT/outputs/post-training-eval-20260711" \
  bash "$ROOT/scripts/run_post_training_eval_queue.sh"

stage scale-1m env WAIT_PID= \
  LOG_DIR="$ROOT/outputs/scale-1m-20260711" \
  bash "$ROOT/scripts/run_scale_1m_queue.sh"

stage legal-target-adaptation env WAIT_PID= \
  LOG_DIR="$ROOT/outputs/legal-adaptation-20260711" \
  bash "$ROOT/scripts/run_legal_adaptation_queue.sh"

stage top-model-sionic env WAIT_PID= \
  LOG_DIR="$ROOT/outputs/top-model-eval-20260711" \
  bash "$ROOT/scripts/run_top_model_sionic_queue.sh"

echo "[$(timestamp)] full campaign queue complete"
