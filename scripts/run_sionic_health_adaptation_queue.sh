#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export TARGET_KIND=health
export TARGET_PHASE=sionic_health_multilingual_100k
export TARGET_DATA_REL=outputs/data/performance-v1/sionic-health-multilingual-100k
export TARGET_ADAPTATION=target-adapted-health-domain
export TARGET_NLIST=512
export TARGET_TRAINING_POINTS=50000
export TARGET_SOURCE_DATASET=LLM-OS-Models/korean-embedding-performance-v1-sionic-health-100k
export TARGET_RUN_NAME=qwen3-embedding-8b-ko-sionic-health50-replay50-lora-r64
export DERIVED_REPO=LLM-OS-Models/korean-embedding-sionic-health-quantile-hn7-replay-v1
export DERIVED_TITLE="Korean Sionic Health Quantile HN7 with General Replay"
export MODEL_REPO=LLM-OS-Models/qwen3-embedding-8b-ko-sionic-health-target-adapted-v1
export CAMPAIGN_STAGE=sionic-health-target
export LOG_DIR="${LOG_DIR:-$ROOT/outputs/sionic-health-adaptation-20260712}"

exec bash "$ROOT/scripts/run_sionic_squad_adaptation_queue.sh"
