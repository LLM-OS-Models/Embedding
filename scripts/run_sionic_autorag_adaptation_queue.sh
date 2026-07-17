#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_runtime.sh"

export TARGET_KIND=autorag
export TARGET_PHASE=sionic_autorag_domain_100k
export TARGET_DATA_REL=outputs/data/performance-v1/sionic-autorag-domain-100k
export TARGET_ADAPTATION=target-adapted-autorag-domain
export TARGET_NLIST=512
export TARGET_TRAINING_POINTS=50000
export TARGET_SOURCE_DATASET=LLM-OS-Models/korean-embedding-performance-v1-sionic-autorag-100k
export TARGET_RUN_NAME=qwen3-embedding-8b-ko-sionic-autorag50-replay50-lora-r64
export DERIVED_REPO=LLM-OS-Models2/korean-embedding-sionic-autorag-quantile-hn7-replay-v1
export DERIVED_TITLE="Korean Sionic AutoRAG Quantile HN7 with General Replay"
export MODEL_REPO=LLM-OS-Models2/qwen3-embedding-8b-ko-sionic-autorag-target-adapted-v1
export CAMPAIGN_STAGE=sionic-autorag-target
export LOG_DIR="${LOG_DIR:-$ROOT/outputs/sionic-autorag-adaptation-20260712}"

exec bash "$ROOT/scripts/run_sionic_squad_adaptation_queue.sh"
