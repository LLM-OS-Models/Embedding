#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export TARGET_KIND=retrieval_family
export TARGET_PHASE=sionic_retrieval_train_family_4146
export TARGET_DATA_REL=outputs/data/performance-v1/sionic-retrieval-train-family-4146
export TARGET_ADAPTATION=target-adapted-retrieval-family
export TARGET_NLIST=128
export TARGET_TRAINING_POINTS=4000
export TARGET_MAX_LENGTH=2048
export TARGET_ENCODE_BATCH_SIZE=32
export TARGET_TRAIN_BATCH_SIZE=2
export TARGET_GRAD_ACCUM_STEPS=32
export TARGET_FALLBACK_BATCH_SIZE=1
export TARGET_FALLBACK_GRAD_ACCUM_STEPS=64
export TARGET_EVAL_BATCH_SIZE=1
export TARGET_SOURCE_DATASET=LLM-OS-Models/korean-embedding-performance-v1-sionic-retrieval-train-family-4146
export TARGET_RUN_NAME=qwen3-embedding-8b-ko-sionic-retrieval-family50-replay50-lora-r64
export DERIVED_REPO=LLM-OS-Models/korean-embedding-sionic-retrieval-family-quantile-hn7-replay-v1
export DERIVED_TITLE="Korean Sionic Retrieval-Family Quantile HN7 with General Replay"
export MODEL_REPO=LLM-OS-Models/qwen3-embedding-8b-ko-sionic-retrieval-family-target-adapted-v1
export CAMPAIGN_STAGE=sionic-retrieval-family-target
export LOG_DIR="${LOG_DIR:-$ROOT/outputs/sionic-retrieval-family-adaptation-20260712}"

exec bash "$ROOT/scripts/run_sionic_squad_adaptation_queue.sh"
