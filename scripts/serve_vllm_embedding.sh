#!/usr/bin/env bash
set -euo pipefail

# Production-oriented OpenAI-compatible embedding service. Override every knob
# through the environment; defaults target one H100 80GB and Qwen3-Embedding-8B.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Embedding-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-embedding-8b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM_BIN="${VLLM_BIN:-$ROOT/.venv-vllm/bin/vllm}"

if [[ ! -x "$VLLM_BIN" ]]; then
  echo "vLLM executable not found: $VLLM_BIN" >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "$VLLM_BIN" serve "$MODEL_ID" \
  --runner pooling \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --disable-log-requests
