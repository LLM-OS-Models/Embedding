#!/usr/bin/env bash
set -uo pipefail

# Continue the performance-first GPU queue after a long baseline process.
# Every stage is independently logged and a failed ablation does not prevent
# later useful stages from running.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAIT_PID="${WAIT_PID:-}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/night-queue-20260711}"
COMSAT_REV="a5cc22b651c1b2e51cdd8bf671774ae93584f0ab"
COMSAT_ROOT="$ROOT/outputs/evaluation/mteb_korean_v1/sionic-ai__comsat-embed-ko-8b-preview/$COMSAT_REV"
COMSAT_CACHE="$COMSAT_ROOT/mteb_cache/results/sionic-ai__comsat-embed-ko-8b-preview/$COMSAT_REV"
MIRACL_RESULT="$COMSAT_CACHE/MIRACLRetrieval.json"
PILOT_DIR="$ROOT/data/processed/ko_triplet_pilot_10k"
PILOT_TRAIN="$PILOT_DIR/train.hn-qwen3-r095-n4.jsonl"
PILOT_VAL="$PILOT_DIR/validation.hn-qwen3-r095-n4.jsonl"
PERF50_DIR="$ROOT/outputs/data/performance-v1/pilot-50k"
PERF200_DIR="$ROOT/outputs/data/performance-v1/ablation-200k"
COMSAT_EMBED_CACHE="$ROOT/outputs/embedding-cache/comsat-official-korean"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

run_stage() {
  local name="$1"
  shift
  echo "[$(timestamp)] START $name"
  "$@"
  local status=$?
  echo "[$(timestamp)] END $name status=$status"
  return "$status"
}

perf50_ready() {
  [[ -s "$PERF50_DIR/train.jsonl" && -s "$PERF50_DIR/manifest.json" ]] || return 1
  [[ "$(jq -r '.phase + ":" + (.built_rows | tostring)' "$PERF50_DIR/manifest.json")" == \
    "pilot_50k:50000" ]]
}

perf200_ready() {
  [[ -s "$PERF200_DIR/train.jsonl" && -s "$PERF200_DIR/manifest.json" ]] || return 1
  [[ "$(jq -r '.phase + ":" + (.built_rows | tostring)' "$PERF200_DIR/manifest.json")" == \
    "ablation_200k:200000" ]]
}

if [[ -f "$ROOT/.env" ]]; then
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "$ROOT/.env" | tail -n 1)"
  export HF_TOKEN
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$ROOT/third_party/mteb${PYTHONPATH:+:$PYTHONPATH}"
export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

if [[ -n "$WAIT_PID" ]]; then
  echo "[$(timestamp)] waiting for pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 10
  done
fi

if [[ ! -s "$MIRACL_RESULT" ]]; then
  for batch in 208 192; do
    run_stage "comsat-miracl-fa2-batch-$batch" \
      "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_mteb_korean_v1.py" \
      --task MIRACLRetrieval --batch-size "$batch" \
      --attn-implementation flash_attention_2 \
      --embedding-cache-dir "$COMSAT_EMBED_CACHE"
    [[ -s "$MIRACL_RESULT" ]] && break
  done
fi

if [[ -s "$MIRACL_RESULT" ]]; then
  run_stage "comsat-official-korean-summary" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/evaluate_mteb_korean_v1.py" \
    --batch-size 192 --attn-implementation flash_attention_2 \
    --embedding-cache-dir "$COMSAT_EMBED_CACHE"
  run_stage "comsat-live-borda" \
    "$ROOT/.venv-mteb/bin/python" "$ROOT/scripts/compare_local_mteb_korean.py" \
    --summary "$COMSAT_ROOT/summary.json" \
    --output "$ROOT/outputs/evaluation/mteb_korean_v1/comsat-live-comparison.json"
else
  echo "[$(timestamp)] MIRACL result missing after fallbacks; continuing with training queue"
fi

if [[ ! -s "$PILOT_TRAIN" ]]; then
  run_stage "mine-pilot10k-train" env \
    INPUT="$PILOT_DIR/train.jsonl" \
    OUTPUT="$PILOT_TRAIN" \
    ENCODE_BATCH_SIZE=128 ATTENTION_IMPLEMENTATION=flash_attention_2 \
    QUERY_BLOCK_SIZE=128 CORPUS_BLOCK_SIZE=4096 \
    "$ROOT/experiments/020_hard_negative/mine_smoke.sh"
fi
if [[ ! -s "$PILOT_VAL" ]]; then
  run_stage "mine-pilot10k-validation" env \
    INPUT="$PILOT_DIR/validation.jsonl" \
    OUTPUT="$PILOT_VAL" \
    ENCODE_BATCH_SIZE=128 ATTENTION_IMPLEMENTATION=flash_attention_2 \
    QUERY_BLOCK_SIZE=128 CORPUS_BLOCK_SIZE=4096 \
    "$ROOT/experiments/020_hard_negative/mine_smoke.sh"
fi

run_lora_training() {
  local run_name="$1"
  local train_file="$2"
  local max_steps="$3"
  local interval=40
  local dataloader_shuffle=true
  [[ "$train_file" == *homogeneous-b16* ]] && dataloader_shuffle=false
  if (( max_steps > 1000 )); then
    interval=250
  fi
  run_stage "$run_name" env \
    RUN_NAME="$run_name" TRAIN_FILE="$train_file" VAL_FILE="$PILOT_VAL" \
    MAX_STEPS="$max_steps" EVAL_STEPS="$interval" SAVE_STEPS="$interval" \
    TRAIN_BATCH_SIZE=16 GRAD_ACCUM_STEPS=4 \
    TRAIN_DATALOADER_SHUFFLE="$dataloader_shuffle" \
    "$ROOT/experiments/020_hard_negative/train_pilot_lora_r64.sh"
  local latest
  latest="$(find "$ROOT/outputs/$run_name" -maxdepth 3 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1)"
  if [[ -z "$latest" || ! -s "$latest/adapter_model.safetensors" ]]; then
    local fallback_name="${run_name}-b8"
    run_stage "$fallback_name" env \
      RUN_NAME="$fallback_name" TRAIN_FILE="$train_file" VAL_FILE="$PILOT_VAL" \
      MAX_STEPS="$max_steps" EVAL_STEPS="$interval" SAVE_STEPS="$interval" \
      TRAIN_BATCH_SIZE=8 GRAD_ACCUM_STEPS=8 \
      TRAIN_DATALOADER_SHUFFLE="$dataloader_shuffle" \
      "$ROOT/experiments/020_hard_negative/train_pilot_lora_r64.sh"
    run_name="$fallback_name"
    latest="$(find "$ROOT/outputs/$run_name" -maxdepth 3 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1)"
  fi
  if [[ -n "$latest" && -s "$latest/adapter_model.safetensors" ]]; then
    run_stage "$run_name-verify" \
      "$ROOT/.venv-train/bin/python" "$ROOT/scripts/verify_adapter.py" \
      --adapter "$latest" --data "$PILOT_VAL" \
      --output "$ROOT/outputs/$run_name/verification.json"
  fi
}

if [[ -s "$PILOT_TRAIN" && -s "$PILOT_VAL" ]]; then
  run_lora_training "qwen3-embedding-8b-ko-hn10k-lora-r64" "$PILOT_TRAIN" 160
fi

if perf50_ready && [[ -s "$PILOT_VAL" ]]; then
  perf50_train="$PERF50_DIR/train.jsonl"
  [[ -s "$PERF50_DIR/homogeneous-b16.manifest.json" ]] && \
    perf50_train="$PERF50_DIR/train.homogeneous-b16.jsonl"
  run_lora_training "qwen3-embedding-8b-ko-performance50k-lora-r64" \
    "$perf50_train" 800
fi

if perf200_ready && [[ -s "$PILOT_VAL" ]]; then
  perf200_train="$PERF200_DIR/train.jsonl"
  [[ -s "$PERF200_DIR/homogeneous-b16.manifest.json" ]] && \
    perf200_train="$PERF200_DIR/train.homogeneous-b16.jsonl"
  run_lora_training "qwen3-embedding-8b-ko-performance200k-lora-r64" \
    "$perf200_train" 3125
fi

if [[ -s "$PILOT_TRAIN" && -s "$PILOT_VAL" ]]; then
  run_stage "qwen3-embedding-8b-ko-hn10k-f2dual-lora-r64" env \
    RUN_NAME=qwen3-embedding-8b-ko-hn10k-f2dual-lora-r64 \
    F2_DUAL_TEMPERATURE=.05 USE_F2_MRL=0 \
    "$ROOT/experiments/080_f2_recipe/train_pilot_f2_dual_lora_r64.sh"
  run_stage "qwen3-embedding-8b-ko-hn10k-f2dual-t002-lora-r64" env \
    RUN_NAME=qwen3-embedding-8b-ko-hn10k-f2dual-t002-lora-r64 \
    F2_DUAL_TEMPERATURE=.02 USE_F2_MRL=0 \
    "$ROOT/experiments/080_f2_recipe/train_pilot_f2_dual_lora_r64.sh"
  run_stage "qwen3-embedding-8b-ko-hn10k-f2dual-mrl-lora-r64" env \
    RUN_NAME=qwen3-embedding-8b-ko-hn10k-f2dual-mrl-lora-r64 \
    F2_DUAL_TEMPERATURE=.05 USE_F2_MRL=1 \
    "$ROOT/experiments/080_f2_recipe/train_pilot_f2_dual_lora_r64.sh"
fi

for mode in lora_r64 dora_r32 last4 galore standard_full; do
  run_stage "memory-probe-$mode" \
    "$ROOT/experiments/070_tuning_strategy/probe_memory.sh" "$mode"
done

# The 50K builder may have completed while the ablations ran.
if perf50_ready && [[ -s "$PILOT_VAL" \
      && ! -d "$ROOT/outputs/qwen3-embedding-8b-ko-performance50k-lora-r64" ]]; then
  perf50_train="$PERF50_DIR/train.jsonl"
  [[ -s "$PERF50_DIR/homogeneous-b16.manifest.json" ]] && \
    perf50_train="$PERF50_DIR/train.homogeneous-b16.jsonl"
  run_lora_training "qwen3-embedding-8b-ko-performance50k-lora-r64" \
    "$perf50_train" 800
fi

if perf200_ready && [[ -s "$PILOT_VAL" \
      && ! -d "$ROOT/outputs/qwen3-embedding-8b-ko-performance200k-lora-r64" ]]; then
  perf200_train="$PERF200_DIR/train.jsonl"
  [[ -s "$PERF200_DIR/homogeneous-b16.manifest.json" ]] && \
    perf200_train="$PERF200_DIR/train.homogeneous-b16.jsonl"
  run_lora_training "qwen3-embedding-8b-ko-performance200k-lora-r64" \
    "$perf200_train" 3125
fi

echo "[$(timestamp)] GPU queue complete"
