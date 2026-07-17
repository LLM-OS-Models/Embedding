#!/usr/bin/env bash

# Shared fail-closed selector for the fast training runtime. Callers must pass
# the exact workload they are about to launch. FA2 is selected only above the
# speed gate; otherwise the exact matched SDPA half may select the same runtime.
# If neither report validates, callers retain the stable SDPA environment.

embedding_check_fa2_admission() {
  local report="$1" train_file="$2" batch_size="$3" accumulation="$4"
  local max_length="$5" lora_rank="$6" lora_alpha="$7" dtype="$8"
  local base_model="$9" base_revision="${10}" hard_negatives="${11}"
  local lora_dropout="${12:-0.05}"
  local root="${ROOT:?ROOT must be defined before sourcing backend_admission.sh}"
  local fa2_env="${FA2_ENV:-$root/.venv-train-fa2}"
  [[ -x "$fa2_env/bin/python" && -s "$report" ]] || return 1
  if declare -F embedding_enable_torch25_swift_compat >/dev/null; then
    embedding_enable_torch25_swift_compat
  fi
  "$fa2_env/bin/python" "$root/scripts/backend_admission.py" check \
    --report "$report" --quiet \
    --train-file "$train_file" --backend flash_attention_2 \
    --batch-size "$batch_size" \
    --gradient-accumulation-steps "$accumulation" \
    --max-length "$max_length" --lora-rank "$lora_rank" \
    --lora-alpha "$lora_alpha" --lora-dropout "$lora_dropout" \
    --dtype "$dtype" --base-model "$base_model" \
    --base-revision "$base_revision" --hard-negatives "$hard_negatives"
}

embedding_check_matched_sdpa() {
  local report="$1" train_file="$2" batch_size="$3" accumulation="$4"
  local max_length="$5" lora_rank="$6" lora_alpha="$7" dtype="$8"
  local base_model="$9" base_revision="${10}" hard_negatives="${11}"
  local lora_dropout="${12:-0.05}"
  local root="${ROOT:?ROOT must be defined before sourcing backend_admission.sh}"
  local fa2_env="${FA2_ENV:-$root/.venv-train-fa2}"

  [[ -x "$fa2_env/bin/python" && -s "$report" ]] || return 1
  if declare -F embedding_enable_torch25_swift_compat >/dev/null; then
    embedding_enable_torch25_swift_compat
  fi
  "$fa2_env/bin/python" "$root/scripts/backend_admission.py" check-sdpa \
    --report "$report" --quiet \
    --train-file "$train_file" --backend flash_attention_2 \
    --batch-size "$batch_size" \
    --gradient-accumulation-steps "$accumulation" \
    --max-length "$max_length" --lora-rank "$lora_rank" \
    --lora-alpha "$lora_alpha" --lora-dropout "$lora_dropout" \
    --dtype "$dtype" --base-model "$base_model" \
    --base-revision "$base_revision" --hard-negatives "$hard_negatives"
}

embedding_select_fa2_backend() {
  local train_file="$1" run_key="$2" batch_size="$3" accumulation="$4"
  local max_length="$5" lora_rank="$6" lora_alpha="$7" dtype="$8"
  local base_model="$9" base_revision="${10}" hard_negatives="${11}"
  local lora_dropout="${12:-0.05}"
  local root="${ROOT:?ROOT must be defined before sourcing backend_admission.sh}"
  local fa2_env="${FA2_ENV:-$root/.venv-train-fa2}"
  local stable_env="${EMBEDDING_TRAIN_ENV:-$root/.venv-train}"
  if [[ ! -x "$stable_env/bin/python" && -x "$fa2_env/bin/python" ]]; then
    stable_env="$fa2_env"
  fi
  local report="$root/outputs/backend-probes/$run_key/admission.json"

  if [[ ! "$run_key" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "unsafe backend admission run key: $run_key" >&2
    return 2
  fi
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  BACKEND_ADMISSION_ENV="$stable_env"
  BACKEND_ADMISSION_ATTN=sdpa
  BACKEND_ADMISSION_REPORT="$report"
  BACKEND_ADMISSION_VERIFIED_REPORT=
  BACKEND_SDPA_VERIFIED_REPORT=
  BACKEND_ADMISSION_SELECTED_KIND=stable_sdpa
  export BACKEND_ADMISSION_ENV BACKEND_ADMISSION_ATTN BACKEND_ADMISSION_REPORT
  export BACKEND_ADMISSION_VERIFIED_REPORT BACKEND_SDPA_VERIFIED_REPORT
  export BACKEND_ADMISSION_SELECTED_KIND

  if [[ "${FORCE_PROBE:-0}" != 1 ]] && \
      embedding_check_fa2_admission "$report" "$train_file" "$batch_size" \
      "$accumulation" "$max_length" "$lora_rank" "$lora_alpha" "$dtype" \
      "$base_model" "$base_revision" "$hard_negatives" "$lora_dropout"; then
    BACKEND_ADMISSION_ENV="$fa2_env"
    BACKEND_ADMISSION_ATTN=flash_attention_2
    BACKEND_ADMISSION_VERIFIED_REPORT="$report"
    BACKEND_ADMISSION_SELECTED_KIND=fa2
    export BACKEND_ADMISSION_ENV BACKEND_ADMISSION_ATTN
    export BACKEND_ADMISSION_VERIFIED_REPORT BACKEND_ADMISSION_SELECTED_KIND
    return 0
  fi

  if [[ "${FORCE_PROBE:-0}" != 1 ]] && \
      embedding_check_matched_sdpa "$report" "$train_file" "$batch_size" \
      "$accumulation" "$max_length" "$lora_rank" "$lora_alpha" "$dtype" \
      "$base_model" "$base_revision" "$hard_negatives" "$lora_dropout"; then
    BACKEND_ADMISSION_ENV="$fa2_env"
    BACKEND_ADMISSION_ATTN=sdpa
    BACKEND_SDPA_VERIFIED_REPORT="$report"
    BACKEND_ADMISSION_SELECTED_KIND=matched_sdpa
    export BACKEND_ADMISSION_ENV BACKEND_ADMISSION_ATTN
    export BACKEND_SDPA_VERIFIED_REPORT BACKEND_ADMISSION_SELECTED_KIND
    return 0
  fi

  if [[ "${BACKEND_ADMISSION_AUTO_PROBE:-1}" == 1 ]]; then
    echo "FA2 admission missing or contract-mismatched; running tailored probe: $run_key" >&2
    env \
      FA2_ENV="$fa2_env" TRAIN_FILE="$train_file" TRAIN_PROVENANCE= \
      RUN_KEY="$run_key" \
      TRAIN_BATCH_SIZE="$batch_size" GRAD_ACCUM_STEPS="$accumulation" \
      MAX_LENGTH="$max_length" LORA_RANK="$lora_rank" \
      LORA_ALPHA="$lora_alpha" LORA_DROPOUT="$lora_dropout" \
      TRAIN_DTYPE="$dtype" BASE_MODEL="$base_model" \
      BASE_REVISION="$base_revision" \
      INFONCE_HARD_NEGATIVES="$hard_negatives" \
      "$root/experiments/070_tuning_strategy/admit_fa2_lora_backend.sh" || true
  fi

  if embedding_check_fa2_admission "$report" "$train_file" "$batch_size" \
      "$accumulation" "$max_length" "$lora_rank" "$lora_alpha" "$dtype" \
      "$base_model" "$base_revision" "$hard_negatives" "$lora_dropout"; then
    BACKEND_ADMISSION_ENV="$fa2_env"
    BACKEND_ADMISSION_ATTN=flash_attention_2
    BACKEND_ADMISSION_VERIFIED_REPORT="$report"
    BACKEND_ADMISSION_SELECTED_KIND=fa2
    export BACKEND_ADMISSION_ENV BACKEND_ADMISSION_ATTN
    export BACKEND_ADMISSION_VERIFIED_REPORT BACKEND_ADMISSION_SELECTED_KIND
    return 0
  fi

  if embedding_check_matched_sdpa "$report" "$train_file" "$batch_size" \
      "$accumulation" "$max_length" "$lora_rank" "$lora_alpha" "$dtype" \
      "$base_model" "$base_revision" "$hard_negatives" "$lora_dropout"; then
    BACKEND_ADMISSION_ENV="$fa2_env"
    BACKEND_ADMISSION_ATTN=sdpa
    BACKEND_SDPA_VERIFIED_REPORT="$report"
    BACKEND_ADMISSION_SELECTED_KIND=matched_sdpa
    export BACKEND_ADMISSION_ENV BACKEND_ADMISSION_ATTN
    export BACKEND_SDPA_VERIFIED_REPORT BACKEND_ADMISSION_SELECTED_KIND
    echo "FA2 speed gate rejected; using exact matched SDPA runtime: $run_key" >&2
    return 0
  fi

  echo "no exact fast-runtime backend admitted; using stable SDPA: $run_key" >&2
  return 1
}
