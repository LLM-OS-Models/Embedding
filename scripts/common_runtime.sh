#!/usr/bin/env bash

# Shared, side-effect-light runtime defaults for first-party campaign entrypoints.
# This file is sourced by queue/training/mining scripts.  It must never print
# environment variables because HF_TOKEN/GITHUB may be present in the caller.

EMBEDDING_RUNTIME_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep downloaded models, datasets, and Hub metadata inside the ignored
# workspace cache.  Callers may still provide an explicit cache location.
export HF_HOME="${HF_HOME:-$EMBEDDING_RUNTIME_ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
# ms-swift passes ModelScope's cache root to datasets.load_dataset even when
# --use_hf=true.  Pin it as well or map caches escape to a shared home cache and
# cannot be reused reliably after a restart on this public machine.
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$EMBEDDING_RUNTIME_ROOT/.cache/modelscope}"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$MODELSCOPE_CACHE"

embedding_effective_cpu_count() {
  local available quota period quota_cpus cgroup_path relative
  available="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
  [[ "$available" =~ ^[1-9][0-9]*$ ]] || available=1

  # cgroup v2.  In a cgroup namespace cpu.max is usually at the mount root;
  # the /proc path fallback also covers hosts exposing the full hierarchy.
  cgroup_path=/sys/fs/cgroup/cpu.max
  relative="$(awk -F: '$1 == "0" { print $3; exit }' /proc/self/cgroup 2>/dev/null || true)"
  if [[ -n "$relative" && -f "/sys/fs/cgroup${relative%/}/cpu.max" ]]; then
    cgroup_path="/sys/fs/cgroup${relative%/}/cpu.max"
  fi
  if [[ -r "$cgroup_path" ]]; then
    read -r quota period < "$cgroup_path" || true
    if [[ "$quota" =~ ^[0-9]+$ && "$period" =~ ^[1-9][0-9]*$ ]]; then
      quota_cpus="$(((quota + period - 1) / period))"
      (( quota_cpus < 1 )) && quota_cpus=1
      (( quota_cpus < available )) && available="$quota_cpus"
    fi
  # cgroup v1 fallback.
  elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us \
      && -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]]; then
    quota="$(< /sys/fs/cgroup/cpu/cpu.cfs_quota_us)"
    period="$(< /sys/fs/cgroup/cpu/cpu.cfs_period_us)"
    if [[ "$quota" =~ ^[0-9]+$ && "$period" =~ ^[1-9][0-9]*$ ]]; then
      quota_cpus="$(((quota + period - 1) / period))"
      (( quota_cpus < 1 )) && quota_cpus=1
      (( quota_cpus < available )) && available="$quota_cpus"
    fi
  fi
  printf '%s\n' "$available"
}

if [[ -z "${EFFECTIVE_CPU_COUNT:-}" ]]; then
  EFFECTIVE_CPU_COUNT="$(embedding_effective_cpu_count)"
fi
if [[ ! "$EFFECTIVE_CPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "EFFECTIVE_CPU_COUNT must be a positive integer" >&2
  return 2 2>/dev/null || exit 2
fi
export EFFECTIVE_CPU_COUNT

embedding_read_dotenv_key() {
  local env_file="$1" wanted="$2" line key value mode owner
  [[ -r "$env_file" && -f "$env_file" && ! -L "$env_file" ]] || return 1
  mode="$(stat -Lc '%a' "$env_file" 2>/dev/null)" || return 1
  owner="$(stat -Lc '%u' "$env_file" 2>/dev/null)" || return 1
  [[ "$mode" == 600 && "$owner" == "$EUID" ]] || return 1
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -n "$line" && "$line" != \#* ]] || continue
    if [[ "$line" == export[[:space:]]* ]]; then
      line="${line#export}"
      line="${line#"${line%%[![:space:]]*}"}"
    fi
    [[ "$line" == *=* ]] || continue
    key="${line%%=*}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    [[ "$key" == "$wanted" ]] || continue
    value="${line#*=}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ ${#value} -ge 2 ]]; then
      if [[ "$value" == \"*\" || "$value" == \'*\' ]]; then
        [[ "${value:0:1}" == "${value: -1}" ]] || return 1
        value="${value:1:${#value}-2}"
      fi
    fi
    [[ -n "$value" ]] || return 1
    printf '%s' "$value"
    return 0
  done < "$env_file"
  return 1
}

embedding_load_github_credential() {
  local env_file="${1:-$EMBEDDING_RUNTIME_ROOT/.env}" loaded status=0
  if [[ -z "${GITHUB:-}" ]]; then
    loaded="$(embedding_read_dotenv_key "$env_file" GITHUB)" || status=$?
    if (( status == 0 )) && [[ -n "$loaded" ]]; then
      GITHUB="$loaded"
      export GITHUB
    else
      status=1
    fi
  fi
  # A Git-only publisher must never retain unrelated Hub credentials.
  unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
  return "$status"
}

embedding_load_hf_credential() {
  local env_file="${1:-$EMBEDDING_RUNTIME_ROOT/.env}" loaded status=0
  loaded="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}"
  if [[ -z "$loaded" ]]; then
    loaded="$(embedding_read_dotenv_key "$env_file" HF_TOKEN)" || {
      loaded="$(embedding_read_dotenv_key "$env_file" HUGGINGFACE_HUB_TOKEN)" || status=$?
    }
  fi
  if (( status == 0 )) && [[ -n "$loaded" ]]; then
    HF_TOKEN="$loaded"
    export HF_TOKEN
  else
    status=1
  fi
  # A Hub-only publisher must never retain unrelated Git credentials or a
  # second token alias.  The caller runs this helper inside a short subshell.
  unset GITHUB GITHUB_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
  return "$status"
}

embedding_require_storage_headroom() {
  local path="$1" min_gib="$2" min_inodes="$3"
  local available_kib available_inodes required_kib
  if [[ ! "$min_gib" =~ ^[1-9][0-9]*$ \
      || ! "$min_inodes" =~ ^[1-9][0-9]*$ ]]; then
    echo "storage headroom thresholds must be positive integers" >&2
    return 2
  fi
  available_kib="$(df -Pk "$path" | awk 'NR == 2 {print $4}')"
  available_inodes="$(df -Pi "$path" | awk 'NR == 2 {print $4}')"
  if [[ ! "$available_kib" =~ ^[0-9]+$ \
      || ! "$available_inodes" =~ ^[0-9]+$ ]]; then
    echo "unable to measure storage headroom: $path" >&2
    return 2
  fi
  required_kib=$((min_gib * 1024 * 1024))
  if (( available_kib < required_kib || available_inodes < min_inodes )); then
    echo "insufficient storage headroom: $path requires ${min_gib}GiB and ${min_inodes} inodes" >&2
    return 3
  fi
}

embedding_require_clean_validation() {
  local validation="$1"
  local manifest="${2:-$(dirname "$validation")/manifest.json}"
  local declared_sha actual_sha declared_rows actual_rows
  [[ -s "$validation" && -s "$manifest" ]] || {
    echo "clean validation artifact is missing" >&2
    return 1
  }
  [[ "$(jq -r '.status // empty' "$manifest")" == complete \
      && "$(jq -r '.artifact_id // empty' "$manifest")" == \
        legal-source-heldout-i-v2-text-strict-training-validation \
      && "$(jq -r '.assertions.source_holdout_contract_verified // false' "$manifest")" == true \
      && "$(jq -r '.assertions.selected_query_training_text_overlap // -1' "$manifest")" == 0 \
      && "$(jq -r '.assertions.selected_positive_training_text_overlap // -1' "$manifest")" == 0 \
      && "$(jq -r '.assertions.selected_negative_training_text_overlap // -1' "$manifest")" == 0 \
      && "$(jq -r '.assertions.selected_source_document_training_provenance_overlap // -1' "$manifest")" == 0 ]] || {
    echo "clean validation leakage contract failed" >&2
    return 1
  }
  declared_sha="$(jq -r '.files["validation.jsonl"].sha256 // empty' "$manifest")"
  declared_rows="$(jq -r '.files["validation.jsonl"].rows // -1' "$manifest")"
  actual_sha="$(sha256sum "$validation" | awk '{print $1}')"
  actual_rows="$(wc -l < "$validation")"
  [[ "$declared_sha" == "$actual_sha" && "$declared_rows" == 512 \
      && "$actual_rows" == 512 ]] || {
    echo "clean validation file hash or row contract failed" >&2
    return 1
  }
}

embedding_resolve_train_runtime() {
  local requested="${TRAIN_ENV:-}" candidate
  EMBEDDING_TRAIN_ENV=
  if [[ -n "$requested" && -x "$requested/bin/python" ]]; then
    EMBEDDING_TRAIN_ENV="$requested"
  else
    for candidate in \
      "$EMBEDDING_RUNTIME_ROOT/.venv-train" \
      "$EMBEDDING_RUNTIME_ROOT/.venv-train-fa2"; do
      if [[ -x "$candidate/bin/python" ]]; then
        EMBEDDING_TRAIN_ENV="$candidate"
        break
      fi
    done
  fi
  if [[ -z "$EMBEDDING_TRAIN_ENV" ]]; then
    echo "no repository-local training Python environment is available" >&2
    return 4
  fi
  if [[ "$EMBEDDING_TRAIN_ENV" == "$EMBEDDING_RUNTIME_ROOT/.venv-train-fa2" ]]; then
    embedding_enable_torch25_swift_compat
  fi
  EMBEDDING_TRAIN_PYTHON="$EMBEDDING_TRAIN_ENV/bin/python"
  export EMBEDDING_TRAIN_ENV EMBEDDING_TRAIN_PYTHON
}

embedding_resolve_general_base() {
  local selection="${GENERAL_SELECTION:-$EMBEDDING_RUNTIME_ROOT/outputs/reranker-kd-20260717-frontier/clean-first-selection.json}"
  local selected_rel selected_abs candidate
  EMBEDDING_GENERAL_BASE=
  if [[ -s "$selection" ]]; then
    selected_rel="$(jq -r '.best.model // empty' "$selection" 2>/dev/null)"
    if [[ -n "$selected_rel" && "$selected_rel" != /* \
        && "$selected_rel" != *../* && "$selected_rel" != ../* ]]; then
      selected_abs="$EMBEDDING_RUNTIME_ROOT/$selected_rel"
      if [[ -s "$selected_abs/merge_report.json" ]]; then
        EMBEDDING_GENERAL_BASE="$selected_abs"
      fi
    fi
  fi
  if [[ -z "$EMBEDDING_GENERAL_BASE" ]]; then
    for candidate in \
      "$EMBEDDING_RUNTIME_ROOT/artifacts/models/qwen3-embedding-8b-ko-performance1m-lora-r64-best-merged" \
      "$EMBEDDING_RUNTIME_ROOT/artifacts/models/qwen3-embedding-8b-ko-performance1m-lora-r64-b8-best-merged"; do
      if [[ -s "$candidate/merge_report.json" ]]; then
        EMBEDDING_GENERAL_BASE="$candidate"
        break
      fi
    done
  fi
  export EMBEDDING_GENERAL_BASE
  [[ -n "$EMBEDDING_GENERAL_BASE" ]]
}

embedding_enable_torch25_swift_compat() {
  local compat="$EMBEDDING_RUNTIME_ROOT/compat/torch25"
  case ":${PYTHONPATH:-}:" in
    *":$compat:"*) ;;
    *) export PYTHONPATH="$compat${PYTHONPATH:+:$PYTHONPATH}" ;;
  esac
}

embedding_configure_hf_access() {
  if [[ "${EMBEDDING_OFFLINE:-0}" == 1 ]]; then
    # Long-running jobs on the shared machine must not retain Hub credentials
    # when every pinned input is already present in the repository-local cache.
    unset HF_TOKEN HUGGINGFACE_HUB_TOKEN
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    return 0
  fi
  if [[ -z "${HF_TOKEN:-}" && -f "$EMBEDDING_RUNTIME_ROOT/.env" ]]; then
    HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "$EMBEDDING_RUNTIME_ROOT/.env" | tail -n 1)"
    export HF_TOKEN
  fi
}
