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
