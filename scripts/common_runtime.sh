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

embedding_enable_torch25_swift_compat() {
  local compat="$EMBEDDING_RUNTIME_ROOT/compat/torch25"
  case ":${PYTHONPATH:-}:" in
    *":$compat:"*) ;;
    *) export PYTHONPATH="$compat${PYTHONPATH:+:$PYTHONPATH}" ;;
  esac
}
