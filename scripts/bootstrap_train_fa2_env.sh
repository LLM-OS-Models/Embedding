#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TARGET="${TRAIN_FA2_ENV:-$ROOT/.venv-train-fa2}"

if [[ ! -x "$TARGET/bin/python" ]]; then
  if ! python3 -m venv --system-site-packages "$TARGET"; then
    BOOTSTRAP="$ROOT/.cache/virtualenv-bootstrap"
    PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.cache/pip}" \
      python3 -m pip install --disable-pip-version-check --no-input \
      --target "$BOOTSTRAP" virtualenv==21.6.1
    PYTHONPATH="$BOOTSTRAP" python3 -m virtualenv --clear \
      --system-site-packages "$TARGET"
  fi
fi
"$TARGET/bin/python" - <<'PY'
import flash_attn
import torch

assert torch.cuda.is_available()
major, minor = torch.cuda.get_device_capability()
assert (major, minor) >= (8, 0)
print({"torch": torch.__version__, "cuda": torch.version.cuda, "flash_attn": flash_attn.__version__})
PY
"$TARGET/bin/python" -m pip install -U pip setuptools wheel
"$TARGET/bin/python" -m pip install -r "$ROOT/requirements/train-fa2-overlay.txt"
"$TARGET/bin/python" -m pip install -e "$ROOT/third_party/ms-swift"
"$TARGET/bin/python" - <<'PY'
import accelerate
import flash_attn
import peft
import sentence_transformers
import swift
import torch
import transformers
import trl

print(
    {
        "status": "import-pass",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_attn": flash_attn.__version__,
        "transformers": transformers.__version__,
        "swift": swift.__version__,
        "peft": peft.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "trl": trl.__version__,
        "accelerate": accelerate.__version__,
    }
)
PY
echo "Import gate passed. A real 8B forward/backward probe is still required before long training."
