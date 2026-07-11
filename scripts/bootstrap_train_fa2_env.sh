#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TARGET="${TRAIN_FA2_ENV:-$ROOT/.venv-train-fa2}"

if [[ ! -x "$TARGET/bin/python" ]]; then
  python3 -m venv --system-site-packages "$TARGET"
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
"$TARGET/bin/python" -m pip install -e "$ROOT/third_party/ms-swift"
"$TARGET/bin/python" - <<'PY'
import accelerate
import flash_attn
import peft
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
        "trl": trl.__version__,
        "accelerate": accelerate.__version__,
    }
)
PY
echo "Import gate passed. A real 8B forward/backward probe is still required before long training."
