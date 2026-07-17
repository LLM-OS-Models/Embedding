#!/usr/bin/env bash
set -euo pipefail

# Rebuild the pinned evaluation/ANN environment entirely on the NFS workspace.
# The host image has no ensurepip, so use the same local virtualenv fallback as
# the FA2 training bootstrap.  System torch/flash-attn are inherited read-only;
# evaluation Python packages are overlaid only inside .venv-mteb.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${MTEB_ENV:-$ROOT/.venv-mteb}"
BOOTSTRAP="$ROOT/.cache/virtualenv-bootstrap"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  if ! python3 -m venv --system-site-packages "$ENV_DIR"; then
    if [[ ! -s "$BOOTSTRAP/virtualenv/__init__.py" ]]; then
      mkdir -p "$BOOTSTRAP"
      python3 -m pip install --disable-pip-version-check --no-input \
        --target "$BOOTSTRAP" 'virtualenv==21.6.1'
    fi
    PYTHONPATH="$BOOTSTRAP" python3 -m virtualenv \
      --system-site-packages "$ENV_DIR"
  fi
fi

"$ENV_DIR/bin/python" -m pip install --disable-pip-version-check --no-input \
  -r "$ROOT/requirements/mteb-extras.txt"
"$ENV_DIR/bin/python" -m pip install --disable-pip-version-check --no-input \
  -e "$ROOT/third_party/mteb"

PYTHONPATH="$ROOT/third_party/mteb${PYTHONPATH:+:$PYTHONPATH}" \
  "$ENV_DIR/bin/python" - <<'PY'
import faiss
import flash_attn
import mteb
import numpy
import sentence_transformers
import torch
import transformers

observed = {
    "mteb": mteb.__version__,
    "faiss": faiss.__version__,
    "numpy": numpy.__version__,
    "sentence_transformers": sentence_transformers.__version__,
    "transformers": transformers.__version__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "flash_attn": flash_attn.__version__,
}
expected = {"mteb": "2.18.0", "faiss": "1.14.3", "numpy": "1.26.4"}
for name, version in expected.items():
    if observed[name] != version:
        raise SystemExit(f"{name} version mismatch: {observed[name]} != {version}")
print({"status": "mteb-import-pass", **observed})
PY
