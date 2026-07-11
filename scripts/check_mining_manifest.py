#!/usr/bin/env python3
"""Exit successfully only when a mining manifest matches the requested encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from mine_faiss_hard_negatives import local_model_weights_sha256
except ModuleNotFoundError:
    from scripts.mine_faiss_hard_negatives import local_model_weights_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="")
    return parser.parse_args()


def manifest_matches(manifest: dict, model: str, revision: str) -> bool:
    return (
        manifest.get("model") == model
        and (manifest.get("revision") or "") == revision
        and manifest.get("model_weights_sha256")
        == local_model_weights_sha256(model)
    )


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest_matches(manifest, args.model, args.revision):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
