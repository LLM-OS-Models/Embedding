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
    parser.add_argument("--selection-strategy")
    parser.add_argument("--candidate-pool-size", type=int)
    parser.add_argument("--num-negatives", type=int)
    return parser.parse_args()


def manifest_matches(
    manifest: dict,
    model: str,
    revision: str,
    selection_strategy: str | None = None,
    candidate_pool_size: int | None = None,
    num_negatives: int | None = None,
) -> bool:
    matches = (
        manifest.get("model") == model
        and (manifest.get("revision") or "") == revision
        and manifest.get("model_weights_sha256")
        == local_model_weights_sha256(model)
    )
    if selection_strategy is not None:
        matches = matches and manifest.get("selection_strategy") == selection_strategy
    if candidate_pool_size is not None:
        matches = matches and manifest.get("candidate_pool_size") == candidate_pool_size
    if num_negatives is not None:
        matches = matches and manifest.get("num_negatives") == num_negatives
    return matches


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest_matches(
        manifest,
        args.model,
        args.revision,
        args.selection_strategy,
        args.candidate_pool_size,
        args.num_negatives,
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
