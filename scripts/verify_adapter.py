#!/usr/bin/env python3
"""Reload a Qwen3 embedding LoRA and verify normalized retrieval behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from swift import __version__ as swift_version
from swift.infer_engine import InferRequest, TransformersEngine


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-disqualified-diagnostic", action="store_true")
    return parser.parse_args()


def disqualification_marker(adapter: Path) -> Path | None:
    resolved = adapter.expanduser().resolve()
    for directory in (resolved, *resolved.parents):
        marker = directory / "DISQUALIFIED.json"
        if marker.is_file() and marker.stat().st_size > 0:
            return marker
    return None


def main() -> None:
    args = parse_args()
    marker = disqualification_marker(args.adapter)
    if marker is not None and not args.allow_disqualified_diagnostic:
        raise RuntimeError(
            "Refusing candidate verification for a disqualified run: "
            f"{marker}. Use --allow-disqualified-diagnostic only for an explicitly "
            "labelled diagnostic check."
        )
    adapter_config = json.loads(
        (args.adapter / "adapter_config.json").read_text(encoding="utf-8")
    )
    reference = adapter_config.get("base_model_name_or_path")
    default_revision = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
    if (
        args.model == "Qwen/Qwen3-Embedding-8B"
        and isinstance(reference, str)
        and Path(reference).is_dir()
        and Path(reference).name != default_revision
    ):
        args.model = reference
    row = json.loads(args.data.read_text(encoding="utf-8").splitlines()[0])
    groups = [
        row["messages"],
        row["positive_messages"][0],
        row["negative_messages"][0],
    ]
    requests = [InferRequest(messages=messages) for messages in groups]
    engine = TransformersEngine(
        args.model,
        task_type="embedding",
        torch_dtype=torch.bfloat16,
        attn_impl="sdpa",
        adapters=[str(args.adapter.resolve())],
        use_hf=True,
    )
    responses = engine.infer(requests)
    embeddings = torch.tensor([response.data[0].embedding for response in responses])
    scores = embeddings @ embeddings.T
    norms = embeddings.norm(dim=1)
    if embeddings.shape != (3, 4096):
        raise RuntimeError(f"Unexpected embedding shape: {tuple(embeddings.shape)}")
    if not torch.isfinite(embeddings).all():
        raise RuntimeError("Non-finite embedding values")
    # The engine serializes BF16 embeddings through Python floats; a few 1e-3
    # of norm drift is expected after that round trip.
    if not torch.allclose(norms, torch.ones_like(norms), atol=5e-3):
        raise RuntimeError(f"Embeddings are not L2 normalized: {norms.tolist()}")
    if scores[0, 1] <= scores[0, 2]:
        raise RuntimeError("Positive similarity is not greater than negative similarity")

    result = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.model,
        "adapter": str(args.adapter),
        "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "embedding_shape": list(embeddings.shape),
        "norms": norms.tolist(),
        "query_positive": float(scores[0, 1]),
        "query_negative": float(scores[0, 2]),
        "margin": float(scores[0, 1] - scores[0, 2]),
        "torch": torch.__version__,
        "swift": swift_version,
        "status": "pass",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
