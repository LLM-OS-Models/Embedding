#!/usr/bin/env python3
"""Package and verify an ms-swift full/partial SentenceTransformer checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    from merge_embedding_adapter import (
        validate_sentence_transformers_contract,
        write_sentence_transformers_contract,
    )
except ModuleNotFoundError:
    from scripts.merge_embedding_adapter import (
        validate_sentence_transformers_contract,
        write_sentence_transformers_contract,
    )


BASE_MODEL = "Qwen/Qwen3-Embedding-8B"
BASE_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--copy-weights", action="store_true")
    return parser.parse_args()


def hash_model_files(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(root.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No model safetensors under {root}")
    for path in files:
        digest.update(path.name.encode() + b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path, copy_weights: bool) -> None:
    # Hugging Face snapshots are relative symlinks into blobs. Hard-link the
    # resolved blob, otherwise the staged relative symlink becomes broken.
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if not copy_weights:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def stage_checkpoint(checkpoint: Path, output: Path, copy_weights: bool) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    root_files = {
        "config.json",
        "config_sentence_transformers.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors.index.json",
        "modules.json",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    copied = 0
    for source in checkpoint.iterdir():
        if source.is_file() and (
            source.name in root_files
            or (source.name.startswith("model") and source.suffix == ".safetensors")
        ):
            link_or_copy(source, output / source.name, copy_weights)
            copied += 1
    for directory in ("1_Pooling", "2_Normalize"):
        source = checkpoint / directory
        if source.exists():
            shutil.copytree(source, output / directory)
    # Normalize has no parameters/config and Hub snapshots cannot preserve an
    # empty directory, while the publication contract intentionally requires it.
    (output / "2_Normalize").mkdir(exist_ok=True)
    if copied == 0 or not (output / "modules.json").is_file():
        raise FileNotFoundError("Checkpoint is not a complete SentenceTransformers model")


def verify(output: Path, args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    model_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    hidden_size = model_config.get("hidden_size")
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError(f"Invalid packaged hidden_size: {hidden_size!r}")
    write_sentence_transformers_contract(output, hidden_size)
    tokenizer_path = output / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_config["padding_side"] = "left"
    tokenizer_path.write_text(
        json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    contract = validate_sentence_transformers_contract(output)
    dtype = getattr(torch, args.dtype)
    model = SentenceTransformer(
        str(output),
        device=args.device,
        model_kwargs={"attn_implementation": args.attn_implementation, "torch_dtype": dtype},
        tokenizer_kwargs={"padding_side": "left"},
    )
    texts = [
        "대한민국의 수도는 어디인가?",
        "대한민국의 수도는 서울특별시이다.",
        "고양이는 포유류 동물이다.",
        "행정처분 취소소송의 제소기간은 법률이 정한다.",
    ]
    vectors = np.asarray(
        model.encode(texts, batch_size=4, normalize_embeddings=True, convert_to_numpy=True),
        dtype=np.float32,
    )
    norms = np.linalg.norm(vectors, axis=1)
    if vectors.shape != (4, 4096) or not np.isfinite(vectors).all():
        raise RuntimeError(f"Invalid packaged embedding shape/value: {vectors.shape}")
    max_norm_error = float(np.max(np.abs(norms - 1.0)))
    positive_margin = float(vectors[0] @ vectors[1] - vectors[0] @ vectors[2])
    if max_norm_error > 1e-4 or positive_margin <= 0:
        raise RuntimeError(
            f"Packaged model probe failed: norm_error={max_norm_error}, margin={positive_margin}"
        )
    return {
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_method": "partial-full-parameter-update",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "model": {"weights_sha256": hash_model_files(output)},
        "sentence_transformers_contract": contract,
        "probe": {
            "metrics": {
                "maximum_norm_error": max_norm_error,
                "positive_margin": positive_margin,
            }
        },
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.building-{uuid.uuid4().hex}"
    try:
        stage_checkpoint(checkpoint, staging, args.copy_weights)
        report = verify(staging, args)
        (staging / "full_tuning_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
