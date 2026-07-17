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
        same_model_reference,
        validate_sentence_transformers_contract,
        write_sentence_transformers_contract,
    )
except ModuleNotFoundError:
    from scripts.merge_embedding_adapter import (
        same_model_reference,
        validate_sentence_transformers_contract,
        write_sentence_transformers_contract,
    )

try:
    from model_lineage import resolve_base_lineage
except ModuleNotFoundError:
    from scripts.model_lineage import resolve_base_lineage


DEFAULT_BASE_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_BASE_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--copy-weights", action="store_true")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base-revision", default=DEFAULT_BASE_REVISION)
    parser.add_argument("--training-contract", type=Path)
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Verify an existing package against the exact source checkpoint/contract.",
    )
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_training_contract(args: argparse.Namespace) -> dict | None:
    if args.training_contract is None:
        return None
    contract_path = args.training_contract.resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(f"Missing training contract: {contract_path}")
    declared = json.loads(contract_path.read_text(encoding="utf-8"))
    if declared.get("schema_version") != 1 or declared.get(
        "artifact_type"
    ) != "embedding-capacity-training-contract":
        raise ValueError("Invalid training contract schema/type")
    if declared.get("status") != "complete":
        raise ValueError("Training contract is not complete")
    if declared.get("base_model") != args.base_model or declared.get(
        "base_revision"
    ) != args.base_revision:
        raise ValueError("Training contract base model/revision mismatch")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir() or not checkpoint.is_relative_to(contract_path.parent):
        raise ValueError("Checkpoint is outside the contracted run directory")
    optimization = declared.get("optimization")
    if declared.get("mode") != "last4" or not isinstance(optimization, dict):
        raise ValueError("Training contract is not the admitted last4 capacity run")
    expected_optimization = {
        "max_steps": 3123,
        "global_batch_size": 64,
        "dataset_shuffle": False,
        "train_dataloader_shuffle": False,
    }
    for field, expected in expected_optimization.items():
        if optimization.get(field) != expected:
            raise ValueError(f"Training contract optimization mismatch: {field}")
    for field in ("train", "validation"):
        evidence = declared.get(field)
        if not isinstance(evidence, dict):
            raise ValueError(f"Missing {field} evidence in training contract")
        source = Path(str(evidence.get("path", ""))).resolve()
        if not source.is_file() or source.stat().st_size != evidence.get("size_bytes"):
            raise ValueError(f"Training contract {field} file/size mismatch")
        if sha256_file(source) != evidence.get("sha256"):
            raise ValueError(f"Training contract {field} hash mismatch")
    completion = declared.get("completion")
    if not isinstance(completion, dict) or completion.get("expected_steps") != 3123:
        raise ValueError("Training contract has no exact completion evidence")
    for field in ("train_log", "logging_jsonl"):
        evidence = completion.get(field)
        if not isinstance(evidence, dict):
            raise ValueError(f"Missing completion evidence: {field}")
        source = Path(str(evidence.get("path", ""))).resolve()
        if not source.is_file() or not source.is_relative_to(contract_path.parent):
            raise ValueError(f"Unsafe completion evidence path: {field}")
        if sha256_file(source) != evidence.get("sha256"):
            raise ValueError(f"Completion evidence hash mismatch: {field}")
    return {
        "path": str(contract_path),
        "sha256": sha256_file(contract_path),
    }


def validate_existing_package(args: argparse.Namespace) -> dict:
    output = args.output_dir.expanduser().resolve()
    report_path = output / "full_tuning_report.json"
    if not output.is_dir() or output.is_symlink() or not report_path.is_file():
        raise FileNotFoundError("Existing full-model package is incomplete or unsafe")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("status") != "pass"
        or report.get("training_method") != "partial-full-parameter-update"
    ):
        raise ValueError("Existing full-model package report did not pass")
    checkpoint = args.checkpoint.expanduser().resolve()
    try:
        recorded_checkpoint = Path(str(report.get("source_checkpoint", ""))).resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError("Existing package source checkpoint is invalid") from error
    if recorded_checkpoint != checkpoint:
        raise ValueError("Existing package belongs to a different source checkpoint")
    training_contract = validate_training_contract(args)
    if report.get("training_contract") != training_contract:
        raise ValueError("Existing package training contract drifted")
    if not same_model_reference(report.get("base_model"), args.base_model):
        raise ValueError("Existing package belongs to a different base model")
    if report.get("base_revision") != args.base_revision:
        raise ValueError("Existing package belongs to a different base revision")
    if report.get("upstream_base_models") != resolve_base_lineage(
        args.base_model, args.base_revision
    ):
        raise ValueError("Existing package upstream lineage drifted")
    validate_sentence_transformers_contract(output)
    output_sha = hash_model_files(output)
    if report.get("model", {}).get("weights_sha256") != output_sha:
        raise ValueError("Existing package model shards drifted from evidence")
    if hash_model_files(checkpoint) != output_sha:
        raise ValueError("Existing package no longer matches source checkpoint shards")
    return {
        "status": "pass",
        "artifact_type": "validated-existing-full-model-package",
        "source_checkpoint": str(checkpoint),
        "model_weights_sha256": output_sha,
        "training_contract_sha256": (
            training_contract["sha256"] if training_contract is not None else None
        ),
    }


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
    training_contract = validate_training_contract(args)
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
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "upstream_base_models": resolve_base_lineage(
            args.base_model, args.base_revision
        ),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "training_contract": training_contract,
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
    if args.validate_existing:
        print(json.dumps(validate_existing_package(args), ensure_ascii=False, indent=2))
        return
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
