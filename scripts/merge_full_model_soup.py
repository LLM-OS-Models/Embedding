#!/usr/bin/env python3
"""Create a basis-safe weighted soup from compatible merged embedding models.

Unlike averaging LoRA A/B factors across independently trained adapters, this
tool averages the already folded full transformer weights.  Tensors are read
from safetensors, accumulated in FP32, emitted in BF16 shard-by-shard, and bound
to exact source/output hashes.  Metadata comes from an explicit reference
model only after architecture, SentenceTransformers contract, key, shape, and
dtype compatibility pass for every source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.merge_embedding_adapter import validate_sentence_transformers_contract
except ImportError:  # pragma: no cover - direct script execution fallback
    from merge_embedding_adapter import validate_sentence_transformers_contract


ARTIFACT_TYPE = "weighted-full-model-embedding-soup"
REPORT_NAME = "soup_report.json"
INDEX_NAME = "model.safetensors.index.json"
SINGLE_WEIGHTS_NAME = "model.safetensors"
EVIDENCE_NAMES = (
    "merge_report.json",
    "full_tuning_report.json",
    REPORT_NAME,
)
EXCLUDED_METADATA_NAMES = {
    "README.md",
    "merge_report.json",
    "full_tuning_report.json",
    REPORT_NAME,
    "publication_manifest.json",
    "private_candidate_manifest.json",
}


@dataclass(frozen=True)
class SourceModel:
    root: Path
    weight: float
    evidence_path: Path
    evidence_sha256: str
    model_weights_sha256: str
    weight_map: dict[str, str]
    shards: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--weight", type=float, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_weights_sha256(root: Path, shards: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in sorted(shards):
        path = root / name
        digest.update(name.encode() + b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def model_evidence(root: Path) -> tuple[Path, dict[str, Any]]:
    present = [root / name for name in EVIDENCE_NAMES if (root / name).is_file()]
    if len(present) != 1:
        raise ValueError("Each soup source must contain exactly one model evidence report")
    evidence = read_json(present[0])
    if evidence.get("status") != "pass":
        raise ValueError("A soup source has no passing model evidence")
    contract = evidence.get("sentence_transformers_contract", {})
    if contract.get("pooling") != "last_token" or contract.get("normalize") is not True:
        raise ValueError("A soup source has an incompatible embedding contract")
    return present[0], evidence


def safetensors_layout(root: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("safetensors is required") from error
    index_path = root / INDEX_NAME
    if index_path.is_file():
        index = read_json(index_path)
        raw_map = index.get("weight_map")
        if not isinstance(raw_map, dict) or not raw_map:
            raise ValueError("Invalid safetensors shard index")
        weight_map = {}
        for key, shard in raw_map.items():
            if not isinstance(key, str) or not key or not isinstance(shard, str):
                raise ValueError("Invalid safetensors shard mapping")
            if Path(shard).name != shard or not shard.endswith(".safetensors"):
                raise ValueError("Unsafe safetensors shard name")
            weight_map[key] = shard
        shards = tuple(sorted(set(weight_map.values())))
    elif (root / SINGLE_WEIGHTS_NAME).is_file():
        shards = (SINGLE_WEIGHTS_NAME,)
        with safe_open(str(root / SINGLE_WEIGHTS_NAME), framework="pt", device="cpu") as handle:
            weight_map = {key: SINGLE_WEIGHTS_NAME for key in handle.keys()}
    else:
        raise FileNotFoundError(f"No model safetensors under {root}")

    observed: dict[str, str] = {}
    for shard in shards:
        path = root / shard
        if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
            raise ValueError(f"Missing or unsafe model shard: {shard}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in observed:
                    raise ValueError(f"Duplicate tensor across shards: {key}")
                observed[key] = shard
    if observed != weight_map:
        raise ValueError("Safetensors index does not match shard payloads")
    return weight_map, shards


def normalized_config(path: Path) -> dict[str, Any]:
    value = read_json(path)
    for key in ("_name_or_path", "transformers_version", "torch_dtype"):
        value.pop(key, None)
    return value


def contract_files(root: Path) -> dict[str, bytes]:
    names = (
        "modules.json",
        "config_sentence_transformers.json",
        "1_Pooling/config.json",
    )
    values = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing SentenceTransformers metadata: {name}")
        values[name] = path.read_bytes()
    return values


def validate_weights(weights: list[float], model_count: int) -> None:
    if len(weights) != model_count or model_count < 2:
        raise ValueError("Provide matching --model/--weight values for at least two models")
    if any(not math.isfinite(weight) or weight <= 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("Soup weights must be finite and within (0, 1]")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Soup weights must sum exactly to 1 within 1e-9")


def validate_sources(models: list[Path], weights: list[float]) -> list[SourceModel]:
    validate_weights(weights, len(models))
    roots = [model.expanduser().resolve() for model in models]
    if len(set(roots)) != len(roots):
        raise ValueError("Soup source models must be distinct")
    reference_config: dict[str, Any] | None = None
    reference_contract: dict[str, bytes] | None = None
    reference_keys: set[str] | None = None
    sources = []
    for root, weight in zip(roots, weights, strict=True):
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"Soup source is missing or unsafe: {root}")
        evidence_path, evidence = model_evidence(root)
        config = normalized_config(root / "config.json")
        contract = contract_files(root)
        weight_map, shards = safetensors_layout(root)
        if reference_config is None:
            reference_config = config
            reference_contract = contract
            reference_keys = set(weight_map)
        elif config != reference_config:
            raise ValueError("Soup source transformer configurations differ")
        elif contract != reference_contract:
            raise ValueError("Soup source SentenceTransformers metadata differs")
        elif set(weight_map) != reference_keys:
            raise ValueError("Soup source tensor key sets differ")
        actual_sha = model_weights_sha256(root, shards)
        if evidence.get("model", {}).get("weights_sha256") != actual_sha:
            raise ValueError("Soup source shards do not match model evidence")
        sources.append(
            SourceModel(
                root=root,
                weight=weight,
                evidence_path=evidence_path,
                evidence_sha256=sha256_file(evidence_path),
                model_weights_sha256=actual_sha,
                weight_map=weight_map,
                shards=shards,
            )
        )
    return sources


def copy_reference_metadata(reference: Path, staging: Path) -> None:
    for child in sorted(reference.iterdir()):
        if child.name in EXCLUDED_METADATA_NAMES or child.name == "evaluation":
            continue
        if child.name == INDEX_NAME or (
            child.name.startswith("model") and child.name.endswith(".safetensors")
        ):
            continue
        if child.is_symlink():
            raise ValueError(f"Reference metadata contains a symlink: {child.name}")
        destination = staging / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        elif child.is_file():
            shutil.copy2(child, destination)
        else:
            raise ValueError(f"Reference metadata is non-regular: {child.name}")


def average_shards(
    sources: list[SourceModel], staging: Path, *, output_dtype: str, torch_threads: int
) -> dict[str, Any]:
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("torch and safetensors are required") from error
    if torch_threads < 1 or torch_threads > 32:
        raise ValueError("--torch-threads must be within [1, 32]")
    torch.set_num_threads(torch_threads)
    emitted_dtype = torch.bfloat16 if output_dtype == "bfloat16" else torch.float32
    reference = sources[0]
    by_reference_shard: dict[str, list[str]] = {name: [] for name in reference.shards}
    for key, shard in reference.weight_map.items():
        by_reference_shard[shard].append(key)

    tensor_count = 0
    parameter_count = 0
    tensor_bytes = 0
    input_dtypes: set[str] = set()
    shard_sizes: dict[str, int] = {}
    with ExitStack() as stack:
        handles: list[dict[str, Any]] = []
        for source in sources:
            handles.append(
                {
                    shard: stack.enter_context(
                        safe_open(str(source.root / shard), framework="pt", device="cpu")
                    )
                    for shard in source.shards
                }
            )
        for shard in reference.shards:
            output_tensors: dict[str, Any] = {}
            for key in sorted(by_reference_shard[shard]):
                tensors = [
                    handles[index][source.weight_map[key]].get_tensor(key)
                    for index, source in enumerate(sources)
                ]
                first = tensors[0]
                for index, tensor in enumerate(tensors[1:], start=1):
                    if tensor.shape != first.shape:
                        raise ValueError(f"Tensor shape drift for {key!r} at source {index}")
                    if tensor.dtype != first.dtype:
                        raise ValueError(f"Tensor dtype drift for {key!r} at source {index}")
                input_dtypes.add(str(first.dtype).removeprefix("torch."))
                if torch.is_floating_point(first):
                    accumulator = first.to(dtype=torch.float32).mul_(sources[0].weight)
                    if not torch.isfinite(accumulator).all():
                        raise ValueError(f"Non-finite source tensor: {key}")
                    for tensor, source in zip(tensors[1:], sources[1:], strict=True):
                        value = tensor.to(dtype=torch.float32)
                        if not torch.isfinite(value).all():
                            raise ValueError(f"Non-finite source tensor: {key}")
                        accumulator.add_(value, alpha=source.weight)
                    if not torch.isfinite(accumulator).all():
                        raise ValueError(f"Non-finite soup tensor: {key}")
                    output = accumulator.to(dtype=emitted_dtype).contiguous()
                else:
                    if any(not torch.equal(first, tensor) for tensor in tensors[1:]):
                        raise ValueError(f"Non-floating tensor drift for {key!r}")
                    output = first.clone().contiguous()
                output_tensors[key] = output
                tensor_count += 1
                parameter_count += output.numel()
                tensor_bytes += output.numel() * output.element_size()
            output_path = staging / shard
            save_file(output_tensors, str(output_path), metadata={"format": "pt"})
            shard_sizes[shard] = output_path.stat().st_size
            del output_tensors
    return {
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "tensor_bytes": tensor_bytes,
        "input_dtypes": sorted(input_dtypes),
        "accumulation_dtype": "float32",
        "output_floating_dtype": output_dtype,
        "torch_threads": torch_threads,
        "shard_sizes": shard_sizes,
    }


def build_soup(args: argparse.Namespace) -> dict[str, Any]:
    sources = validate_sources(args.model, args.weight)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.soup-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        copy_reference_metadata(sources[0].root, staging)
        tensor_report = average_shards(
            sources,
            staging,
            output_dtype=args.output_dtype,
            torch_threads=args.torch_threads,
        )
        if len(sources[0].shards) > 1:
            write_json(
                staging / INDEX_NAME,
                {
                    "metadata": {"total_size": tensor_report["tensor_bytes"]},
                    "weight_map": sources[0].weight_map,
                },
            )
        contract = validate_sentence_transformers_contract(staging)
        output_sha = model_weights_sha256(staging, sources[0].shards)
        report = {
            "schema_version": 1,
            "artifact_type": ARTIFACT_TYPE,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "training_method": "weighted-full-model-soup",
            "sources": [
                {
                    "model": str(source.root),
                    "weight": source.weight,
                    "weights_sha256": source.model_weights_sha256,
                    "evidence_file": source.evidence_path.name,
                    "evidence_sha256": source.evidence_sha256,
                }
                for source in sources
            ],
            "soup": {
                "method": "weighted_arithmetic_mean",
                "weight_sum": sum(source.weight for source in sources),
                **tensor_report,
            },
            "model": {
                "weights_sha256": output_sha,
                "shards": {
                    name: {
                        "sha256": sha256_file(staging / name),
                        "size_bytes": (staging / name).stat().st_size,
                    }
                    for name in sources[0].shards
                },
            },
            "sentence_transformers_contract": contract,
            "environment": {
                "python": sys.version,
                "output_dtype": args.output_dtype,
            },
        }
        write_json(staging / REPORT_NAME, report)
        os.replace(staging, output_dir)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    report = build_soup(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
