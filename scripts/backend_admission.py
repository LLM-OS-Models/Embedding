#!/usr/bin/env python3
"""Create and validate fail-closed training-backend admission contracts.

An ``admitted`` boolean only applies to the exact workload and runtime that
was actually probed.  This module keeps contract construction and validation
in one place so queue scripts cannot silently reuse an unrelated FA2 result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 2
WORKLOAD_CONTRACT = "embedding-fa2-training-workload-v1"
RUNTIME_CONTRACT = "embedding-fa2-runtime-v1"
MINIMUM_REQUIRED_SPEEDUP = 1.05
MINIMUM_PROBE_STEPS = 5


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256_bytes(encoded)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sampled_weight_identity(path: Path) -> dict[str, Any]:
    """Fingerprint large weight files without hashing every multi-GB shard."""

    size = path.stat().st_size
    sample_size = 1024 * 1024
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(sample_size))
        if size > sample_size:
            handle.seek(max(0, size - sample_size))
            digest.update(handle.read(sample_size))
    return {
        "size_bytes": size,
        "edge_sample_sha256": digest.hexdigest(),
    }


def base_identity(base_model: str, base_revision: str) -> dict[str, Any]:
    candidate = Path(base_model).expanduser()
    if candidate.exists():
        resolved = candidate.resolve()
        if not resolved.is_dir():
            raise ValueError(f"Local base model must be a directory: {resolved}")
        identity_files: dict[str, str] = {}
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "merge_report.json",
            "full_tuning_report.json",
        ):
            path = resolved / name
            if path.is_file():
                identity_files[name] = sha256_file(path)
        weights = {
            path.name: _sampled_weight_identity(path)
            for path in sorted(resolved.glob("*.safetensors"))
            if path.is_file()
        }
        if not identity_files and not weights:
            raise ValueError(
                f"Local base model has no fingerprintable config/report/weights: {resolved}"
            )
        return {
            "kind": "local",
            "path": str(resolved),
            "revision": base_revision,
            "identity_files_sha256": identity_files,
            "weight_shards": weights,
        }
    if not base_model or "/" not in base_model:
        raise ValueError(f"Remote base model must be a Hub ID: {base_model!r}")
    if not base_revision:
        raise ValueError("Remote base model admission requires an exact revision")
    return {
        "kind": "huggingface",
        "model_id": base_model,
        "revision": base_revision,
    }


def build_workload_contract(
    *,
    train_file: Path,
    backend: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    dtype: str,
    base_model: str,
    base_revision: str,
    hard_negatives: int,
    train_sha256: str | None = None,
) -> dict[str, Any]:
    train_file = train_file.expanduser().resolve()
    if not train_file.is_file():
        raise FileNotFoundError(f"Training workload does not exist: {train_file}")
    for name, value in (
        ("batch_size", batch_size),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
        ("max_length", max_length),
        ("lora_rank", lora_rank),
        ("lora_alpha", lora_alpha),
        ("hard_negatives", hard_negatives),
    ):
        _positive_integer(value, name)
    if backend != "flash_attention_2":
        raise ValueError("This admission contract only admits flash_attention_2")
    if dtype not in {"bfloat16", "float16"}:
        raise ValueError(f"Unsupported FA2 training dtype: {dtype}")
    if not 0 <= lora_dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    return {
        "contract": WORKLOAD_CONTRACT,
        "backend": backend,
        "train_file": str(train_file),
        "train_sha256": train_sha256 or sha256_file(train_file),
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_length": max_length,
        "tuner_type": "lora",
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": "all-linear",
        "dtype": dtype,
        "gradient_checkpointing": True,
        "lazy_tokenize": True,
        "load_from_cache_file": False,
        "dataset_shuffle": False,
        "train_dataloader_shuffle": False,
        "strict": True,
        "loss_type": "infonce",
        "infonce_use_batch": True,
        "infonce_hard_negatives": hard_negatives,
        "world_size": 1,
        "base": base_identity(base_model, base_revision),
    }


def _module_version(module_name: str) -> tuple[str, str]:
    module = importlib.import_module(module_name)
    version = str(getattr(module, "__version__", "unknown"))
    location = str(Path(getattr(module, "__file__", "unknown")).resolve())
    return version, location


def _git_identity(path: Path) -> dict[str, Any] | None:
    try:
        root = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        head = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        diff = subprocess.check_output(
            ["git", "-C", root, "diff", "--binary", "HEAD", "--"],
            stderr=subprocess.DEVNULL,
        )
        status = subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain=v1", "--untracked-files=no"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return {
        "root": str(Path(root).resolve()),
        "head": head,
        "tracked_diff_sha256": sha256_bytes(diff),
        "tracked_status_sha256": sha256_bytes(status),
    }


def collect_runtime_fingerprint() -> dict[str, Any]:
    import torch

    flash_version, flash_location = _module_version("flash_attn")
    swift_version, swift_location = _module_version("swift")
    transformers_version, transformers_location = _module_version("transformers")
    cuda_available = bool(torch.cuda.is_available())
    device: dict[str, Any] | None = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        device = {
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "multi_processor_count": properties.multi_processor_count,
        }
    return {
        "contract": RUNTIME_CONTRACT,
        "python_version": platform.python_version(),
        "python_executable": os.path.abspath(sys.executable),
        "python_prefix": os.path.abspath(sys.prefix),
        "platform": platform.platform(),
        "packages": {
            "torch": str(torch.__version__),
            "torch_git_version": str(getattr(torch.version, "git_version", None)),
            "torch_cuda": str(torch.version.cuda),
            "torch_cxx11_abi": bool(
                getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", False)
            ),
            "flash_attn": flash_version,
            "flash_attn_module": flash_location,
            "swift": swift_version,
            "swift_module": swift_location,
            "transformers": transformers_version,
            "transformers_module": transformers_location,
        },
        "swift_git": _git_identity(Path(swift_location).parent),
        "cuda_available": cuda_available,
        "cuda_device": device,
        "runtime_environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", ""
            ),
        },
    }


def validate_admission_report(
    report: Mapping[str, Any],
    *,
    expected_contract: Mapping[str, Any],
    current_runtime: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version={report.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )
    if report.get("admitted") is not True:
        errors.append("report is not admitted")
    if report.get("real_8b_backward_probe") is not True:
        errors.append("real_8b_backward_probe is not true")
    if report.get("process_status") != 0:
        errors.append("FA2 probe process did not exit 0")
    if report.get("matched_sdpa_process_status") != 0:
        errors.append("matched SDPA probe process did not exit 0")
    if report.get("baseline_source") != "matched_subset_same_environment":
        errors.append("baseline is not a matched SDPA measurement")
    if not isinstance(report.get("probe_steps"), int) or report.get(
        "probe_steps", 0
    ) < MINIMUM_PROBE_STEPS:
        errors.append("probe_steps is below the admission minimum")
    baseline = report.get("baseline_sdpa_seconds_per_step")
    measured = report.get("measured_seconds_per_step")
    required_speedup = report.get("required_speedup")
    threshold = report.get("admission_threshold_seconds_per_step")
    numeric = (baseline, measured, required_speedup, threshold)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
        for value in numeric
    ):
        errors.append("admission timing evidence is missing or invalid")
    else:
        expected_threshold = baseline / required_speedup
        if required_speedup < MINIMUM_REQUIRED_SPEEDUP:
            errors.append("required speedup is below project policy")
        if abs(threshold - expected_threshold) > 1e-9:
            errors.append("admission threshold is inconsistent with SDPA timing")
        if measured > expected_threshold:
            errors.append("FA2 timing does not satisfy the recorded speedup gate")
    recorded_contract = report.get("workload_contract")
    if recorded_contract != expected_contract:
        errors.append("workload contract mismatch")
    expected_contract_sha = canonical_sha256(expected_contract)
    if report.get("workload_contract_sha256") != expected_contract_sha:
        errors.append("workload contract SHA256 mismatch")
    recorded_runtime = report.get("runtime_fingerprint")
    if recorded_runtime != current_runtime:
        errors.append("runtime fingerprint mismatch")
    expected_runtime_sha = canonical_sha256(current_runtime)
    if report.get("runtime_fingerprint_sha256") != expected_runtime_sha:
        errors.append("runtime fingerprint SHA256 mismatch")
    if report.get("backend") != expected_contract.get("backend"):
        errors.append("top-level backend does not match contract")
    return errors


def _contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--backend", default="flash_attention_2")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--lora-alpha", type=int, required=True)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision", default="")
    parser.add_argument("--hard-negatives", type=int, required=True)


def _contract_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return build_workload_contract(
        train_file=args.train_file,
        backend=args.backend,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dtype=args.dtype,
        base_model=args.base_model,
        base_revision=args.base_revision,
        hard_negatives=args.hard_negatives,
    )


def _read_runtime(args: argparse.Namespace) -> dict[str, Any]:
    if args.runtime_json:
        value = json.loads(args.runtime_json)
        if not isinstance(value, dict):
            raise ValueError("--runtime-json must contain a JSON object")
        return value
    return collect_runtime_fingerprint()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--report", type=Path, required=True)
    check.add_argument("--runtime-json")
    check.add_argument("--quiet", action="store_true")
    _contract_arguments(check)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = _contract_from_args(args)
    runtime = _read_runtime(args)
    if not args.report.is_file():
        if not args.quiet:
            print(f"missing admission report: {args.report}", file=sys.stderr)
        return 1
    report = json.loads(args.report.read_text(encoding="utf-8"))
    errors = validate_admission_report(
        report, expected_contract=contract, current_runtime=runtime
    )
    if errors:
        if not args.quiet:
            print(json.dumps({"admitted": False, "errors": errors}, indent=2))
        return 1
    if not args.quiet:
        print(
            json.dumps(
                {
                    "admitted": True,
                    "report": str(args.report.resolve()),
                    "workload_contract_sha256": canonical_sha256(contract),
                    "runtime_fingerprint_sha256": canonical_sha256(runtime),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
