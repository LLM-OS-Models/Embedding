#!/usr/bin/env python3
"""Average compatible LoRA checkpoints from one training trajectory in FP32.

Checkpoint averaging is intentionally performed on adapter parameters, before
the normal safe PEFT fold into the base model.  The tool fails closed on mixed
training versions, configuration/key/shape drift, non-finite tensors, symlinks,
or disqualified runs.  Outputs are staged and atomically renamed so a restart
can never mistake a partial adapter for a candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CHECKPOINT_RE = re.compile(r"checkpoint-([1-9][0-9]*)$")
WEIGHTS_NAME = "adapter_model.safetensors"
CONFIG_NAME = "adapter_config.json"
REPORT_NAME = "average_report.json"
ARTIFACT_TYPE = "fp32-lora-checkpoint-average"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average the latest compatible LoRA checkpoints in the training "
            "version containing an anchor checkpoint."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--last-n", type=int, default=5)
    parser.add_argument("--minimum-checkpoints", type=int, default=2)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid checkpoint directory name: {path}")
    return int(match.group(1))


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def find_disqualification_marker(path: Path, stop: Path) -> Path | None:
    current = path
    while True:
        marker = current / "DISQUALIFIED.json"
        if marker.is_file() and marker.stat().st_size > 0:
            return marker
        if current == stop:
            return None
        if current.parent == current or not is_relative_to(current.parent, stop):
            return None
        current = current.parent


def validate_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required checkpoint file is missing or unsafe: {path}")


def complete_checkpoint(path: Path) -> bool:
    return (
        path.is_dir()
        and not path.is_symlink()
        and (path / WEIGHTS_NAME).is_file()
        and not (path / WEIGHTS_NAME).is_symlink()
        and (path / CONFIG_NAME).is_file()
        and not (path / CONFIG_NAME).is_symlink()
    )


def select_checkpoints(
    *,
    run_dir: Path,
    anchor_checkpoint: Path,
    last_n: int,
    minimum_checkpoints: int,
) -> list[Path]:
    if last_n < 2:
        raise ValueError("--last-n must be at least 2")
    if minimum_checkpoints < 2 or minimum_checkpoints > last_n:
        raise ValueError("--minimum-checkpoints must be between 2 and --last-n")
    run_dir = run_dir.expanduser().resolve()
    anchor_checkpoint = anchor_checkpoint.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not is_relative_to(anchor_checkpoint, run_dir):
        raise ValueError("Anchor checkpoint escapes the run directory")
    if not complete_checkpoint(anchor_checkpoint):
        raise ValueError(f"Anchor checkpoint is incomplete: {anchor_checkpoint}")
    checkpoint_step(anchor_checkpoint)
    marker = find_disqualification_marker(anchor_checkpoint, run_dir)
    if marker is not None:
        raise RuntimeError(f"Refusing to average a disqualified run: {marker}")

    # Averaging across retry/version directories would mix separate optimizer
    # trajectories.  Restrict discovery to the anchor's exact parent.
    candidates = sorted(
        (
            path.resolve()
            for path in anchor_checkpoint.parent.glob("checkpoint-*")
            if complete_checkpoint(path)
        ),
        key=lambda path: (checkpoint_step(path), str(path)),
    )
    if len(candidates) < minimum_checkpoints:
        raise ValueError(
            f"Only {len(candidates)} complete checkpoints are available; "
            f"need at least {minimum_checkpoints}"
        )
    selected = candidates[-last_n:]
    if anchor_checkpoint not in candidates:
        raise ValueError("Anchor checkpoint disappeared during discovery")
    return selected


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_configs(checkpoints: Iterable[Path]) -> tuple[bytes, str, dict[str, Any]]:
    checkpoints = list(checkpoints)
    config_paths = [checkpoint / CONFIG_NAME for checkpoint in checkpoints]
    for path in config_paths:
        validate_regular_file(path)
    config_bytes = config_paths[0].read_bytes()
    if any(path.read_bytes() != config_bytes for path in config_paths[1:]):
        raise ValueError("Adapter configurations differ across checkpoints")
    config = load_json_object(config_paths[0])
    if config.get("peft_type") != "LORA":
        raise ValueError("Only PEFT LoRA checkpoints can be averaged")
    rank = config.get("r")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("Adapter rank must be a positive integer")
    return config_bytes, hashlib.sha256(config_bytes).hexdigest(), config


def average_safetensors(
    source_paths: list[Path], output_path: Path
) -> dict[str, Any]:
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - production dependency guard
        raise RuntimeError("torch and safetensors are required") from error

    for path in source_paths:
        validate_regular_file(path)
    output_tensors: dict[str, Any] = {}
    input_dtypes: set[str] = set()
    floating_tensor_count = 0
    nonfloating_tensor_count = 0
    with ExitStack() as stack:
        handles = [
            stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
            for path in source_paths
        ]
        keys = list(handles[0].keys())
        if not keys:
            raise ValueError("Adapter weights contain no tensors")
        expected_keys = set(keys)
        for index, handle in enumerate(handles[1:], start=1):
            if set(handle.keys()) != expected_keys:
                raise ValueError(f"Tensor key drift in checkpoint index {index}")

        for key in sorted(keys):
            tensors = [handle.get_tensor(key) for handle in handles]
            reference = tensors[0]
            for index, tensor in enumerate(tensors[1:], start=1):
                if tensor.shape != reference.shape:
                    raise ValueError(
                        f"Tensor shape drift for {key!r} in checkpoint index {index}"
                    )
                if tensor.dtype != reference.dtype:
                    raise ValueError(
                        f"Tensor dtype drift for {key!r} in checkpoint index {index}"
                    )
            input_dtypes.add(str(reference.dtype).removeprefix("torch."))
            if torch.is_floating_point(reference):
                accumulator = reference.to(dtype=torch.float32)
                if not torch.isfinite(accumulator).all():
                    raise ValueError(f"Non-finite source tensor: {key}")
                for tensor in tensors[1:]:
                    value = tensor.to(dtype=torch.float32)
                    if not torch.isfinite(value).all():
                        raise ValueError(f"Non-finite source tensor: {key}")
                    accumulator.add_(value)
                accumulator.div_(len(tensors))
                if not torch.isfinite(accumulator).all():
                    raise ValueError(f"Non-finite averaged tensor: {key}")
                output_tensors[key] = accumulator.contiguous()
                floating_tensor_count += 1
            else:
                if any(not torch.equal(reference, tensor) for tensor in tensors[1:]):
                    raise ValueError(f"Non-floating tensor drift for {key!r}")
                output_tensors[key] = reference.clone().contiguous()
                nonfloating_tensor_count += 1

    save_file(
        output_tensors,
        str(output_path),
        metadata={
            "artifact_type": ARTIFACT_TYPE,
            "averaging_dtype": "float32",
            "checkpoint_count": str(len(source_paths)),
        },
    )
    return {
        "tensor_count": len(output_tensors),
        "floating_tensor_count": floating_tensor_count,
        "nonfloating_tensor_count": nonfloating_tensor_count,
        "input_dtypes": sorted(input_dtypes),
        "output_floating_dtype": "float32",
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_average(
    *,
    run_dir: Path,
    anchor_checkpoint: Path,
    output_dir: Path,
    last_n: int,
    minimum_checkpoints: int,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    anchor_checkpoint = anchor_checkpoint.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    selected = select_checkpoints(
        run_dir=run_dir,
        anchor_checkpoint=anchor_checkpoint,
        last_n=last_n,
        minimum_checkpoints=minimum_checkpoints,
    )
    config_bytes, config_sha256, config = validate_configs(selected)
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.averaging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        output_weights = staging / WEIGHTS_NAME
        tensor_report = average_safetensors(
            [checkpoint / WEIGHTS_NAME for checkpoint in selected], output_weights
        )
        (staging / CONFIG_NAME).write_bytes(config_bytes)
        sources = []
        for checkpoint in selected:
            weight_path = checkpoint / WEIGHTS_NAME
            sources.append(
                {
                    "checkpoint": str(checkpoint),
                    "step": checkpoint_step(checkpoint),
                    "weights_sha256": sha256_file(weight_path),
                    "weights_bytes": weight_path.stat().st_size,
                }
            )
        report = {
            "schema_version": 1,
            "artifact_type": ARTIFACT_TYPE,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "run_dir": str(run_dir),
            "training_version_dir": str(selected[0].parent),
            "anchor_checkpoint": str(anchor_checkpoint),
            "selection": {
                "policy": "latest_available_same_training_trajectory",
                "requested_last_n": last_n,
                "minimum_checkpoints": minimum_checkpoints,
                "checkpoint_count": len(selected),
                "steps": [checkpoint_step(path) for path in selected],
            },
            "sources": sources,
            "adapter_config": {
                "sha256": config_sha256,
                "peft_type": config.get("peft_type"),
                "rank": config.get("r"),
                "lora_alpha": config.get("lora_alpha"),
                "base_model_name_or_path": config.get("base_model_name_or_path"),
            },
            "averaging": {
                "method": "arithmetic_mean",
                "accumulation_dtype": "float32",
                **tensor_report,
            },
            "output": {
                "weights_filename": WEIGHTS_NAME,
                "weights_sha256": sha256_file(output_weights),
                "weights_bytes": output_weights.stat().st_size,
                "config_filename": CONFIG_NAME,
                "config_sha256": config_sha256,
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
    report = build_average(
        run_dir=args.run_dir,
        anchor_checkpoint=args.anchor_checkpoint,
        output_dir=args.output_dir,
        last_n=args.last_n,
        minimum_checkpoints=args.minimum_checkpoints,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
