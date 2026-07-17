#!/usr/bin/env python3
"""Select the complete Nemotron public LoRA checkpoint with minimum heldout loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_MODEL = "nvidia/Nemotron-3-Embed-8B-BF16"
BASE_REVISION = "2b29550c4ab0646bb6bb47032dda54ea11f6dfe2"
DEFAULT_STEPS = (50, 100, 150, 200, 250, 300)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-steps",
        default=",".join(map(str, DEFAULT_STEPS)),
        help="Comma-separated complete checkpoint steps required for selection",
    )
    return parser.parse_args()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_steps(value: str) -> tuple[int, ...]:
    fields = value.split(",")
    if not fields or any(not re.fullmatch(r"[1-9][0-9]*", field) for field in fields):
        raise ValueError("--expected-steps must be comma-separated positive integers")
    steps = tuple(int(field) for field in fields)
    if tuple(sorted(set(steps))) != steps:
        raise ValueError("--expected-steps must be unique and increasing")
    return steps


def same_step_eval_loss(state: dict[str, Any], step: int) -> float:
    if state.get("global_step") != step:
        raise ValueError(f"checkpoint-{step} trainer global_step mismatch")
    history = state.get("log_history")
    if not isinstance(history, list):
        raise ValueError(f"checkpoint-{step} has no log_history")
    losses = [
        float(row["eval_loss"])
        for row in history
        if isinstance(row, dict)
        and row.get("step") == step
        and isinstance(row.get("eval_loss"), (int, float))
        and not isinstance(row.get("eval_loss"), bool)
        and math.isfinite(float(row["eval_loss"]))
    ]
    if not losses:
        raise ValueError(f"checkpoint-{step} has no finite same-step eval_loss")
    return losses[-1]


def select(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    manifest_path = args.training_manifest.resolve()
    run_contract_path = run_dir / "run_contract.json"
    completion_path = run_dir / "training-complete.json"
    for path in (run_contract_path, completion_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = read_object(run_contract_path)
    completion = read_object(completion_path)
    manifest = read_object(manifest_path)
    if contract.get("base_model") != BASE_MODEL or contract.get("base_revision") != BASE_REVISION:
        raise ValueError("Nemotron base contract drifted")
    if completion.get("status") != "complete":
        raise ValueError("Training completion marker did not pass")
    if manifest.get("release_eligible") is not True or manifest.get("release_blockers"):
        raise ValueError("Training manifest is not release eligible")
    if manifest.get("visibility") != "public":
        raise ValueError("Training manifest is not public")
    if contract.get("training_data", {}).get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("Run contract belongs to a different training manifest")

    candidates: list[dict[str, Any]] = []
    for step in parse_steps(args.expected_steps):
        checkpoint = run_dir / f"checkpoint-{step}"
        required = (
            checkpoint / "adapter_model.safetensors",
            checkpoint / "adapter_config.json",
            checkpoint / "trainer_state.json",
            checkpoint / "optimizer.pt",
            checkpoint / "scheduler.pt",
        )
        if any(not path.is_file() or path.stat().st_size < 1 for path in required):
            raise ValueError(f"checkpoint-{step} is incomplete")
        adapter_config = read_object(checkpoint / "adapter_config.json")
        base_reference = str(adapter_config.get("base_model_name_or_path", ""))
        if BASE_REVISION not in base_reference and base_reference.rstrip("/") != BASE_MODEL:
            raise ValueError(f"checkpoint-{step} adapter base reference drifted")
        candidates.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "eval_loss": same_step_eval_loss(
                    read_object(checkpoint / "trainer_state.json"), step
                ),
                "adapter_weights_sha256": sha256(
                    checkpoint / "adapter_model.safetensors"
                ),
                "adapter_config_sha256": sha256(checkpoint / "adapter_config.json"),
            }
        )
    winner = min(candidates, key=lambda row: (row["eval_loss"], row["step"]))
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "selection_signal": "minimum finite same-step independent heldout eval_loss",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "training_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
            "visibility": "public",
            "release_eligible": True,
        },
        "candidates": candidates,
        "selected": winner,
        "public_benchmark_used_for_selection": False,
    }


def main() -> None:
    args = parse_args()
    result = select(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
