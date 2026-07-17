#!/usr/bin/env python3
"""Fail closed unless a Trainer checkpoint matches the requested run contract."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"checkpoint-([1-9][0-9]*)$")


def parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision", default="")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--grad-accum-steps", type=int, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--lora-alpha", type=float, required=True)
    parser.add_argument("--lora-dropout", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--loss-type", required=True)
    parser.add_argument("--dataset-shuffle", type=parse_bool, required=True)
    parser.add_argument("--train-dataloader-shuffle", type=parse_bool, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 2:
        raise ValueError("Resume metadata is missing or unsafe")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Resume metadata is not an object")
    return value


def one_resolved_path(value: Any) -> Path:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError("Resume dataset declaration is malformed")
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("Resume dataset declaration must contain one path")
    return Path(values[0]).expanduser().resolve()


def same_model(actual: Any, expected: str, revision: str) -> bool:
    if not isinstance(actual, str) or not actual:
        return False
    if actual.rstrip("/") == expected.rstrip("/"):
        return True
    if Path(expected).is_absolute():
        return Path(actual).expanduser().resolve() == Path(expected).resolve()
    actual_path = Path(actual).expanduser()
    return (
        bool(revision)
        and actual_path.name == revision
        and actual_path.parent.name == "snapshots"
    )


def require_equal(actual: Any, expected: Any, field: str) -> None:
    if isinstance(expected, float):
        try:
            matches = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            matches = False
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(f"Resume training contract mismatch: {field}")


def validate(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if (
        args.checkpoint.is_symlink()
        or not checkpoint.is_dir()
        or not checkpoint.is_relative_to(run_dir)
    ):
        raise ValueError("Resume checkpoint is outside the run directory")
    match = STEP_RE.fullmatch(checkpoint.name)
    if not match:
        raise ValueError("Resume checkpoint name is invalid")
    step = int(match.group(1))
    state = load_json(checkpoint / "trainer_state.json")
    training = load_json(checkpoint / "args.json")
    require_equal(state.get("global_step"), step, "global_step")
    require_equal(state.get("max_steps"), args.max_steps, "max_steps")
    if one_resolved_path(training.get("dataset")) != args.train_file.resolve():
        raise ValueError("Resume training contract mismatch: dataset")
    if one_resolved_path(training.get("val_dataset")) != args.val_file.resolve():
        raise ValueError("Resume training contract mismatch: val_dataset")
    if not same_model(training.get("model"), args.base_model, args.base_revision):
        raise ValueError("Resume training contract mismatch: model")
    actual_revision = training.get("model_revision") or ""
    require_equal(actual_revision, args.base_revision, "model_revision")
    expected = {
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "max_length": args.max_length,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
        "loss_type": args.loss_type,
        "dataset_shuffle": args.dataset_shuffle,
        "train_dataloader_shuffle": args.train_dataloader_shuffle,
        "seed": args.seed,
    }
    for field, value in expected.items():
        require_equal(training.get(field), value, field)
    return step


def main() -> None:
    args = parse_args()
    step = validate(args)
    print(json.dumps({"status": "pass", "resume_step": step}, sort_keys=True))


if __name__ == "__main__":
    main()
