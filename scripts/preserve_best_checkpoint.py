#!/usr/bin/env python3
"""Preserve the best checkpoint from an already-running Trainer process."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

try:
    from select_best_checkpoint import (
        checkpoint_step,
        complete_checkpoint,
        evaluation_losses,
    )
except ModuleNotFoundError:
    from scripts.select_best_checkpoint import (
        checkpoint_step,
        complete_checkpoint,
        evaluation_losses,
    )


ESSENTIAL_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "additional_config.json",
    "README.md",
    "args.json",
    "training_args.bin",
    "trainer_state.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--watch-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser.parse_args()


def process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def checkpoint_loss(path: Path) -> float | None:
    state = path / "trainer_state.json"
    if not state.is_file():
        return None
    values = evaluation_losses(json.loads(state.read_text(encoding="utf-8")))
    return values.get(checkpoint_step(path))


def hardlink_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def preserve_current_best(run_dir: Path) -> dict[str, Any] | None:
    checkpoints = sorted(
        (
            path
            for path in run_dir.glob("**/checkpoint-*")
            if path.is_dir() and complete_checkpoint(path, "adapter")
        ),
        key=lambda path: (checkpoint_step(path), str(path)),
    )
    candidates = [
        (loss, checkpoint_step(path), str(path), path)
        for path in checkpoints
        if (loss := checkpoint_loss(path)) is not None
    ]
    if not candidates:
        return None
    loss, step, _, best = min(candidates)
    if "-preserved" in best.parent.name:
        return {"status": "already_preserved", "step": step, "eval_loss": loss}

    preserved_parent = best.parent.with_name(best.parent.name + "-preserved")
    destination = preserved_parent / best.name
    if destination.is_dir() and complete_checkpoint(destination, "adapter"):
        return {"status": "already_preserved", "step": step, "eval_loss": loss}
    staging = preserved_parent / (best.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name in ESSENTIAL_FILES:
        source = best / name
        if source.is_file():
            hardlink_or_copy(source, staging / name)
    if not complete_checkpoint(staging, "adapter"):
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(f"Incomplete preserved checkpoint staged from {best}")
    preserved_parent.mkdir(parents=True, exist_ok=True)
    logging = best.parent / "logging.jsonl"
    if logging.is_file() and not (preserved_parent / "logging.jsonl").exists():
        hardlink_or_copy(logging, preserved_parent / "logging.jsonl")
    os.replace(staging, destination)
    report = {
        "status": "preserved",
        "source": str(best),
        "destination": str(destination),
        "step": step,
        "eval_loss": loss,
    }
    (preserved_parent / "preservation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    run_dir = args.run_dir.resolve()
    last: tuple[int, float] | None = None
    while True:
        report = preserve_current_best(run_dir)
        if report is not None:
            current = (int(report["step"]), float(report["eval_loss"]))
            if current != last:
                print(json.dumps(report, ensure_ascii=False), flush=True)
                last = current
        if not process_alive(args.watch_pid):
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
