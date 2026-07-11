#!/usr/bin/env python3
"""Select a complete adapter checkpoint by minimum recorded evaluation loss."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"checkpoint-(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--print-path", action="store_true")
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = STEP_RE.search(path.name)
    if not match:
        raise ValueError(f"Invalid checkpoint directory: {path}")
    return int(match.group(1))


def evaluation_losses(state: dict[str, Any]) -> dict[int, float]:
    losses: dict[int, float] = {}
    for row in state.get("log_history", []):
        if "step" not in row or "eval_loss" not in row:
            continue
        try:
            step = int(row["step"])
            loss = float(row["eval_loss"])
        except (TypeError, ValueError):
            continue
        if loss == loss and abs(loss) != float("inf"):
            losses[step] = loss
    return losses


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoints = sorted(
        (
            path
            for path in run_dir.glob("**/checkpoint-*")
            if path.is_dir()
            and (path / "adapter_model.safetensors").is_file()
            and (path / "adapter_config.json").is_file()
        ),
        key=checkpoint_step,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No complete adapter checkpoint under {run_dir}")

    losses: dict[int, float] = {}
    for checkpoint in checkpoints:
        state_path = checkpoint / "trainer_state.json"
        if state_path.is_file():
            losses.update(
                evaluation_losses(json.loads(state_path.read_text(encoding="utf-8")))
            )
    by_step = {checkpoint_step(path): path for path in checkpoints}
    eligible = [(loss, step) for step, loss in losses.items() if step in by_step]
    if eligible:
        best_loss, best_step = min(eligible)
        reason = "minimum_eval_loss"
    else:
        best_step = max(by_step)
        best_loss = None
        reason = "latest_complete_checkpoint_no_eval_loss"
    best = by_step[best_step]
    report = {
        "run_dir": str(run_dir),
        "selected_checkpoint": str(best),
        "selected_step": best_step,
        "selected_eval_loss": best_loss,
        "reason": reason,
        "complete_checkpoints": [str(path) for path in checkpoints],
        "eval_losses": {str(step): loss for step, loss in sorted(losses.items())},
    }
    if args.print_path:
        print(best)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
