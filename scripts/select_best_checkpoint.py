#!/usr/bin/env python3
"""Select a complete adapter or full checkpoint by evaluation loss."""

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
    parser.add_argument(
        "--checkpoint-kind", choices=("adapter", "full", "auto"), default="adapter"
    )
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


def complete_checkpoint(path: Path, kind: str) -> bool:
    adapter = (path / "adapter_model.safetensors").is_file() and (
        path / "adapter_config.json"
    ).is_file()
    full = (
        any(path.glob("model*.safetensors"))
        and (path / "config.json").is_file()
        and (path / "modules.json").is_file()
        and (path / "1_Pooling/config.json").is_file()
    )
    return {"adapter": adapter, "full": full, "auto": adapter or full}[kind]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoints = sorted(
        (
            path
            for path in run_dir.glob("**/checkpoint-*")
            if path.is_dir() and complete_checkpoint(path, args.checkpoint_kind)
        ),
        key=checkpoint_step,
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No complete {args.checkpoint_kind} checkpoint under {run_dir}"
        )

    checkpoint_losses: dict[Path, float] = {}
    for checkpoint in checkpoints:
        state_path = checkpoint / "trainer_state.json"
        if state_path.is_file():
            losses = evaluation_losses(json.loads(state_path.read_text(encoding="utf-8")))
            step = checkpoint_step(checkpoint)
            if step in losses:
                checkpoint_losses[checkpoint] = losses[step]
    eligible = [
        (loss, checkpoint_step(path), str(path), path)
        for path, loss in checkpoint_losses.items()
    ]
    if eligible:
        best_loss, best_step, _, best = min(eligible)
        reason = "minimum_eval_loss"
    else:
        best = max(
            checkpoints,
            key=lambda path: (checkpoint_step(path), path.stat().st_mtime_ns, str(path)),
        )
        best_step = checkpoint_step(best)
        best_loss = None
        reason = "latest_complete_checkpoint_no_eval_loss"
    report = {
        "run_dir": str(run_dir),
        "selected_checkpoint": str(best),
        "selected_step": best_step,
        "selected_eval_loss": best_loss,
        "reason": reason,
        "checkpoint_kind": args.checkpoint_kind,
        "complete_checkpoints": [str(path) for path in checkpoints],
        "eval_losses": {
            str(path): loss
            for path, loss in sorted(checkpoint_losses.items(), key=lambda item: str(item[0]))
        },
    }
    if args.print_path:
        print(best)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
