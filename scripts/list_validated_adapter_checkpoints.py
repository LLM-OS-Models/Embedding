#!/usr/bin/env python3
"""List every integrity-checked adapter checkpoint from one Trainer version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from average_lora_checkpoints import (
        ARCHIVE_NAME,
        checkpoint_step,
        complete_archived_checkpoint,
        complete_checkpoint,
        find_disqualification_marker,
        is_relative_to,
    )
except ModuleNotFoundError:
    from scripts.average_lora_checkpoints import (
        ARCHIVE_NAME,
        checkpoint_step,
        complete_archived_checkpoint,
        complete_checkpoint,
        find_disqualification_marker,
        is_relative_to,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--print-paths", action="store_true")
    return parser.parse_args()


def discover(run_dir: Path, anchor: Path) -> tuple[list[Path], bool]:
    run_dir = run_dir.expanduser().resolve()
    anchor = anchor.expanduser().resolve()
    if not run_dir.is_dir() or not complete_checkpoint(anchor):
        raise ValueError("Run directory or anchor checkpoint is incomplete")
    if not is_relative_to(anchor, run_dir):
        raise ValueError("Anchor checkpoint escapes the run directory")
    checkpoint_step(anchor)
    marker = find_disqualification_marker(anchor, run_dir)
    if marker is not None:
        raise RuntimeError(f"Refusing a disqualified run: {marker}")
    version_name = anchor.parent.name
    archive_version = run_dir / ARCHIVE_NAME / version_name
    if archive_version.is_symlink():
        raise ValueError("Archive version directory must not be a symlink")
    archived: list[Path] = []
    if archive_version.exists():
        if not archive_version.is_dir():
            raise ValueError("Archive version path is not a directory")
        candidates = list(archive_version.glob("checkpoint-*"))
        invalid = [path for path in candidates if not complete_archived_checkpoint(path)]
        if invalid:
            raise ValueError("One or more archived checkpoints failed integrity validation")
        archived = [path.resolve() for path in candidates]
    live = [
        path.resolve()
        for path in anchor.parent.glob("checkpoint-*")
        if complete_checkpoint(path)
    ]
    # Once an archive exists it is the canonical complete history.  Live
    # directories are subject to Trainer save_total_limit eviction.
    selected = archived if archived else live
    selected.sort(key=lambda path: checkpoint_step(path))
    steps = [checkpoint_step(path) for path in selected]
    if not selected or len(steps) != len(set(steps)):
        raise ValueError("No checkpoints or duplicate checkpoint steps")
    return selected, bool(archived)


def main() -> None:
    args = parse_args()
    checkpoints, archived = discover(args.run_dir, args.anchor_checkpoint)
    if args.print_paths:
        for checkpoint in checkpoints:
            print(checkpoint)
        return
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir.expanduser().resolve()),
                "anchor_checkpoint": str(args.anchor_checkpoint.expanduser().resolve()),
                "archive_used": archived,
                "checkpoints": [
                    {"step": checkpoint_step(path), "path": str(path)}
                    for path in checkpoints
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
