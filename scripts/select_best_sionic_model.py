#!/usr/bin/env python3
"""Rank complete local Sionic-9 summaries and return the best model path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-model", action="store_true")
    parser.add_argument(
        "--disqualification-root",
        type=Path,
        help="Outputs root containing <run>/DISQUALIFIED.json markers.",
    )
    return parser.parse_args()


def model_run_name(model: str) -> str | None:
    name = Path(model).name
    for suffix in ("-best-merged", "-best-full"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def disqualification_marker(model: str, root: Path | None) -> Path | None:
    if root is None:
        return None
    run_name = model_run_name(model)
    if not run_name:
        return None
    marker = root.expanduser().resolve() / run_name / "DISQUALIFIED.json"
    return marker if marker.is_file() and marker.stat().st_size > 0 else None


def main() -> None:
    args = parse_args()
    candidates = []
    excluded = []
    for path in args.root.glob("*/summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("completed_tasks") != 9 or summary.get("average") is None:
            continue
        marker = disqualification_marker(
            str(summary.get("model", "")), args.disqualification_root
        )
        if marker is not None:
            excluded.append(
                {
                    "model": summary.get("model"),
                    "summary": str(path.resolve()),
                    "marker": str(marker),
                    "reason": "run-level DISQUALIFIED.json",
                }
            )
            continue
        candidates.append(
            {
                "model": summary["model"],
                "average": float(summary["average"]),
                "scores": summary["scores"],
                "summary": str(path.resolve()),
            }
        )
    if not candidates:
        raise RuntimeError(f"No complete Sionic-9 summaries under {args.root}")
    candidates.sort(key=lambda row: (-row["average"], row["model"]))
    report = {
        "selection_metric": "Sionic Korean retrieval 9-task macro NDCG@10",
        "best": candidates[0],
        "ranking": candidates,
        "excluded": excluded,
        "comsat_card_reference": 0.793,
        "beats_comsat_card_reference": candidates[0]["average"] > 0.793,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.print_model:
        print(candidates[0]["model"])
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
