#!/usr/bin/env python3
"""Validate Qwen's registered-loader Korean-v1 result and update README."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


QWEN = "Qwen/Qwen3-Embedding-8B"
REVISION = "4e423935c619ae4df87b646a3ce949610c66241c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def validate(summary: dict[str, Any], comparison: dict[str, Any]) -> None:
    if (
        summary.get("model") != QWEN
        or summary.get("requested_revision") != REVISION
        or summary.get("protocol_id") != "mteb-korean-v1-mteb-2.18.0"
        or summary.get("complete") is not True
        or summary.get("completed_tasks") != 6
        or summary.get("environment", {}).get("registered_loader") is not True
    ):
        raise ValueError("Qwen official summary contract is incomplete or mismatched")
    local = comparison.get("local", {})
    if local.get("model") != QWEN or local.get("revision") != REVISION:
        raise ValueError("Live-board comparison belongs to another model/revision")
    reproduction = comparison.get("official_rank_reproduction", {})
    if reproduction.get("matched") != reproduction.get("total"):
        raise ValueError("Official live ranks were not fully reproduced")


def result_row(summary: dict[str, Any], comparison: dict[str, Any]) -> str:
    rank = int(comparison["local"]["rank_borda_if_inserted"])
    retrieval = 100.0 * float(summary["means_by_type"]["Retrieval"])
    return (
        f"| 로컬 재현 | `{QWEN}` | **{rank} if inserted** | "
        f"**{float(summary['mean_task_leaderboard_points']):.2f}** | "
        f"**{float(summary['mean_task_type_leaderboard_points']):.2f}** | "
        f"**{retrieval:.2f}** | registry 감사 | registered-loader 동일 protocol, 6/6 완료 |"
    )


def update_readme(path: Path, row: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    prefix = f"| 비교 | `{QWEN}` |"
    replacement_prefix = f"| 로컬 재현 | `{QWEN}` |"
    indices = [
        index
        for index, line in enumerate(lines)
        if line.startswith(prefix) or line.startswith(replacement_prefix)
    ]
    if len(indices) != 1:
        raise ValueError(f"Expected one Qwen official README row, found {len(indices)}")
    lines[indices[0]] = row
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = read_json(args.summary)
    comparison = read_json(args.comparison)
    validate(summary, comparison)
    row = result_row(summary, comparison)
    update_readme(args.readme, row)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "live_comparison": comparison,
        "readme_row": row,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
