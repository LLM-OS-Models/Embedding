#!/usr/bin/env python3
"""Insert one complete local Korean-v1 run into the live MTEB Borda table.

This does not submit a result or mutate the official leaderboard. It validates
that its local Borda implementation exactly reproduces every live rank before
adding the local row, then writes a dated comparison snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_ENDPOINT = (
    "https://mteb-leaderboard-backend.hf.space/v1/benchmarks/"
    "MTEB%28kor%2C%20v1%29/scores"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--neighbors", type=int, default=3)
    return parser.parse_args()


def average_ranks_descending(values: list[tuple[int, float]]) -> dict[int, float]:
    """Return one-based average ranks, matching Polars rank(method='average')."""

    ordered = sorted(values, key=lambda item: (-item[1], item[0]))
    result: dict[int, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for index, _ in ordered[cursor:end]:
            result[index] = average_rank
        cursor = end
    return result


def borda(rows: list[dict[str, Any]], tasks: list[str]) -> tuple[list[int], list[float], dict[str, dict[int, float]]]:
    n_models = len(rows)
    points = [0.0] * n_models
    ranks_by_task: dict[str, dict[int, float]] = {}
    for task in tasks:
        scored = [
            (index, float(row["scoresByTask"][task]))
            for index, row in enumerate(rows)
            if task in row.get("scoresByTask", {})
        ]
        task_ranks = average_ranks_descending(scored)
        ranks_by_task[task] = task_ranks
        for index, rank in task_ranks.items():
            points[index] += n_models - rank
    final_ranks = [1 + sum(other > value for other in points) for value in points]
    return final_ranks, points, ranks_by_task


def load_local(path: Path, tasks: list[str]) -> tuple[dict[str, Any], dict[str, float]]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("benchmark") != "MTEB(kor, v1)":
        raise ValueError("Local summary is not MTEB(kor, v1)")
    if not summary.get("complete") or summary.get("completed_tasks") != len(tasks):
        raise ValueError("Local summary must contain a complete six-task run")
    scores = {
        task: float(summary["scores"][task]["score"])
        for task in tasks
    }
    if set(scores) != set(tasks):
        raise ValueError("Local task membership does not match the live benchmark")
    return summary, scores


def main() -> None:
    args = parse_args()
    response = requests.get(args.endpoint, timeout=60)
    response.raise_for_status()
    payload = response.json()
    tasks = list(payload["tasks"])
    official_rows = list(payload["rows"])

    official_ranks, _, _ = borda(official_rows, tasks)
    mismatches = [
        {
            "model": row["model"]["name"],
            "backend_rank": row["rank"],
            "recomputed_rank": computed,
        }
        for row, computed in zip(official_rows, official_ranks, strict=True)
        if row["rank"] != computed
    ]
    if mismatches:
        raise RuntimeError(
            f"Local Borda implementation does not reproduce the live backend: {mismatches[:3]}"
        )

    local_summary, local_scores = load_local(args.summary, tasks)
    local_name = local_summary["model"]
    if any(row["model"]["name"] == local_name for row in official_rows):
        raise ValueError(
            f"{local_name} already has an official row; do not insert a duplicate local result"
        )

    local_row = {
        "model": {"name": local_name},
        "scoresByTask": local_scores,
        "meanTask": float(local_summary["mean_task"]),
        "meanTaskType": float(local_summary["mean_task_type"]),
        "source": "official-protocol-local-reproduction",
    }
    augmented = official_rows + [local_row]
    ranks, points, ranks_by_task = borda(augmented, tasks)
    local_index = len(augmented) - 1
    local_rank = ranks[local_index]

    ordered_indices = sorted(
        range(len(augmented)),
        key=lambda index: (ranks[index], -points[index], augmented[index]["model"]["name"]),
    )
    ordered_position = ordered_indices.index(local_index)
    start = max(0, ordered_position - args.neighbors)
    end = min(len(ordered_indices), ordered_position + args.neighbors + 1)
    neighbors = []
    for index in ordered_indices[start:end]:
        row = augmented[index]
        neighbors.append(
            {
                "rank_with_local_row": ranks[index],
                "borda_points": points[index],
                "model": row["model"]["name"],
                "mean_task": row.get("meanTask"),
                "mean_task_type": row.get("meanTaskType"),
                "is_local": index == local_index,
            }
        )

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": payload["benchmarkName"],
        "official_endpoint": args.endpoint,
        "official_response_sha256": hashlib.sha256(response.content).hexdigest(),
        "official_rows": len(official_rows),
        "complete_official_rows": sum(
            set(tasks) <= set(row.get("scoresByTask", {})) for row in official_rows
        ),
        "official_rank_reproduction": {
            "matched": len(official_rows),
            "total": len(official_rows),
        },
        "local": {
            "model": local_name,
            "revision": local_summary.get("resolved_revision"),
            "source": "official-protocol-local-reproduction; not an official submission",
            "rank_borda_if_inserted": local_rank,
            "borda_points": points[local_index],
            "mean_task": local_summary["mean_task"],
            "mean_task_type": local_summary["mean_task_type"],
            "scores_by_task": local_scores,
            "ranks_by_task_if_inserted": {
                task: ranks_by_task[task][local_index] for task in tasks
            },
        },
        "neighbors": neighbors,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
