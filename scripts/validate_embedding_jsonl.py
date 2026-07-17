#!/usr/bin/env python3
"""Validate the strict ms-swift InfoNCE JSONL shape used in this repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def message_text(messages: Any, field: str) -> str:
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError(f"{field} must contain exactly one message")
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        raise ValueError(f"{field} must contain one user message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{field} content must be a non-empty string")
    return content


def nested_message_text(groups: Any, field: str) -> list[str]:
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"{field} must be a non-empty list")
    return [message_text(group, f"{field}[{index}]") for index, group in enumerate(groups)]


def validate_teacher_scores(
    values: Any, *, negatives: int, path: Path, line_number: int
) -> None:
    if not isinstance(values, list) or len(values) != negatives + 1:
        raise ValueError(
            f"{path}:{line_number}: teacher_scores must align with positive+negatives"
        )
    scores = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}:{line_number}: teacher_scores must be numeric")
        score = float(value)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(
                f"{path}:{line_number}: teacher_scores must be finite probabilities"
            )
        scores.append(score)
    if scores[0] <= max(scores[1:]):
        raise ValueError(
            f"{path}:{line_number}: teacher positive must outrank every negative"
        )


def validate(path: Path, *, require_teacher_scores: bool = False) -> dict[str, Any]:
    digest = hashlib.sha256()
    identities: set[str] = set()
    rows = 0
    teacher_rows = 0
    with path.open("rb") as binary:
        for chunk in iter(lambda: binary.read(1024 * 1024), b""):
            digest.update(chunk)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            row = json.loads(line)
            required = {"messages", "positive_messages", "negative_messages"}
            fields = frozenset(row)
            if fields not in {
                frozenset(required),
                frozenset(required | {"teacher_scores"}),
            }:
                raise ValueError(f"{path}:{line_number}: unexpected fields {sorted(row)}")
            query = message_text(row["messages"], "messages")
            positives = nested_message_text(row["positive_messages"], "positive_messages")
            negatives = nested_message_text(row["negative_messages"], "negative_messages")
            if len(positives) != 1:
                raise ValueError(f"{path}:{line_number}: exactly one positive is required")
            if positives[0] in negatives:
                raise ValueError(f"{path}:{line_number}: positive duplicated as negative")
            if "teacher_scores" in row:
                teacher_rows += 1
                validate_teacher_scores(
                    row["teacher_scores"],
                    negatives=len(negatives),
                    path=path,
                    line_number=line_number,
                )
            identity = hashlib.sha256(
                "\0".join((query, positives[0], *negatives)).encode("utf-8")
            ).hexdigest()
            if identity in identities:
                raise ValueError(f"{path}:{line_number}: duplicate row")
            identities.add(identity)
            rows += 1
    if rows < 2:
        raise ValueError(f"{path}: at least two rows are required")
    if require_teacher_scores and teacher_rows != rows:
        raise ValueError(f"{path}: every row must contain teacher_scores")
    return {
        "path": str(path),
        "rows": rows,
        "teacher_score_rows": teacher_rows,
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--require-teacher-scores", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            [
                validate(path, require_teacher_scores=args.require_teacher_scores)
                for path in args.paths
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
