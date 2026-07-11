#!/usr/bin/env python3
"""Validate the strict ms-swift InfoNCE JSONL shape used in this repository."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def validate(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    identities: set[str] = set()
    rows = 0
    with path.open("rb") as binary:
        for chunk in iter(lambda: binary.read(1024 * 1024), b""):
            digest.update(chunk)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            row = json.loads(line)
            if set(row) != {"messages", "positive_messages", "negative_messages"}:
                raise ValueError(f"{path}:{line_number}: unexpected fields {sorted(row)}")
            query = message_text(row["messages"], "messages")
            positives = nested_message_text(row["positive_messages"], "positive_messages")
            negatives = nested_message_text(row["negative_messages"], "negative_messages")
            if len(positives) != 1:
                raise ValueError(f"{path}:{line_number}: exactly one positive is required")
            if positives[0] in negatives:
                raise ValueError(f"{path}:{line_number}: positive duplicated as negative")
            identity = hashlib.sha256(
                "\0".join((query, positives[0], *negatives)).encode("utf-8")
            ).hexdigest()
            if identity in identities:
                raise ValueError(f"{path}:{line_number}: duplicate row")
            identities.add(identity)
            rows += 1
    if rows < 2:
        raise ValueError(f"{path}: at least two rows are required")
    return {"path": str(path), "rows": rows, "sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([validate(path) for path in args.paths], indent=2))


if __name__ == "__main__":
    main()
