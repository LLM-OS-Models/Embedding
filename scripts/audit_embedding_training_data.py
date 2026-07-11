#!/usr/bin/env python3
"""Stream a strict embedding curriculum and quantify data-quality contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


QUERY_MARKER = "\nQuery:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-batch-size", type=int, default=16)
    return parser.parse_args()


def text(sequence: Any) -> str:
    if (
        not isinstance(sequence, list)
        or len(sequence) != 1
        or not isinstance(sequence[0], dict)
        or sequence[0].get("role") != "user"
        or not isinstance(sequence[0].get("content"), str)
    ):
        raise ValueError("Expected one user message")
    return sequence[0]["content"]


def query_body(value: str) -> tuple[str, str]:
    if QUERY_MARKER not in value:
        return "unprompted", value.strip()
    instruction, body = value.split(QUERY_MARKER, 1)
    return instruction.strip(), body.strip()


def query_style(value: str) -> str:
    compact = value.strip()
    if compact.count(",") >= 2 or compact.count("，") >= 2:
        return "comma_keyword_list"
    question_signals = (
        "?",
        "까",
        "나요",
        "인가요",
        "어떻게",
        "무엇",
        "뭐",
        "왜",
        "언제",
        "어디",
        "누가",
        "몇",
    )
    if any(signal in compact for signal in question_signals):
        return "natural_question"
    if len(compact) <= 40:
        return "short_search_or_title"
    return "long_statement_or_search"


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    position = math.ceil(quantile * len(values)) - 1
    return sorted(values)[max(0, min(position, len(values) - 1))]


def length_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "mean": 0.0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.expected_batch_size < 1:
        raise ValueError("--expected-batch-size must be positive")
    expected_fields = {"messages", "positive_messages", "negative_messages"}
    source_counts: Counter[str] = Counter()
    negative_counts: Counter[int] = Counter()
    instruction_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    exposure_counts: Counter[str] = Counter()
    query_lengths: list[int] = []
    positive_lengths: list[int] = []
    negative_lengths: list[int] = []
    query_hashes: set[bytes] = set()
    positive_hashes: set[bytes] = set()
    duplicate_queries = 0
    duplicate_positives = 0
    row_hash_mismatches = 0
    batch_contract_violations = 0
    rows = 0

    with args.train.open(encoding="utf-8") as train_handle, args.provenance.open(
        encoding="utf-8"
    ) as provenance_handle:
        while True:
            train_line = train_handle.readline()
            provenance_line = provenance_handle.readline()
            if not train_line and not provenance_line:
                break
            rows += 1
            if not train_line or not provenance_line:
                raise ValueError(f"Train/provenance length mismatch at row {rows}")
            row = json.loads(train_line)
            provenance = json.loads(provenance_line)
            if not isinstance(row, dict) or set(row) != expected_fields:
                raise ValueError(f"Strict schema violation at row {rows}")
            query = text(row["messages"])
            positives = row["positive_messages"]
            negatives = row["negative_messages"]
            if not isinstance(positives, list) or len(positives) != 1:
                raise ValueError(f"Expected one positive at row {rows}")
            if not isinstance(negatives, list) or not negatives:
                raise ValueError(f"Expected negatives at row {rows}")
            positive = text(positives[0])
            negative_texts = [text(sequence) for sequence in negatives]
            instruction, body = query_body(query)
            instruction_counts[instruction] += 1
            style_counts[query_style(body)] += 1
            negative_counts[len(negative_texts)] += 1
            query_lengths.append(len(body))
            positive_lengths.append(len(positive))
            negative_lengths.extend(len(value) for value in negative_texts)

            query_digest = hashlib.sha256(body.encode()).digest()
            positive_digest = hashlib.sha256(positive.encode()).digest()
            duplicate_queries += query_digest in query_hashes
            duplicate_positives += positive_digest in positive_hashes
            query_hashes.add(query_digest)
            positive_hashes.add(positive_digest)

            compact = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            if (
                provenance.get("row_sha256")
                != hashlib.sha256(compact.encode()).hexdigest()
            ):
                row_hash_mismatches += 1
            source = provenance.get("source_id")
            if not isinstance(source, str) or not source:
                raise ValueError(f"Missing source_id at row {rows}")
            source_counts[source] += 1
            for task in provenance.get("trained_on_tasks") or []:
                exposure_counts[str(task)] += 1
            batch = provenance.get("homogeneous_batch")
            if not isinstance(batch, dict):
                batch_contract_violations += 1
            elif (
                batch.get("batch_size") != args.expected_batch_size
                or batch.get("source_id") != source
                or batch.get("output_row_index") != rows - 1
                or batch.get("batch_index") != (rows - 1) // args.expected_batch_size
            ):
                batch_contract_violations += 1

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "train": {"path": str(args.train), "sha256": sha256(args.train)},
            "provenance": {
                "path": str(args.provenance),
                "sha256": sha256(args.provenance),
            },
        },
        "rows": rows,
        "source_counts": dict(source_counts.most_common()),
        "negative_count_distribution": {
            str(key): value for key, value in sorted(negative_counts.items())
        },
        "instruction_counts": dict(instruction_counts.most_common()),
        "query_style_heuristic": dict(style_counts.most_common()),
        "benchmark_train_exposure_rows": dict(exposure_counts.most_common()),
        "character_lengths": {
            "query_body": length_summary(query_lengths),
            "positive": length_summary(positive_lengths),
            "negative": length_summary(negative_lengths),
        },
        "exact_duplicates": {
            "query_beyond_first_occurrence": duplicate_queries,
            "positive_beyond_first_occurrence": duplicate_positives,
        },
        "contract_checks": {
            "row_sha256_mismatches": row_hash_mismatches,
            "homogeneous_batch_violations": batch_contract_violations,
            "status": (
                "pass"
                if row_hash_mismatches == 0 and batch_contract_violations == 0
                else "fail"
            ),
        },
        "style_note": "Heuristic counts guide ablations; they are not human relevance labels.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["contract_checks"]["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
