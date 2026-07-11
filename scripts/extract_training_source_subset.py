#!/usr/bin/env python3
"""Extract an auditable source subset from aligned training/provenance JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--expected-rows", type=int)
    return parser.parse_args()


def canonical(value: Any) -> bytes:
    # Match the source builder/auditor contract: compact JSON in parsed key order.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_sha(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(row)).hexdigest()


def extract(args: argparse.Namespace) -> dict[str, Any]:
    train = args.train.resolve()
    provenance = args.provenance.resolve()
    output_dir = args.output_dir.resolve()
    selected = set(args.source)
    if len(selected) != len(args.source):
        raise ValueError("--source values must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_train = output_dir / "train.jsonl"
    output_provenance = output_dir / "provenance.jsonl"
    counts: Counter[str] = Counter()
    input_rows = 0
    output_rows = 0
    with train.open(encoding="utf-8") as rows, provenance.open(encoding="utf-8") as meta, \
        output_train.open("w", encoding="utf-8") as train_out, \
        output_provenance.open("w", encoding="utf-8") as provenance_out:
        while True:
            row_line = rows.readline()
            meta_line = meta.readline()
            if not row_line and not meta_line:
                break
            if not row_line or not meta_line:
                raise RuntimeError("Input train/provenance line counts differ")
            row = json.loads(row_line)
            record = json.loads(meta_line)
            if record.get("row_index") != input_rows:
                raise RuntimeError(f"Input provenance row_index drift at {input_rows}")
            declared_sha = record.get("row_sha256")
            actual_sha = row_sha(row)
            if declared_sha != actual_sha:
                raise RuntimeError(f"Input row SHA mismatch at {input_rows}")
            source = record.get("source_id")
            if source in selected:
                projected = dict(record)
                projected["parent_row_index"] = input_rows
                projected["row_index"] = output_rows
                projected["source_subset_phase"] = args.phase
                train_out.write(row_line if row_line.endswith("\n") else row_line + "\n")
                provenance_out.write(
                    json.dumps(projected, ensure_ascii=False, sort_keys=True) + "\n"
                )
                counts[source] += 1
                output_rows += 1
            input_rows += 1
    missing = selected - set(counts)
    if missing:
        raise RuntimeError(f"Requested sources absent from input: {sorted(missing)}")
    if args.expected_rows is not None and output_rows != args.expected_rows:
        raise RuntimeError(
            f"Expected {args.expected_rows} selected rows, found {output_rows}"
        )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "built_rows": output_rows,
        "selection": {
            "contract": "exact source_id membership; stable parent order",
            "sources": sorted(selected),
            "source_counts": dict(sorted(counts.items())),
        },
        "inputs": {
            "train": {"path": str(train), "sha256": sha256(train), "rows": input_rows},
            "provenance": {"path": str(provenance), "sha256": sha256(provenance), "rows": input_rows},
        },
        "files": {
            "train.jsonl": {"rows": output_rows, "sha256": sha256(output_train)},
            "provenance.jsonl": {
                "rows": output_rows,
                "sha256": sha256(output_provenance),
            },
        },
        "release_eligible": False,
        "visibility": "public research/non-commercial performance track",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    manifest = extract(parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
