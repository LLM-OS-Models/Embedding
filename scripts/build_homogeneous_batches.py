#!/usr/bin/env python3
"""Reorder strict rows into shuffled source-homogeneous microbatches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed: int, *values: str) -> int:
    digest = hashlib.sha256("\0".join((str(seed), *values)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def source_id(provenance: dict) -> str:
    value = provenance.get("source_id") or provenance.get("source")
    if not isinstance(value, str) or not value:
        nested = provenance.get("provenance", {})
        value = nested.get("repository") if isinstance(nested, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError("Provenance row has no source_id/source/repository")
    return value


def read_aligned(train: Path, provenance: Path) -> dict[str, list[tuple[str, str]]]:
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with train.open(encoding="utf-8") as train_handle, provenance.open(
        encoding="utf-8"
    ) as provenance_handle:
        line_number = 0
        while True:
            train_line = train_handle.readline()
            provenance_line = provenance_handle.readline()
            if not train_line and not provenance_line:
                break
            line_number += 1
            if not train_line or not provenance_line:
                raise ValueError(f"Train/provenance length mismatch at row {line_number}")
            train_row = json.loads(train_line)
            expected = {"messages", "positive_messages", "negative_messages"}
            if not isinstance(train_row, dict) or set(train_row) != expected:
                raise ValueError(f"Invalid strict row at line {line_number}")
            provenance_row = json.loads(provenance_line)
            groups[source_id(provenance_row)].append(
                (
                    json.dumps(train_row, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(provenance_row, ensure_ascii=False, separators=(",", ":")),
                )
            )
    return groups


def atomic_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    return os.fdopen(fd, "w", encoding="utf-8"), Path(name)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    train = args.train.resolve()
    provenance = args.provenance.resolve()
    groups = read_aligned(train, provenance)
    batches: list[tuple[str, list[tuple[str, str]]]] = []
    dropped: Counter[str] = Counter()
    input_counts: Counter[str] = Counter()
    for source, rows in sorted(groups.items()):
        input_counts[source] = len(rows)
        random.Random(stable_seed(args.seed, "within", source)).shuffle(rows)
        usable = len(rows) - len(rows) % args.batch_size
        dropped[source] = len(rows) - usable
        for start in range(0, usable, args.batch_size):
            batches.append((source, rows[start : start + args.batch_size]))
    random.Random(stable_seed(args.seed, "batch-order")).shuffle(batches)
    if not batches:
        raise ValueError("No complete homogeneous batch can be emitted")

    train_handle, train_temp = atomic_writer(args.output.resolve())
    provenance_handle, provenance_temp = atomic_writer(args.provenance_output.resolve())
    output_counts: Counter[str] = Counter()
    try:
        output_index = 0
        for batch_index, (source, rows) in enumerate(batches):
            if len(rows) != args.batch_size:
                raise AssertionError("Internal non-homogeneous batch")
            for train_line, provenance_line in rows:
                audit = json.loads(provenance_line)
                audit["homogeneous_batch"] = {
                    "batch_index": batch_index,
                    "batch_size": args.batch_size,
                    "source_id": source,
                    "output_row_index": output_index,
                }
                train_handle.write(train_line + "\n")
                provenance_handle.write(
                    json.dumps(audit, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                output_counts[source] += 1
                output_index += 1
        for handle in (train_handle, provenance_handle):
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        train_handle.close()
        provenance_handle.close()
        train_temp.unlink(missing_ok=True)
        provenance_temp.unlink(missing_ok=True)
        raise
    else:
        train_handle.close()
        provenance_handle.close()
        os.replace(train_temp, args.output.resolve())
        os.replace(provenance_temp, args.provenance_output.resolve())

    output_rows = sum(output_counts.values())
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "input_rows": sum(input_counts.values()),
        "output_rows": output_rows,
        "complete_batches": len(batches),
        "dropped_source_remainders": dict(dropped),
        "input_source_counts": dict(input_counts),
        "output_source_counts": dict(output_counts),
        "order_contract": "shuffle within each source, split complete source-homogeneous batches, shuffle batches globally, trainer shuffle disabled",
        "inputs": {
            "train": {"path": str(train), "sha256": sha256(train)},
            "provenance": {"path": str(provenance), "sha256": sha256(provenance)},
        },
        "outputs": {
            "train": {"path": str(args.output.resolve()), "sha256": sha256(args.output)},
            "provenance": {
                "path": str(args.provenance_output.resolve()),
                "sha256": sha256(args.provenance_output),
            },
        },
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
