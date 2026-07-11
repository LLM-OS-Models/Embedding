#!/usr/bin/env python3
"""Mix any number of source-homogeneous curricula by complete microbatch.

Each --component uses ROLE=TRAIN=PROVENANCE=ROWS. Rows must be a multiple of
the shared batch size. Text is copied by byte offset; only provenance receives
an additional deterministic multidomain batch record.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from build_replay_curriculum import (
        BatchRef,
        atomic_binary_writer,
        scan_batches,
        sha256,
    )
except ModuleNotFoundError:
    from scripts.build_replay_curriculum import (
        BatchRef,
        atomic_binary_writer,
        scan_batches,
        sha256,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        action="append",
        required=True,
        help="ROLE=TRAIN_JSONL=PROVENANCE_JSONL=ROWS; repeat for each domain",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adaptation-label", required=True)
    return parser.parse_args()


def parse_component(value: str) -> tuple[str, Path, Path, int]:
    parts = value.split("=", 3)
    if len(parts) != 4:
        raise ValueError(
            "--component must be ROLE=TRAIN_JSONL=PROVENANCE_JSONL=ROWS"
        )
    role, train, provenance, raw_rows = parts
    if not role or any(character.isspace() for character in role):
        raise ValueError(f"Invalid component role: {role!r}")
    try:
        rows = int(raw_rows)
    except ValueError as error:
        raise ValueError(f"Invalid component row count: {raw_rows!r}") from error
    return role, Path(train).resolve(), Path(provenance).resolve(), rows


def build(args: argparse.Namespace) -> dict:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    components = [parse_component(value) for value in args.component]
    roles = [role for role, *_ in components]
    if len(roles) != len(set(roles)):
        raise ValueError("Component roles must be unique")
    if len(components) < 2:
        raise ValueError("At least two components are required")

    refs: list[BatchRef] = []
    paths: dict[str, tuple[Path, Path]] = {}
    requested_rows: dict[str, int] = {}
    for role, train, provenance, rows in components:
        if not train.is_file() or not provenance.is_file():
            raise FileNotFoundError(f"Missing component files for {role}")
        refs.extend(
            scan_batches(role, train, provenance, rows, args.batch_size)
        )
        paths[role] = (train, provenance)
        requested_rows[role] = rows
    random.Random(args.seed).shuffle(refs)

    handles = {
        role: (train.open("rb"), provenance.open("rb"))
        for role, (train, provenance) in paths.items()
    }
    train_out, train_temp = atomic_binary_writer(args.output.resolve())
    provenance_out, provenance_temp = atomic_binary_writer(
        args.provenance_output.resolve()
    )
    role_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    try:
        output_index = 0
        for batch_index, ref in enumerate(refs):
            train, provenance = handles[ref.role]
            train.seek(ref.train_offset)
            provenance.seek(ref.provenance_offset)
            for _ in range(args.batch_size):
                train_line = train.readline()
                provenance_row = json.loads(provenance.readline())
                provenance_row["multidomain_curriculum_batch"] = {
                    "batch_index": batch_index,
                    "batch_size": args.batch_size,
                    "role": ref.role,
                    "source_id": ref.source,
                    "output_row_index": output_index,
                }
                train_out.write(train_line)
                provenance_out.write(
                    json.dumps(
                        provenance_row, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                    + b"\n"
                )
                role_counts[ref.role] += 1
                source_counts[f"{ref.role}:{ref.source}"] += 1
                output_index += 1
        for handle in (train_out, provenance_out):
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        train_out.close()
        provenance_out.close()
        train_temp.unlink(missing_ok=True)
        provenance_temp.unlink(missing_ok=True)
        raise
    else:
        train_out.close()
        provenance_out.close()
        os.replace(train_temp, args.output.resolve())
        os.replace(provenance_temp, args.provenance_output.resolve())
    finally:
        for train, provenance in handles.values():
            train.close()
            provenance.close()

    output_rows = len(refs) * args.batch_size
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "benchmark_adaptation": args.adaptation_label,
        "batch_size": args.batch_size,
        "output_rows": output_rows,
        "complete_batches": len(refs),
        "role_counts": dict(sorted(role_counts.items())),
        "role_fractions": {
            role: count / output_rows for role, count in sorted(role_counts.items())
        },
        "source_counts": dict(sorted(source_counts.items())),
        "order_contract": (
            "select complete source-homogeneous batches from every component, "
            "shuffle batch references globally, preserve row order inside each "
            "batch, trainer shuffle disabled"
        ),
        "inputs": {
            role: {
                "requested_rows": requested_rows[role],
                "train": {"path": str(train), "sha256": sha256(train)},
                "provenance": {
                    "path": str(provenance),
                    "sha256": sha256(provenance),
                },
            }
            for role, (train, provenance) in paths.items()
        },
        "outputs": {
            "train": {
                "path": str(args.output.resolve()),
                "sha256": sha256(args.output.resolve()),
            },
            "provenance": {
                "path": str(args.provenance_output.resolve()),
                "sha256": sha256(args.provenance_output.resolve()),
            },
        },
    }


def main() -> None:
    args = parse_args()
    manifest = build(args)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
