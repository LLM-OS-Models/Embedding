#!/usr/bin/env python3
"""Build a deterministic curriculum from homogeneous primary and replay batches.

The input files must already be ordered as source-homogeneous microbatches.  We
select complete batches, shuffle only the batch references, and copy the rows
without holding multi-gigabyte training examples in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class BatchRef:
    role: str
    train_offset: int
    provenance_offset: int
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-train", type=Path, required=True)
    parser.add_argument("--primary-provenance", type=Path, required=True)
    parser.add_argument("--primary-rows", type=int, required=True)
    parser.add_argument("--replay-train", type=Path, required=True)
    parser.add_argument("--replay-provenance", type=Path, required=True)
    parser.add_argument("--replay-rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--adaptation-label",
        default="replay-curriculum",
        help="Disclosure label copied into the manifest/model card evidence",
    )
    return parser.parse_args()


def source_id(row: dict) -> str:
    value = row.get("source_id") or row.get("source")
    if not isinstance(value, str) or not value:
        nested = row.get("provenance", {})
        value = nested.get("repository") if isinstance(nested, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError("Provenance row has no source_id/source/repository")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_batches(
    role: str,
    train_path: Path,
    provenance_path: Path,
    requested_rows: int,
    batch_size: int,
) -> list[BatchRef]:
    if requested_rows < 1 or requested_rows % batch_size:
        raise ValueError(f"{role} rows must be a positive multiple of batch size")
    requested_batches = requested_rows // batch_size
    refs: list[BatchRef] = []
    with train_path.open("rb") as train, provenance_path.open("rb") as provenance:
        while len(refs) < requested_batches:
            train_offset = train.tell()
            provenance_offset = provenance.tell()
            sources: set[str] = set()
            for _ in range(batch_size):
                train_line = train.readline()
                provenance_line = provenance.readline()
                if not train_line or not provenance_line:
                    raise ValueError(
                        f"{role} contains fewer than requested {requested_rows} aligned rows"
                    )
                try:
                    provenance_row = json.loads(provenance_line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON while scanning {role}") from error
                if not train_line.startswith(b'{"messages":'):
                    raise ValueError(f"Invalid strict training row in {role}")
                sources.add(source_id(provenance_row))
            if len(sources) != 1:
                raise ValueError(f"{role} input batch is not source-homogeneous: {sources}")
            refs.append(
                BatchRef(
                    role=role,
                    train_offset=train_offset,
                    provenance_offset=provenance_offset,
                    source=next(iter(sources)),
                )
            )
    return refs


def atomic_binary_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    return os.fdopen(fd, "wb"), Path(name)


def copy_curriculum(args: argparse.Namespace, refs: list[BatchRef]) -> dict:
    paths = {
        "primary": (args.primary_train.resolve(), args.primary_provenance.resolve()),
        "replay": (args.replay_train.resolve(), args.replay_provenance.resolve()),
    }
    handles = {
        role: (train.open("rb"), provenance.open("rb"))
        for role, (train, provenance) in paths.items()
    }
    train_out, train_temp = atomic_binary_writer(args.output.resolve())
    provenance_out, provenance_temp = atomic_binary_writer(args.provenance_output.resolve())
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
                provenance_row["curriculum_batch"] = {
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
        "role_counts": dict(role_counts),
        "role_fractions": {
            role: count / output_rows for role, count in sorted(role_counts.items())
        },
        "source_counts": dict(sorted(source_counts.items())),
        "order_contract": (
            "select complete source-homogeneous batches from each role, shuffle batch "
            "references globally, preserve row order inside every batch, trainer shuffle disabled"
        ),
        "inputs": {
            role: {
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
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    primary = scan_batches(
        "primary",
        args.primary_train.resolve(),
        args.primary_provenance.resolve(),
        args.primary_rows,
        args.batch_size,
    )
    replay = scan_batches(
        "replay",
        args.replay_train.resolve(),
        args.replay_provenance.resolve(),
        args.replay_rows,
        args.batch_size,
    )
    refs = primary + replay
    random.Random(args.seed).shuffle(refs)
    manifest = copy_curriculum(args, refs)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
