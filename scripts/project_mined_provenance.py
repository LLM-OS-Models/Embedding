#!/usr/bin/env python3
"""Project aligned input provenance through a miner's row-index audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-provenance", type=Path, required=True)
    parser.add_argument("--mining-audit", type=Path, required=True)
    parser.add_argument(
        "--mined-train",
        type=Path,
        required=True,
        help=(
            "Mined training JSONL. Mining rewrites negatives, so the input "
            "provenance row_sha256 no longer identifies the row it travels "
            "with; it is recomputed against this file and the pre-mining value "
            "is preserved as source_row_sha256."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=args.output.parent, prefix=f".{args.output.name}.", suffix=".tmp"
    )
    output_rows = 0
    input_rows = 0
    with (
        os.fdopen(fd, "w", encoding="utf-8") as output_handle,
        args.mining_audit.open(encoding="utf-8") as audit_handle,
        args.input_provenance.open(encoding="utf-8") as provenance_handle,
        args.mined_train.open(encoding="utf-8") as mined_train_handle,
    ):
        while True:
            line = audit_handle.readline()
            provenance_line = provenance_handle.readline()
            if not line and not provenance_line:
                break
            input_rows += 1
            if not line or not provenance_line:
                raise ValueError(f"Mining audit/provenance length mismatch at row {input_rows}")
            audit = json.loads(line)
            input_index = audit.get("input_row_index")
            output_index = audit.get("output_row_index")
            if input_index != input_rows - 1:
                raise ValueError(
                    f"Mining audit must cover input rows in order: expected {input_rows - 1}, "
                    f"got {input_index}"
                )
            if output_index is None:
                continue
            if output_index != output_rows:
                raise ValueError(
                    f"Non-contiguous output index: expected {output_rows}, got {output_index}"
                )
            mined_line = mined_train_handle.readline()
            if not mined_line:
                raise ValueError(
                    f"Mined training file ended before output row {output_rows}"
                )
            mined_row = json.loads(mined_line)
            mined_compact = json.dumps(
                mined_row, ensure_ascii=False, separators=(",", ":")
            )
            row = dict(json.loads(provenance_line))
            source_row_hash = row.get("row_sha256")
            if source_row_hash is not None:
                row["source_row_sha256"] = source_row_hash
            row["row_sha256"] = hashlib.sha256(
                mined_compact.encode("utf-8")
            ).hexdigest()
            row["row_index"] = output_index
            row["mining_projection"] = {
                "input_row_index": input_index,
                "output_row_index": output_index,
                "positive_score": audit.get("positive_score"),
                "positive_relative_threshold": audit.get("threshold"),
                "ann_search_k": audit.get("ann_search_k"),
                "selected": audit.get("selected"),
                "target_adapted": True,
            }
            output_handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            output_rows += 1
        if mined_train_handle.readline():
            raise ValueError("Mined training file has more rows than the audit")
        output_handle.flush()
        os.fsync(output_handle.fileno())
    os.replace(temporary, args.output)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "inputs": {
            "provenance": {
                "path": str(args.input_provenance.resolve()),
                "sha256": sha256(args.input_provenance),
            },
            "mined_train": {
                "path": str(args.mined_train.resolve()),
                "sha256": sha256(args.mined_train),
            },
            "mining_audit": {
                "path": str(args.mining_audit.resolve()),
                "sha256": sha256(args.mining_audit),
            },
        },
        "output": {"path": str(args.output.resolve()), "sha256": sha256(args.output)},
        "target_adapted": True,
    }
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
