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
    provenance_rows = [
        json.loads(line) for line in args.input_provenance.read_text(encoding="utf-8").splitlines()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=args.output.parent, prefix=f".{args.output.name}.", suffix=".tmp"
    )
    output_rows = 0
    seen_inputs: set[int] = set()
    with os.fdopen(fd, "w", encoding="utf-8") as output_handle, args.mining_audit.open(
        encoding="utf-8"
    ) as audit_handle:
        for audit_line_number, line in enumerate(audit_handle, 1):
            audit = json.loads(line)
            input_index = audit.get("input_row_index")
            output_index = audit.get("output_row_index")
            if not isinstance(input_index, int) or not 0 <= input_index < len(provenance_rows):
                raise ValueError(f"Invalid input index at audit line {audit_line_number}")
            if input_index in seen_inputs:
                raise ValueError(f"Duplicate input index in mining audit: {input_index}")
            seen_inputs.add(input_index)
            if output_index is None:
                continue
            if output_index != output_rows:
                raise ValueError(
                    f"Non-contiguous output index: expected {output_rows}, got {output_index}"
                )
            row = dict(provenance_rows[input_index])
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
        output_handle.flush()
        os.fsync(output_handle.fileno())
    if len(seen_inputs) != len(provenance_rows):
        Path(temporary).unlink(missing_ok=True)
        raise ValueError(
            f"Mining audit covered {len(seen_inputs)}/{len(provenance_rows)} input rows"
        )
    os.replace(temporary, args.output)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": len(provenance_rows),
        "output_rows": output_rows,
        "inputs": {
            "provenance": {
                "path": str(args.input_provenance.resolve()),
                "sha256": sha256(args.input_provenance),
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
