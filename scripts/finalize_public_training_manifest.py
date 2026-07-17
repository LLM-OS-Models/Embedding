#!/usr/bin/env python3
"""Finalize transformed train/provenance files into a public rights manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--transform-manifest", type=Path, action="append", required=True)
    parser.add_argument("--benchmark-overlap-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--required-next-stage", default="ready for public model training")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def file_record(manifest: dict[str, Any], role: str) -> dict[str, Any] | None:
    value = manifest.get("outputs", {}).get(role)
    return value if isinstance(value, dict) else None


def validate_overlap(audit: dict[str, Any], train: Path, provenance: Path) -> None:
    if audit.get("rows") != line_count(train):
        raise ValueError("Benchmark overlap audit row count differs from final train")
    inputs = audit.get("inputs", {})
    if inputs.get("train", {}).get("sha256") != sha256(train):
        raise ValueError("Benchmark overlap audit does not cover final train")
    if inputs.get("provenance", {}).get("sha256") != sha256(provenance):
        raise ValueError("Benchmark overlap audit does not cover final provenance")
    if audit.get("unique_critical_query_or_evaluation_matches") != 0:
        raise ValueError("Final train has critical benchmark text overlap")
    if audit.get("unique_retrieval_corpus_matches") != 0:
        raise ValueError("Final train has retrieval corpus text overlap")


def line_count(path: Path) -> int:
    rows = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            rows += block.count(b"\n")
    return rows


def validate_provenance(
    provenance: Path, approved: dict[str, dict[str, Any]]
) -> tuple[int, dict[str, int]]:
    rows = 0
    counts: dict[str, int] = {}
    with provenance.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"provenance line {line_number} is not an object")
            for field in ("source", "revision", "license"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"provenance line {line_number} has no {field}")
            if row.get("redistribution_allowed") is not True:
                raise ValueError(f"provenance line {line_number} is not releasable")
            policy = approved.get(row["source"])
            if not isinstance(policy, dict):
                raise ValueError(f"provenance line {line_number} has unapproved source")
            if row["revision"] != policy.get("revision") or row["license"] != policy.get(
                "license"
            ):
                raise ValueError(f"provenance line {line_number} rights drifted")
            counts[row["source"]] = counts.get(row["source"], 0) + 1
            rows += 1
    return rows, dict(sorted(counts.items()))


def build(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.train,
        args.provenance,
        args.source_manifest,
        args.benchmark_overlap_audit,
        *args.transform_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = read_object(args.source_manifest)
    if (
        source.get("release_eligible") is not True
        or source.get("release_blockers")
        or source.get("visibility") != "public"
    ):
        raise ValueError("Source manifest is not public-release eligible")
    approved = {
        row["source"]: row
        for row in source.get("sources", [])
        if isinstance(row, dict) and isinstance(row.get("source"), str)
    }
    if not approved:
        raise ValueError("Source manifest has no approved source records")
    rows = line_count(args.train)
    provenance_rows, source_counts = validate_provenance(args.provenance, approved)
    if rows < 2 or rows != provenance_rows:
        raise ValueError("Final train/provenance row counts differ or are too small")
    transforms = [read_object(path) for path in args.transform_manifest]
    latest = transforms[-1]
    train_record = file_record(latest, "train")
    provenance_record = file_record(latest, "provenance")
    if not train_record or train_record.get("sha256") != sha256(args.train):
        raise ValueError("Latest transform manifest does not declare final train SHA")
    if not provenance_record or provenance_record.get("sha256") != sha256(args.provenance):
        raise ValueError("Latest transform manifest does not declare final provenance SHA")
    overlap = read_object(args.benchmark_overlap_audit)
    validate_overlap(overlap, args.train, args.provenance)
    manifest = {
        "schema_version": 2,
        "artifact_id": args.artifact_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "visibility": "public",
        "release_eligible": True,
        "release_blockers": [],
        "dataset_license": source.get("dataset_license", "other"),
        "output_rows": rows,
        "batch_size": latest.get("batch_size"),
        "benchmark_adaptation": latest.get("benchmark_adaptation"),
        "source_counts": source_counts,
        "sources": [approved[name] for name in source_counts],
        "source_manifest": {
            "path": str(args.source_manifest.resolve()),
            "sha256": sha256(args.source_manifest),
        },
        "transforms": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in args.transform_manifest
        ],
        "benchmark_overlap_audit": {
            "path": str(args.benchmark_overlap_audit.resolve()),
            "sha256": sha256(args.benchmark_overlap_audit),
            "unique_critical_query_or_evaluation_matches": 0,
            "unique_retrieval_corpus_matches": 0,
        },
        "outputs": {
            "train": {"path": str(args.train.resolve()), "rows": rows, "sha256": sha256(args.train)},
            "provenance": {
                "path": str(args.provenance.resolve()),
                "rows": rows,
                "sha256": sha256(args.provenance),
            },
        },
        "required_next_stage": args.required_next_stage,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    manifest = build(parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
