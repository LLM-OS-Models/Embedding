#!/usr/bin/env python3
"""Build a rights-annotated, benchmark-clean public legal training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit_training_benchmark_overlap import (
    blocklist_files,
    load_blocked,
    message_content,
    nested_contents,
    semantic_query_body,
    text_digest,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--output-provenance", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--rights-config",
        type=Path,
        default=ROOT / "configs/public_legal_source_rights_v1.json",
    )
    parser.add_argument(
        "--blocklist-root",
        type=Path,
        default=ROOT / "outputs/decontamination/benchmark_blocklist",
    )
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


def temp_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    return os.fdopen(fd, "w", encoding="utf-8"), Path(name)


def row_texts(row: dict[str, Any], line_number: int) -> list[str]:
    query = message_content(row.get("messages"), "messages", line_number)
    return [
        query,
        semantic_query_body(query),
        *nested_contents(row.get("positive_messages"), "positive_messages", line_number),
        *nested_contents(row.get("negative_messages"), "negative_messages", line_number),
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    rights = read_object(args.rights_config)
    rights_sources = rights.get("sources")
    if not isinstance(rights_sources, dict) or not rights_sources:
        raise ValueError("Rights config has no sources")
    for source, policy in rights_sources.items():
        if not isinstance(policy, dict) or policy.get("redistribution_allowed") is not True:
            raise ValueError(f"Source is not redistribution-approved: {source}")
        for field in ("revision", "source_url", "license"):
            if not isinstance(policy.get(field), str) or not policy[field].strip():
                raise ValueError(f"Source {source} has no {field}")

    files = blocklist_files(args.blocklist_root)
    blocked, blocked_occurrences = load_blocked(files)
    train_out, train_tmp = temp_writer(args.output_train.resolve())
    provenance_out, provenance_tmp = temp_writer(args.output_provenance.resolve())
    input_rows = output_rows = removed_rows = 0
    source_counts: Counter[str] = Counter()
    try:
        with args.train.open(encoding="utf-8") as train_in, args.provenance.open(
            encoding="utf-8"
        ) as provenance_in:
            for line_number, train_line in enumerate(train_in, 1):
                provenance_line = provenance_in.readline()
                if not provenance_line:
                    raise ValueError("Provenance has fewer rows than training data")
                row = json.loads(train_line)
                provenance = json.loads(provenance_line)
                if not isinstance(row, dict) or not isinstance(provenance, dict):
                    raise ValueError(f"line {line_number}: rows must be objects")
                input_rows += 1
                if any(text_digest(text) in blocked for text in row_texts(row, line_number)):
                    removed_rows += 1
                    continue
                nested = provenance.get("provenance")
                if not isinstance(nested, dict):
                    raise ValueError(f"line {line_number}: missing nested provenance")
                source = str(provenance.get("source") or nested.get("repository") or "")
                policy = rights_sources.get(source)
                if not isinstance(policy, dict):
                    raise ValueError(f"line {line_number}: unreviewed source {source!r}")
                revision = str(nested.get("revision") or "")
                if revision != policy["revision"]:
                    raise ValueError(
                        f"line {line_number}: revision drift for {source}: {revision}"
                    )
                public_provenance = dict(provenance)
                public_provenance.update(
                    {
                        "source": source,
                        "revision": revision,
                        "source_url": policy["source_url"],
                        "license": policy["license"],
                        "redistribution_allowed": True,
                        "rights_basis": policy.get("rights_basis", []),
                    }
                )
                train_out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                provenance_out.write(
                    json.dumps(public_provenance, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                output_rows += 1
                source_counts[source] += 1
            if provenance_in.readline():
                raise ValueError("Provenance has more rows than training data")
        for handle in (train_out, provenance_out):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        os.replace(train_tmp, args.output_train.resolve())
        os.replace(provenance_tmp, args.output_provenance.resolve())
    except BaseException:
        train_out.close()
        provenance_out.close()
        train_tmp.unlink(missing_ok=True)
        provenance_tmp.unlink(missing_ok=True)
        raise
    if output_rows < 2:
        raise ValueError("Fewer than two rows survived decontamination")

    manifest = {
        "schema_version": 2,
        "artifact_id": "public-legal-source-training-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "visibility": "public",
        "release_eligible": True,
        "release_blockers": [],
        "dataset_license": rights.get("dataset_license", "other"),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "removed_benchmark_overlap_rows": removed_rows,
        "source_counts": dict(sorted(source_counts.items())),
        "sources": [
            {"source": source, **policy}
            for source, policy in sorted(rights_sources.items())
            if source_counts[source]
        ],
        "inputs": {
            "train": {"path": str(args.train.resolve()), "sha256": sha256(args.train)},
            "provenance": {
                "path": str(args.provenance.resolve()),
                "sha256": sha256(args.provenance),
            },
            "rights_config": {
                "path": str(args.rights_config.resolve()),
                "sha256": sha256(args.rights_config),
            },
        },
        "benchmark_blocklist": {
            "root": str(args.blocklist_root.resolve()),
            "manifest_sha256": sha256(args.blocklist_root / "manifest.json"),
            "blocked_hash_occurrences": blocked_occurrences,
            "policy": "remove row on any normalized query, positive, or negative text match",
        },
        "outputs": {
            "train": {
                "path": str(args.output_train.resolve()),
                "rows": output_rows,
                "sha256": sha256(args.output_train),
            },
            "provenance": {
                "path": str(args.output_provenance.resolve()),
                "rows": output_rows,
                "sha256": sha256(args.output_provenance),
            },
        },
        "required_next_stage": "current-student hard-negative mining with rights inheritance",
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    manifest = build(parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
