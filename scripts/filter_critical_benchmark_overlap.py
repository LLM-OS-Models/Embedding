#!/usr/bin/env python3
"""Drop training rows whose text collides with benchmark query/evaluation text.

The combined target curriculum mixes six mined components.  Current-student
mining replaces negatives with corpus documents, and some of those documents
are also benchmark query or evaluation text, which
``audit_training_benchmark_overlap.py`` correctly refuses.  Filtering happens on
the mined (pre-ordering) file so the homogeneous batch contract is rebuilt
afterwards rather than broken by row removal.

A row is dropped when any of its query/positive/negative texts hashes into the
blocklist as ``query_text``, or as ``evaluation_text`` of a task the row's own
source does not declare in ``trained_on_tasks``.  Corpus-text overlap is left
alone: an explicit train split legitimately shares a corpus with its eval split
and the model is then reported as target-adapted, not clean zero-shot.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_training_benchmark_overlap import (  # noqa: E402
    DEFAULT_BLOCKLIST,
    blocklist_files,
    message_content,
    nested_contents,
    read_digest_lines,
    semantic_query_body,
    sha256_file,
    text_digest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--blocklist-root", type=Path, default=DEFAULT_BLOCKLIST)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args()


def load_kind_hashes(blocklist_root: Path) -> tuple[dict[bytes, set[str]], set[bytes]]:
    """Return evaluation-text hashes by task and the full query-text hash set."""

    evaluation_tasks: dict[bytes, set[str]] = defaultdict(set)
    query_hashes: set[bytes] = set()
    for path, kind in blocklist_files(blocklist_root):
        if kind == "corpus_text":
            continue
        task = path.parent.relative_to(blocklist_root).as_posix().split("/")[1]
        for digest in read_digest_lines(path):
            if kind == "query_text":
                query_hashes.add(digest)
            else:
                evaluation_tasks[digest].add(task)
    return dict(evaluation_tasks), query_hashes


def main() -> None:
    args = parse_args()
    evaluation_tasks, query_hashes = load_kind_hashes(args.blocklist_root)

    kept = 0
    dropped = 0
    dropped_by_role: Counter[str] = Counter()
    dropped_by_source: Counter[str] = Counter()

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    train_tmp = args.train_output.with_suffix(args.train_output.suffix + ".tmp")
    provenance_tmp = args.provenance_output.with_suffix(
        args.provenance_output.suffix + ".tmp"
    )

    with (
        args.train.open(encoding="utf-8") as train_handle,
        args.provenance.open(encoding="utf-8") as provenance_handle,
        train_tmp.open("w", encoding="utf-8") as train_out,
        provenance_tmp.open("w", encoding="utf-8") as provenance_out,
    ):
        for line_number, line in enumerate(train_handle, 1):
            provenance_line = provenance_handle.readline()
            if not provenance_line:
                raise ValueError("Provenance has fewer rows than training data")
            row: dict[str, Any] = json.loads(line)
            provenance_row: dict[str, Any] = json.loads(provenance_line)
            source = str(
                provenance_row.get("source_id")
                or provenance_row.get("source")
                or "unknown"
            )
            declared = {
                str(task) for task in provenance_row.get("trained_on_tasks") or []
            }

            query = message_content(row.get("messages"), "messages", line_number)
            values = [
                ("query_full", query),
                ("query_body", semantic_query_body(query)),
                *(
                    ("positive", value)
                    for value in nested_contents(
                        row.get("positive_messages"), "positive_messages", line_number
                    )
                ),
                *(
                    ("negative", value)
                    for value in nested_contents(
                        row.get("negative_messages"), "negative_messages", line_number
                    )
                ),
            ]

            critical_role = None
            for role, value in values:
                digest = text_digest(value)
                if digest in query_hashes:
                    critical_role = role
                    break
                tasks = evaluation_tasks.get(digest)
                if tasks and not tasks <= declared:
                    critical_role = role
                    break

            if critical_role is not None:
                dropped += 1
                dropped_by_role[critical_role] += 1
                dropped_by_source[source] += 1
                continue

            train_out.write(line if line.endswith("\n") else line + "\n")
            provenance_out.write(
                provenance_line
                if provenance_line.endswith("\n")
                else provenance_line + "\n"
            )
            kept += 1
        if provenance_handle.readline():
            raise ValueError("Provenance has more rows than training data")

    train_tmp.replace(args.train_output)
    provenance_tmp.replace(args.provenance_output)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "drop-rows-with-benchmark-query-or-unexpected-evaluation-text",
        "inputs": {
            "train": {"path": str(args.train.resolve()), "sha256": sha256_file(args.train)},
            "provenance": {
                "path": str(args.provenance.resolve()),
                "sha256": sha256_file(args.provenance),
            },
        },
        "outputs": {
            "train": {
                "path": str(args.train_output.resolve()),
                "sha256": sha256_file(args.train_output),
            },
            "provenance": {
                "path": str(args.provenance_output.resolve()),
                "sha256": sha256_file(args.provenance_output),
            },
        },
        "rows_kept": kept,
        "rows_dropped": dropped,
        "dropped_by_role": dict(sorted(dropped_by_role.items())),
        "dropped_by_source": dict(sorted(dropped_by_source.items())),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in ("rows_kept", "rows_dropped", "dropped_by_role")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
