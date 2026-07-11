#!/usr/bin/env python3
"""Audit strict embedding JSONL against the text-only benchmark blocklist.

The report never stores source or benchmark text. Query/evaluation-text matches
are critical. Retrieval-corpus matches are reported separately because an
explicit task train split can legitimately share a corpus with its eval split;
such a model is target-adapted, not clean zero-shot.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLOCKLIST = ROOT / "outputs/decontamination/benchmark_blocklist"
ZERO_WIDTH_TRANSLATION = str.maketrans("", "", "\u200b\u200c\u200d\u2060\ufeff")
TEXT_HASH_FILES = {
    "query_text.sha256.gz": "query_text",
    "corpus_text.sha256.gz": "corpus_text",
    "evaluation_text.sha256.gz": "evaluation_text",
}
CRITICAL_KINDS = {"query_text", "evaluation_text"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--blocklist-root", type=Path, default=DEFAULT_BLOCKLIST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-critical", action="store_true")
    parser.add_argument("--fail-on-any-text", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.translate(ZERO_WIDTH_TRANSLATION)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(normalized.split())


def text_digest(value: str) -> bytes:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).digest()


def semantic_query_body(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("Instruct:") and "Query:" in stripped:
        return stripped.rpartition("Query:")[2].strip()
    return stripped


def message_content(value: Any, field: str, line_number: int) -> str:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"line {line_number}: {field} must contain one message")
    message = value[0]
    if (
        not isinstance(message, dict)
        or message.get("role") != "user"
        or not isinstance(message.get("content"), str)
    ):
        raise ValueError(f"line {line_number}: invalid {field} message")
    return message["content"]


def nested_contents(value: Any, field: str, line_number: int) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"line {line_number}: {field} must be a non-empty list")
    return [
        message_content(group, f"{field}[{index}]", line_number)
        for index, group in enumerate(value)
    ]


def blocklist_files(root: Path) -> list[tuple[Path, str]]:
    files = [
        (path, TEXT_HASH_FILES[path.name])
        for path in root.rglob("*.sha256.gz")
        if path.name in TEXT_HASH_FILES
    ]
    if not files:
        raise FileNotFoundError(f"No benchmark text hashes under {root}")
    return sorted(files)


def read_digest_lines(path: Path) -> Iterable[bytes]:
    with gzip.open(path, "rt", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, 1):
            value = line.strip()
            if len(value) != 64:
                raise ValueError(f"{path}:{line_number}: invalid SHA-256")
            yield bytes.fromhex(value)


def load_blocked(files: list[tuple[Path, str]]) -> tuple[set[bytes], dict[str, int]]:
    blocked: set[bytes] = set()
    occurrences: Counter[str] = Counter()
    for path, kind in files:
        for digest in read_digest_lines(path):
            blocked.add(digest)
            occurrences[kind] += 1
    return blocked, dict(sorted(occurrences.items()))


def audit(
    train: Path,
    provenance: Path | None,
    blocklist_root: Path,
) -> dict[str, Any]:
    files = blocklist_files(blocklist_root)
    blocked, blocked_occurrences = load_blocked(files)
    checked: Counter[str] = Counter()
    matched_occurrences: Counter[str] = Counter()
    matched: dict[bytes, Counter[str]] = defaultdict(Counter)
    matched_sources: dict[bytes, Counter[str]] = defaultdict(Counter)
    rows = 0

    provenance_handle = provenance.open(encoding="utf-8") if provenance else None
    try:
        with train.open(encoding="utf-8") as train_handle:
            for line_number, line in enumerate(train_handle, 1):
                if not line.strip():
                    raise ValueError(f"{train}:{line_number}: blank line")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{train}:{line_number}: row must be an object")
                query = message_content(row.get("messages"), "messages", line_number)
                positives = nested_contents(
                    row.get("positive_messages"), "positive_messages", line_number
                )
                negatives = nested_contents(
                    row.get("negative_messages"), "negative_messages", line_number
                )
                source = "unknown"
                if provenance_handle is not None:
                    provenance_line = provenance_handle.readline()
                    if not provenance_line:
                        raise ValueError("Provenance has fewer rows than training data")
                    provenance_row = json.loads(provenance_line)
                    source = str(
                        provenance_row.get("source_id")
                        or provenance_row.get("source")
                        or "unknown"
                    )
                values = [
                    ("query_full", query),
                    ("query_body", semantic_query_body(query)),
                    *(("positive", value) for value in positives),
                    *(("negative", value) for value in negatives),
                ]
                for role, value in values:
                    digest = text_digest(value)
                    checked[role] += 1
                    if digest in blocked:
                        matched_occurrences[role] += 1
                        matched[digest][role] += 1
                        matched_sources[digest][source] += 1
                rows += 1
        if provenance_handle is not None and provenance_handle.readline():
            raise ValueError("Provenance has more rows than training data")
    finally:
        if provenance_handle is not None:
            provenance_handle.close()

    locations: dict[bytes, list[dict[str, str]]] = defaultdict(list)
    matched_hashes = set(matched)
    for path, kind in files:
        for digest in read_digest_lines(path):
            if digest in matched_hashes:
                locations[digest].append(
                    {
                        "kind": kind,
                        "task_path": str(path.parent.relative_to(blocklist_root)),
                    }
                )

    unique_critical: set[bytes] = set()
    unique_corpus: set[bytes] = set()
    matches = []
    for digest in sorted(matched):
        kinds = {location["kind"] for location in locations[digest]}
        if kinds & CRITICAL_KINDS:
            unique_critical.add(digest)
        if "corpus_text" in kinds:
            unique_corpus.add(digest)
        matches.append(
            {
                "sha256": digest.hex(),
                "training_role_occurrences": dict(sorted(matched[digest].items())),
                "source_occurrences": dict(sorted(matched_sources[digest].items())),
                "benchmark_locations": locations[digest],
            }
        )

    if unique_critical:
        status = "critical_query_or_evaluation_text_overlap"
    elif unique_corpus:
        status = "pass_with_retrieval_corpus_exposure"
    else:
        status = "pass_no_exact_text_overlap"
    root_manifest = blocklist_root / "manifest.json"
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "train": {"path": str(train), "sha256": sha256_file(train)},
            "provenance": (
                {"path": str(provenance), "sha256": sha256_file(provenance)}
                if provenance
                else None
            ),
            "blocklist_root": str(blocklist_root),
            "blocklist_manifest_sha256": (
                sha256_file(root_manifest) if root_manifest.is_file() else None
            ),
        },
        "rows": rows,
        "blocklist": {
            "files": len(files),
            "unique_text_hashes": len(blocked),
            "hash_occurrences_by_kind": blocked_occurrences,
        },
        "checked_training_text_occurrences": dict(sorted(checked.items())),
        "matched_training_text_occurrences": dict(sorted(matched_occurrences.items())),
        "unique_matches": len(matched),
        "unique_critical_query_or_evaluation_matches": len(unique_critical),
        "unique_retrieval_corpus_matches": len(unique_corpus),
        "matches": matches,
        "status": status,
        "interpretation": (
            "query_text/evaluation_text overlap is critical; corpus-only overlap can "
            "arise from an explicitly disclosed shared task-train corpus and prevents "
            "a clean zero-shot claim"
        ),
    }


def main() -> None:
    args = parse_args()
    report = audit(
        args.train.resolve(),
        args.provenance.resolve() if args.provenance else None,
        args.blocklist_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_any_text and report["unique_matches"]:
        raise SystemExit(3)
    if args.fail_on_critical and report["unique_critical_query_or_evaluation_matches"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
