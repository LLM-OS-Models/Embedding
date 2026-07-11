#!/usr/bin/env python3
"""Prepare a small, deterministic ms-swift embedding-training split.

This script is intentionally for pipeline validation.  The default source does
not declare a dataset license, so its output must not be used for a public
release candidate without a separate rights review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_DATASET = "nlpai-lab/ko-triplet-v1.0"
DEFAULT_REVISION = "1f5d72d21ae8309b5221a588b13930b423385bff"
DEFAULT_INSTRUCTION = (
    "Given a Korean web search query, retrieve relevant passages that answer the query"
)
HANGUL_RE = re.compile(r"[가-힣]")
WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=288)
    parser.add_argument("--val-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--min-query-chars", type=int, default=4)
    parser.add_argument("--min-document-chars", type=int, default=20)
    parser.add_argument("--max-document-chars", type=int, default=80_000)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return WHITESPACE_RE.sub(" ", text).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_example(query: str, positive: str, negative: str, instruction: str) -> dict[str, Any]:
    instructed_query = f"Instruct: {instruction}\nQuery:{query}"
    return {
        "messages": [{"role": "user", "content": instructed_query}],
        "positive_messages": [[{"role": "user", "content": positive}]],
        "negative_messages": [[{"role": "user", "content": negative}]],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.limit <= args.val_size or args.val_size < 2:
        raise ValueError("limit must be greater than val-size, and val-size must be at least 2")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream = load_dataset(
        args.dataset,
        revision=args.revision,
        split=args.split,
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejected = {
        "missing_or_short": 0,
        "non_korean_query": 0,
        "duplicate": 0,
        "positive_equals_negative": 0,
        "document_too_long": 0,
    }

    for raw in stream:
        query = normalize_text(raw.get("query"))
        positive = normalize_text(raw.get("document"))
        negative = normalize_text(raw.get("hard_negative"))
        if (
            len(query) < args.min_query_chars
            or len(positive) < args.min_document_chars
            or len(negative) < args.min_document_chars
        ):
            rejected["missing_or_short"] += 1
            continue
        if not HANGUL_RE.search(query):
            rejected["non_korean_query"] += 1
            continue
        if max(len(positive), len(negative)) > args.max_document_chars:
            rejected["document_too_long"] += 1
            continue
        if positive == negative:
            rejected["positive_equals_negative"] += 1
            continue
        identity = text_hash("\0".join((query, positive, negative)))
        if identity in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(identity)
        rows.append(format_example(query, positive, negative, args.instruction))
        if len(rows) >= args.limit:
            break

    if len(rows) != args.limit:
        raise RuntimeError(f"Only collected {len(rows)} usable rows; requested {args.limit}")

    train_rows = rows[: -args.val_size]
    val_rows = rows[-args.val_size :]
    train_path = args.output_dir / "train.jsonl"
    val_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "pipeline-validation-only",
        "release_eligible": False,
        "release_blocker": "source dataset card does not declare an explicit license",
        "source": {
            "dataset": args.dataset,
            "revision": args.revision,
            "split": args.split,
            "url": f"https://huggingface.co/datasets/{args.dataset}/tree/{args.revision}",
            "declared_license": None,
        },
        "sampling": {
            "seed": args.seed,
            "shuffle_buffer": args.shuffle_buffer,
            "requested": args.limit,
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
            "rejected_before_completion": rejected,
        },
        "format": {
            "trainer": "ms-swift InfoNCE",
            "instruction": args.instruction,
            "positive_count": 1,
            "explicit_hard_negative_count": 1,
        },
        "files": {
            "train.jsonl": {"sha256": file_hash(train_path), "rows": len(train_rows)},
            "validation.jsonl": {"sha256": file_hash(val_path), "rows": len(val_rows)},
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
