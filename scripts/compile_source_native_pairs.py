#!/usr/bin/env python3
"""Compile source-native query/positive candidates into strict embedding rows.

The deterministic negatives produced here are bootstrap candidates only. A
current-student dense miner and reranker should replace them before final use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


QUERY_INSTRUCTION = (
    "Instruct: Given a Korean legal or public-administration search query, "
    "retrieve the passage that provides the requested rule or holding\nQuery: "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--negatives-per-row", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


def load_candidates(paths: list[Path], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_pairs: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                required = {"id", "query", "positive", "pair_type", "provenance"}
                if not isinstance(row, dict) or not required <= set(row):
                    raise ValueError(f"{path}:{line_number}: invalid source candidate")
                if row["id"] in seen_ids:
                    raise ValueError(f"{path}:{line_number}: duplicate id {row['id']}")
                identity = stable_hash(row["query"].strip(), row["positive"].strip())
                if identity in seen_pairs:
                    continue
                seen_ids.add(row["id"])
                seen_pairs.add(identity)
                rows.append(row)
    rows.sort(key=lambda row: stable_hash(str(seed), row["id"]))
    return rows


def source_key(row: dict[str, Any]) -> str:
    provenance = row["provenance"]
    return str(provenance.get("repository") or provenance.get("repository_url") or "unknown")


def choose_negatives(
    row: dict[str, Any], pool: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    if len(pool) <= count:
        raise ValueError(f"Source pool is too small for {row['id']}")
    selected = []
    positive = row["positive"].strip()
    query = row["query"].strip()
    # Pools are sorted once in main. Per-row hashing selects a deterministic
    # cyclic starting point without the previous O(rows * pool log pool) sort.
    start = int(stable_hash(str(seed), row["id"], "negative-offset")[:16], 16) % len(pool)
    for offset in range(len(pool)):
        candidate = pool[(start + offset) % len(pool)]
        if candidate["id"] == row["id"]:
            continue
        text = candidate["positive"].strip()
        if not text or text == positive or text == query:
            continue
        selected.append(candidate)
        if len(selected) == count:
            return selected
    raise ValueError(f"Not enough unique negatives for {row['id']}")


def atomic_text_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    return os.fdopen(fd, "w", encoding="utf-8"), Path(name)


def main() -> None:
    args = parse_args()
    if args.negatives_per_row < 1:
        raise ValueError("--negatives-per-row must be positive")
    candidates = load_candidates([path.resolve() for path in args.input], args.seed)
    if args.max_rows:
        candidates = candidates[: args.max_rows]
    if len(candidates) < 2:
        raise ValueError("At least two candidates are required")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_source[source_key(row)].append(row)
    for pool in by_source.values():
        pool.sort(key=lambda row: stable_hash(str(args.seed), "source-pool", row["id"]))

    train_handle, train_temp = atomic_text_writer(args.output.resolve())
    provenance_handle, provenance_temp = atomic_text_writer(
        args.provenance_output.resolve()
    )
    source_counts: Counter[str] = Counter()
    try:
        for index, row in enumerate(candidates):
            source = source_key(row)
            negatives = choose_negatives(
                row, by_source[source], args.negatives_per_row, args.seed
            )
            strict = {
                "messages": [
                    {"role": "user", "content": QUERY_INSTRUCTION + row["query"].strip()}
                ],
                "positive_messages": [
                    [{"role": "user", "content": row["positive"].strip()}]
                ],
                "negative_messages": [
                    [{"role": "user", "content": candidate["positive"].strip()}]
                    for candidate in negatives
                ],
            }
            audit = {
                "row_index": index,
                "source_candidate_id": row["id"],
                "source": source,
                "pair_type": row["pair_type"],
                "label_origin": row.get("label_origin"),
                "provenance": row["provenance"],
                "bootstrap_negative_candidate_ids": [item["id"] for item in negatives],
                "bootstrap_negative_policy": "seeded same-source candidate; replace with current-student mining",
                "benchmark_exposure": "target-like legal/public-administration data; not clean zero-shot for LawIRKo/AutoRAG legal",
            }
            train_handle.write(json.dumps(strict, ensure_ascii=False, separators=(",", ":")) + "\n")
            provenance_handle.write(
                json.dumps(audit, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            source_counts[source] += 1
        train_handle.flush()
        os.fsync(train_handle.fileno())
        provenance_handle.flush()
        os.fsync(provenance_handle.fileno())
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

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(candidates),
        "seed": args.seed,
        "negatives_per_row": args.negatives_per_row,
        "source_counts": dict(sorted(source_counts.items())),
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path.resolve())}
            for path in args.input
        ],
        "files": {
            args.output.name: {"rows": len(candidates), "sha256": sha256(args.output)},
            args.provenance_output.name: {
                "rows": len(candidates),
                "sha256": sha256(args.provenance_output),
            },
        },
        "query_instruction": QUERY_INSTRUCTION,
        "release_eligible": False,
        "use_policy": "performance-noncommercial-target-adapted",
        "required_next_stage": "current-student dense retrieval plus reranker false-negative filtering",
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
