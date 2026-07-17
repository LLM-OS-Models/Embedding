#!/usr/bin/env python3
"""Build a fixed finance/knowledge retrieval selector from sealed source splits.

The artifact is deliberately not a public benchmark and is never training
input. Every query and candidate passage is removed when its normalized text
appears in any declared current/future training role or in the pinned public
benchmark blocklist. Selection is deterministic by source ID hash and the
manifest binds every input and emitted byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from scripts.audit_training_benchmark_overlap import (
        blocklist_files,
        load_blocked,
        message_content,
        nested_contents,
        semantic_query_body,
        text_digest,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from audit_training_benchmark_overlap import (
        blocklist_files,
        load_blocked,
        message_content,
        nested_contents,
        semantic_query_body,
        text_digest,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/multidomain-selection-heldout-v1"
TRAINING_HASH_CACHE = ROOT / "outputs/evaluation/.multidomain-selection-training-hash-cache-v1"
FINANCE_VALIDATION = (
    ROOT
    / "outputs/assets/bcai-finance-kor-embedding-triplet/data/validation-00000-of-00001.parquet"
)
KOTSQA_TEST = ROOT / "outputs/assets/kotsqa-v2/test.parquet"
FINANCE_SHA256 = "eb0d31a4d8ae6fb6ca81d8813f7a2a4389e31cc1f0233f295ffa9202058e8c63"
KOTSQA_SHA256 = "5e8719d1d48b45dc9d325dd448a201b6d739c1e9921274faae66c2883656e949"
CREATED_AT_UTC = "2026-07-17T09:37:53.810201+00:00"
MANIFEST_SHA256 = "86fea553c6652388b1f67160c0e2e6b7626acf8929f86c1a2708156bd89b3c46"
DEFAULT_JSONL_TRAINING = (
    ROOT / "outputs/data/performance-v1/ablation-200k/train.jsonl",
    ROOT / "outputs/data/performance-v1/performance-1m/train.jsonl",
    ROOT / "outputs/data/performance-v1/sionic-retrieval-train-family-4146/train.jsonl",
    ROOT / "outputs/data/performance-v1/sionic-squad-train-60k/train.jsonl",
    ROOT / "outputs/data/performance-v1/sionic-health-multilingual-100k/train.jsonl",
    ROOT / "outputs/data/performance-v1/sionic-autorag-domain-100k/train.jsonl",
    ROOT / "outputs/data/legal-performance-v1/train.bootstrap.jsonl",
)
DEFAULT_PARQUET_TRAINING = (
    ROOT
    / "outputs/assets/bcai-finance-kor-embedding-triplet/data/train-00000-of-00001.parquet",
    ROOT / "outputs/assets/kotsqa-v2/train.parquet",
)
ZERO_WIDTH_TRANSLATION = str.maketrans("", "", "\u200b\u200c\u200d\u2060\ufeff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--finance-rows", type=int, default=900)
    parser.add_argument("--knowledge-rows", type=int, default=1000)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Record workspace-relative provenance without leaking one host layout."""

    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Selection input is outside the workspace: {path.name}") from error


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).translate(ZERO_WIDTH_TRANSLATION)
    return " ".join(value.replace("\r\n", "\n").replace("\r", "\n").split())


def normalized_digest(value: str) -> bytes:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).digest()


def iter_jsonl_training_texts(path: Path) -> Iterable[str]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            query = message_content(row.get("messages"), "messages", line_number)
            yield query
            yield semantic_query_body(query)
            yield from nested_contents(
                row.get("positive_messages"), "positive_messages", line_number
            )
            yield from nested_contents(
                row.get("negative_messages"), "negative_messages", line_number
            )


def parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def load_forbidden_training_hashes(
    jsonl_paths: Iterable[Path], parquet_paths: Iterable[Path]
) -> tuple[set[bytes], list[dict[str, Any]]]:
    jsonl_paths = tuple(jsonl_paths)
    parquet_paths = tuple(parquet_paths)
    descriptors: list[dict[str, Any]] = []
    for path, kind in (
        *((path, "strict-embedding-jsonl") for path in jsonl_paths),
        *((path, "sealed-source-train-parquet") for path in parquet_paths),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Declared training role is unavailable: {path}")
        descriptors.append(
            {"path": str(path.resolve()), "sha256": sha256_file(path), "kind": kind}
        )
    cache_key = hashlib.sha256(
        json.dumps(descriptors, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cache_dir = TRAINING_HASH_CACHE / cache_key
    cache_payload = cache_dir / "digests.bin"
    cache_manifest = cache_dir / "manifest.json"
    if cache_payload.is_file() and cache_manifest.is_file():
        manifest = json.loads(cache_manifest.read_text(encoding="utf-8"))
        payload = cache_payload.read_bytes()
        digest_count = manifest.get("digest_count")
        if (
            manifest.get("cache_key") == cache_key
            and manifest.get("inputs") == descriptors
            and isinstance(digest_count, int)
            and digest_count * 32 == len(payload)
            and manifest.get("payload_sha256") == hashlib.sha256(payload).hexdigest()
        ):
            return (
                {payload[offset : offset + 32] for offset in range(0, len(payload), 32)},
                manifest["training_inputs"],
            )
    forbidden: set[bytes] = set()
    inputs: list[dict[str, Any]] = []
    for path, descriptor in zip(jsonl_paths, descriptors[: len(jsonl_paths)], strict=True):
        rows = 0
        for value in iter_jsonl_training_texts(path):
            forbidden.add(text_digest(value))
            rows += 1
        inputs.append(
            {
                "path": str(path.resolve()),
                "sha256": descriptor["sha256"],
                "text_occurrences": rows,
                "kind": "strict-embedding-jsonl",
            }
        )
    for path, descriptor in zip(
        parquet_paths, descriptors[len(jsonl_paths) :], strict=True
    ):
        rows = parquet_rows(path)
        occurrences = 0
        for row in rows:
            if {"anchor", "positive", "negative"} <= row.keys():
                values = (row["anchor"], row["positive"], row["negative"])
            elif {"question", "passages"} <= row.keys():
                values = (row["question"], *(row["passages"] or []))
            else:
                raise ValueError(f"Unsupported sealed-training parquet schema: {path}")
            for value in values:
                if not isinstance(value, str) or not normalize_text(value):
                    raise ValueError(f"Empty training text in {path}")
                forbidden.add(normalized_digest(value))
                occurrences += 1
        inputs.append(
            {
                "path": str(path.resolve()),
                "sha256": descriptor["sha256"],
                "text_occurrences": occurrences,
                "kind": "sealed-source-train-parquet",
            }
        )
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{cache_key}.", dir=cache_dir.parent))
    try:
        payload_path = staging / "digests.bin"
        with payload_path.open("wb") as handle:
            for digest in sorted(forbidden):
                handle.write(digest)
        payload_sha = sha256_file(payload_path)
        (staging / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cache_key": cache_key,
                    "inputs": descriptors,
                    "training_inputs": inputs,
                    "digest_count": len(forbidden),
                    "payload_sha256": payload_sha,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if cache_dir.exists():
            shutil.rmtree(staging)
        else:
            os.replace(staging, cache_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return forbidden, inputs


def relevant_knowledge_passages(row: dict[str, Any]) -> set[int]:
    answers = [normalize_text(value).casefold() for value in row.get("answers") or []]
    answers = [value for value in answers if value]
    return {
        index
        for index, passage in enumerate(row.get("passages") or [])
        if isinstance(passage, str)
        and any(answer in normalize_text(passage).casefold() for answer in answers)
    }


def add_corpus(
    corpus: dict[tuple[str, bytes], dict[str, str]], *, domain: str, text: str
) -> str:
    digest = normalized_digest(text)
    key = (domain, digest)
    corpus_id = f"{domain}-{digest.hex()[:24]}"
    existing = corpus.get(key)
    row = {"_id": corpus_id, "text": normalize_text(text), "domain": domain}
    if existing is not None and existing != row:
        raise ValueError("Normalized corpus hash collision")
    corpus[key] = row
    return corpus_id


def build_rows(
    *, forbidden: set[bytes], blocked: set[bytes], finance_count: int, knowledge_count: int
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if sha256_file(FINANCE_VALIDATION) != FINANCE_SHA256:
        raise ValueError("Pinned finance validation bytes drifted")
    if sha256_file(KOTSQA_TEST) != KOTSQA_SHA256:
        raise ValueError("Pinned KoTSQA test bytes drifted")
    excluded = forbidden | blocked
    queries: list[dict[str, str]] = []
    qrels: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    corpus: dict[tuple[str, bytes], dict[str, str]] = {}
    seen_queries: set[bytes] = set()
    stats = {
        "finance_forbidden_or_duplicate": 0,
        "finance_corpus_training_text_occurrences": 0,
        "knowledge_no_exact_answer_passage": 0,
        "knowledge_forbidden_or_duplicate": 0,
    }

    finance_rows = parquet_rows(FINANCE_VALIDATION)
    finance_rows.sort(
        key=lambda row: hashlib.sha256(str(row["anchor_chunk_id"]).encode()).hexdigest()
    )
    for source_index, row in enumerate(finance_rows):
        values = [row["anchor"], row["positive"], row["negative"]]
        digests = [normalized_digest(value) for value in values]
        if (
            digests[0] in forbidden
            or any(digest in blocked for digest in digests)
            or digests[0] in seen_queries
            or len(set(digests)) != len(digests)
        ):
            stats["finance_forbidden_or_duplicate"] += 1
            continue
        stats["finance_corpus_training_text_occurrences"] += sum(
            digest in forbidden for digest in digests[1:]
        )
        query_id = f"finance-{len([q for q in queries if q['domain'] == 'finance']):04d}"
        positive_id = add_corpus(corpus, domain="finance", text=row["positive"])
        add_corpus(corpus, domain="finance", text=row["negative"])
        queries.append(
            {"_id": query_id, "text": normalize_text(row["anchor"]), "domain": "finance"}
        )
        qrels.append({"query-id": query_id, "corpus-id": positive_id, "score": 1})
        provenance.append(
            {
                "query_id": query_id,
                "domain": "finance",
                "source": "BCCard/BCAI-Finance-Kor-Embedding-Triplet",
                "revision": "f63d59969dba9916bd34c86c82112331890b11da",
                "split": "validation",
                "source_row": source_index,
                "source_id_sha256": hashlib.sha256(
                    str(row["anchor_chunk_id"]).encode()
                ).hexdigest(),
            }
        )
        seen_queries.add(digests[0])
        if sum(query["domain"] == "finance" for query in queries) == finance_count:
            break

    knowledge_rows = parquet_rows(KOTSQA_TEST)
    knowledge_rows.sort(
        key=lambda row: hashlib.sha256(str(row["id"]).encode()).hexdigest()
    )
    knowledge_selected = 0
    for source_index, row in enumerate(knowledge_rows):
        relevant = relevant_knowledge_passages(row)
        if not relevant:
            stats["knowledge_no_exact_answer_passage"] += 1
            continue
        passages = row.get("passages") or []
        values = [row["question"], *passages]
        if any(not isinstance(value, str) or not normalize_text(value) for value in values):
            stats["knowledge_forbidden_or_duplicate"] += 1
            continue
        digests = [normalized_digest(value) for value in values]
        if (
            any(digest in excluded for digest in digests)
            or digests[0] in seen_queries
            or len(set(digests[1:])) != len(digests[1:])
        ):
            stats["knowledge_forbidden_or_duplicate"] += 1
            continue
        query_id = f"knowledge-{knowledge_selected:04d}"
        passage_ids = [
            add_corpus(corpus, domain="knowledge", text=passage) for passage in passages
        ]
        queries.append(
            {"_id": query_id, "text": normalize_text(row["question"]), "domain": "knowledge"}
        )
        for index in sorted(relevant):
            qrels.append(
                {"query-id": query_id, "corpus-id": passage_ids[index], "score": 1}
            )
        provenance.append(
            {
                "query_id": query_id,
                "domain": "knowledge",
                "source": "etri-lirs/KoTSQA-v.2.0",
                "revision": "ff9349df469a765b4561959e36ef1b3f377765cd",
                "split": "test-used-as-fixed-internal-selection",
                "source_row": source_index,
                "source_id_sha256": hashlib.sha256(str(row["id"]).encode()).hexdigest(),
                "relevant_passages": len(relevant),
            }
        )
        seen_queries.add(digests[0])
        knowledge_selected += 1
        if knowledge_selected == knowledge_count:
            break

    domain_counts = {
        domain: sum(query["domain"] == domain for query in queries)
        for domain in ("finance", "knowledge")
    }
    if domain_counts != {"finance": finance_count, "knowledge": knowledge_count}:
        raise RuntimeError(f"Insufficient eligible selection rows: {domain_counts}")
    corpus_rows = sorted(corpus.values(), key=lambda row: row["_id"])
    return queries, corpus_rows, qrels, provenance, {**stats, **domain_counts}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def verify_output(output: Path) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("Multidomain selection manifest identity drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("protocol_id") != "multidomain-selection-heldout-v1":
        raise ValueError("Multidomain selection manifest is not complete")
    for name, descriptor in manifest.get("files", {}).items():
        path = output / name
        if not path.is_file() or sha256_file(path) != descriptor.get("sha256"):
            raise ValueError(f"Multidomain output hash drifted: {name}")
        with path.open("rb") as handle:
            rows = sum(1 for _ in handle)
        if rows != descriptor.get("rows"):
            raise ValueError(f"Multidomain output row count drifted: {name}")
    return manifest


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    if args.verify_only:
        print(json.dumps(verify_output(output), ensure_ascii=False, indent=2))
        return
    if args.finance_rows < 1 or args.knowledge_rows < 1:
        raise ValueError("Domain row targets must be positive")
    if output.exists():
        verify_output(output)
        print(f"Reused verified multidomain selection artifact: {output}")
        return
    forbidden, training_inputs = load_forbidden_training_hashes(
        DEFAULT_JSONL_TRAINING, DEFAULT_PARQUET_TRAINING
    )
    block_files = blocklist_files(ROOT / "outputs/decontamination/benchmark_blocklist")
    blocked, blocked_occurrences = load_blocked(block_files)
    queries, corpus, qrels, provenance, stats = build_rows(
        forbidden=forbidden,
        blocked=blocked,
        finance_count=args.finance_rows,
        knowledge_count=args.knowledge_rows,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        files: dict[str, dict[str, Any]] = {}
        for name, rows in (
            ("queries.jsonl", queries),
            ("corpus.jsonl", corpus),
            ("qrels.jsonl", qrels),
            ("provenance.jsonl", provenance),
        ):
            count = write_jsonl(staging / name, rows)
            files[name] = {"rows": count, "sha256": sha256_file(staging / name)}
        manifest = {
            "schema_version": 1,
            "protocol_id": "multidomain-selection-heldout-v1",
            "created_at_utc": CREATED_AT_UTC,
            "status": "complete",
            "purpose": "fixed internal model selection; never training; not a public benchmark",
            "selection_rule": "source rows ordered by SHA-256 of immutable source ID",
            "domains": {
                "finance": {
                    "queries": stats["finance"],
                    "independence": "query-heldout; corpus exposure disclosed",
                    "corpus_training_text_occurrences": stats[
                        "finance_corpus_training_text_occurrences"
                    ],
                },
                "knowledge": {
                    "queries": stats["knowledge"],
                    "independence": "query-and-corpus exact-text-heldout",
                },
            },
            "source_inputs": [
                {
                    "path": portable_path(FINANCE_VALIDATION),
                    "sha256": FINANCE_SHA256,
                    "revision": "f63d59969dba9916bd34c86c82112331890b11da",
                    "split": "validation",
                },
                {
                    "path": portable_path(KOTSQA_TEST),
                    "sha256": KOTSQA_SHA256,
                    "revision": "ff9349df469a765b4561959e36ef1b3f377765cd",
                    "split": "test-used-as-fixed-internal-selection",
                },
            ],
            "training_roles_excluded": [
                {**row, "path": portable_path(Path(row["path"]))}
                for row in training_inputs
            ],
            "benchmark_blocklist": {
                "root": portable_path(
                    ROOT / "outputs/decontamination/benchmark_blocklist"
                ),
                "manifest_sha256": sha256_file(
                    ROOT / "outputs/decontamination/benchmark_blocklist/manifest.json"
                ),
                "hash_occurrences_by_kind": blocked_occurrences,
            },
            "assertions": {
                "all_selected_query_exact_training_text_overlap": 0,
                "knowledge_query_and_corpus_exact_training_text_overlap": 0,
                "all_selected_query_and_corpus_benchmark_blocklist_overlap": 0,
                "public_benchmark_score_used_for_selection": False,
            },
            "exclusion_counts": {
                key: value
                for key, value in stats.items()
                if key.endswith("forbidden_or_duplicate")
                or key.endswith("no_exact_answer_passage")
            },
            "files": files,
            "limitations": [
                "finance is query-heldout but shares domain corpus text with BCAI train",
                "knowledge is exact-text heldout; neither domain is source-document Grade I or unseen-source Grade Z",
                "KoTSQA relevance is inferred only when a normalized answer occurs in a passage",
                "finance and knowledge are selection domains; legal is evaluated by the separate Grade-I board",
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(verify_output(output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
