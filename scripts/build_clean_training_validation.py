#!/usr/bin/env python3
"""Build and verify a strict training-validation split from the legal holdout.

The source artifact remains the comprehensive 10K retrieval evaluation set.
This script derives a smaller source-balanced Trainer validation set without
weakening the source-document or benchmark-exclusion contracts.  Every source
query and document is compared, under the same normalization policy, with all
roles in every declared training JSONL before it can be selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/evaluation/legal-source-heldout-i-v2-text-strict"
DEFAULT_OUTPUT = ROOT / "outputs/data/validation/legal-source-heldout-i-v2-text-strict-512"
DEFAULT_INSTRUCTION = (
    "Instruct: Given a Korean web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)
ZERO_WIDTH_TRANSLATION = str.maketrans("", "", "\u200b\u200c\u200d\u2060\ufeff")
WORD_RE = re.compile(r"[0-9A-Za-z가-힣]+")
STRICT_FIELDS = {"messages", "positive_messages", "negative_messages"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-data", type=Path, action="append", default=[])
    parser.add_argument("--training-provenance", type=Path, action="append", default=[])
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--negative-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    return parser.parse_args()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.translate(ZERO_WIDTH_TRANSLATION)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(value.split())


def text_digest(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def validate_source(source_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], dict]:
    manifest_path = source_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assertions = manifest.get("assertions", {})
    required_zero = (
        "selected_query_hash_overlap_with_benchmark",
        "selected_positive_hash_overlap_with_benchmark",
        "selected_source_candidate_id_overlap_with_training",
        "selected_source_document_sha256_overlap_with_training",
        "selected_query_hash_overlap_with_training_text",
        "selected_positive_hash_overlap_with_training_text",
    )
    if (
        manifest.get("status") != "complete"
        or manifest.get("independence", {}).get("grade") != "I"
        or manifest.get("independence", {}).get("not_grade") != "Z"
        or any(assertions.get(key) != 0 for key in required_zero)
    ):
        raise ValueError("Source holdout did not pass its independence contract")
    for name in ("queries.jsonl", "corpus.jsonl", "qrels.jsonl", "provenance.jsonl"):
        declared = manifest.get("files", {}).get(name, {})
        path = source_dir / name
        if declared.get("sha256") != sha256_file(path):
            raise ValueError(f"Source holdout hash mismatch: {name}")
    queries = read_jsonl(source_dir / "queries.jsonl")
    corpus = read_jsonl(source_dir / "corpus.jsonl")
    qrels = read_jsonl(source_dir / "qrels.jsonl")
    if not queries or len(queries) != len(corpus) or len(queries) != len(qrels):
        raise ValueError("Source query/corpus/qrel row counts are inconsistent")
    query_by_id = {str(row["_id"]): row for row in queries}
    corpus_by_id = {str(row["_id"]): row for row in corpus}
    if len(query_by_id) != len(queries) or len(corpus_by_id) != len(corpus):
        raise ValueError("Source IDs are not unique")
    positives: dict[str, str] = {}
    for row in qrels:
        query_id = str(row["query-id"])
        corpus_id = str(row["corpus-id"])
        if row.get("score") != 1 or query_id in positives:
            raise ValueError("Expected exactly one score-1 qrel per query")
        positives[query_id] = corpus_id
    if set(positives) != set(query_by_id) or not set(positives.values()) <= set(corpus_by_id):
        raise ValueError("Source qrel identifiers are inconsistent")
    return queries, corpus, positives, manifest


def candidate_hashes(
    queries: list[dict[str, Any]], corpus: list[dict[str, Any]]
) -> tuple[dict[str, set[tuple[str, str]]], dict[str, str], dict[str, str]]:
    targets: dict[str, set[tuple[str, str]]] = defaultdict(set)
    query_hashes: dict[str, str] = {}
    corpus_hashes: dict[str, str] = {}
    for row in queries:
        item_id = str(row["_id"])
        digest = text_digest(str(row["text"]))
        query_hashes[item_id] = digest
        targets[digest].add(("query", item_id))
    for row in corpus:
        item_id = str(row["_id"])
        digest = text_digest(str(row["text"]))
        corpus_hashes[item_id] = digest
        targets[digest].add(("document", item_id))
    return targets, query_hashes, corpus_hashes


def audit_training_data(
    paths: list[Path], targets: dict[str, set[tuple[str, str]]]
) -> tuple[set[tuple[str, str]], list[dict[str, Any]]]:
    blocked: set[tuple[str, str]] = set()
    reports: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        role_occurrences: Counter[str] = Counter()
        matched_occurrences: Counter[str] = Counter()
        matched_items: set[tuple[str, str]] = set()
        rows = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"{path}:{line_number}: blank line")
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != STRICT_FIELDS:
                    raise ValueError(f"{path}:{line_number}: strict schema violation")
                query = message_content(row["messages"], "messages", line_number)
                positives = nested_contents(row["positive_messages"], "positive_messages", line_number)
                negatives = nested_contents(row["negative_messages"], "negative_messages", line_number)
                values = [
                    ("query_full", query),
                    ("query_body", semantic_query_body(query)),
                    *(("positive", value) for value in positives),
                    *(("negative", value) for value in negatives),
                ]
                for role, value in values:
                    role_occurrences[role] += 1
                    matches = targets.get(text_digest(value), ())
                    if matches:
                        matched_occurrences[role] += 1
                        matched_items.update(matches)
                rows += 1
        blocked.update(matched_items)
        reports.append(
            {
                "path": relative_path(path),
                "bytes": path.stat().st_size,
                "rows": rows,
                "sha256": sha256_file(path),
                "checked_role_occurrences": dict(sorted(role_occurrences.items())),
                "matched_role_occurrences": dict(sorted(matched_occurrences.items())),
                "matched_unique_queries": sum(kind == "query" for kind, _ in matched_items),
                "matched_unique_documents": sum(kind == "document" for kind, _ in matched_items),
            }
        )
    return blocked, reports


def recursive_document_hashes(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "source_document_sha256" and isinstance(child, str):
                yield child
            else:
                yield from recursive_document_hashes(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_document_hashes(child)


def audit_training_provenance(
    paths: list[Path], source_documents: set[str]
) -> tuple[set[str], list[dict[str, Any]]]:
    matched_all: set[str] = set()
    reports: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        matched: set[str] = set()
        seen_document_occurrences = 0
        rows = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"{path}:{line_number}: blank line")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                for digest in recursive_document_hashes(row):
                    seen_document_occurrences += 1
                    if digest in source_documents:
                        matched.add(digest)
                rows += 1
        matched_all.update(matched)
        reports.append(
            {
                "path": relative_path(path),
                "bytes": path.stat().st_size,
                "rows": rows,
                "sha256": sha256_file(path),
                "source_document_occurrences": seen_document_occurrences,
                "matched_unique_source_documents": len(matched),
            }
        )
    return matched_all, reports


def stable_key(seed: int, *values: str) -> str:
    material = "\0".join((str(seed), *values))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def lexical_features(value: str) -> set[str]:
    normalized = normalize_text(value).lower()
    compact = "".join(normalized.split())
    features = {f"w:{token}" for token in WORD_RE.findall(normalized) if len(token) >= 2}
    features.update(f"c2:{compact[index:index + 2]}" for index in range(max(0, len(compact) - 1)))
    return features


def strict_row(query: str, positive: str, negatives: list[str], instruction: str) -> dict[str, Any]:
    message = lambda value: [{"role": "user", "content": value}]
    return {
        "messages": message(instruction + query),
        "positive_messages": [message(positive)],
        "negative_messages": [message(value) for value in negatives],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.target_size < 1 or args.negative_count < 1:
        raise ValueError("target-size and negative-count must be positive")
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    queries, corpus, positives, source_manifest = validate_source(source)
    query_by_id = {str(row["_id"]): row for row in queries}
    corpus_by_id = {str(row["_id"]): row for row in corpus}
    targets, query_hashes, corpus_hashes = candidate_hashes(queries, corpus)
    blocked_items, training_reports = audit_training_data(args.training_data, targets)
    all_source_documents = {
        str(row.get("metadata", {}).get("source_document_sha256")) for row in corpus
    }
    if None in all_source_documents or "None" in all_source_documents:
        raise ValueError("Source corpus metadata is missing source_document_sha256")
    blocked_documents, provenance_reports = audit_training_provenance(
        args.training_provenance, all_source_documents
    )

    eligible_queries: dict[str, list[str]] = defaultdict(list)
    eligible_corpus: dict[str, list[str]] = defaultdict(list)
    for query_id, query in query_by_id.items():
        corpus_id = positives[query_id]
        query_meta = query.get("metadata", {})
        corpus_meta = corpus_by_id[corpus_id].get("metadata", {})
        repository = str(query_meta.get("repository"))
        if repository != str(corpus_meta.get("repository")):
            raise ValueError("Positive query/corpus repositories differ")
        document_sha = str(corpus_meta.get("source_document_sha256"))
        if (
            ("query", query_id) in blocked_items
            or ("document", corpus_id) in blocked_items
            or document_sha in blocked_documents
        ):
            continue
        eligible_queries[repository].append(query_id)
    for corpus_id, document in corpus_by_id.items():
        meta = document.get("metadata", {})
        repository = str(meta.get("repository"))
        document_sha = str(meta.get("source_document_sha256"))
        if ("document", corpus_id) not in blocked_items and document_sha not in blocked_documents:
            eligible_corpus[repository].append(corpus_id)

    repositories = sorted(eligible_queries)
    if not repositories or args.target_size % len(repositories):
        raise ValueError("target-size must divide evenly across source repositories")
    per_repository = args.target_size // len(repositories)
    selected_query_ids: list[str] = []
    for repository in repositories:
        ranked = sorted(
            eligible_queries[repository],
            key=lambda query_id: stable_key(args.seed, repository, query_id, positives[query_id]),
        )
        if len(ranked) < per_repository:
            raise ValueError(f"Insufficient eligible queries for {repository}")
        selected_query_ids.extend(ranked[:per_repository])
    selected_query_ids.sort(
        key=lambda query_id: stable_key(args.seed, "output", query_id, positives[query_id])
    )

    feature_by_corpus: dict[str, set[str]] = {}
    document_frequency: Counter[str] = Counter()
    for corpus_id, document in corpus_by_id.items():
        title = str(document.get("title") or "")
        features = lexical_features(title + "\n" + str(document["text"]))
        feature_by_corpus[corpus_id] = features
        document_frequency.update(features)
    corpus_count = len(corpus_by_id)
    idf = {
        feature: math.log((corpus_count + 1) / (frequency + 1)) + 1.0
        for feature, frequency in document_frequency.items()
    }

    output.mkdir(parents=True, exist_ok=True)
    validation_path = output / "validation.jsonl"
    provenance_path = output / "provenance.jsonl"
    validation_lines: list[str] = []
    provenance_lines: list[str] = []
    selected_docs: set[str] = set()
    selected_negative_docs: set[str] = set()
    repository_counts: Counter[str] = Counter()
    for row_index, query_id in enumerate(selected_query_ids):
        query = query_by_id[query_id]
        positive_id = positives[query_id]
        positive = corpus_by_id[positive_id]
        repository = str(query["metadata"]["repository"])
        query_features = lexical_features(str(query["text"]))
        candidates: list[tuple[float, str, str]] = []
        for corpus_id in eligible_corpus[repository]:
            if corpus_id == positive_id:
                continue
            overlap = query_features & feature_by_corpus[corpus_id]
            score = sum(idf[feature] for feature in overlap)
            candidates.append(
                (-score, stable_key(args.seed, "negative", query_id, corpus_id), corpus_id)
            )
        candidates.sort()
        negative_ids = [item[2] for item in candidates[: args.negative_count]]
        if len(negative_ids) != args.negative_count:
            raise ValueError(f"Insufficient negatives for query {query_id}")
        negatives = [str(corpus_by_id[item]["text"]) for item in negative_ids]
        strict = strict_row(str(query["text"]), str(positive["text"]), negatives, args.instruction)
        strict_line = compact_json(strict)
        positive_document_sha = str(positive["metadata"]["source_document_sha256"])
        negative_document_shas = [
            str(corpus_by_id[item]["metadata"]["source_document_sha256"])
            for item in negative_ids
        ]
        selected_docs.add(positive_document_sha)
        selected_negative_docs.update(negative_document_shas)
        repository_counts[repository] += 1
        provenance = {
            "row_index": row_index,
            "row_sha256": hashlib.sha256(strict_line.encode("utf-8")).hexdigest(),
            "query_id": query_id,
            "positive_corpus_id": positive_id,
            "negative_corpus_ids": negative_ids,
            "repository": repository,
            "source_candidate_id": query["metadata"].get("source_candidate_id"),
            "source_document_sha256": positive_document_sha,
            "negative_source_document_sha256": negative_document_shas,
            "independence_grade": "I",
            "independence_label": "same-repository whole-source-document-held-out",
            "selection_key_sha256": stable_key(args.seed, repository, query_id, positive_id),
            "negative_policy": "same-repository highest IDF-weighted normalized word/character-bigram overlap; stable SHA-256 tie-break",
        }
        validation_lines.append(strict_line + "\n")
        provenance_lines.append(compact_json(provenance) + "\n")
    validation_path.write_text("".join(validation_lines), encoding="utf-8")
    provenance_path.write_text("".join(provenance_lines), encoding="utf-8")

    files = {}
    for name in ("validation.jsonl", "provenance.jsonl"):
        path = output / name
        files[name] = {
            "bytes": path.stat().st_size,
            "rows": args.target_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": 1,
        "artifact_id": "legal-source-heldout-i-v2-text-strict-training-validation",
        "status": "complete",
        "parameters": {
            "seed": args.seed,
            "target_size": args.target_size,
            "negative_count": args.negative_count,
            "instruction_sha256": hashlib.sha256(args.instruction.encode("utf-8")).hexdigest(),
            "source_balance": "equal stable-hash quota across repositories",
            "negative_policy": "same-repository IDF-weighted lexical hard negatives with stable SHA-256 tie-break",
        },
        "source_holdout": {
            "path": relative_path(source),
            "manifest_sha256": sha256_file(source / "manifest.json"),
            "artifact_id": source_manifest.get("artifact_id"),
            "independence_grade": "I",
            "not_grade": "Z",
            "training_query_text_overlap": source_manifest["assertions"]["selected_query_hash_overlap_with_training_text"],
            "training_positive_text_overlap": source_manifest["assertions"]["selected_positive_hash_overlap_with_training_text"],
            "benchmark_query_overlap": source_manifest["assertions"]["selected_query_hash_overlap_with_benchmark"],
            "benchmark_positive_overlap": source_manifest["assertions"]["selected_positive_hash_overlap_with_benchmark"],
            "training_source_document_overlap": source_manifest["assertions"]["selected_source_document_sha256_overlap_with_training"],
        },
        "training_data_audit": training_reports,
        "training_provenance_audit": provenance_reports,
        "selection": {
            "eligible_query_counts": dict(sorted((key, len(value)) for key, value in eligible_queries.items())),
            "eligible_corpus_counts": dict(sorted((key, len(value)) for key, value in eligible_corpus.items())),
            "selected_repository_counts": dict(sorted(repository_counts.items())),
            "selected_unique_positive_source_documents": len(selected_docs),
            "selected_unique_negative_source_documents": len(selected_negative_docs),
        },
        "assertions": {
            "source_holdout_contract_verified": True,
            "selected_query_training_text_overlap": 0,
            "selected_positive_training_text_overlap": 0,
            "selected_negative_training_text_overlap": 0,
            "selected_source_document_training_provenance_overlap": 0,
            "selected_unique_query_hashes": len({query_hashes[item] for item in selected_query_ids}),
            "selected_unique_positive_hashes": len({corpus_hashes[positives[item]] for item in selected_query_ids}),
            "selected_unique_positive_source_documents": len(selected_docs),
            "negative_is_never_positive_for_same_row": True,
            "public_benchmark_used_for_model_selection": False,
        },
        "files": files,
        "claims": {
            "allowed": "same-repository whole-source-document-held-out (I) Trainer validation",
            "forbidden": ["clean zero-shot", "unseen-source", "independence grade Z"],
            "role": "finite-loss/completion monitoring and future internal checkpoint signal; comprehensive model selection remains the full 10K clean board",
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(compact_json(manifest) + "\n", encoding="utf-8")
    return manifest


def verify(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    manifest_path = output / "manifest.json"
    declared = json.loads(manifest_path.read_text(encoding="utf-8"))
    if declared.get("status") != "complete":
        raise ValueError("Validation manifest is not complete")
    for name, metadata in declared.get("files", {}).items():
        path = output / name
        if metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"Output hash mismatch: {name}")
    validation = read_jsonl(output / "validation.jsonl")
    provenance = read_jsonl(output / "provenance.jsonl")
    if len(validation) != args.target_size or len(provenance) != args.target_size:
        raise ValueError("Validation output row count mismatch")
    targets: dict[str, set[tuple[str, str]]] = defaultdict(set)
    source_documents: set[str] = set()
    for line_number, (row, evidence) in enumerate(zip(validation, provenance, strict=True), 1):
        if set(row) != STRICT_FIELDS:
            raise ValueError(f"validation row {line_number}: strict schema violation")
        query = message_content(row["messages"], "messages", line_number)
        positive = nested_contents(row["positive_messages"], "positive_messages", line_number)
        negatives = nested_contents(row["negative_messages"], "negative_messages", line_number)
        if len(positive) != 1 or len(negatives) != args.negative_count:
            raise ValueError(f"validation row {line_number}: role count mismatch")
        item_key = str(line_number)
        targets[text_digest(semantic_query_body(query))].add(("query", item_key))
        targets[text_digest(positive[0])].add(("positive", item_key))
        for negative in negatives:
            targets[text_digest(negative)].add(("negative", item_key))
        source_documents.add(str(evidence["source_document_sha256"]))
        source_documents.update(str(value) for value in evidence["negative_source_document_sha256"])
        strict_line = compact_json(row)
        if evidence.get("row_index") != line_number - 1 or evidence.get("row_sha256") != hashlib.sha256(strict_line.encode("utf-8")).hexdigest():
            raise ValueError(f"validation row {line_number}: provenance mismatch")
    blocked, training_reports = audit_training_data(args.training_data, targets)
    blocked_documents, provenance_reports = audit_training_provenance(
        args.training_provenance, source_documents
    )
    if blocked or blocked_documents:
        raise ValueError("Validation overlaps declared training inputs")
    expected_training = [(item["path"], item["sha256"]) for item in declared["training_data_audit"]]
    actual_training = [(item["path"], item["sha256"]) for item in training_reports]
    expected_provenance = [(item["path"], item["sha256"]) for item in declared["training_provenance_audit"]]
    actual_provenance = [(item["path"], item["sha256"]) for item in provenance_reports]
    if expected_training != actual_training or expected_provenance != actual_provenance:
        raise ValueError("Declared training audit inputs changed")
    validate_source(args.source_dir.resolve())
    return declared


def main() -> None:
    args = parse_args()
    result = build(args) if args.command == "build" else verify(args)
    print(compact_json(result))


if __name__ == "__main__":
    main()
