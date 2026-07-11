#!/usr/bin/env python3
"""Build and verify a deterministic Korean legal/public retrieval holdout.

This is a same-repository, whole-source-document-held-out (grade I) artifact.
It is deliberately not described as unseen-source or grade Z.  Training
provenance excludes complete ``source_document_sha256`` values; the pinned
benchmark blocklist additionally excludes exact normalized query/positive
hashes.  No GPU or model inference is used.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/legal_source_holdout_v1.json"
HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
H1_RE = re.compile(r"(?m)^#\s+([^#\n].*?)\s*$")
ZERO_WIDTH_TRANSLATION = str.maketrans("", "", "\u200b\u200c\u200d\u2060\ufeff")


def canonical_line(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def normalize_text(value: str, policy: Mapping[str, Any]) -> str:
    normalized = unicodedata.normalize(str(policy["unicode_form"]), value)
    if policy.get("remove_zero_width", True):
        normalized = normalized.translate(ZERO_WIDTH_TRANSLATION)
    if policy.get("normalize_line_endings", True):
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    if policy.get("collapse_whitespace", True):
        normalized = " ".join(normalized.split())
    else:
        normalized = normalized.strip()
    if policy.get("casefold", False):
        normalized = normalized.casefold()
    return normalized


def normalized_sha256(value: str, policy: Mapping[str, Any]) -> str:
    return sha256_text(normalize_text(value, policy))


def require_text(value: Any, field: str, minimum: int = 1) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n")).strip()
    if len(normalized) < minimum:
        raise ValueError(f"{field} must contain at least {minimum} characters")
    return normalized


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != 1:
        raise ValueError(f"unsupported config schema: {config.get('schema_version')!r}")
    if config.get("independence", {}).get("grade") != "I":
        raise ValueError("this builder only supports independence grade I")
    if config.get("independence", {}).get("not_grade") != "Z":
        raise ValueError("config must explicitly state that this is not grade Z")
    return config


class AtomicJSONL:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None
        self.digest = hashlib.sha256()
        self.rows = 0
        self.bytes = 0

    def __enter__(self) -> "AtomicJSONL":
        self.handle = self.path.open("wb")
        return self

    def write(self, value: Any) -> None:
        if self.handle is None:
            raise RuntimeError("writer is not open")
        encoded = canonical_line(value)
        self.handle.write(encoded)
        self.digest.update(encoded)
        self.rows += 1
        self.bytes += len(encoded)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            if exc_type is None:
                self.handle.flush()
                os.fsync(self.handle.fileno())
            self.handle.close()

    def metadata(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "bytes": self.bytes,
            "sha256": self.digest.hexdigest(),
        }


def atomic_write_json(path: Path, value: Any) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with path.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def prepare_destination(destination: Path, overwrite: bool) -> Path:
    destination = destination.resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))


def finalize_destination(staging: Path, destination: Path, overwrite: bool) -> None:
    destination = destination.resolve()
    backup: Path | None = None
    try:
        if destination.exists():
            if not overwrite:
                raise FileExistsError(destination)
            backup = destination.with_name(f".{destination.name}.backup-{os.getpid()}")
            if backup.exists():
                shutil.rmtree(backup)
            destination.replace(backup)
        staging.replace(destination)
        if backup is not None:
            shutil.rmtree(backup)
    except BaseException:
        if not destination.exists() and backup is not None and backup.exists():
            backup.replace(destination)
        raise


def resolve_candidate_paths(config: dict[str, Any], explicit: Sequence[Path]) -> list[Path]:
    if explicit:
        paths = [path.resolve() for path in explicit]
    else:
        paths = sorted(ROOT.glob(config["inputs"]["candidate_glob"]))
    if not paths:
        raise FileNotFoundError("no candidate JSONL files resolved")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"candidate files missing: {missing}")
    if len(paths) != len(set(paths)):
        raise ValueError("candidate path list contains duplicates")
    return sorted(paths)


def source_date(provenance: dict[str, Any], config: dict[str, Any]) -> dict[str, str] | None:
    metadata = provenance.get("metadata")
    if not isinstance(metadata, dict):
        return None
    for field in config["source_date_field_priority"]:
        value = metadata.get(field)
        if isinstance(value, str) and value.strip():
            return {"field": field, "value": value.strip()}
    return None


def validate_provenance(
    provenance: Any, field: str, config: dict[str, Any]
) -> tuple[dict[str, Any], str, str, str, dict[str, str] | None]:
    if not isinstance(provenance, dict):
        raise ValueError(f"{field} must be an object")
    repository = require_text(provenance.get("repository"), f"{field}.repository")
    revision = require_text(provenance.get("revision"), f"{field}.revision")
    document_hash = require_text(
        provenance.get("source_document_sha256"), f"{field}.source_document_sha256"
    )
    if not HEX_40_RE.fullmatch(revision):
        raise ValueError(f"{field}.revision is not a pinned 40-character Git SHA")
    if not HEX_64_RE.fullmatch(document_hash):
        raise ValueError(f"{field}.source_document_sha256 is not SHA-256")
    require_text(provenance.get("path"), f"{field}.path")
    date = source_date(provenance, config)
    if config["defaults"]["require_source_date"] and date is None:
        raise ValueError(f"{field} has no configured source date")
    return provenance, repository, revision, document_hash, date


def load_training_exclusions(
    path: Path, config: dict[str, Any]
) -> tuple[set[str], set[str], set[str], dict[str, Any]]:
    candidate_ids: set[str] = set()
    document_hashes: set[str] = set()
    repositories: set[str] = set()
    rows = 0
    for line_number, raw in read_jsonl(path):
        candidate_id = require_text(
            raw.get("source_candidate_id"), f"{path}:{line_number}.source_candidate_id"
        )
        provenance, repository, _, document_hash, _ = validate_provenance(
            raw.get("provenance"), f"{path}:{line_number}.provenance", config
        )
        if candidate_id in candidate_ids:
            raise ValueError(f"{path}:{line_number}: duplicate source_candidate_id")
        candidate_ids.add(candidate_id)
        document_hashes.add(document_hash)
        repositories.add(repository)
        rows += 1
        if provenance["source_document_sha256"] != document_hash:
            raise AssertionError("unreachable provenance normalization mismatch")
    if not rows:
        raise ValueError(f"training provenance is empty: {path}")
    metadata = {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "rows": rows,
        "unique_source_candidate_ids": len(candidate_ids),
        "unique_source_document_sha256": len(document_hashes),
        "repositories": sorted(repositories),
    }
    return candidate_ids, document_hashes, repositories, metadata


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE candidates (
            candidate_id TEXT PRIMARY KEY,
            repository TEXT NOT NULL,
            stable_rank BLOB NOT NULL,
            query_hash TEXT NOT NULL,
            positive_hash TEXT NOT NULL,
            document_hash TEXT NOT NULL,
            payload TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        "CREATE INDEX candidates_source_rank ON candidates(repository, stable_rank, candidate_id)"
    )


def stage_candidates(
    paths: Sequence[Path],
    connection: sqlite3.Connection,
    config: dict[str, Any],
    normalization: dict[str, Any],
    train_ids: set[str],
    train_documents: set[str],
    seed: int,
) -> tuple[dict[str, Any], set[str]]:
    counters: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_hashes: set[str] = set()
    input_files: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    minimum_query = int(config["defaults"]["minimum_query_characters"])
    minimum_positive = int(config["defaults"]["minimum_positive_characters"])
    batch: list[tuple[str, str, bytes, str, str, str, str]] = []
    for path in paths:
        file_rows = 0
        for line_number, raw in read_jsonl(path):
            candidate_id = require_text(raw.get("id"), f"{path}:{line_number}.id")
            if candidate_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate candidate id {candidate_id}")
            seen_ids.add(candidate_id)
            query = require_text(raw.get("query"), f"{path}:{line_number}.query", minimum_query)
            positive = require_text(
                raw.get("positive"), f"{path}:{line_number}.positive", minimum_positive
            )
            pair_type = require_text(raw.get("pair_type"), f"{path}:{line_number}.pair_type")
            label_origin = require_text(
                raw.get("label_origin"), f"{path}:{line_number}.label_origin"
            )
            provenance, repository, revision, document_hash, date = validate_provenance(
                raw.get("provenance"), f"{path}:{line_number}.provenance", config
            )
            counters["candidate_rows_seen"] += 1
            per_source[repository]["candidate_rows_seen"] += 1
            file_rows += 1
            trained_id = candidate_id in train_ids
            trained_document = document_hash in train_documents
            if trained_id:
                counters["candidate_ids_present_in_training"] += 1
                per_source[repository]["candidate_ids_present_in_training"] += 1
            if trained_document:
                counters["excluded_entire_training_source_document"] += 1
                per_source[repository]["excluded_entire_training_source_document"] += 1
                continue
            if trained_id:
                raise RuntimeError(
                    f"training candidate id {candidate_id} reappeared with a different source document hash"
                )
            query_hash = normalized_sha256(query, normalization)
            positive_hash = normalized_sha256(positive, normalization)
            rank = hashlib.sha256(
                "\0".join(
                    (str(seed), repository, candidate_id, query_hash, positive_hash)
                ).encode("utf-8")
            ).digest()
            payload = {
                "id": candidate_id,
                "query": query,
                "positive": positive,
                "pair_type": pair_type,
                "label_origin": label_origin,
                "provenance": provenance,
                "repository": repository,
                "revision": revision,
                "source_document_sha256": document_hash,
                "source_date": date,
                "normalized_query_sha256": query_hash,
                "normalized_positive_sha256": positive_hash,
            }
            batch.append(
                (
                    candidate_id,
                    repository,
                    rank,
                    query_hash,
                    positive_hash,
                    document_hash,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                )
            )
            candidate_hashes.add(query_hash)
            candidate_hashes.add(positive_hash)
            counters["eligible_before_benchmark_exact_exclusion"] += 1
            per_source[repository]["eligible_before_benchmark_exact_exclusion"] += 1
            if len(batch) >= 5000:
                connection.executemany("INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
        input_files.append(
            {
                "path": relative_path(path),
                "sha256": sha256_file(path),
                "rows": file_rows,
            }
        )
    if batch:
        connection.executemany("INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
    connection.commit()
    summary = {
        "files": input_files,
        "counters": dict(sorted(counters.items())),
        "per_repository": {
            source: dict(sorted(values.items()))
            for source, values in sorted(per_source.items())
        },
        "unique_candidate_ids_seen": len(seen_ids),
        "unique_candidate_text_hashes_before_benchmark": len(candidate_hashes),
    }
    return summary, candidate_hashes


def benchmark_hash_paths(root: Path, config: dict[str, Any]) -> list[Path]:
    names = set(config["inputs"]["benchmark_hash_file_names"])
    paths = [
        path
        for path in root.rglob("*.sha256.gz")
        if path.name in names and ".cache" not in path.parts
    ]
    if not paths:
        raise FileNotFoundError(f"no benchmark text hash files under {root}")
    return sorted(paths)


def scan_benchmark_intersections(
    root: Path, config: dict[str, Any], candidate_hashes: set[str]
) -> tuple[set[str], dict[str, Any]]:
    paths = benchmark_hash_paths(root, config)
    blocked: set[str] = set()
    files: list[dict[str, Any]] = []
    kinds: Counter[str] = Counter()
    for path in paths:
        records = 0
        matched: set[str] = set()
        previous = ""
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                digest = line.strip()
                if not HEX_64_RE.fullmatch(digest):
                    raise ValueError(f"{path}:{line_number}: invalid SHA-256 line")
                if previous and digest <= previous:
                    raise ValueError(f"{path}:{line_number}: hashes are not strictly sorted")
                previous = digest
                records += 1
                if digest in candidate_hashes:
                    matched.add(digest)
        blocked.update(matched)
        kinds[path.name] += records
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "kind": path.name.removesuffix(".sha256.gz"),
                "records": records,
                "candidate_hash_matches": len(matched),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    root_manifest = root / "manifest.json"
    metadata = {
        "root": relative_path(root),
        "root_manifest_sha256": sha256_file(root_manifest) if root_manifest.is_file() else None,
        "normalization_policy_path": relative_path(ROOT / config["inputs"]["decontamination_policy"]),
        "files": files,
        "records_by_file_name": dict(sorted(kinds.items())),
        "candidate_hash_intersections": len(blocked),
    }
    return blocked, metadata


def title_from_positive(positive: str, provenance: dict[str, Any], query: str) -> str:
    match = H1_RE.search(positive)
    if match:
        return match.group(1).strip()
    components = provenance.get("query_components")
    if isinstance(components, list) and components and isinstance(components[0], str):
        return components[0].strip()
    return query[:160].strip()


def source_balanced_selection(
    connection: sqlite3.Connection,
    blocked_hashes: set[str],
    target_size: int,
    maximum_per_document: int,
    required_sources: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    available_sources = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT repository FROM candidates ORDER BY repository"
        )
    ]
    missing = sorted(set(required_sources) - set(available_sources))
    if missing:
        return [], {
            "status": "missing_required_repositories_before_benchmark",
            "missing_required_repositories": missing,
            "available_repositories": available_sources,
        }
    cursors = {
        source: iter(
            connection.execute(
                "SELECT payload FROM candidates WHERE repository=? "
                "ORDER BY stable_rank, candidate_id",
                (source,),
            )
        )
        for source in available_sources
    }
    exhausted: set[str] = set()
    selected: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    seen_positives: set[str] = set()
    document_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    skips: Counter[str] = Counter()
    while len(selected) < target_size and len(exhausted) < len(available_sources):
        progress = False
        for source in available_sources:
            if len(selected) >= target_size:
                break
            if source in exhausted:
                continue
            while True:
                try:
                    payload = json.loads(next(cursors[source])[0])
                except StopIteration:
                    exhausted.add(source)
                    break
                query_hash = payload["normalized_query_sha256"]
                positive_hash = payload["normalized_positive_sha256"]
                document_hash = payload["source_document_sha256"]
                if query_hash in blocked_hashes:
                    skips["benchmark_exact_query_hash"] += 1
                    continue
                if positive_hash in blocked_hashes:
                    skips["benchmark_exact_positive_hash"] += 1
                    continue
                if query_hash in seen_queries:
                    skips["duplicate_normalized_query"] += 1
                    continue
                if positive_hash in seen_positives:
                    skips["duplicate_normalized_positive"] += 1
                    continue
                if document_counts[document_hash] >= maximum_per_document:
                    skips["source_document_pair_cap"] += 1
                    continue
                seen_queries.add(query_hash)
                seen_positives.add(positive_hash)
                document_counts[document_hash] += 1
                source_counts[source] += 1
                selected.append(payload)
                progress = True
                break
        if not progress and len(exhausted) < len(available_sources):
            raise RuntimeError("selection made no progress without exhausting all source cursors")
    metadata = {
        "status": "complete" if len(selected) == target_size else "shortfall",
        "selected_rows": len(selected),
        "selected_unique_source_documents": len(document_counts),
        "selected_repository_counts": dict(sorted(source_counts.items())),
        "selection_skips": dict(sorted(skips.items())),
        "available_repositories": available_sources,
        "exhausted_repositories": sorted(exhausted),
    }
    return selected, metadata


def stable_output_ids(payload: dict[str, Any]) -> tuple[str, str]:
    query_id = "legal-i-q-" + sha256_text(
        "\0".join(
            (
                payload["id"],
                payload["normalized_query_sha256"],
                payload["source_document_sha256"],
            )
        )
    )[:24]
    corpus_id = "legal-i-d-" + sha256_text(
        "\0".join(
            (
                payload["id"],
                payload["normalized_positive_sha256"],
                payload["source_document_sha256"],
            )
        )
    )[:24]
    return query_id, corpus_id


def selection_reason() -> str:
    return (
        "source-native structural pair; entire source_document_sha256 absent from training; "
        "source_candidate_id absent from training; normalized query/positive SHA-256 absent "
        "from pinned benchmark exact blocklists; seeded source-balanced stable-rank selection"
    )


def write_dataset(
    staging: Path,
    selected: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reason = selection_reason()
    source_counts: Counter[str] = Counter()
    revisions: dict[str, set[str]] = defaultdict(set)
    dates: dict[str, list[str]] = defaultdict(list)
    with ExitStack() as stack:
        queries = stack.enter_context(AtomicJSONL(staging / "queries.jsonl"))
        corpus = stack.enter_context(AtomicJSONL(staging / "corpus.jsonl"))
        qrels = stack.enter_context(AtomicJSONL(staging / "qrels.jsonl"))
        provenance_writer = stack.enter_context(AtomicJSONL(staging / "provenance.jsonl"))
        for row_index, payload in enumerate(selected):
            query_id, corpus_id = stable_output_ids(payload)
            repository = payload["repository"]
            revision = payload["revision"]
            date = payload["source_date"]
            metadata = {
                "source_candidate_id": payload["id"],
                "source_document_sha256": payload["source_document_sha256"],
                "repository": repository,
                "revision": revision,
                "source_date": date,
                "pair_type": payload["pair_type"],
                "selection_reason": reason,
                "independence_grade": "I",
                "independence_label": "same-repository source-document-held-out",
            }
            queries.write({"_id": query_id, "text": payload["query"], "metadata": metadata})
            corpus.write(
                {
                    "_id": corpus_id,
                    "title": title_from_positive(
                        payload["positive"], payload["provenance"], payload["query"]
                    ),
                    "text": payload["positive"],
                    "metadata": metadata,
                }
            )
            qrels.write(
                {
                    "query-id": query_id,
                    "corpus-id": corpus_id,
                    "score": int(config["defaults"]["relevance_score"]),
                }
            )
            provenance_writer.write(
                {
                    "row_index": row_index,
                    "query_id": query_id,
                    "corpus_id": corpus_id,
                    "source_candidate_id": payload["id"],
                    "source_document_sha256": payload["source_document_sha256"],
                    "repository": repository,
                    "revision": revision,
                    "source_date": date,
                    "pair_type": payload["pair_type"],
                    "label_origin": payload["label_origin"],
                    "emitted_query_sha256": sha256_text(payload["query"]),
                    "emitted_positive_sha256": sha256_text(payload["positive"]),
                    "normalized_query_sha256": payload["normalized_query_sha256"],
                    "normalized_positive_sha256": payload["normalized_positive_sha256"],
                    "selection_reason": reason,
                    "independence_grade": "I",
                    "independence_label": "same-repository source-document-held-out",
                    "provenance": payload["provenance"],
                }
            )
            source_counts[repository] += 1
            revisions[repository].add(revision)
            if date is not None:
                dates[repository].append(date["value"])
        files = {
            "queries.jsonl": queries.metadata(),
            "corpus.jsonl": corpus.metadata(),
            "qrels.jsonl": qrels.metadata(),
            "provenance.jsonl": provenance_writer.metadata(),
        }
    source_summary = {
        source: {
            "rows": source_counts[source],
            "revisions": sorted(revisions[source]),
            "source_date_min": min(dates[source]) if dates[source] else None,
            "source_date_max": max(dates[source]) if dates[source] else None,
        }
        for source in sorted(source_counts)
    }
    return files, source_summary


def base_manifest(
    config_path: Path,
    config: dict[str, Any],
    target_size: int,
    seed: int,
    training: dict[str, Any],
    candidates: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_id": config["artifact_id"],
        "independence": config["independence"],
        "builder": {
            "path": relative_path(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
            "runtime": f"python-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
        "config": {"path": relative_path(config_path), "sha256": sha256_file(config_path)},
        "parameters": {
            "target_size": target_size,
            "seed": seed,
            "maximum_pairs_per_source_document": config["defaults"][
                "maximum_pairs_per_source_document"
            ],
            "source_balance": config["selection"]["source_balance"],
        },
        "inputs": {
            "training_provenance": training,
            "candidate_sources": candidates,
        },
        "claims": {
            "allowed": "same-repository whole-source-document-held-out legal/public retrieval (grade I)",
            "forbidden": "unseen-source, clean zero-shot, or grade Z",
            "relevance": "source-native structural relation, not an independent human relevance judgment",
            "benchmark_exclusion": "exact normalized SHA-256 only; this does not claim semantic or near-duplicate exclusion",
        },
    }


def write_blocked_manifest(
    destination: Path,
    overwrite: bool,
    manifest: dict[str, Any],
    status: str,
    reason: str,
) -> None:
    staging = prepare_destination(destination, overwrite)
    try:
        manifest.update(
            {
                "status": status,
                "blocked_reason": reason,
                "files": {},
                "assertions": {
                    "weakened_holdout_emitted": False,
                    "training_document_exclusion_relaxed": False,
                    "independence_grade_promoted_to_Z": False,
                },
            }
        )
        atomic_write_json(staging / "manifest.json", manifest)
        finalize_destination(staging, destination, overwrite)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def command_build(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    target_size = args.target_size or int(config["defaults"]["target_size"])
    seed = args.seed if args.seed is not None else int(config["defaults"]["seed"])
    if target_size < 2:
        raise ValueError("target size must be at least 2")
    candidate_paths = resolve_candidate_paths(config, args.candidate)
    training_path = (
        args.training_provenance.resolve()
        if args.training_provenance
        else (ROOT / config["inputs"]["training_provenance"]).resolve()
    )
    blocklist_root = (
        args.blocklist_root.resolve()
        if args.blocklist_root
        else (ROOT / config["inputs"]["benchmark_blocklist_root"]).resolve()
    )
    policy_path = (ROOT / config["inputs"]["decontamination_policy"]).resolve()
    policy = read_json(policy_path)
    normalization = policy["normalization"]
    train_ids, train_documents, train_repositories, training_meta = load_training_exclusions(
        training_path, config
    )
    required_sources = config["required_repositories"]
    missing_train_sources = sorted(set(required_sources) - train_repositories)
    if missing_train_sources:
        raise ValueError(
            f"training provenance is missing required repositories: {missing_train_sources}"
        )
    with tempfile.TemporaryDirectory(prefix="legal-holdout-index-", dir=args.work_dir) as temporary:
        connection = sqlite3.connect(str(Path(temporary) / "candidates.sqlite3"))
        initialize_database(connection)
        try:
            candidate_meta, candidate_hashes = stage_candidates(
                candidate_paths,
                connection,
                config,
                normalization,
                train_ids,
                train_documents,
                seed,
            )
            manifest = base_manifest(
                args.config, config, target_size, seed, training_meta, candidate_meta
            )
            eligible_before = candidate_meta["counters"].get(
                "eligible_before_benchmark_exact_exclusion", 0
            )
            if eligible_before < target_size:
                manifest["counts"] = {
                    "eligible_before_benchmark_exact_exclusion": eligible_before,
                    "target_size": target_size,
                    "shortfall_before_benchmark": target_size - eligible_before,
                }
                manifest["benchmark_blocklist"] = {
                    "root": relative_path(blocklist_root),
                    "root_manifest_sha256": sha256_file(blocklist_root / "manifest.json")
                    if (blocklist_root / "manifest.json").is_file()
                    else None,
                    "scan_status": "not_needed_because_source_document_exclusion_already_caused_shortfall",
                }
                write_blocked_manifest(
                    args.output_dir,
                    args.overwrite,
                    manifest,
                    "blocked_insufficient_source_document_heldout_candidates",
                    "Candidate inputs do not contain enough rows from source documents absent from training provenance.",
                )
                sys.stderr.write(
                    f"blocked: {eligible_before} source-document-held-out candidates for target {target_size}; "
                    f"manifest written to {args.output_dir / 'manifest.json'}\n"
                )
                return 2
            blocked_hashes, blocklist_meta = scan_benchmark_intersections(
                blocklist_root, config, candidate_hashes
            )
            selected, selection_meta = source_balanced_selection(
                connection,
                blocked_hashes,
                target_size,
                int(config["defaults"]["maximum_pairs_per_source_document"]),
                required_sources,
            )
            manifest["benchmark_blocklist"] = {
                **blocklist_meta,
                "policy_path": relative_path(policy_path),
                "policy_sha256": sha256_file(policy_path),
                "normalization": normalization,
            }
            manifest["selection"] = selection_meta
            if len(selected) < target_size:
                manifest["counts"] = {
                    "eligible_before_benchmark_exact_exclusion": eligible_before,
                    "selected_after_all_exclusions": len(selected),
                    "target_size": target_size,
                    "shortfall_after_all_exclusions": target_size - len(selected),
                }
                write_blocked_manifest(
                    args.output_dir,
                    args.overwrite,
                    manifest,
                    "blocked_insufficient_post_decontamination_candidates",
                    "Benchmark exact exclusion, deduplication, document caps, or source requirements leave fewer rows than requested.",
                )
                sys.stderr.write(
                    f"blocked: {len(selected)} post-decontamination candidates for target {target_size}\n"
                )
                return 2
            staging = prepare_destination(args.output_dir, args.overwrite)
            try:
                files, source_summary = write_dataset(staging, selected, config)
                selected_ids = {row["id"] for row in selected}
                selected_documents = {row["source_document_sha256"] for row in selected}
                selected_query_hashes = {row["normalized_query_sha256"] for row in selected}
                selected_positive_hashes = {
                    row["normalized_positive_sha256"] for row in selected
                }
                assertions = {
                    "selected_source_candidate_id_overlap_with_training": len(
                        selected_ids & train_ids
                    ),
                    "selected_source_document_sha256_overlap_with_training": len(
                        selected_documents & train_documents
                    ),
                    "selected_query_hash_overlap_with_benchmark": len(
                        selected_query_hashes & blocked_hashes
                    ),
                    "selected_positive_hash_overlap_with_benchmark": len(
                        selected_positive_hashes & blocked_hashes
                    ),
                    "selected_unique_query_hashes": len(selected_query_hashes),
                    "selected_unique_positive_hashes": len(selected_positive_hashes),
                    "selected_unique_source_documents": len(selected_documents),
                    "same_repository_as_training": sorted(
                        {row["repository"] for row in selected}
                    )
                    == sorted(required_sources),
                    "independence_grade_is_I_not_Z": True,
                }
                overlap_keys = [
                    key
                    for key, value in assertions.items()
                    if (
                        key.endswith("overlap_with_training")
                        or key.endswith("overlap_with_benchmark")
                    )
                    if value != 0
                ]
                if overlap_keys:
                    raise AssertionError(f"holdout exclusion assertions failed: {overlap_keys}")
                manifest.update(
                    {
                        "status": "complete",
                        "counts": {
                            "queries": len(selected),
                            "corpus": len(selected),
                            "qrels": len(selected),
                            "target_size": target_size,
                        },
                        "source_summary": source_summary,
                        "assertions": assertions,
                        "files": files,
                    }
                )
                atomic_write_json(staging / "manifest.json", manifest)
                finalize_destination(staging, args.output_dir, args.overwrite)
            except BaseException:
                if staging.exists():
                    shutil.rmtree(staging)
                raise
        finally:
            connection.close()
    sys.stdout.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return 0


def load_output_by_id(path: Path, id_field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line_number, row in read_jsonl(path):
        value = require_text(row.get(id_field), f"{path}:{line_number}.{id_field}")
        if value in result:
            raise ValueError(f"{path}:{line_number}: duplicate {id_field} {value}")
        result[value] = row
    return result


def command_verify(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output_dir = args.output_dir.resolve()
    manifest = read_json(output_dir / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"cannot verify non-complete artifact: {manifest.get('status')}")
    if manifest.get("independence", {}).get("grade") != "I":
        raise ValueError("manifest independence grade is not I")
    config_sha = sha256_file(args.config)
    if manifest.get("config", {}).get("sha256") != config_sha:
        raise ValueError("manifest was not built with the current holdout config")
    files = manifest.get("files", {})
    for name in ("queries.jsonl", "corpus.jsonl", "qrels.jsonl", "provenance.jsonl"):
        path = output_dir / name
        declared = files.get(name)
        if not isinstance(declared, dict):
            raise ValueError(f"manifest is missing {name}")
        actual = sha256_file(path)
        if actual != declared.get("sha256"):
            raise ValueError(f"{name} SHA mismatch: {actual} != {declared.get('sha256')}")
    training_path = (
        args.training_provenance.resolve()
        if args.training_provenance
        else (ROOT / config["inputs"]["training_provenance"]).resolve()
    )
    blocklist_root = (
        args.blocklist_root.resolve()
        if args.blocklist_root
        else (ROOT / config["inputs"]["benchmark_blocklist_root"]).resolve()
    )
    policy_path = (ROOT / config["inputs"]["decontamination_policy"]).resolve()
    normalization = read_json(policy_path)["normalization"]
    train_ids, train_documents, train_repositories, training_meta = load_training_exclusions(
        training_path, config
    )
    if manifest.get("inputs", {}).get("training_provenance", {}).get(
        "sha256"
    ) != training_meta["sha256"]:
        raise ValueError("training provenance SHA differs from the build manifest")
    queries = load_output_by_id(output_dir / "queries.jsonl", "_id")
    corpus = load_output_by_id(output_dir / "corpus.jsonl", "_id")
    provenance_rows: list[dict[str, Any]] = []
    query_hashes: set[str] = set()
    positive_hashes: set[str] = set()
    source_counts: Counter[str] = Counter()
    document_hashes: set[str] = set()
    candidate_ids: set[str] = set()
    provenance_query_ids: set[str] = set()
    provenance_corpus_ids: set[str] = set()
    expected_relations: dict[str, str] = {}
    for line_number, row in read_jsonl(output_dir / "provenance.jsonl"):
        if row.get("row_index") != line_number - 1:
            raise ValueError("provenance row_index is not contiguous")
        query_id = require_text(row.get("query_id"), "query_id")
        corpus_id = require_text(row.get("corpus_id"), "corpus_id")
        if query_id not in queries or corpus_id not in corpus:
            raise ValueError("provenance references missing query/corpus")
        if query_id in provenance_query_ids or corpus_id in provenance_corpus_ids:
            raise ValueError("provenance query/corpus ids are not one-to-one")
        provenance_query_ids.add(query_id)
        provenance_corpus_ids.add(corpus_id)
        expected_relations[query_id] = corpus_id
        candidate_id = require_text(row.get("source_candidate_id"), "source_candidate_id")
        document_hash = require_text(row.get("source_document_sha256"), "source_document_sha256")
        if candidate_id in candidate_ids:
            raise ValueError("duplicate source_candidate_id in provenance")
        if document_hash in document_hashes:
            raise ValueError("source-document pair cap violated in provenance")
        if candidate_id in train_ids:
            raise AssertionError(f"training source_candidate_id leaked: {candidate_id}")
        if document_hash in train_documents:
            raise AssertionError(f"training source document leaked: {document_hash}")
        query_text = require_text(queries[query_id].get("text"), "query.text")
        positive_text = require_text(corpus[corpus_id].get("text"), "corpus.text")
        query_hash = normalized_sha256(query_text, normalization)
        positive_hash = normalized_sha256(positive_text, normalization)
        if query_hash != row.get("normalized_query_sha256"):
            raise ValueError("normalized query hash mismatch")
        if positive_hash != row.get("normalized_positive_sha256"):
            raise ValueError("normalized positive hash mismatch")
        if sha256_text(query_text) != row.get("emitted_query_sha256"):
            raise ValueError("emitted query hash mismatch")
        if sha256_text(positive_text) != row.get("emitted_positive_sha256"):
            raise ValueError("emitted positive hash mismatch")
        if row.get("independence_grade") != "I":
            raise ValueError("row independence grade is not I")
        expected_query_id, expected_corpus_id = stable_output_ids(
            {
                "id": candidate_id,
                "normalized_query_sha256": query_hash,
                "normalized_positive_sha256": positive_hash,
                "source_document_sha256": document_hash,
            }
        )
        if query_id != expected_query_id or corpus_id != expected_corpus_id:
            raise ValueError("query/corpus id is not the deterministic content-derived id")
        nested, repository, revision, nested_document, date = validate_provenance(
            row.get("provenance"), "provenance", config
        )
        if repository != row.get("repository") or revision != row.get("revision"):
            raise ValueError("flattened source provenance mismatch")
        if nested_document != document_hash or date != row.get("source_date"):
            raise ValueError("source document/date provenance mismatch")
        metadata = queries[query_id].get("metadata")
        corpus_metadata = corpus[corpus_id].get("metadata")
        if metadata != corpus_metadata:
            raise ValueError("query/corpus metadata mismatch")
        if not isinstance(metadata, dict):
            raise ValueError("query/corpus metadata must be an object")
        if metadata.get("source_document_sha256") != nested["source_document_sha256"]:
            raise ValueError("query metadata source document mismatch")
        if metadata.get("selection_reason") != row.get("selection_reason"):
            raise ValueError("selection reason mismatch")
        if metadata.get("independence_grade") != "I":
            raise ValueError("metadata independence grade is not I")
        if query_hash in query_hashes or positive_hash in positive_hashes:
            raise ValueError("duplicate normalized query or positive hash")
        query_hashes.add(query_hash)
        positive_hashes.add(positive_hash)
        document_hashes.add(document_hash)
        candidate_ids.add(candidate_id)
        source_counts[repository] += 1
        provenance_rows.append(row)
    qrel_count = 0
    seen_qrel_queries: set[str] = set()
    seen_qrel_corpus: set[str] = set()
    for _, qrel in read_jsonl(output_dir / "qrels.jsonl"):
        query_id = require_text(qrel.get("query-id"), "qrel.query-id")
        corpus_id = require_text(qrel.get("corpus-id"), "qrel.corpus-id")
        if query_id not in queries or corpus_id not in corpus:
            raise ValueError("qrel references missing query/corpus")
        if expected_relations.get(query_id) != corpus_id:
            raise ValueError("qrel does not match the source-native provenance relation")
        if qrel.get("score") != int(config["defaults"]["relevance_score"]):
            raise ValueError("unexpected qrel score")
        if query_id in seen_qrel_queries or corpus_id in seen_qrel_corpus:
            raise ValueError("qrels are not one-to-one")
        seen_qrel_queries.add(query_id)
        seen_qrel_corpus.add(corpus_id)
        qrel_count += 1
    row_count = len(provenance_rows)
    if not (len(queries) == len(corpus) == qrel_count == row_count):
        raise ValueError("query/corpus/qrel/provenance row counts differ")
    candidate_hashes = query_hashes | positive_hashes
    blocked, block_meta = scan_benchmark_intersections(
        blocklist_root, config, candidate_hashes
    )
    declared_blocklist = manifest.get("benchmark_blocklist", {})
    identity_fields = ("path", "kind", "records", "bytes", "sha256")
    declared_file_identities = [
        {key: row.get(key) for key in identity_fields}
        for row in declared_blocklist.get("files", [])
    ]
    actual_file_identities = [
        {key: row.get(key) for key in identity_fields} for row in block_meta["files"]
    ]
    if declared_file_identities != actual_file_identities:
        raise ValueError("benchmark blocklist files differ from the build manifest")
    if declared_blocklist.get("root_manifest_sha256") != block_meta[
        "root_manifest_sha256"
    ]:
        raise ValueError("benchmark blocklist root manifest differs from the build manifest")
    if query_hashes & blocked:
        raise AssertionError("benchmark exact query hash leaked")
    if positive_hashes & blocked:
        raise AssertionError("benchmark exact positive hash leaked")
    required_sources = set(config["required_repositories"])
    if set(source_counts) != required_sources:
        raise ValueError(
            f"output repository set differs: {sorted(source_counts)} != {sorted(required_sources)}"
        )
    if not set(source_counts).issubset(train_repositories):
        raise ValueError("output is not same-repository relative to training provenance")
    declared_counts = manifest.get("counts", {})
    if any(declared_counts.get(key) != row_count for key in ("queries", "corpus", "qrels")):
        raise ValueError("manifest row counts differ from outputs")
    declared_source_counts = manifest.get("selection", {}).get(
        "selected_repository_counts"
    )
    if declared_source_counts != dict(sorted(source_counts.items())):
        raise ValueError("manifest repository counts differ from outputs")
    result = {
        "verified": True,
        "rows": row_count,
        "repositories": dict(sorted(source_counts.items())),
        "unique_source_documents": len(document_hashes),
        "unique_source_candidate_ids": len(candidate_ids),
        "training_source_candidate_overlap": len(candidate_ids & train_ids),
        "training_source_document_overlap": len(document_hashes & train_documents),
        "benchmark_query_hash_overlap": len(query_hashes & blocked),
        "benchmark_positive_hash_overlap": len(positive_hashes & blocked),
        "benchmark_hash_files_scanned": len(block_meta["files"]),
        "independence_grade": "I",
        "not_grade": "Z",
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
    }
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--candidate", type=Path, action="append", default=[])
    build.add_argument("--training-provenance", type=Path)
    build.add_argument("--blocklist-root", type=Path)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--work-dir", type=Path)
    build.add_argument("--target-size", type=int)
    build.add_argument("--seed", type=int)
    build.add_argument("--overwrite", action="store_true")
    build.set_defaults(function=command_build)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--training-provenance", type=Path)
    verify.add_argument("--blocklist-root", type=Path)
    verify.add_argument("--output-dir", type=Path, required=True)
    verify.set_defaults(function=command_verify)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.config = args.config.resolve()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
