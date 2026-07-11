#!/usr/bin/env python3
"""Build privacy-preserving blocklists for the pinned Korean benchmarks.

The output contains normalized SHA-256 digests and optional MinHash signatures,
never benchmark text or source IDs.  ``--list-only`` and ``--dry-run`` resolve
the exact MTEB tasks without downloading datasets.  A real run downloads only
the selected task data and commits each task directory atomically.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOLS = (
    ROOT / "configs/sionic9_protocol.json",
    ROOT / "configs/mteb_korean_v1_protocol.json",
)
DEFAULT_POLICY = ROOT / "configs/decontamination_policy.json"
DEFAULT_OUTPUT = ROOT / "outputs/decontamination/benchmark_blocklist"
PINNED_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
ZERO_WIDTH_TRANSLATION = str.maketrans("", "", "\u200b\u200c\u200d\u2060\ufeff")


@dataclasses.dataclass(frozen=True)
class TaskPlan:
    protocol_path: Path
    protocol_id: str
    protocol_sha256: str
    label: str
    name: str
    task_type: str
    split: str
    subset: str
    dataset: dict[str, Any]
    license: Any
    mteb_git_revision: str
    task: Any = dataclasses.field(compare=False, repr=False)

    @property
    def key(self) -> str:
        return f"{self.protocol_id}:{self.label}:{self.subset}:{self.split}"

    def public_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "protocol_id": self.protocol_id,
            "protocol_path": relative_or_absolute(self.protocol_path),
            "protocol_sha256": self.protocol_sha256,
            "label": self.label,
            "name": self.name,
            "type": self.task_type,
            "split": self.split,
            "subset": self.subset,
            "dataset": self.dataset,
            "license": self.license,
            "mteb_git_revision": self.mteb_git_revision,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build text-free SHA-256 blocklists for pinned Korean benchmarks"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        action="append",
        help="Protocol JSON; repeat as needed (default: Sionic 9 and MTEB Korean v1)",
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--task",
        action="append",
        help=(
            "Task label/name/key; repeat to select tasks. A protocol-qualified "
            "selector may use <protocol_id>:<label>."
        ),
    )
    parser.add_argument("--num-proc", type=int)
    parser.add_argument(
        "--minhash",
        choices=("off", "auto", "required"),
        help="Override policy MinHash mode; auto uses datasketch only if installed",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Validate and print resolved metadata; never load benchmark datasets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show run/resume decisions; never load benchmark datasets or write output",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip matching completed task artifacts (default: true)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace completed artifacts with a newly built version",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_policy(policy: Mapping[str, Any]) -> None:
    required = {
        "policy_id",
        "schema_version",
        "normalization",
        "scope",
        "text_field_names",
        "id_field_names",
        "minhash",
        "artifact_policy",
    }
    missing = required - set(policy)
    if missing:
        raise ValueError(f"Decontamination policy is missing keys: {sorted(missing)}")
    if policy["scope"].get("retrieval_splits") != "protocol_eval_split_only":
        raise ValueError("Only protocol_eval_split_only retrieval scope is supported")
    required_scope_flags = (
        "hash_text_fields_separately",
        "hash_ids",
        "hash_qrel_relations",
    )
    disabled = [name for name in required_scope_flags if not policy["scope"].get(name)]
    if disabled:
        raise ValueError(
            f"A benchmark blocklist may not disable required coverage: {disabled}"
        )
    if policy["minhash"].get("shingle_unit") != "character":
        raise ValueError("Only character MinHash shingles are currently supported")
    artifact_policy = policy["artifact_policy"]
    if artifact_policy.get("contains_source_text") is not False:
        raise ValueError("Artifact policy must forbid source text")
    if artifact_policy.get("contains_source_ids") is not False:
        raise ValueError("Artifact policy must forbid source IDs")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_or_absolute(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_component(value: str) -> str:
    cleaned = SAFE_COMPONENT_RE.sub("_", value).strip("._")
    if not cleaned:
        raise ValueError(f"Unsafe empty path component derived from {value!r}")
    return cleaned


def normalize_text(value: str, policy: Mapping[str, Any]) -> str:
    form = str(policy["unicode_form"])
    normalized = unicodedata.normalize(form, value)
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


def validate_mteb_checkout(expected_revisions: set[str]) -> str:
    if len(expected_revisions) != 1:
        raise ValueError(
            "All protocols in one artifact must pin one MTEB git revision; found "
            f"{sorted(expected_revisions)}"
        )
    expected = next(iter(expected_revisions))
    checkout = ROOT / "third_party/mteb"
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"Cannot inspect pinned MTEB checkout at {checkout}"
        ) from error
    if actual != expected:
        raise RuntimeError(f"MTEB git mismatch: expected {expected}, found {actual}")
    return actual


def normalized_dataset_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in ("path", "name", "revision")
        if key in value and value[key] is not None
    }


def validate_dataset_revision(plan: TaskPlan) -> None:
    revision = str(plan.dataset.get("revision", ""))
    if not PINNED_REVISION_RE.fullmatch(revision):
        raise RuntimeError(
            f"{plan.key} does not resolve to a pinned 40-character dataset revision: "
            f"{revision!r}"
        )


def resolve_sionic_protocol(
    mteb: Any, path: Path, protocol: dict[str, Any]
) -> list[TaskPlan]:
    specs = protocol.get("tasks", [])
    if len(specs) != 9:
        raise ValueError(f"{path} must contain the exact nine Sionic tasks")
    plans: list[TaskPlan] = []
    protocol_sha = sha256_file(path)
    for spec in specs:
        task = mteb.get_task(
            spec["name"],
            eval_splits=[spec["split"]],
            hf_subsets=spec.get("hf_subsets"),
        )
        if "dataset_revision" in spec:
            task.metadata.dataset["revision"] = spec["dataset_revision"]
        subsets = list(task.hf_subsets)
        if len(subsets) != 1:
            raise RuntimeError(
                f"{spec['name']} must resolve to exactly one subset, found {subsets}"
            )
        # ``get_task(eval_splits=...)`` narrows the instance property while the
        # immutable metadata may legitimately advertise additional splits.
        if list(task.eval_splits) != [spec["split"]]:
            raise RuntimeError(
                f"Split drift for {spec['name']}: expected {spec['split']}, "
                f"found {list(task.eval_splits)}"
            )
        plan = TaskPlan(
            protocol_path=path,
            protocol_id=protocol["protocol_id"],
            protocol_sha256=protocol_sha,
            label=spec["label"],
            name=spec["name"],
            task_type=str(task.metadata.type),
            split=spec["split"],
            subset=subsets[0],
            dataset=normalized_dataset_metadata(dict(task.metadata.dataset)),
            license=task.metadata.license,
            mteb_git_revision=protocol["mteb_git_revision"],
            task=task,
        )
        validate_dataset_revision(plan)
        plans.append(plan)
    return plans


def resolve_official_protocol(
    mteb: Any, path: Path, protocol: dict[str, Any]
) -> list[TaskPlan]:
    if protocol.get("benchmark") != "MTEB(kor, v1)":
        raise ValueError(f"Unsupported benchmark protocol: {path}")
    if protocol.get("mteb_version") != mteb.__version__:
        raise RuntimeError(
            f"MTEB version mismatch: expected {protocol.get('mteb_version')}, "
            f"found {mteb.__version__}"
        )
    specs = protocol.get("tasks", [])
    if len(specs) != 6:
        raise ValueError(f"{path} must contain exactly six official Korean-v1 tasks")
    benchmark = mteb.get_benchmark(protocol["benchmark"])
    benchmark_tasks = {task.metadata.name: task for task in benchmark.tasks}
    expected_names = {spec["name"] for spec in specs}
    if set(benchmark_tasks) != expected_names:
        raise RuntimeError(
            "Official benchmark membership drifted: "
            f"expected={sorted(expected_names)}, found={sorted(benchmark_tasks)}"
        )
    plans: list[TaskPlan] = []
    protocol_sha = sha256_file(path)
    for spec in specs:
        task = benchmark_tasks[spec["name"]]
        actual = {
            "name": task.metadata.name,
            "type": str(task.metadata.type),
            "split": list(task.metadata.eval_splits),
            "hf_subsets": list(task.hf_subsets),
            "dataset": normalized_dataset_metadata(dict(task.metadata.dataset)),
            "main_score": task.metadata.main_score,
            "task_prompt": task.metadata.prompt,
            "instruction_fallback": task.abstask_prompt,
        }
        expected = {
            "name": spec["name"],
            "type": spec["type"],
            "split": [spec["split"]],
            "hf_subsets": spec["hf_subsets"],
            "dataset": normalized_dataset_metadata(spec["dataset"]),
            "main_score": spec["main_score"],
            "task_prompt": spec["task_prompt"],
            "instruction_fallback": spec["instruction_fallback"],
        }
        if actual != expected:
            raise RuntimeError(
                f"Installed MTEB metadata drifted for {spec['name']}: "
                f"expected={expected!r}, found={actual!r}"
            )
        subsets = list(task.hf_subsets)
        if len(subsets) != 1:
            raise RuntimeError(
                f"{spec['name']} must resolve to exactly one Korean subset: {subsets}"
            )
        plan = TaskPlan(
            protocol_path=path,
            protocol_id=protocol["protocol_id"],
            protocol_sha256=protocol_sha,
            label=spec["name"],
            name=spec["name"],
            task_type=spec["type"],
            split=spec["split"],
            subset=subsets[0],
            dataset=actual["dataset"],
            license=task.metadata.license,
            mteb_git_revision=protocol["mteb_git_revision"],
            task=task,
        )
        validate_dataset_revision(plan)
        plans.append(plan)
    return plans


def resolve_plans(protocol_paths: Sequence[Path]) -> tuple[list[TaskPlan], str]:
    protocols: list[tuple[Path, dict[str, Any]]] = []
    revisions: set[str] = set()
    for raw_path in protocol_paths:
        path = raw_path.resolve()
        protocol = read_json(path)
        revision = protocol.get("mteb_git_revision")
        if not isinstance(revision, str) or not PINNED_REVISION_RE.fullmatch(revision):
            raise ValueError(f"Protocol does not pin MTEB git SHA: {path}")
        revisions.add(revision)
        protocols.append((path, protocol))
    revision = validate_mteb_checkout(revisions)

    import mteb

    plans: list[TaskPlan] = []
    for path, protocol in protocols:
        if protocol.get("benchmark") == "MTEB(kor, v1)":
            plans.extend(resolve_official_protocol(mteb, path, protocol))
        elif len(protocol.get("tasks", [])) == 9:
            plans.extend(resolve_sionic_protocol(mteb, path, protocol))
        else:
            raise ValueError(f"Unrecognized benchmark protocol: {path}")
    keys = [plan.key for plan in plans]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Resolved task keys are not unique")
    return plans, revision


def selector_matches(selector: str, plan: TaskPlan) -> bool:
    candidates = {
        plan.key,
        plan.label,
        plan.name,
        f"{plan.protocol_id}:{plan.label}",
        f"{plan.protocol_id}:{plan.name}",
    }
    return selector in candidates


def select_plans(
    plans: Sequence[TaskPlan], selectors: Sequence[str] | None
) -> list[TaskPlan]:
    if not selectors:
        return list(plans)
    selected = [
        plan for plan in plans if any(selector_matches(s, plan) for s in selectors)
    ]
    unmatched = [s for s in selectors if not any(selector_matches(s, p) for p in plans)]
    if unmatched:
        raise ValueError(f"Unknown task selectors: {unmatched}")
    return selected


def import_minhash(mode: str) -> Any | None:
    if mode == "off":
        return None
    try:
        from datasketch import MinHash
    except ImportError:
        if mode == "required":
            raise RuntimeError(
                "--minhash required needs the optional 'datasketch' package"
            ) from None
        return None
    return MinHash


class HashAccumulator:
    """Disk-backed unique hash accumulator; source strings are never persisted."""

    def __init__(
        self,
        database: Path,
        normalization: Mapping[str, Any],
        minhash_config: Mapping[str, Any],
        minhash_class: Any | None,
    ) -> None:
        self.normalization = normalization
        self.minhash_config = minhash_config
        self.minhash_class = minhash_class
        self.connection = sqlite3.connect(database)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute(
            "CREATE TABLE hashes (kind TEXT NOT NULL, digest BLOB NOT NULL, "
            "PRIMARY KEY (kind, digest)) WITHOUT ROWID"
        )
        if minhash_class is not None:
            self.connection.execute(
                "CREATE TABLE minhash (kind TEXT NOT NULL, digest BLOB NOT NULL, "
                "signature BLOB NOT NULL, PRIMARY KEY (kind, digest)) WITHOUT ROWID"
            )
        self.occurrences: Counter[str] = Counter()
        self.records: Counter[str] = Counter()
        self.splits: Counter[str] = Counter()

    def close(self) -> None:
        self.connection.close()

    def add_normalized(self, kind: str, normalized: str) -> bytes | None:
        if not normalized:
            return None
        self.occurrences[kind] += 1
        digest = hashlib.sha256(normalized.encode("utf-8")).digest()
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO hashes(kind, digest) VALUES (?, ?)",
            (kind, digest),
        )
        if (
            cursor.rowcount
            and self.minhash_class is not None
            and kind.endswith("_text")
        ):
            signature = self._minhash(normalized)
            if signature is not None:
                self.connection.execute(
                    "INSERT INTO minhash(kind, digest, signature) VALUES (?, ?, ?)",
                    (kind, digest, signature),
                )
        return digest

    def add_text(self, kind: str, value: Any) -> bytes | None:
        if not isinstance(value, str):
            return None
        return self.add_normalized(kind, normalize_text(value, self.normalization))

    def add_id(self, kind: str, value: Any) -> bytes | None:
        if value is None:
            return None
        return self.add_normalized(kind, normalize_text(str(value), self.normalization))

    def add_relation(self, kind: str, value: Mapping[str, Any]) -> bytes:
        self.occurrences[kind] += 1
        digest = hashlib.sha256(canonical_json(value)).digest()
        self.connection.execute(
            "INSERT OR IGNORE INTO hashes(kind, digest) VALUES (?, ?)",
            (kind, digest),
        )
        return digest

    def _minhash(self, normalized: str) -> bytes | None:
        size = int(self.minhash_config["shingle_size"])
        minimum = int(self.minhash_config["minimum_normalized_characters"])
        if len(normalized) < max(size, minimum):
            return None
        instance = self.minhash_class(
            num_perm=int(self.minhash_config["num_perm"]),
            seed=int(self.minhash_config["seed"]),
        )
        for index in range(len(normalized) - size + 1):
            instance.update(normalized[index : index + size].encode("utf-8"))
        return instance.hashvalues.astype("<u8", copy=False).tobytes()

    def kinds(self) -> list[str]:
        return [
            row[0]
            for row in self.connection.execute(
                "SELECT DISTINCT kind FROM hashes ORDER BY kind"
            )
        ]

    def unique_count(self, kind: str) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM hashes WHERE kind = ?", (kind,)
            ).fetchone()[0]
        )

    def export(self, destination: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        counts: dict[str, Any] = {}
        files: list[dict[str, Any]] = []
        for kind in self.kinds():
            path = destination / f"{safe_component(kind)}.sha256.gz"
            with deterministic_gzip_text(path) as handle:
                rows = self.connection.execute(
                    "SELECT digest FROM hashes WHERE kind = ? ORDER BY digest", (kind,)
                )
                for (digest,) in rows:
                    handle.write(bytes(digest).hex() + "\n")
            unique = self.unique_count(kind)
            counts[kind] = {
                "occurrences": self.occurrences[kind],
                "unique_sha256": unique,
            }
            files.append(file_manifest(path, unique, kind))

        if self.minhash_class is not None:
            minhash_path = destination / "text.minhash.jsonl.gz"
            signatures = 0
            with deterministic_gzip_text(minhash_path) as handle:
                rows = self.connection.execute(
                    "SELECT kind, digest, signature FROM minhash ORDER BY kind, digest"
                )
                for kind, digest, signature in rows:
                    row = {
                        "kind": kind,
                        "sha256": bytes(digest).hex(),
                        "signature_le_u64_base64": base64.b64encode(signature).decode(
                            "ascii"
                        ),
                    }
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                    handle.write("\n")
                    signatures += 1
            files.append(file_manifest(minhash_path, signatures, "minhash_signature"))
            counts["minhash_signatures"] = signatures
        return counts, files


@contextlib.contextmanager
def deterministic_gzip_text(path: Path) -> Iterator[io.TextIOWrapper]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def file_manifest(path: Path, records: int, kind: str) -> dict[str, Any]:
    return {
        "path": path.name,
        "kind": kind,
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def scalar_text_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from scalar_text_values(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for nested in value:
            yield from scalar_text_values(nested)


def extract_row_texts(row: Mapping[str, Any], allowed_fields: set[str]) -> list[str]:
    values: list[str] = []
    for key, value in row.items():
        if str(key).lower() in allowed_fields:
            values.extend(scalar_text_values(value))
    return values


def first_present(row: Mapping[str, Any], fields: Iterable[str]) -> Any | None:
    for field in fields:
        if field in row and row[field] is not None:
            return row[field]
    return None


def iter_dataset_rows(dataset: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(dataset, Mapping) and not hasattr(dataset, "column_names"):
        for record_id, value in dataset.items():
            if isinstance(value, Mapping):
                yield {"id": record_id, **value}
            else:
                yield {"id": record_id, "text": value}
        return
    for row in dataset:
        if not isinstance(row, Mapping):
            raise TypeError(f"Unexpected dataset row type: {type(row)!r}")
        yield row


def hash_retrieval_split(
    data: Mapping[str, Any],
    accumulator: HashAccumulator,
    split_key: str,
    policy: Mapping[str, Any],
) -> None:
    text_fields = {str(x).lower() for x in policy["text_field_names"]}
    id_fields = [str(x) for x in policy["id_field_names"]]
    scope = policy["scope"]
    query_rows_with_id = 0
    corpus_rows_with_id = 0

    for row in iter_dataset_rows(data["queries"]):
        accumulator.records["query_records"] += 1
        accumulator.splits[split_key] += 1
        record_id = first_present(row, id_fields)
        if record_id is not None:
            query_rows_with_id += 1
            accumulator.add_id("query_id", record_id)
        texts = extract_row_texts(row, text_fields)
        for text in texts:
            accumulator.add_text("query_text", text)
        instruction = row.get("instruction")
        query_text = first_present(row, ("text", "query", "question"))
        if (
            scope.get("hash_query_instruction_combination", True)
            and isinstance(instruction, str)
            and isinstance(query_text, str)
        ):
            accumulator.add_text("query_text", f"{instruction}\n{query_text}")

    for row in iter_dataset_rows(data["corpus"]):
        accumulator.records["corpus_records"] += 1
        accumulator.splits[split_key] += 1
        record_id = first_present(row, id_fields)
        if record_id is not None:
            corpus_rows_with_id += 1
            accumulator.add_id("corpus_id", record_id)
        texts = extract_row_texts(row, text_fields)
        for text in texts:
            accumulator.add_text("corpus_text", text)
        title = row.get("title")
        body = first_present(row, ("text", "passage", "document", "context"))
        if (
            scope.get("hash_title_text_combination", True)
            and isinstance(title, str)
            and title.strip()
            and isinstance(body, str)
            and body.strip()
        ):
            accumulator.add_text("corpus_text", f"{title}\n{body}")

    qrels = data.get("relevant_docs") or {}
    for query_id, documents in qrels.items():
        normalized_query = normalize_text(str(query_id), accumulator.normalization)
        accumulator.add_id("query_id", query_id)
        if not isinstance(documents, Mapping):
            raise TypeError("relevant_docs values must map corpus IDs to scores")
        for corpus_id, score in documents.items():
            normalized_corpus = normalize_text(
                str(corpus_id), accumulator.normalization
            )
            accumulator.add_id("corpus_id", corpus_id)
            accumulator.add_relation(
                "qrel_relation",
                {
                    "query_id": normalized_query,
                    "corpus_id": normalized_corpus,
                    "score": score,
                },
            )

    if scope.get("hash_reranking_candidate_relations", True):
        for query_id, documents in (data.get("top_ranked") or {}).items():
            normalized_query = normalize_text(str(query_id), accumulator.normalization)
            for rank, corpus_id in enumerate(documents):
                normalized_corpus = normalize_text(
                    str(corpus_id), accumulator.normalization
                )
                accumulator.add_relation(
                    "candidate_relation",
                    {
                        "query_id": normalized_query,
                        "corpus_id": normalized_corpus,
                        "rank": rank,
                    },
                )

    # These are row counts, not source IDs; unique ID counts come from the
    # disk-backed hash tables and do not require retaining raw IDs in memory.
    accumulator.records["query_rows_with_id"] += query_rows_with_id
    accumulator.records["corpus_rows_with_id"] += corpus_rows_with_id


def nested_retrieval_split(task: Any, plan: TaskPlan) -> Mapping[str, Any]:
    if hasattr(task, "queries") and hasattr(task, "corpus"):
        task.convert_v1_dataset_format_to_v2(num_proc=None)
    try:
        return task.dataset[plan.subset][plan.split]
    except (KeyError, TypeError) as error:
        available = {
            str(subset): list(splits)
            for subset, splits in getattr(task, "dataset", {}).items()
        }
        raise RuntimeError(
            f"Cannot find {plan.subset}/{plan.split} after loading {plan.key}; "
            f"available={available}"
        ) from error


def iter_non_retrieval_splits(
    task: Any, plan: TaskPlan, scope: str
) -> Iterator[tuple[str, Any]]:
    dataset = task.dataset
    if (
        isinstance(dataset, Mapping)
        and plan.subset in dataset
        and isinstance(dataset[plan.subset], Mapping)
        and plan.split in dataset[plan.subset]
    ):
        dataset = dataset[plan.subset]
    if not isinstance(dataset, Mapping):
        raise TypeError(f"Unexpected non-retrieval dataset type: {type(dataset)!r}")
    if scope == "all_loaded_splits":
        selected_splits = list(dataset)
    elif scope == "protocol_eval_split_only":
        selected_splits = [plan.split]
    else:
        raise ValueError(f"Unknown non_retrieval_splits policy: {scope}")
    for split in selected_splits:
        if split not in dataset:
            raise RuntimeError(f"Missing split {split!r} in {plan.key}")
        yield str(split), dataset[split]


def hash_non_retrieval_task(
    task: Any,
    plan: TaskPlan,
    accumulator: HashAccumulator,
    policy: Mapping[str, Any],
) -> None:
    text_fields = {str(x).lower() for x in policy["text_field_names"]}
    id_fields = [str(x) for x in policy["id_field_names"]]
    split_scope = str(policy["scope"]["non_retrieval_splits"])
    for split, dataset in iter_non_retrieval_splits(task, plan, split_scope):
        for row_index, row in enumerate(iter_dataset_rows(dataset)):
            accumulator.records["evaluation_records"] += 1
            accumulator.splits[f"{plan.subset}/{split}"] += 1
            record_id = first_present(row, id_fields)
            if record_id is not None:
                accumulator.add_id("example_id", record_id)
            text_digests: list[str] = []
            for text in extract_row_texts(row, text_fields):
                digest = accumulator.add_text("evaluation_text", text)
                if digest is not None:
                    text_digests.append(digest.hex())
            # Hash the pair/example structure without storing labels, scores, or text.
            if text_digests:
                accumulator.add_relation(
                    "example_relation",
                    {
                        "split": split,
                        "row": row_index,
                        "text_sha256": text_digests,
                    },
                )


def task_directory(output: Path, plan: TaskPlan) -> Path:
    return (
        output
        / safe_component(plan.protocol_id)
        / safe_component(plan.label)
        / safe_component(plan.subset)
        / safe_component(plan.split)
    )


def task_fingerprint(
    plan: TaskPlan, policy: Mapping[str, Any], minhash_enabled: bool
) -> str:
    return sha256_bytes(
        canonical_json(
            {
                "plan": plan.public_dict(),
                "policy": policy,
                "minhash_enabled": minhash_enabled,
            }
        )
    )


def completed_manifest(path: Path) -> dict[str, Any] | None:
    manifest_path = path / "manifest.json"
    success_path = path / "_SUCCESS"
    if not manifest_path.is_file() or not success_path.is_file():
        return None
    manifest = read_json(manifest_path)
    if success_path.read_text(encoding="ascii").strip() != sha256_file(manifest_path):
        raise RuntimeError(f"Completion marker does not match manifest: {path}")
    for entry in manifest.get("files", []):
        artifact = path / entry["path"]
        if not artifact.is_file() or sha256_file(artifact) != entry["sha256"]:
            raise RuntimeError(f"Artifact integrity check failed: {artifact}")
    return manifest


def plan_action(
    destination: Path,
    fingerprint: str,
    resume: bool,
    overwrite: bool,
) -> str:
    if not destination.exists():
        return "build"
    manifest = completed_manifest(destination)
    if manifest is not None and manifest.get("task_fingerprint") == fingerprint:
        if resume and not overwrite:
            return "skip-complete"
        return "replace"
    if overwrite:
        return "replace"
    if manifest is None:
        return "error-incomplete-output"
    return "error-fingerprint-mismatch"


def build_task(
    plan: TaskPlan,
    policy: Mapping[str, Any],
    policy_path: Path,
    destination: Path,
    minhash_class: Any | None,
    num_proc: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.building-", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        database = temporary / ".hashes.sqlite3"
        accumulator = HashAccumulator(
            database,
            normalization=policy["normalization"],
            minhash_config=policy["minhash"],
            minhash_class=minhash_class,
        )
        try:
            plan.task.load_data(num_proc=num_proc)
            if plan.task_type in {"Retrieval", "Reranking"}:
                data = nested_retrieval_split(plan.task, plan)
                hash_retrieval_split(
                    data,
                    accumulator,
                    f"{plan.subset}/{plan.split}",
                    policy,
                )
            else:
                hash_non_retrieval_task(plan.task, plan, accumulator, policy)
            accumulator.connection.commit()
            counts, files = accumulator.export(temporary)
            records = dict(sorted(accumulator.records.items()))
            split_records = dict(sorted(accumulator.splits.items()))
        finally:
            accumulator.close()
        database.unlink(missing_ok=True)

        fingerprint = task_fingerprint(plan, policy, minhash_class is not None)
        manifest = {
            "schema_version": int(policy["schema_version"]),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_fingerprint": fingerprint,
            "task": plan.public_dict(),
            "policy": {
                "path": relative_or_absolute(policy_path),
                "sha256": sha256_file(policy_path),
                "policy_id": policy["policy_id"],
                "normalization": policy["normalization"],
                "scope": policy["scope"],
                "artifact_policy": policy["artifact_policy"],
            },
            "minhash": {
                **policy["minhash"],
                "enabled": minhash_class is not None,
                "implementation": (
                    "datasketch.MinHash" if minhash_class is not None else None
                ),
            },
            "records": records,
            "records_by_subset_split": split_records,
            "hash_counts": counts,
            "files": files,
            "contains_source_text": False,
            "contains_source_ids": False,
        }
        manifest_path = temporary / "manifest.json"
        write_json_atomic(manifest_path, manifest)
        (temporary / "_SUCCESS").write_text(
            sha256_file(manifest_path) + "\n", encoding="ascii"
        )

        if destination.exists():
            if not overwrite:
                raise FileExistsError(destination)
            backup = destination.with_name(f".{destination.name}.old-{os.getpid()}")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(destination, backup)
            try:
                os.replace(temporary, destination)
            except BaseException:
                os.replace(backup, destination)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temporary, destination)
        return manifest


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rebuild_root_manifest(
    output: Path, policy: Mapping[str, Any], policy_path: Path
) -> None:
    task_manifests: list[dict[str, Any]] = []
    for path in sorted(output.glob("*/*/*/*/manifest.json")):
        task_manifest = completed_manifest(path.parent)
        if task_manifest is None:
            continue
        task_manifests.append(
            {
                "path": str(path.parent.relative_to(output)),
                "task_fingerprint": task_manifest["task_fingerprint"],
                "key": task_manifest["task"]["key"],
                "dataset": task_manifest["task"]["dataset"],
                "license": task_manifest["task"]["license"],
                "hash_counts": task_manifest["hash_counts"],
                "manifest_sha256": sha256_file(path),
            }
        )
    root_manifest = {
        "schema_version": int(policy["schema_version"]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_id": policy["policy_id"],
        "policy_path": relative_or_absolute(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "artifact_policy": policy["artifact_policy"],
        "completed_tasks": len(task_manifests),
        "tasks": task_manifests,
    }
    write_json_atomic(output / "manifest.json", root_manifest)


def main() -> None:
    args = parse_args()
    protocol_paths = tuple(args.protocol or DEFAULT_PROTOCOLS)
    policy_path = args.policy.resolve()
    policy = read_json(policy_path)
    validate_policy(policy)
    plans, mteb_revision = resolve_plans(protocol_paths)
    selected = select_plans(plans, args.task)
    minhash_mode = args.minhash or str(policy["minhash"].get("default", "off"))
    if minhash_mode not in {"off", "auto", "required"}:
        raise ValueError(f"Invalid MinHash mode in policy: {minhash_mode}")
    minhash_class = import_minhash(minhash_mode)
    minhash_enabled = minhash_class is not None
    output = args.output_dir.resolve()

    resolved = {
        "mteb_git_revision": mteb_revision,
        "mteb_version": __import__("mteb").__version__,
        "policy": {
            "path": relative_or_absolute(policy_path),
            "sha256": sha256_file(policy_path),
            "policy_id": policy["policy_id"],
        },
        "output_dir": relative_or_absolute(output),
        "minhash": {
            "requested": minhash_mode,
            "enabled": minhash_enabled,
            "implementation": "datasketch.MinHash" if minhash_enabled else None,
        },
        "selected_tasks": [plan.public_dict() for plan in selected],
    }
    if args.list_only:
        print(json.dumps(resolved, ensure_ascii=False, indent=2, default=str))
        return

    decisions: list[dict[str, Any]] = []
    for plan in selected:
        destination = task_directory(output, plan)
        fingerprint = task_fingerprint(plan, policy, minhash_enabled)
        decisions.append(
            {
                "key": plan.key,
                "destination": relative_or_absolute(destination),
                "task_fingerprint": fingerprint,
                "action": plan_action(
                    destination, fingerprint, args.resume, args.overwrite
                ),
            }
        )
    if args.dry_run:
        print(
            json.dumps(
                {**resolved, "decisions": decisions},
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return

    failures = [row for row in decisions if row["action"].startswith("error-")]
    if failures:
        raise RuntimeError(
            "Existing output is incomplete or has a different fingerprint; "
            f"use --overwrite after inspection: {failures}"
        )

    results: list[dict[str, Any]] = []
    for plan, decision in zip(selected, decisions, strict=True):
        destination = task_directory(output, plan)
        if decision["action"] == "skip-complete":
            manifest = completed_manifest(destination)
            assert manifest is not None
            results.append(
                {
                    "key": plan.key,
                    "action": "skipped",
                    "destination": relative_or_absolute(destination),
                    "hash_counts": manifest["hash_counts"],
                }
            )
            continue
        manifest = build_task(
            plan=plan,
            policy=policy,
            policy_path=policy_path,
            destination=destination,
            minhash_class=minhash_class,
            num_proc=args.num_proc,
            overwrite=decision["action"] == "replace",
        )
        rebuild_root_manifest(output, policy, policy_path)
        results.append(
            {
                "key": plan.key,
                "action": "built",
                "destination": relative_or_absolute(destination),
                "hash_counts": manifest["hash_counts"],
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    if output.exists():
        rebuild_root_manifest(output, policy, policy_path)
    print(
        json.dumps({"complete": True, "results": results}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
