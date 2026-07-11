#!/usr/bin/env python3
"""Build a deterministic, provenance-tracked ms-swift embedding data mix.

The normal execution path streams Hugging Face sources. ``--list`` and
``--dry-run`` only validate/print the pinned plan and perform no data download.
Evaluation split protection is intentionally enforced even for the repository's
private, performance-first track.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import random
import re
import shutil
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import quote


DEFAULT_CONFIG = Path("configs/performance_data_mix_v1.json")
HANGUL_RE = re.compile(r"[가-힣]")
WHITESPACE_RE = re.compile(r"\s+")
NEGATIVE_FIELD_RE = re.compile(r"^negative_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", default="pilot_50k")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--list", action="store_true", help="List validated phases and sources"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the selected plan without contacting dataset hosts",
    )
    parser.add_argument("--only-source", action="append", default=[])
    parser.add_argument(
        "--sample-cap",
        type=int,
        help="Cap every selected source; intended only for converter smoke tests",
    )
    parser.add_argument("--negatives-per-row", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return WHITESPACE_RE.sub(" ", text).strip()


def stable_hash(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


def stable_seed(base_seed: int, value: str) -> int:
    return int(stable_hash(str(base_seed), value)[:16], 16)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_value(row: dict[str, Any], field: str) -> Any:
    value: Any = row
    for key in field.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing nested field {field!r}")
        value = value[key]
    return value


def instructed_query(query: str, instruction: str) -> str:
    if query.lstrip().startswith("Instruct:"):
        return query
    return f"Instruct: {instruction}\nQuery: {query}"


def semantic_query_body(query: str) -> str:
    """Return user text, excluding an upstream instruction prefix if present."""

    if query.lstrip().startswith("Instruct:") and "Query:" in query:
        return query.rpartition("Query:")[2].strip()
    return query.strip()


def format_row(query: str, positive: str, negatives: list[str]) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": query}],
        "positive_messages": [[{"role": "user", "content": positive}]],
        "negative_messages": [
            [{"role": "user", "content": negative}] for negative in negatives
        ],
    }


def read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def blocked_by_policy(config: dict[str, Any], source: dict[str, Any]) -> str | None:
    repo_id = source.get("repo_id")
    split = source.get("split")
    file_name = source.get("file")
    for rule in config.get("blocked_training_inputs", []):
        if not repo_id or rule.get("repo_id") != repo_id:
            continue
        blocked_splits = rule.get("blocked_splits", [])
        if "all" in blocked_splits or (split and split in blocked_splits):
            return f"{repo_id}:{split or 'all'} is blocked"
        for pattern in rule.get("blocked_files", []):
            if file_name and fnmatch.fnmatch(file_name, pattern):
                return f"{repo_id}/{file_name} matches blocked pattern {pattern}"
    return None


def validate_config(config: dict[str, Any]) -> None:
    required = {"schema_version", "mix_id", "seed", "phases", "sources"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config missing keys: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("Only schema_version=1 is supported")
    sources = config["sources"]
    supported_adapters = {
        "triplet_fields",
        "f2_fields",
        "kalm_lists",
        "classification_label",
        "sts_pairs",
        "qa_context",
        "beir_join",
    }
    for source_id, source in sources.items():
        revision = source.get("revision", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{source_id}: revision must be a pinned 40-character SHA")
        if source.get("adapter") not in supported_adapters:
            raise ValueError(
                f"{source_id}: unsupported adapter {source.get('adapter')!r}"
            )
        if not isinstance(source.get("trained_on_tasks"), list):
            raise ValueError(f"{source_id}: trained_on_tasks must be a list")
        blocked = blocked_by_policy(config, source)
        if blocked:
            raise ValueError(f"{source_id}: {blocked}")
    for phase_name, phase in config["phases"].items():
        caps = phase.get("source_caps", {})
        unknown = set(caps) - set(sources)
        if unknown:
            raise ValueError(f"{phase_name}: unknown sources {sorted(unknown)}")
        if any(not isinstance(cap, int) or cap <= 0 for cap in caps.values()):
            raise ValueError(f"{phase_name}: all source caps must be positive integers")
        planned = sum(caps.values())
        if planned != phase.get("target_rows"):
            raise ValueError(
                f"{phase_name}: source caps sum to {planned}, target is {phase.get('target_rows')}"
            )
        for source_id, cap in caps.items():
            observed = sources[source_id].get("observed_rows")
            if observed is not None and cap > observed:
                raise ValueError(
                    f"{phase_name}/{source_id}: cap {cap} exceeds observed rows {observed}"
                )


def selected_caps(config: dict[str, Any], args: argparse.Namespace) -> dict[str, int]:
    if args.phase not in config["phases"]:
        raise ValueError(f"Unknown phase {args.phase!r}")
    caps = dict(config["phases"][args.phase]["source_caps"])
    if args.only_source:
        requested = set(args.only_source)
        unknown = requested - set(caps)
        if unknown:
            raise ValueError(f"Sources not present in {args.phase}: {sorted(unknown)}")
        caps = {
            source_id: cap for source_id, cap in caps.items() if source_id in requested
        }
    if args.sample_cap is not None:
        if args.sample_cap <= 0:
            raise ValueError("--sample-cap must be positive")
        caps = {source_id: min(cap, args.sample_cap) for source_id, cap in caps.items()}
    return caps


def plan_summary(
    config: dict[str, Any], phase_name: str, caps: dict[str, int]
) -> dict[str, Any]:
    rows = []
    exposed_tasks: set[str] = set()
    for source_id, cap in caps.items():
        source = config["sources"][source_id]
        tasks = source["trained_on_tasks"]
        exposed_tasks.update(tasks)
        rows.append(
            {
                "source_id": source_id,
                "rows": cap,
                "loader": source["loader"],
                "adapter": source["adapter"],
                "repo_id": source["repo_id"],
                "revision": source["revision"],
                "file_or_split": source.get("file", source.get("split")),
                "trained_on_tasks": tasks,
                "benchmark_exposure": source["benchmark_exposure"],
            }
        )
    return {
        "mix_id": config["mix_id"],
        "phase": phase_name,
        "configured_target_rows": config["phases"][phase_name]["target_rows"],
        "selected_target_rows": sum(caps.values()),
        "default_visibility": config["default_visibility"],
        "default_use_policy": config["default_use_policy"],
        "trained_on_tasks": sorted(exposed_tasks),
        "sources": rows,
    }


def load_hf_dataset_stream(
    source: dict[str, Any], seed: int, shuffle: bool = True
) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    if source["loader"] == "hf_parquet":
        url = (
            f"https://huggingface.co/datasets/{source['repo_id']}/resolve/"
            f"{source['revision']}/{quote(source['file'])}"
        )
        stream = load_dataset(
            "parquet",
            data_files={"train": url},
            split="train",
            streaming=True,
            token=token,
        )
    else:
        kwargs: dict[str, Any] = {
            "path": source["repo_id"],
            "revision": source["revision"],
            "split": source["split"],
            "streaming": True,
            "token": token,
        }
        if source.get("config") and source["config"] != "default":
            kwargs["name"] = source["config"]
        stream = load_dataset(**kwargs)
    if shuffle:
        stream = stream.shuffle(
            seed=seed,
            buffer_size=int(source.get("shuffle_buffer", 10_000)),
        )
    return stream


def iter_triplet_fields(
    source: dict[str, Any], seed: int
) -> Iterator[tuple[str, str, list[str]]]:
    schema = source["schema"]
    for row in load_hf_dataset_stream(source, seed):
        yield (
            row.get(schema["query_field"]),
            row.get(schema["positive_field"]),
            [row.get(field) for field in schema["negative_fields"]],
        )


def iter_f2_fields(
    source: dict[str, Any], seed: int
) -> Iterator[tuple[str, str, list[str]]]:
    for row in load_hf_dataset_stream(source, seed):
        negative_fields = sorted(
            (field for field in row if NEGATIVE_FIELD_RE.match(field)),
            key=lambda field: int(NEGATIVE_FIELD_RE.match(field).group(1)),  # type: ignore[union-attr]
        )
        yield row.get("query"), row.get("passage"), [
            row.get(k) for k in negative_fields
        ]


def iter_kalm_lists(
    source: dict[str, Any], seed: int
) -> Iterator[tuple[str, str, list[str]]]:
    schema = source["schema"]
    for row in load_hf_dataset_stream(source, seed):
        positives = row.get(schema["positive_list_field"]) or []
        if not positives:
            yield row.get(schema["query_field"]), None, []
            continue
        yield (
            row.get(schema["query_field"]),
            positives[0],
            list(row.get(schema["negative_list_field"]) or []),
        )


def iter_classification(
    source: dict[str, Any], seed: int
) -> Iterator[tuple[str, str, list[str]]]:
    schema = source["schema"]
    label_values = schema["label_values"]
    for row in load_hf_dataset_stream(source, seed):
        label = row.get(schema["label_field"])
        positive = (
            label_values[label] if isinstance(label, int) else normalize_text(label)
        )
        negatives = [candidate for candidate in label_values if candidate != positive]
        yield row.get(schema["text_field"]), positive, negatives


def iter_sts_pairs(
    source: dict[str, Any], seed: int
) -> Iterator[tuple[str, str, list[str]]]:
    schema = source["schema"]
    negative_pool: list[str] = []
    for row in load_hf_dataset_stream(source, seed, shuffle=False):
        score = float(nested_value(row, schema["score_field"]))
        if score <= float(schema["negative_threshold"]):
            candidate = normalize_text(row.get(schema["sentence2_field"]))
            if candidate:
                negative_pool.append(candidate)
    if not negative_pool:
        raise RuntimeError(
            f"{source['repo_id']}: STS adapter found no low-score negatives"
        )
    for row in load_hf_dataset_stream(source, seed, shuffle=True):
        score = float(nested_value(row, schema["score_field"]))
        if score < float(schema["positive_threshold"]):
            continue
        query = row.get(schema["sentence1_field"])
        positive = row.get(schema["sentence2_field"])
        index = stable_seed(seed, normalize_text(query)) % len(negative_pool)
        yield query, positive, [negative_pool[index]]


def iter_qa_context(
    source: dict[str, Any], seed: int
) -> Iterator[tuple[str, str, list[str]]]:
    """Convert extractive QA train rows into retrieval examples.

    The first pass builds a deterministic pool of unique contexts. The second
    pass emits question -> answer-bearing context pairs plus bootstrap
    negatives from other contexts. These bootstrap negatives are deliberately
    simple: scale/target-adaptation queues must remine them with the current
    embedding model before using this source for a final candidate.
    """

    schema = source["schema"]
    context_field = schema["context_field"]
    question_field = schema["question_field"]
    bootstrap_negatives = int(source.get("bootstrap_negatives", 7))
    if bootstrap_negatives <= 0:
        raise ValueError("qa_context bootstrap_negatives must be positive")

    context_pool: list[str] = []
    context_seen: set[str] = set()
    for row in load_hf_dataset_stream(source, seed, shuffle=False):
        context = normalize_text(row.get(context_field))
        if context and context not in context_seen:
            context_seen.add(context)
            context_pool.append(context)
    if len(context_pool) <= bootstrap_negatives:
        raise RuntimeError(
            f"{source['repo_id']}: QA adapter needs more than "
            f"{bootstrap_negatives} unique contexts"
        )

    for row in load_hf_dataset_stream(source, seed, shuffle=True):
        query = normalize_text(row.get(question_field))
        positive = normalize_text(row.get(context_field))
        if not query or not positive:
            yield query, positive, []
            continue
        start = stable_seed(seed, stable_hash(query, positive)) % len(context_pool)
        negatives: list[str] = []
        for offset in range(len(context_pool)):
            candidate = context_pool[(start + offset) % len(context_pool)]
            if candidate == positive:
                continue
            negatives.append(candidate)
            if len(negatives) >= bootstrap_negatives:
                break
        yield query, positive, negatives


def iter_beir_join(
    source: dict[str, Any], seed: int
) -> Iterator[tuple[str, str, list[str]]]:
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")

    def load_part(part: dict[str, Any]):
        return load_dataset(
            source["repo_id"],
            name=part["config"],
            split=part["split"],
            revision=source["revision"],
            token=token,
        )

    qrels_spec = source["qrels"]
    queries_spec = source["queries"]
    corpus_spec = source["corpus"]
    qrels = load_part(qrels_spec)
    queries_ds = load_part(queries_spec)
    corpus_ds = load_part(corpus_spec)
    queries = {
        str(row[queries_spec["id_field"]]): normalize_text(
            row[queries_spec["text_field"]]
        )
        for row in queries_ds
    }
    corpus: dict[str, str] = {}
    for row in corpus_ds:
        title = normalize_text(row.get(corpus_spec.get("title_field", "")))
        body = normalize_text(row.get(corpus_spec["text_field"]))
        text = normalize_text(f"{title} {body}") if title else body
        corpus[str(row[corpus_spec["id_field"]])] = text
    positive_threshold = float(source.get("positive_score_gt", 0))
    positives: dict[str, list[str]] = defaultdict(list)
    negatives: dict[str, list[str]] = defaultdict(list)
    for row in qrels:
        query_id = str(row[qrels_spec["query_id_field"]])
        doc_id = str(row[qrels_spec["doc_id_field"]])
        score = float(row[qrels_spec["score_field"]])
        if score > positive_threshold:
            positives[query_id].append(doc_id)
        else:
            negatives[query_id].append(doc_id)
    query_ids = sorted(positives)
    random.Random(seed).shuffle(query_ids)
    corpus_ids = sorted(corpus)
    for query_id in query_ids:
        if query_id not in queries:
            continue
        positive_ids = [doc_id for doc_id in positives[query_id] if doc_id in corpus]
        if not positive_ids:
            continue
        positive_id = positive_ids[stable_seed(seed, query_id) % len(positive_ids)]
        negative_ids = [
            doc_id
            for doc_id in negatives.get(query_id, [])
            if doc_id in corpus and doc_id not in positive_ids
        ]
        if not negative_ids:
            start = stable_seed(seed, f"negative:{query_id}") % len(corpus_ids)
            for offset in range(len(corpus_ids)):
                candidate = corpus_ids[(start + offset) % len(corpus_ids)]
                if candidate not in positive_ids:
                    negative_ids = [candidate]
                    break
        yield queries[query_id], corpus[positive_id], [corpus[x] for x in negative_ids]


def source_examples(
    source: dict[str, Any], seed: int
) -> Iterator[tuple[str, str, list[str]]]:
    adapter = source["adapter"]
    if adapter == "triplet_fields":
        return iter_triplet_fields(source, seed)
    if adapter == "f2_fields":
        return iter_f2_fields(source, seed)
    if adapter == "kalm_lists":
        return iter_kalm_lists(source, seed)
    if adapter == "classification_label":
        return iter_classification(source, seed)
    if adapter == "sts_pairs":
        return iter_sts_pairs(source, seed)
    if adapter == "qa_context":
        return iter_qa_context(source, seed)
    if adapter == "beir_join":
        return iter_beir_join(source, seed)
    raise AssertionError(f"Unsupported adapter: {adapter}")


def cleaned_example(
    raw: tuple[Any, Any, list[Any]],
    source: dict[str, Any],
    defaults: dict[str, Any],
    seed: int,
    negatives_per_row: int,
) -> tuple[str, str, list[str]] | tuple[None, str, list[str]]:
    query = normalize_text(raw[0])
    positive = normalize_text(raw[1])
    negatives = []
    seen_negatives: set[str] = set()
    for value in raw[2]:
        negative = normalize_text(value)
        if not negative or negative == positive or negative in seen_negatives:
            continue
        seen_negatives.add(negative)
        negatives.append(negative)
    min_query_chars = int(source.get("min_query_chars", defaults["min_query_chars"]))
    min_document_chars = int(
        source.get("min_document_chars", defaults["min_document_chars"])
    )
    max_document_chars = int(
        source.get("max_document_chars", defaults["max_document_chars"])
    )
    query_body = semantic_query_body(query)
    if (
        len(query_body) < min_query_chars
        or len(positive) < min_document_chars
        or not negatives
    ):
        return None, "missing_or_short", []
    if source.get(
        "require_hangul", defaults["require_hangul"]
    ) and not HANGUL_RE.search(query_body):
        return None, "non_korean_query", []
    negatives = [
        negative
        for negative in negatives
        if len(negative) >= min_document_chars and len(negative) <= max_document_chars
    ]
    if not negatives:
        return None, "missing_or_short_negative", []
    if len(positive) > max_document_chars:
        return None, "document_too_long", []
    if len(negatives) > negatives_per_row:
        rng = random.Random(stable_seed(seed, stable_hash(query, positive)))
        negatives = rng.sample(negatives, negatives_per_row)
    instruction = source.get("query_instruction", defaults["query_instruction"])
    return instructed_query(query, instruction), positive, negatives


def write_mix(
    config: dict[str, Any], args: argparse.Namespace, caps: dict[str, int]
) -> dict[str, Any]:
    if args.output_dir is None:
        raise ValueError("--output-dir is required for a real build")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already exists; pass --overwrite to replace it"
        )
    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    defaults = dict(config["defaults"])
    negatives_per_row = args.negatives_per_row or int(defaults["negatives_per_row"])
    if negatives_per_row <= 0:
        raise ValueError("negatives-per-row must be positive")
    train_path = temporary / "train.jsonl"
    provenance_path = temporary / "provenance.jsonl"
    global_pairs: set[str] = set()
    source_stats: dict[str, Any] = {}
    total_rows = 0
    try:
        with train_path.open(
            "w", encoding="utf-8"
        ) as train_handle, provenance_path.open(
            "w", encoding="utf-8"
        ) as provenance_handle:
            for source_id, target in caps.items():
                source = dict(config["sources"][source_id])
                source.setdefault("shuffle_buffer", defaults["shuffle_buffer"])
                seed = stable_seed(int(config["seed"]), source_id)
                accepted = 0
                examined = 0
                rejected: dict[str, int] = defaultdict(int)
                for raw in source_examples(source, seed):
                    examined += 1
                    cleaned = cleaned_example(
                        raw, source, defaults, seed, negatives_per_row
                    )
                    if cleaned[0] is None:
                        rejected[cleaned[1]] += 1
                        continue
                    query, positive, negatives = cleaned
                    pair_hash = stable_hash(query, positive)
                    if pair_hash in global_pairs:
                        rejected["duplicate_query_positive"] += 1
                        continue
                    global_pairs.add(pair_hash)
                    row = format_row(query, positive, negatives)
                    row_json = json.dumps(
                        row, ensure_ascii=False, separators=(",", ":")
                    )
                    row_hash = hashlib.sha256(row_json.encode("utf-8")).hexdigest()
                    train_handle.write(row_json + "\n")
                    provenance = {
                        "row_index": total_rows,
                        "row_sha256": row_hash,
                        "source_id": source_id,
                        "repo_id": source["repo_id"],
                        "revision": source["revision"],
                        "file": source.get("file"),
                        "split": source.get("split"),
                        "trained_on_tasks": source["trained_on_tasks"],
                        "benchmark_exposure": source["benchmark_exposure"],
                        "release_eligible": source.get("release_eligible", False),
                    }
                    provenance_handle.write(
                        json.dumps(
                            provenance, ensure_ascii=False, separators=(",", ":")
                        )
                        + "\n"
                    )
                    total_rows += 1
                    accepted += 1
                    if accepted >= target:
                        break
                if accepted != target:
                    raise RuntimeError(
                        f"{source_id}: collected {accepted}/{target} after examining {examined}; "
                        f"rejected={dict(rejected)}"
                    )
                source_stats[source_id] = {
                    "requested": target,
                    "accepted": accepted,
                    "examined": examined,
                    "rejected": dict(sorted(rejected.items())),
                    "trained_on_tasks": source["trained_on_tasks"],
                    "benchmark_exposure": source["benchmark_exposure"],
                }
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mix_id": config["mix_id"],
            "phase": args.phase,
            "configured_target_rows": config["phases"][args.phase]["target_rows"],
            "built_rows": total_rows,
            "seed": config["seed"],
            "negatives_per_row_max": negatives_per_row,
            "visibility": config["default_visibility"],
            "use_policy": config["default_use_policy"],
            "release_eligible": False,
            "release_blockers": [
                "performance track intentionally includes sources with missing/custom/noncommercial terms",
                "official benchmark train-task exposure must be disclosed",
                "a separate rights-safe retrain or distillation pass is required before public release",
            ],
            "source_stats": source_stats,
            "files": {
                "train.jsonl": {
                    "rows": total_rows,
                    "sha256": file_hash(train_path),
                },
                "provenance.jsonl": {
                    "rows": total_rows,
                    "sha256": file_hash(provenance_path),
                },
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary.replace(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    validate_config(config)
    if args.list:
        listing = {
            name: plan_summary(config, name, dict(phase["source_caps"]))
            for name, phase in config["phases"].items()
        }
        print(json.dumps(listing, ensure_ascii=False, indent=2))
        return
    caps = selected_caps(config, args)
    summary = plan_summary(config, args.phase, caps)
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    manifest = write_mix(config, args, caps)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
