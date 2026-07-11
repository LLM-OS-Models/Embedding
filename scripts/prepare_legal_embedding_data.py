#!/usr/bin/env python3
"""Build deterministic, source-grounded Korean legal embedding candidates.

This extractor deliberately does not ask an LLM to invent questions and does not
create negatives.  It only uses relationships already encoded by the source:

* statute / administrative-rule / ordinance title + article heading -> article
* precedent issue (판시사항) -> holding summary (판결요지)

The output is a candidate pool, not a claim that every row is a manually judged
relevance label.  Benchmark decontamination and negative mining belong downstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


FRONTMATTER_RE = re.compile(r"\A---\r?\n(?P<meta>.*?)\r?\n---\r?\n", re.DOTALL)
TOP_LEVEL_SCALAR_RE = re.compile(r"^(?P<key>[^\s:#][^:]*):(?:\s*(?P<value>.*))?$")
H1_RE = re.compile(r"(?m)^#\s+([^#\n].*?)\s*$")
ARTICLE_RE = re.compile(
    r"(?m)^#####\s+(?P<heading>제[0-9]+조(?:의[0-9]+)?(?:\s*\([^\n)]*\))?)\s*$"
)
LEVEL2_RE = re.compile(r"(?m)^##\s+(?P<heading>[^#\n].*?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/legal_data_sources_v1.json"),
        help="Pinned source configuration.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Directory containing cloned repositories (default: config value).",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source id to process; repeatable. Default: every source with an extractor.",
    )
    parser.add_argument("--output", type=Path, help="Destination JSONL.")
    parser.add_argument("--manifest", type=Path, help="Optional deterministic run manifest.")
    parser.add_argument(
        "--min-positive-chars",
        type=int,
        default=64,
        help="Drop structurally paired positives shorter than this value.",
    )
    parser.add_argument(
        "--max-query-chars",
        type=int,
        default=0,
        help="Source-text character cap; 0 keeps the full query candidate.",
    )
    parser.add_argument(
        "--max-positive-chars",
        type=int,
        default=0,
        help="Source-text character cap; 0 keeps the full positive candidate.",
    )
    parser.add_argument(
        "--max-files-per-source",
        type=int,
        default=0,
        help="Testing aid; 0 processes every selected file.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Global deterministic record cap; 0 emits all candidates.",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep exact query/positive duplicates (default: first occurrence wins).",
    )
    parser.add_argument(
        "--skip-revision-check",
        action="store_true",
        help="Allow a local checkout that does not match the pinned commit.",
    )
    parser.add_argument(
        "--verify-inventory",
        action="store_true",
        help="Verify configured document counts and bytes before extraction.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and count without writing JSONL.",
    )
    args = parser.parse_args()
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is set")
    if args.shard_count < 1:
        parser.error("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must satisfy 0 <= index < count")
    for name in (
        "min_positive_chars",
        "max_query_chars",
        "max_positive_chars",
        "max_files_per_source",
        "max_records",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    return args


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError(f"unsupported config schema: {config.get('schema_version')!r}")
    return config


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def parse_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
        if value is not None:
            value = value.replace("''", "'") if value else value
    return normalize(value)


def parse_document(raw_text: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_RE.match(raw_text)
    if not match:
        return {}, normalize(raw_text)
    metadata: dict[str, str] = {}
    for line in match.group("meta").splitlines():
        scalar = TOP_LEVEL_SCALAR_RE.match(line)
        if scalar:
            metadata[normalize(scalar.group("key"))] = parse_scalar(scalar.group("value") or "")
    return metadata, normalize(raw_text[match.end() :])


def source_title(metadata: dict[str, str], body: str, title_field: str) -> str:
    title = metadata.get(title_field, "")
    if title:
        return title
    match = H1_RE.search(body)
    return normalize(match.group(1)) if match else ""


def markdown_sections(body: str, heading_re: re.Pattern[str]) -> Iterator[tuple[str, str]]:
    matches = list(heading_re.finditer(body))
    level2_positions = [m.start() for m in LEVEL2_RE.finditer(body)] if heading_re is ARTICLE_RE else []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        if level2_positions:
            next_level2 = next((position for position in level2_positions if position > match.start()), None)
            if next_level2 is not None:
                end = min(end, next_level2)
        heading = normalize(match.group("heading"))
        section = normalize(body[match.start() : end])
        yield heading, section


def named_level2_sections(body: str) -> dict[str, str]:
    matches = list(LEVEL2_RE.finditer(body))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        heading = normalize(match.group("heading"))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections.setdefault(heading, normalize(body[match.end() : end]))
    return sections


def cap_source_text(text: str, limit: int) -> tuple[str, bool]:
    if limit == 0 or len(text) <= limit:
        return text, False
    return text[:limit].rstrip(), True


def stable_id(*parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"legal-v1-{digest[:24]}"


def selected_metadata(metadata: dict[str, str], keys: Iterable[str]) -> dict[str, str]:
    return {key: metadata[key] for key in keys if metadata.get(key, "")}


def base_provenance(
    source: dict[str, Any],
    relative_path: str,
    metadata: dict[str, str],
    document_sha256: str,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "repository": source["repository"],
        "repository_url": source["url"],
        "revision": source["revision"],
        "path": relative_path,
        "source_document_sha256": document_sha256,
        "metadata": selected_metadata(metadata, source.get("identifier_fields", [])),
        "license_as_declared": source["license_as_declared"]["content"],
    }
    source_url = metadata.get(source.get("source_url_field", ""), "")
    if source_url:
        provenance["source_url"] = source_url
    return provenance


def extract_article_records(
    source: dict[str, Any],
    relative_path: str,
    metadata: dict[str, str],
    body: str,
    document_sha256: str,
    min_positive_chars: int,
    max_query_chars: int,
    max_positive_chars: int,
) -> Iterator[dict[str, Any]]:
    if metadata.get("본문출처") == "parsing-failed":
        return
    title = source_title(metadata, body, source["title_field"])
    if not title:
        return
    for ordinal, (heading, source_section) in enumerate(markdown_sections(body, ARTICLE_RE)):
        query, query_truncated = cap_source_text(normalize(f"{title} {heading}"), max_query_chars)
        positive_source = normalize(f"# {title}\n\n{source_section}")
        positive, positive_truncated = cap_source_text(positive_source, max_positive_chars)
        if len(query) < 2 or len(positive) < min_positive_chars:
            continue
        provenance = base_provenance(source, relative_path, metadata, document_sha256)
        provenance.update(
            {
                "section_heading": heading,
                "section_ordinal": ordinal,
                "query_components": [title, heading],
                "source_text_truncated": {
                    "query": query_truncated,
                    "positive": positive_truncated,
                },
            }
        )
        yield {
            "id": stable_id(
                source["id"], source["revision"], relative_path, "title_article", str(ordinal)
            ),
            "query": query,
            "positive": positive,
            "pair_type": "source_title_and_article_heading_to_article",
            "label_origin": "source_document_structure_not_manual_relevance_judgment",
            "provenance": provenance,
        }


def extract_precedent_record(
    source: dict[str, Any],
    relative_path: str,
    metadata: dict[str, str],
    body: str,
    document_sha256: str,
    min_positive_chars: int,
    max_query_chars: int,
    max_positive_chars: int,
) -> Iterator[dict[str, Any]]:
    title = source_title(metadata, body, source["title_field"])
    sections = named_level2_sections(body)
    issue = sections.get("판시사항", "")
    holding = sections.get("판결요지", "")
    if not title or not issue or not holding:
        return
    query, query_truncated = cap_source_text(issue, max_query_chars)
    positive_source = normalize(f"# {title}\n\n## 판결요지\n\n{holding}")
    positive, positive_truncated = cap_source_text(positive_source, max_positive_chars)
    if len(query) < 2 or len(positive) < min_positive_chars:
        return
    provenance = base_provenance(source, relative_path, metadata, document_sha256)
    provenance.update(
        {
            "query_section_heading": "판시사항",
            "positive_section_heading": "판결요지",
            "source_text_truncated": {
                "query": query_truncated,
                "positive": positive_truncated,
            },
        }
    )
    yield {
        "id": stable_id(source["id"], source["revision"], relative_path, "issue_holding", "0"),
        "query": query,
        "positive": positive,
        "pair_type": "source_precedent_issue_to_holding_summary",
        "label_origin": "source_document_structure_not_manual_relevance_judgment",
        "provenance": provenance,
    }


def iter_source_files(repo_root: Path, globs: Iterable[str]) -> list[Path]:
    paths: dict[str, Path] = {}
    for pattern in globs:
        for path in repo_root.glob(pattern):
            if path.is_file():
                relative = path.relative_to(repo_root).as_posix()
                paths[relative] = path
    return [paths[key] for key in sorted(paths)]


def file_in_shard(source_id: str, relative_path: str, count: int, index: int) -> bool:
    digest = hashlib.sha256(f"{source_id}\0{relative_path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count == index


def checkout_revision(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def pair_digest(record: dict[str, Any]) -> bytes:
    return hashlib.sha256(
        (record["query"] + "\0" + record["positive"]).encode("utf-8")
    ).digest()


def json_line(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    source_root = args.source_root or Path(config["source_root_default"])
    configured = {source["id"]: source for source in config["sources"]}
    wanted = args.source or [
        source["id"] for source in config["sources"] if source.get("extractor") not in {None, "none"}
    ]
    unknown = sorted(set(wanted) - set(configured))
    if unknown:
        raise ValueError(f"unknown source ids: {', '.join(unknown)}")
    sources = [configured[source_id] for source_id in wanted]
    non_extractable = [source["id"] for source in sources if source.get("extractor") in {None, "none"}]
    if non_extractable:
        raise ValueError(f"sources have no extractor: {', '.join(non_extractable)}")

    stats: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = {}
    seen_pairs: set[bytes] = set()
    output_hash = hashlib.sha256()
    temp_output: Path | None = None
    output_handle = None

    if not args.dry_run:
        assert args.output is not None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp_output = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
        output_handle = temp_output.open("wb")

    reached_limit = False
    try:
        for source in sources:
            source_stats: Counter[str] = Counter()
            per_source[source["id"]] = source_stats
            repo_root = source_root / source["local_subdir"]
            if not repo_root.is_dir():
                raise FileNotFoundError(f"missing source checkout: {repo_root}")
            actual_revision = checkout_revision(repo_root)
            if actual_revision != source["revision"] and not args.skip_revision_check:
                raise RuntimeError(
                    f"revision mismatch for {source['id']}: expected {source['revision']}, got {actual_revision}"
                )
            files = iter_source_files(repo_root, source["file_globs"])
            if args.verify_inventory:
                inventory = source["snapshot_inventory"]
                actual_bytes = sum(path.stat().st_size for path in files)
                if len(files) != inventory["documents"] or actual_bytes != inventory["document_bytes"]:
                    raise RuntimeError(
                        f"inventory mismatch for {source['id']}: "
                        f"expected {inventory['documents']} files/{inventory['document_bytes']} bytes, "
                        f"got {len(files)} files/{actual_bytes} bytes"
                    )
            processed_for_source = 0
            for path in files:
                relative_path = path.relative_to(repo_root).as_posix()
                if not file_in_shard(
                    source["id"], relative_path, args.shard_count, args.shard_index
                ):
                    source_stats["files_outside_shard"] += 1
                    continue
                if args.max_files_per_source and processed_for_source >= args.max_files_per_source:
                    break
                processed_for_source += 1
                raw_bytes = path.read_bytes()
                document_sha256 = hashlib.sha256(raw_bytes).hexdigest()
                raw_text = raw_bytes.decode("utf-8")
                metadata, body = parse_document(raw_text)
                source_stats["files_processed"] += 1
                if metadata.get("본문출처") == "parsing-failed":
                    source_stats["files_parsing_failed"] += 1
                if source["extractor"] == "legal_articles":
                    records = extract_article_records(
                        source,
                        relative_path,
                        metadata,
                        body,
                        document_sha256,
                        args.min_positive_chars,
                        args.max_query_chars,
                        args.max_positive_chars,
                    )
                elif source["extractor"] == "precedent_issue_holding":
                    records = extract_precedent_record(
                        source,
                        relative_path,
                        metadata,
                        body,
                        document_sha256,
                        args.min_positive_chars,
                        args.max_query_chars,
                        args.max_positive_chars,
                    )
                else:
                    raise ValueError(f"unsupported extractor: {source['extractor']}")
                file_record_count = 0
                for record in records:
                    digest = pair_digest(record)
                    if not args.keep_duplicates and digest in seen_pairs:
                        stats["duplicates_skipped"] += 1
                        source_stats["duplicates_skipped"] += 1
                        continue
                    if not args.keep_duplicates:
                        seen_pairs.add(digest)
                    encoded = json_line(record)
                    if output_handle is not None:
                        output_handle.write(encoded)
                    output_hash.update(encoded)
                    stats["records_emitted"] += 1
                    stats["output_bytes"] += len(encoded)
                    source_stats["records_emitted"] += 1
                    file_record_count += 1
                    if args.max_records and stats["records_emitted"] >= args.max_records:
                        reached_limit = True
                        break
                if file_record_count == 0:
                    source_stats["files_without_emitted_record"] += 1
                if reached_limit:
                    break
            if reached_limit:
                break
        if output_handle is not None:
            output_handle.flush()
            os.fsync(output_handle.fileno())
            output_handle.close()
            output_handle = None
            assert temp_output is not None and args.output is not None
            temp_output.replace(args.output)
    except BaseException:
        if output_handle is not None:
            output_handle.close()
        if temp_output is not None:
            temp_output.unlink(missing_ok=True)
        raise

    manifest = {
        "schema_version": 1,
        "config_id": config["config_id"],
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "selected_sources": [source["id"] for source in sources],
        "source_revisions": {source["id"]: source["revision"] for source in sources},
        "parameters": {
            "min_positive_chars": args.min_positive_chars,
            "max_query_chars": args.max_query_chars,
            "max_positive_chars": args.max_positive_chars,
            "max_files_per_source": args.max_files_per_source,
            "max_records": args.max_records,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "deduplicate_exact_pairs": not args.keep_duplicates,
            "dry_run": args.dry_run,
        },
        "summary": dict(sorted(stats.items())),
        "per_source": {
            source_id: dict(sorted(source_stats.items()))
            for source_id, source_stats in per_source.items()
        },
        "output_sha256": output_hash.hexdigest(),
        "limit_reached": reached_limit,
    }
    rendered_manifest = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(rendered_manifest, encoding="utf-8")
    sys.stdout.write(rendered_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
