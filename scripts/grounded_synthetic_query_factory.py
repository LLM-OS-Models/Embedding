#!/usr/bin/env python3
"""Build grounded Korean synthetic-query data with auditable hard negatives.

The factory has four intentionally separate stages:

* ``prepare`` makes deterministic chat-completion requests from source-native
  query/positive candidate JSONL.
* ``generate`` resolves those requests through a local OpenAI-compatible
  endpoint or an offline response JSONL, then applies deterministic grounding
  validation.
* ``compile`` joins externally produced teacher/reranker scores, applies a
  positive-aware false-negative filter, and writes strict ms-swift JSONL plus a
  provenance sidecar and manifest.
* ``verify`` revalidates the three compiled artifacts and their hashes.

Only Python's standard library is required.  No hosted service, secret, model,
or benchmark data is implicitly accessed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
WHITESPACE_RE = re.compile(r"\s+")
CODE_FENCE_RE = re.compile(r"\A\s*```(?:json)?\s*(.*?)\s*```\s*\Z", re.DOTALL | re.IGNORECASE)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_source_text(value: str) -> str:
    return unicodedata.normalize(
        "NFC", value.replace("\r\n", "\n").replace("\r", "\n")
    ).strip()


def normalize_match_text(value: str) -> str:
    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{sha256_text(chr(0).join(parts))[:24]}"


def require_string(value: Any, field: str, *, minimum: int = 1) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    result = normalize_source_text(value)
    if len(result) < minimum:
        raise ValueError(f"{field} must contain at least {minimum} characters")
    return result


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError(f"unsupported config schema: {config.get('schema_version')!r}")
    required = {
        "factory_id",
        "seed",
        "input_contract",
        "generation",
        "validation",
        "teacher_scoring",
        "hard_negative_selection",
        "output_contract",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config is missing fields: {missing}")
    styles = config["generation"].get("styles")
    if not isinstance(styles, list) or not styles:
        raise ValueError("generation.styles must be a non-empty list")
    style_ids = [style.get("id") for style in styles]
    if any(not isinstance(item, str) or not item for item in style_ids):
        raise ValueError("every generation style needs a non-empty id")
    if len(style_ids) != len(set(style_ids)):
        raise ValueError("generation style ids must be unique")
    return config


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


class AtomicJSONL:
    def __init__(self, path: Path):
        self.path = path
        self.temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        self.handle = None
        self.digest = hashlib.sha256()
        self.rows = 0
        self.bytes = 0

    def __enter__(self) -> "AtomicJSONL":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.temp.open("wb")
        return self

    def write(self, value: Any) -> str:
        if self.handle is None:
            raise RuntimeError("writer is not open")
        encoded = canonical_bytes(value)
        self.handle.write(encoded)
        self.digest.update(encoded)
        self.rows += 1
        self.bytes += len(encoded)
        return sha256_bytes(encoded)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            if exc_type is None:
                self.handle.flush()
                os.fsync(self.handle.fileno())
            self.handle.close()
        if exc_type is None:
            self.temp.replace(self.path)
        else:
            self.temp.unlink(missing_ok=True)

    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "rows": self.rows,
            "bytes": self.bytes,
            "sha256": self.digest.hexdigest(),
        }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        with temp.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def validate_candidate(
    raw: dict[str, Any], config: dict[str, Any], path: Path, line_number: int
) -> dict[str, Any]:
    contract = config["input_contract"]
    missing = [field for field in contract["required_fields"] if field not in raw]
    if missing:
        raise ValueError(f"{path}:{line_number}: candidate fields missing: {missing}")
    candidate_id = require_string(raw["id"], "id")
    source_query = require_string(
        raw["query"],
        "query",
        minimum=int(contract["minimum_source_query_characters"]),
    )
    positive = require_string(
        raw["positive"],
        "positive",
        minimum=int(contract["minimum_positive_characters"]),
    )
    pair_type = require_string(raw["pair_type"], "pair_type")
    label_origin = require_string(raw["label_origin"], "label_origin")
    provenance = raw["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError(f"{path}:{line_number}: provenance must be an object")
    return {
        "id": candidate_id,
        "query": source_query,
        "positive": positive,
        "pair_type": pair_type,
        "label_origin": label_origin,
        "provenance": provenance,
    }


def allowed_locators(candidate: dict[str, Any]) -> list[str]:
    provenance = candidate["provenance"]
    section = provenance.get("section_heading") or provenance.get(
        "positive_section_heading"
    )
    path = provenance.get("path")
    candidates: list[Any] = [
        f"{path}#{section}" if path and section else None,
        path,
        provenance.get("source_url"),
        section,
    ]
    result: list[str] = []
    for value in candidates:
        if isinstance(value, str) and value.strip():
            normalized = normalize_source_text(value)
            if normalized not in result:
                result.append(normalized)
    if not result:
        result.append(candidate["id"])
    return result


def prompt_for_candidate(candidate: dict[str, Any], style: dict[str, Any]) -> str:
    locators = allowed_locators(candidate)
    contract = {
        "query": "새 한국어 검색 질의",
        "answer": "evidence_quote 안에서 그대로 복사한 짧은 답",
        "evidence_quote": "아래 positive에서 그대로 복사한 연속 구절",
        "citation": {
            "source_candidate_id": candidate["id"],
            "locator": f"다음 중 정확히 하나: {locators}",
        },
    }
    return (
        "다음 source-native 검색 pair를 한 개의 새 검색 학습 예제로 바꿔라.\n"
        f"질의 스타일: {style['id']}\n"
        f"스타일 지침: {style['instruction']}\n\n"
        f"source_candidate_id: {candidate['id']}\n"
        f"허용 citation locator: {json.dumps(locators, ensure_ascii=False)}\n"
        f"기존 source query:\n{candidate['query']}\n\n"
        f"positive 원문:\n{candidate['positive']}\n\n"
        "규칙:\n"
        "1. positive 원문만으로 답할 수 있는 질의를 쓴다.\n"
        "2. answer는 evidence_quote 내부의 연속 문자열을 글자 그대로 복사한다.\n"
        "3. evidence_quote도 positive 원문의 연속 문자열을 글자 그대로 복사한다.\n"
        "4. citation.source_candidate_id는 위 id와 정확히 같아야 한다.\n"
        "5. citation.locator는 허용 목록의 문자열 하나와 정확히 같아야 한다.\n"
        "6. Markdown code fence나 설명 없이 JSON 객체 하나만 출력한다.\n"
        f"출력 구조: {json.dumps(contract, ensure_ascii=False)}"
    )


def request_record(
    candidate: dict[str, Any], style: dict[str, Any], config: dict[str, Any], model: str
) -> dict[str, Any]:
    generation = config["generation"]
    positive_sha = sha256_text(candidate["positive"])
    request_id = stable_id(
        "gsq-request",
        config["factory_id"],
        candidate["id"],
        style["id"],
        positive_sha,
    )
    seed = int.from_bytes(hashlib.sha256(request_id.encode("utf-8")).digest()[:4], "big")
    api_request = {
        "model": model,
        "messages": [
            {"role": "system", "content": generation["system_prompt"]},
            {"role": "user", "content": prompt_for_candidate(candidate, style)},
        ],
        "temperature": float(generation["temperature"]),
        "top_p": float(generation["top_p"]),
        "max_tokens": int(generation["max_tokens"]),
        "seed": seed,
        "response_format": {"type": "json_object"},
    }
    return {
        "schema_version": 1,
        "request_id": request_id,
        "source_candidate_id": candidate["id"],
        "style": style["id"],
        "source_native_query": candidate["query"],
        "positive": candidate["positive"],
        "positive_sha256": positive_sha,
        "pair_type": candidate["pair_type"],
        "label_origin": candidate["label_origin"],
        "source_provenance": candidate["provenance"],
        "allowed_citation_locators": allowed_locators(candidate),
        "api_request": api_request,
    }


def command_prepare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model = args.model or config["generation"]["model"]
    selected_styles = set(args.style or [item["id"] for item in config["generation"]["styles"]])
    styles = [item for item in config["generation"]["styles"] if item["id"] in selected_styles]
    unknown = selected_styles - {item["id"] for item in styles}
    if unknown:
        raise ValueError(f"unknown styles: {sorted(unknown)}")
    seen_ids: set[str] = set()
    counters: Counter[str] = Counter()
    with AtomicJSONL(args.requests) as writer:
        for line_number, raw in read_jsonl(args.input):
            candidate = validate_candidate(raw, config, args.input, line_number)
            if candidate["id"] in seen_ids:
                raise ValueError(f"{args.input}:{line_number}: duplicate id {candidate['id']}")
            seen_ids.add(candidate["id"])
            if args.shard_count > 1:
                shard = int.from_bytes(
                    hashlib.sha256(candidate["id"].encode("utf-8")).digest()[:8], "big"
                ) % args.shard_count
                if shard != args.shard_index:
                    counters["candidates_outside_shard"] += 1
                    continue
            if args.max_candidates and counters["candidates_selected"] >= args.max_candidates:
                break
            counters["candidates_selected"] += 1
            for style in styles:
                writer.write(request_record(candidate, style, config, model))
                counters[f"style:{style['id']}"] += 1
                counters["requests_emitted"] += 1
        metadata = writer.metadata()
    summary = {
        "schema_version": 1,
        "stage": "prepare",
        "factory_id": config["factory_id"],
        "config": {"path": str(args.config), "sha256": file_sha256(args.config)},
        "input": {"path": str(args.input), "sha256": file_sha256(args.input)},
        "model": model,
        "styles": [item["id"] for item in styles],
        "shard": {"count": args.shard_count, "index": args.shard_index},
        "counters": dict(sorted(counters.items())),
        "output": metadata,
    }
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return 0


def extract_response_content(raw: dict[str, Any]) -> Any:
    value: Any = raw.get("response", raw)
    if isinstance(value, dict) and all(
        field in value for field in ("query", "answer", "evidence_quote", "citation")
    ):
        return value
    if isinstance(value, dict) and isinstance(value.get("choices"), list) and value["choices"]:
        choice = value["choices"][0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict) and "content" in message:
                return message["content"]
            if "text" in choice:
                return choice["text"]
    if isinstance(value, dict):
        for key in ("content", "output", "text"):
            if key in value:
                return value[key]
    return value


def parse_response_object(raw: dict[str, Any]) -> tuple[dict[str, Any], str]:
    content = extract_response_content(raw)
    raw_bytes = json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8")
    raw_hash = sha256_bytes(raw_bytes)
    if isinstance(content, dict):
        result = content
    elif isinstance(content, str):
        match = CODE_FENCE_RE.match(content)
        if match:
            content = match.group(1)
        decoder = json.JSONDecoder()
        start = content.find("{")
        if start < 0:
            raise ValueError("response content has no JSON object")
        result, end = decoder.raw_decode(content[start:])
        if content[start + end :].strip():
            raise ValueError("response contains text after the JSON object")
    else:
        raise ValueError("response content must be a JSON object or a JSON string")
    if not isinstance(result, dict):
        raise ValueError("parsed response is not an object")
    return result, raw_hash


def character_coverage(query: str, positive: str) -> float:
    query_chars = "".join(normalize_match_text(query).split())
    positive_chars = "".join(normalize_match_text(positive).split())
    if not query_chars:
        return 1.0
    longest_copy = difflib.SequenceMatcher(
        None, query_chars, positive_chars, autojunk=False
    ).find_longest_match().size
    return longest_copy / len(query_chars)


def validate_generation(
    request: dict[str, Any], raw_response: dict[str, Any], config: dict[str, Any], mode: str
) -> dict[str, Any]:
    response, raw_hash = parse_response_object(raw_response)
    required = set(config["generation"]["response_contract"]["required_fields"])
    if set(response) != required:
        raise ValueError(
            f"response fields must be exactly {sorted(required)}, got {sorted(response)}"
        )
    validation = config["validation"]
    query = require_string(
        response["query"], "query", minimum=int(validation["minimum_query_characters"])
    )
    answer = require_string(
        response["answer"], "answer", minimum=int(validation["minimum_answer_characters"])
    )
    evidence = require_string(
        response["evidence_quote"],
        "evidence_quote",
        minimum=int(validation["minimum_evidence_characters"]),
    )
    if len(query) > int(validation["maximum_query_characters"]):
        raise ValueError("query exceeds maximum_query_characters")
    if len(answer) > int(validation["maximum_answer_characters"]):
        raise ValueError("answer exceeds maximum_answer_characters")
    if len(evidence) > int(validation["maximum_evidence_characters"]):
        raise ValueError("evidence_quote exceeds maximum_evidence_characters")
    if validation["require_hangul_in_query"] and not HANGUL_RE.search(query):
        raise ValueError("query contains no Hangul syllable")
    answer_match = normalize_match_text(answer)
    evidence_match = normalize_match_text(evidence)
    positive_match = normalize_match_text(request["positive"])
    if validation["require_exact_answer_in_evidence"] and answer_match not in evidence_match:
        raise ValueError("answer is not an exact normalized span of evidence_quote")
    if validation["require_exact_evidence_in_positive"] and evidence_match not in positive_match:
        raise ValueError("evidence_quote is not an exact normalized span of positive")
    if validation["reject_query_equal_to_source_query"] and normalize_match_text(
        query
    ) == normalize_match_text(request["source_native_query"]):
        raise ValueError("generated query exactly equals source-native query")
    coverage = character_coverage(query, request["positive"])
    if coverage > float(validation["maximum_query_positive_character_coverage"]):
        raise ValueError(
            f"query-positive character coverage {coverage:.6f} exceeds configured maximum"
        )
    citation = response["citation"]
    citation_fields = set(config["generation"]["response_contract"]["citation_fields"])
    if not isinstance(citation, dict) or set(citation) != citation_fields:
        raise ValueError(f"citation fields must be exactly {sorted(citation_fields)}")
    citation_candidate = require_string(citation["source_candidate_id"], "citation.source_candidate_id")
    locator = require_string(citation["locator"], "citation.locator")
    if citation_candidate != request["source_candidate_id"]:
        raise ValueError("citation.source_candidate_id does not match request")
    if locator not in request["allowed_citation_locators"]:
        raise ValueError("citation.locator is not in the request allowlist")
    generated_id = stable_id(
        "gsq",
        request["request_id"],
        query,
        request["positive_sha256"],
    )
    return {
        "schema_version": 1,
        "generated_id": generated_id,
        "request_id": request["request_id"],
        "source_candidate_id": request["source_candidate_id"],
        "style": request["style"],
        "query": query,
        "positive": request["positive"],
        "positive_sha256": request["positive_sha256"],
        "answer": answer,
        "evidence_quote": evidence,
        "citation": {
            "source_candidate_id": citation_candidate,
            "locator": locator,
        },
        "source_native_query": request["source_native_query"],
        "pair_type": request["pair_type"],
        "label_origin": request["label_origin"],
        "source_provenance": request["source_provenance"],
        "generation": {
            "mode": mode,
            "model": request["api_request"]["model"],
            "temperature": request["api_request"].get("temperature"),
            "seed": request["api_request"].get("seed"),
            "raw_response_sha256": raw_hash,
        },
        "validation": {
            "exact_answer_in_evidence": True,
            "exact_evidence_in_positive": True,
            "citation_allowlist_match": True,
            "query_positive_character_coverage": round(coverage, 8),
        },
    }


def load_offline_responses(path: Path) -> dict[str, dict[str, Any]]:
    responses: dict[str, dict[str, Any]] = {}
    for line_number, raw in read_jsonl(path):
        request_id = require_string(raw.get("request_id"), "request_id")
        if request_id in responses:
            raise ValueError(f"{path}:{line_number}: duplicate request_id {request_id}")
        responses[request_id] = raw
    return responses


def endpoint_url(endpoint: dict[str, Any]) -> str:
    return endpoint["base_url"].rstrip("/") + "/" + endpoint["chat_completions_path"].lstrip("/")


def call_endpoint(
    request_record_value: dict[str, Any], endpoint: dict[str, Any]
) -> dict[str, Any]:
    body = json.dumps(request_record_value["api_request"], ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    key_name = endpoint.get("api_key_env")
    if key_name and os.environ.get(key_name):
        headers["Authorization"] = f"Bearer {os.environ[key_name]}"
    request = urllib.request.Request(endpoint_url(endpoint), data=body, headers=headers, method="POST")
    attempts = int(endpoint["max_attempts"])
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=float(endpoint["timeout_seconds"])) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("endpoint returned a non-object JSON response")
                return {"request_id": request_record_value["request_id"], "response": payload}
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 4))
    assert error is not None
    raise RuntimeError(f"endpoint failed after {attempts} attempts: {error}") from error


def command_generate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    requests = [raw for _, raw in read_jsonl(args.requests)]
    request_ids = [raw.get("request_id") for raw in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request JSONL contains duplicate request_id values")
    if args.mode == "offline":
        if args.responses is None:
            raise ValueError("--responses is required in offline mode")
        offline = load_offline_responses(args.responses)
        unexpected = set(offline) - set(request_ids)
        if unexpected and not args.allow_extra_responses:
            raise ValueError(f"offline file contains {len(unexpected)} unexpected request ids")
        resolved: dict[str, dict[str, Any]] = offline
    else:
        endpoint = dict(config["generation"]["endpoint"])
        if args.endpoint_base_url:
            endpoint["base_url"] = args.endpoint_base_url
        concurrency = args.concurrency or int(endpoint["concurrency"])
        resolved = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_id = {
                executor.submit(call_endpoint, request, endpoint): request["request_id"]
                for request in requests
            }
            for future in concurrent.futures.as_completed(future_to_id):
                request_id = future_to_id[future]
                try:
                    resolved[request_id] = future.result()
                except Exception as exc:
                    resolved[request_id] = {"request_id": request_id, "endpoint_error": str(exc)}
    counters: Counter[str] = Counter()
    duplicate_keys: set[tuple[str, str]] = set()
    with ExitStack() as stack:
        accepted_writer = stack.enter_context(AtomicJSONL(args.validated))
        rejected_writer = stack.enter_context(AtomicJSONL(args.rejected)) if args.rejected else None
        for request in requests:
            request_id = request["request_id"]
            raw_response = resolved.get(request_id)
            if raw_response is None:
                reason = "missing_response"
                error = "no response found for request_id"
            elif "endpoint_error" in raw_response:
                reason = "endpoint_error"
                error = str(raw_response["endpoint_error"])
            else:
                try:
                    generated = validate_generation(request, raw_response, config, args.mode)
                    duplicate_key = (
                        generated["source_candidate_id"],
                        normalize_match_text(generated["query"]),
                    )
                    if (
                        config["validation"]["reject_duplicate_generated_queries_per_source"]
                        and duplicate_key in duplicate_keys
                    ):
                        raise ValueError("duplicate generated query for the same source candidate")
                    duplicate_keys.add(duplicate_key)
                    accepted_writer.write(generated)
                    counters["accepted"] += 1
                    counters[f"accepted_style:{generated['style']}"] += 1
                    continue
                except Exception as exc:
                    reason = "validation_error"
                    error = str(exc)
            counters["rejected"] += 1
            counters[f"rejected_reason:{reason}"] += 1
            if rejected_writer is not None:
                rejected_writer.write(
                    {
                        "request_id": request_id,
                        "source_candidate_id": request.get("source_candidate_id"),
                        "style": request.get("style"),
                        "reason": reason,
                        "error": error,
                    }
                )
        accepted_meta = accepted_writer.metadata()
        rejected_meta = rejected_writer.metadata() if rejected_writer is not None else None
    summary = {
        "schema_version": 1,
        "stage": "generate",
        "mode": args.mode,
        "factory_id": config["factory_id"],
        "config": {"path": str(args.config), "sha256": file_sha256(args.config)},
        "requests": {"path": str(args.requests), "sha256": file_sha256(args.requests)},
        "responses": (
            {"path": str(args.responses), "sha256": file_sha256(args.responses)}
            if args.responses
            else {"source": "local_openai_compatible_endpoint"}
        ),
        "counters": dict(sorted(counters.items())),
        "validated": accepted_meta,
        "rejected": rejected_meta,
    }
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return 0


def initialize_candidate_index(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            id TEXT PRIMARY KEY,
            positive TEXT NOT NULL,
            positive_normalized TEXT NOT NULL,
            positive_sha256 TEXT NOT NULL,
            provenance_json TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )


def build_candidate_index(
    path: Path, config: dict[str, Any], database: Path
) -> tuple[sqlite3.Connection, int]:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(database))
    initialize_candidate_index(connection)
    existing = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    metadata = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='index_metadata'"
    ).fetchone()
    input_hash = file_sha256(path)
    if metadata:
        stored = dict(connection.execute("SELECT key, value FROM index_metadata"))
        if stored.get("input_sha256") == input_hash and int(stored.get("rows", "0")) == existing:
            return connection, int(existing)
        connection.execute("DELETE FROM candidates")
        connection.execute("DELETE FROM index_metadata")
    else:
        connection.execute("CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        if existing:
            connection.execute("DELETE FROM candidates")
    rows = 0
    batch: list[tuple[str, str, str, str, str]] = []
    for line_number, raw in read_jsonl(path):
        candidate = validate_candidate(raw, config, path, line_number)
        positive = candidate["positive"]
        batch.append(
            (
                candidate["id"],
                positive,
                normalize_match_text(positive),
                sha256_text(positive),
                json.dumps(candidate["provenance"], ensure_ascii=False, sort_keys=True),
            )
        )
        if len(batch) >= 5000:
            try:
                connection.executemany("INSERT INTO candidates VALUES (?, ?, ?, ?, ?)", batch)
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"candidate source contains duplicate ids near line {line_number}") from exc
            rows += len(batch)
            batch.clear()
    if batch:
        try:
            connection.executemany("INSERT INTO candidates VALUES (?, ?, ?, ?, ?)", batch)
        except sqlite3.IntegrityError as exc:
            raise ValueError("candidate source contains duplicate ids") from exc
        rows += len(batch)
    connection.executemany(
        "INSERT INTO index_metadata VALUES (?, ?)",
        [("input_sha256", input_hash), ("rows", str(rows))],
    )
    connection.commit()
    return connection, rows


def initialize_score_index(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE scores (generated_id TEXT PRIMARY KEY, payload TEXT NOT NULL) WITHOUT ROWID")


def build_score_index(path: Path, database: Path) -> tuple[sqlite3.Connection, int]:
    connection = sqlite3.connect(str(database))
    initialize_score_index(connection)
    batch: list[tuple[str, str]] = []
    rows = 0
    for line_number, raw in read_jsonl(path):
        generated_id = require_string(raw.get("generated_id"), "generated_id")
        documents = raw.get("documents")
        if not isinstance(documents, list) or not documents:
            raise ValueError(f"{path}:{line_number}: documents must be a non-empty list")
        batch.append((generated_id, json.dumps(raw, ensure_ascii=False, sort_keys=True)))
        if len(batch) >= 5000:
            try:
                connection.executemany("INSERT INTO scores VALUES (?, ?)", batch)
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"{path}:{line_number}: duplicate generated_id") from exc
            rows += len(batch)
            batch.clear()
    if batch:
        try:
            connection.executemany("INSERT INTO scores VALUES (?, ?)", batch)
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"{path}: duplicate generated_id") from exc
        rows += len(batch)
    connection.commit()
    return connection, rows


@dataclass(frozen=True)
class ScoredDocument:
    candidate_id: str
    score: float
    text: str
    normalized: str
    text_sha256: str
    provenance: dict[str, Any]
    original_rank: int


class PolicyDrop(ValueError):
    """A valid row that the configured quality/quantity policy rejects."""


def choose_score_field(score_row: dict[str, Any], config: dict[str, Any]) -> str:
    documents = score_row["documents"]
    explicit = score_row.get("score_field")
    priorities = config["teacher_scoring"]["score_field_priority"]
    if explicit is not None:
        if explicit not in priorities:
            raise ValueError(f"score_field {explicit!r} is not in configured priority list")
        return explicit
    for field in priorities:
        if all(isinstance(item, dict) and field in item for item in documents):
            return field
    raise ValueError("no configured score field is present on every scored document")


def parse_score(value: Any, field: str, config: dict[str, Any]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    minimum, maximum = config["teacher_scoring"]["score_range"]
    if result < float(minimum) or result > float(maximum):
        raise ValueError(f"{field}={result} lies outside configured score range")
    return result


def candidate_from_index(
    connection: sqlite3.Connection, candidate_id: str
) -> tuple[str, str, str, dict[str, Any]]:
    row = connection.execute(
        "SELECT positive, positive_normalized, positive_sha256, provenance_json FROM candidates WHERE id=?",
        (candidate_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"teacher score references missing candidate id {candidate_id}")
    return row[0], row[1], row[2], json.loads(row[3])


def select_hard_negatives(
    generated: dict[str, Any],
    score_row: dict[str, Any],
    candidate_db: sqlite3.Connection,
    config: dict[str, Any],
) -> tuple[list[ScoredDocument], dict[str, Any]]:
    score_field = choose_score_field(score_row, config)
    documents = score_row["documents"]
    seen_ids: set[str] = set()
    scored: list[ScoredDocument] = []
    for rank, raw in enumerate(documents, 1):
        if not isinstance(raw, dict):
            raise ValueError("every scored document must be an object")
        candidate_id = require_string(raw.get("candidate_id"), "documents[].candidate_id")
        if candidate_id in seen_ids:
            raise ValueError(f"score row repeats candidate id {candidate_id}")
        seen_ids.add(candidate_id)
        score = parse_score(raw.get(score_field), score_field, config)
        text, normalized, text_hash, provenance = candidate_from_index(candidate_db, candidate_id)
        scored.append(
            ScoredDocument(
                candidate_id=candidate_id,
                score=score,
                text=text,
                normalized=normalized,
                text_sha256=text_hash,
                provenance=provenance,
                original_rank=rank,
            )
        )
    positive_id = generated["source_candidate_id"]
    positive_rows = [item for item in scored if item.candidate_id == positive_id]
    if len(positive_rows) != 1:
        raise ValueError("score row must include the source positive exactly once")
    positive = positive_rows[0]
    teacher = config["teacher_scoring"]
    if positive.score < float(teacher["minimum_positive_score"]):
        raise PolicyDrop(
            f"positive score {positive.score} is below minimum {teacher['minimum_positive_score']}"
        )
    positive_hash = generated["positive_sha256"]
    if positive.text_sha256 != positive_hash or positive.text != generated["positive"]:
        raise ValueError("validated positive no longer matches the pinned source candidate")
    policy = config["hard_negative_selection"]
    relative_limit = positive.score * float(policy["positive_relative_ratio"])
    absolute_limit = positive.score - float(policy["absolute_positive_margin"])
    upper_limit = min(relative_limit, absolute_limit)
    query_normalized = normalize_match_text(generated["query"])
    eligible: list[ScoredDocument] = []
    exclusion_counts: Counter[str] = Counter()
    for item in scored:
        if policy["exclude_same_candidate"] and item.candidate_id == positive_id:
            exclusion_counts["same_candidate"] += 1
            continue
        if policy["exclude_duplicate_positive_text"] and item.normalized == positive.normalized:
            exclusion_counts["duplicate_positive_text"] += 1
            continue
        if policy["exclude_query_text_match"] and item.normalized == query_normalized:
            exclusion_counts["query_text_match"] += 1
            continue
        if item.score > upper_limit:
            exclusion_counts["positive_relative_filter"] += 1
            continue
        if item.score < float(policy["minimum_negative_score"]):
            exclusion_counts["below_minimum_negative_score"] += 1
            continue
        eligible.append(item)
    eligible.sort(key=lambda item: (-item.score, item.candidate_id))
    unique_eligible: list[ScoredDocument] = []
    seen_negative_texts: set[str] = set()
    for item in eligible:
        if item.normalized in seen_negative_texts:
            exclusion_counts["duplicate_negative_text"] += 1
            continue
        seen_negative_texts.add(item.normalized)
        unique_eligible.append(item)
    eligible = unique_eligible
    pool_size = min(int(policy["candidate_pool_size"]), len(eligible))
    pool = eligible[:pool_size]
    needed = int(policy["negatives_per_query"])
    if len(pool) < needed:
        raise PolicyDrop(f"insufficient eligible negatives: need {needed}, found {len(pool)}")
    strategy = policy["selection_strategy"]
    if strategy == "top_k":
        selected = pool[:needed]
    elif strategy == "hash_sample_from_top_pool":
        seed = str(config["seed"])
        selected = sorted(
            pool,
            key=lambda item: sha256_text(
                "\0".join((seed, generated["generated_id"], item.candidate_id))
            ),
        )[:needed]
        selected.sort(key=lambda item: (-item.score, item.candidate_id))
    else:
        raise ValueError(f"unsupported selection strategy: {strategy}")
    audit = {
        "score_field": score_field,
        "scorer": score_row.get("scorer"),
        "positive_score": positive.score,
        "relative_score_limit": relative_limit,
        "absolute_score_limit": absolute_limit,
        "effective_score_limit": upper_limit,
        "scored_document_count": len(scored),
        "eligible_count": len(eligible),
        "top_pool_count": len(pool),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
    }
    return selected, audit


def ms_swift_row(query: str, positive: str, negatives: Sequence[str]) -> dict[str, Any]:
    message = lambda text: [{"role": "user", "content": text}]
    return {
        "messages": message(query),
        "positive_messages": [message(positive)],
        "negative_messages": [message(negative) for negative in negatives],
    }


def command_compile(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.output.resolve() in {args.audit.resolve(), args.manifest.resolve()}:
        raise ValueError("--output, --audit, and --manifest must be distinct")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    candidate_db_path = args.candidate_index or args.work_dir / "candidate-index.sqlite3"
    score_db_path = args.work_dir / f"scores-{os.getpid()}.sqlite3"
    if score_db_path.exists():
        score_db_path.unlink()
    candidate_db, candidate_count = build_candidate_index(args.candidates, config, candidate_db_path)
    score_db, score_count = build_score_index(args.scores, score_db_path)
    counters: Counter[str] = Counter()
    seen_generated: set[str] = set()
    seen_train_rows: set[str] = set()
    scorer_identities: set[str] = set()
    generator_identities: set[str] = set()
    try:
        with ExitStack() as stack:
            train_writer = stack.enter_context(AtomicJSONL(args.output))
            audit_writer = stack.enter_context(AtomicJSONL(args.audit))
            for line_number, generated in read_jsonl(args.validated):
                generated_id = require_string(generated.get("generated_id"), "generated_id")
                if generated_id in seen_generated:
                    raise ValueError(f"{args.validated}:{line_number}: duplicate generated_id")
                seen_generated.add(generated_id)
                counters["validated_seen"] += 1
                generation_identity = generated.get("generation")
                if not isinstance(generation_identity, dict):
                    raise ValueError(f"{generated_id}: generation must be an object")
                generator_identities.add(
                    json.dumps(
                        {
                            key: generation_identity.get(key)
                            for key in ("mode", "model", "temperature")
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                payload = score_db.execute(
                    "SELECT payload FROM scores WHERE generated_id=?", (generated_id,)
                ).fetchone()
                if payload is None:
                    counters["dropped_missing_score_row"] += 1
                    if args.require_all_scores:
                        raise ValueError(f"no teacher score row for {generated_id}")
                    continue
                counters["score_rows_matched"] += 1
                score_row = json.loads(payload[0])
                try:
                    negatives, selection_audit = select_hard_negatives(
                        generated, score_row, candidate_db, config
                    )
                except PolicyDrop as exc:
                    counters["dropped_score_or_negative_policy"] += 1
                    if config["hard_negative_selection"]["insufficient_policy"] == "error":
                        raise ValueError(f"{generated_id}: {exc}") from exc
                    continue
                scorer = selection_audit.get("scorer")
                if scorer is not None:
                    scorer_identities.add(
                        json.dumps(scorer, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    )
                train_row = ms_swift_row(
                    generated["query"], generated["positive"], [item.text for item in negatives]
                )
                encoded = canonical_bytes(train_row)
                identity = sha256_bytes(encoded)
                if identity in seen_train_rows:
                    counters["dropped_duplicate_train_row"] += 1
                    if config["hard_negative_selection"]["duplicate_row_policy"] == "error":
                        raise ValueError(f"duplicate strict training row at {generated_id}")
                    continue
                seen_train_rows.add(identity)
                train_line = train_writer.rows + 1
                train_writer.write(train_row)
                row_id = stable_id("gsq-row", generated_id, *(item.candidate_id for item in negatives))
                audit_writer.write(
                    {
                        "schema_version": 1,
                        "row_id": row_id,
                        "train_line_number": train_line,
                        "train_row_sha256": identity,
                        "generated_id": generated_id,
                        "request_id": generated["request_id"],
                        "source_candidate_id": generated["source_candidate_id"],
                        "style": generated["style"],
                        "query_sha256": sha256_text(generated["query"]),
                        "positive_sha256": generated["positive_sha256"],
                        "grounding": {
                            "answer": generated["answer"],
                            "evidence_quote": generated["evidence_quote"],
                            "citation": generated["citation"],
                            "validation": generated["validation"],
                        },
                        "generation": generated["generation"],
                        "source": {
                            "source_native_query": generated["source_native_query"],
                            "pair_type": generated["pair_type"],
                            "label_origin": generated["label_origin"],
                            "provenance": generated["source_provenance"],
                        },
                        "teacher_selection": selection_audit,
                        "negatives": [
                            {
                                "candidate_id": item.candidate_id,
                                "score": item.score,
                                "original_rank": item.original_rank,
                                "text_sha256": item.text_sha256,
                                "provenance": item.provenance,
                            }
                            for item in negatives
                        ],
                    }
                )
                counters["rows_emitted"] += 1
                counters[f"style:{generated['style']}"] += 1
            if train_writer.rows < 2:
                raise ValueError(
                    f"compile emitted {train_writer.rows} rows; at least two are required"
                )
            train_meta = train_writer.metadata()
            audit_meta = audit_writer.metadata()
        unused_scores = score_count - counters["score_rows_matched"]
        counters["unused_score_rows"] = unused_scores
        manifest = {
            "schema_version": 1,
            "stage": "compiled_grounded_synthetic_query_data",
            "factory_id": config["factory_id"],
            "seed": config["seed"],
            "config": {"path": str(args.config), "sha256": file_sha256(args.config)},
            "inputs": {
                "candidates": {
                    "path": str(args.candidates),
                    "sha256": file_sha256(args.candidates),
                    "indexed_rows": candidate_count,
                },
                "validated_generations": {
                    "path": str(args.validated),
                    "sha256": file_sha256(args.validated),
                },
                "teacher_scores": {
                    "path": str(args.scores),
                    "sha256": file_sha256(args.scores),
                    "indexed_rows": score_count,
                },
            },
            "teacher_scorers": [json.loads(item) for item in sorted(scorer_identities)],
            "generators": [json.loads(item) for item in sorted(generator_identities)],
            "selection_policy": config["hard_negative_selection"],
            "teacher_policy": config["teacher_scoring"],
            "counters": dict(sorted(counters.items())),
            "files": {
                args.output.name: train_meta,
                args.audit.name: audit_meta,
            },
            "claims": {
                "grounding": "Every emitted answer is an exact normalized span of an exact normalized source-positive evidence span.",
                "relevance": "Teacher scores and deterministic filters are training signals, not human relevance judgments.",
                "benchmark_cleanliness": "Not implied; inspect source exposure and the separate decontamination manifest.",
            },
        }
        atomic_write_json(args.manifest, manifest)
        sys.stdout.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    finally:
        candidate_db.close()
        score_db.close()
        score_db_path.unlink(missing_ok=True)
    return 0


def one_message(value: Any, field: str) -> str:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{field} must contain exactly one message")
    message = value[0]
    if not isinstance(message, dict) or set(message) != {"role", "content"}:
        raise ValueError(f"{field} contains an invalid message")
    if message["role"] != "user":
        raise ValueError(f"{field} role must be user")
    return require_string(message["content"], f"{field}.content")


def command_verify(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    for path in (args.output, args.audit):
        declared = files.get(path.name)
        if not declared:
            raise ValueError(f"manifest has no entry for {path.name}")
        actual = file_sha256(path)
        if actual != declared.get("sha256"):
            raise ValueError(f"hash mismatch for {path}: {actual} != {declared.get('sha256')}")
    train_iterator = read_jsonl(args.output)
    audit_iterator = read_jsonl(args.audit)
    rows = 0
    identities: set[str] = set()
    while True:
        try:
            train_line, train = next(train_iterator)
        except StopIteration:
            train_item = None
        else:
            train_item = (train_line, train)
        try:
            audit_line, audit = next(audit_iterator)
        except StopIteration:
            audit_item = None
        else:
            audit_item = (audit_line, audit)
        if train_item is None or audit_item is None:
            if train_item is not None or audit_item is not None:
                raise ValueError("training and audit JSONL have different row counts")
            break
        train_line, train = train_item
        audit_line, audit = audit_item
        expected_fields = {"messages", "positive_messages", "negative_messages"}
        if set(train) != expected_fields:
            raise ValueError(f"{args.output}:{train_line}: unexpected fields {sorted(train)}")
        query = one_message(train["messages"], "messages")
        positives_raw = train["positive_messages"]
        negatives_raw = train["negative_messages"]
        if not isinstance(positives_raw, list) or len(positives_raw) != 1:
            raise ValueError("exactly one positive is required")
        if not isinstance(negatives_raw, list) or not negatives_raw:
            raise ValueError("at least one negative is required")
        positive = one_message(positives_raw[0], "positive_messages[0]")
        negatives = [one_message(item, f"negative_messages[{i}]") for i, item in enumerate(negatives_raw)]
        if positive in negatives:
            raise ValueError("positive is duplicated as a negative")
        identity = sha256_bytes(canonical_bytes(train))
        if identity in identities:
            raise ValueError("duplicate strict training row")
        identities.add(identity)
        if audit.get("train_line_number") != train_line or audit_line != train_line:
            raise ValueError("audit line alignment mismatch")
        if audit.get("train_row_sha256") != identity:
            raise ValueError("audit train_row_sha256 mismatch")
        if audit.get("query_sha256") != sha256_text(query):
            raise ValueError("audit query_sha256 mismatch")
        if audit.get("positive_sha256") != sha256_text(positive):
            raise ValueError("audit positive_sha256 mismatch")
        if len(audit.get("negatives", [])) != len(negatives):
            raise ValueError("audit negative count mismatch")
        for text, negative in zip(negatives, audit["negatives"]):
            if negative.get("text_sha256") != sha256_text(text):
                raise ValueError("audit negative text hash mismatch")
        rows += 1
    if rows < 2:
        raise ValueError("compiled dataset must contain at least two rows")
    expected_rows = manifest.get("counters", {}).get("rows_emitted")
    if expected_rows != rows:
        raise ValueError(f"manifest row count {expected_rows} != actual {rows}")
    result = {
        "verified": True,
        "rows": rows,
        "training_sha256": file_sha256(args.output),
        "audit_sha256": file_sha256(args.audit),
        "manifest_sha256": file_sha256(args.manifest),
    }
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic_query_factory_v1.json"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create deterministic chat requests")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--requests", type=Path, required=True)
    prepare.add_argument("--model")
    prepare.add_argument("--style", action="append", default=[])
    prepare.add_argument("--max-candidates", type=int, default=0)
    prepare.add_argument("--shard-count", type=int, default=1)
    prepare.add_argument("--shard-index", type=int, default=0)
    prepare.set_defaults(function=command_prepare)

    generate = subparsers.add_parser(
        "generate", help="Resolve request JSONL and validate grounded generations"
    )
    generate.add_argument("--requests", type=Path, required=True)
    generate.add_argument("--mode", choices=("offline", "endpoint"), required=True)
    generate.add_argument("--responses", type=Path)
    generate.add_argument("--validated", type=Path, required=True)
    generate.add_argument("--rejected", type=Path)
    generate.add_argument("--allow-extra-responses", action="store_true")
    generate.add_argument("--endpoint-base-url")
    generate.add_argument("--concurrency", type=int)
    generate.set_defaults(function=command_generate)

    compile_parser = subparsers.add_parser(
        "compile", help="Apply teacher scores and emit strict ms-swift JSONL"
    )
    compile_parser.add_argument("--candidates", type=Path, required=True)
    compile_parser.add_argument("--validated", type=Path, required=True)
    compile_parser.add_argument("--scores", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--audit", type=Path, required=True)
    compile_parser.add_argument("--manifest", type=Path, required=True)
    compile_parser.add_argument("--work-dir", type=Path, required=True)
    compile_parser.add_argument("--candidate-index", type=Path)
    compile_parser.add_argument("--require-all-scores", action=argparse.BooleanOptionalAction, default=True)
    compile_parser.set_defaults(function=command_compile)

    verify = subparsers.add_parser("verify", help="Verify compiled hashes, schema, and sidecar")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--audit", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.set_defaults(function=command_verify)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if args.command == "prepare":
        if args.max_candidates < 0:
            raise ValueError("--max-candidates must be >= 0")
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise ValueError("shard arguments must satisfy count >= 1 and 0 <= index < count")
    if args.command == "generate" and args.concurrency is not None and args.concurrency < 1:
        raise ValueError("--concurrency must be positive")


def main() -> int:
    args = build_parser().parse_args()
    validate_cli(args)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
