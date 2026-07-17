#!/usr/bin/env python3
"""Compile a verified Qwen3-reranker cache into strict listwise-KD JSONL."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import cache_qwen3_reranker_scores as scorer
from scripts.validate_embedding_jsonl import validate as validate_embedding_jsonl


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SelectionPolicy:
    candidate_pool_size: int = 200
    negatives_per_query: int = 15
    positive_relative_ratio: float = 0.95
    absolute_positive_margin: float = 0.02
    minimum_positive_score: float = 0.5
    minimum_negative_score: float = 0.0

    def __post_init__(self) -> None:
        if self.candidate_pool_size < self.negatives_per_query or self.negatives_per_query < 1:
            raise ValueError("candidate pool must cover at least one selected negative")
        for name in (
            "positive_relative_ratio",
            "absolute_positive_margin",
            "minimum_positive_score",
            "minimum_negative_score",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite in [0,1]")
        if self.minimum_negative_score >= self.minimum_positive_score:
            raise ValueError("minimum negative score must be below the positive gate")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evenly_spaced_rank_indices(pool_size: int, selected_size: int) -> list[int]:
    if pool_size < selected_size or selected_size < 1:
        raise ValueError("rank quantiles require pool >= selected >= 1")
    if selected_size == 1:
        return [0]
    indices = [
        (index * (pool_size - 1)) // (selected_size - 1)
        for index in range(selected_size)
    ]
    if len(set(indices)) != selected_size:
        raise RuntimeError("rank quantile indices collided")
    return indices


def strict_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at line {line_number}")
            value = scorer.strict_json_loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            yield value


def ms_swift_kd_row(
    request: scorer.InputRow,
    selected: Sequence[tuple[scorer.DocumentInput, float]],
    positive_score: float,
) -> dict[str, Any]:
    message = lambda text: [{"role": "user", "content": text}]
    return {
        "messages": message(request.query),
        "positive_messages": [message(request.positive.text)],
        "negative_messages": [message(document.text) for document, _ in selected],
        "teacher_scores": [positive_score, *(score for _, score in selected)],
    }


def compile_row(
    request: scorer.InputRow,
    score_row: dict[str, Any],
    policy: SelectionPolicy,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    score_field = score_row.get("score_field")
    row_scorer = score_row.get("scorer")
    if score_field != scorer.PRODUCTION_SCORE_FIELD or not isinstance(row_scorer, dict):
        raise ValueError("only production reranker score rows are admissible")
    if (
        row_scorer.get("model") != scorer.MODEL_ID
        or row_scorer.get("revision") != scorer.MODEL_REVISION
        or row_scorer.get("backend") != "pinned-local-qwen3-reranker"
    ):
        raise ValueError("score row does not use the pinned production teacher")
    scorer.validate_output_row(
        score_row,
        request,
        scorer=row_scorer,
        score_field=score_field,
    )
    documents = score_row["documents"]
    positive_score = float(documents[0][score_field])
    if positive_score < policy.minimum_positive_score:
        return None, {"drop_reason": "positive_below_gate", "positive_score": positive_score}
    relative_limit = positive_score * policy.positive_relative_ratio
    absolute_limit = positive_score - policy.absolute_positive_margin
    effective_limit = min(relative_limit, absolute_limit)
    eligible: list[tuple[scorer.DocumentInput, float, int]] = []
    exclusions: Counter[str] = Counter()
    seen_text_hashes: set[str] = set()
    for rank, (document, scored) in enumerate(
        zip(request.candidates, documents[1:], strict=True), 1
    ):
        score = float(scored[score_field])
        text_hash = scored["text_sha256"]
        if text_hash in seen_text_hashes:
            exclusions["duplicate_candidate_text"] += 1
            continue
        seen_text_hashes.add(text_hash)
        if score > effective_limit:
            exclusions["positive_relative_or_margin_filter"] += 1
            continue
        if score < policy.minimum_negative_score:
            exclusions["below_negative_gate"] += 1
            continue
        eligible.append((document, score, rank))
    eligible.sort(key=lambda item: (-item[1], item[0].candidate_id))
    pool = eligible[: policy.candidate_pool_size]
    if len(pool) < policy.negatives_per_query:
        return None, {
            "drop_reason": "insufficient_eligible_negatives",
            "positive_score": positive_score,
            "eligible_count": len(eligible),
            "exclusions": dict(sorted(exclusions.items())),
        }
    indices = evenly_spaced_rank_indices(len(pool), policy.negatives_per_query)
    selected_full = [pool[index] for index in indices]
    selected = [(document, score) for document, score, _ in selected_full]
    output = ms_swift_kd_row(request, selected, positive_score)
    audit = {
        "drop_reason": None,
        "generated_id": request.generated_id,
        "query_sha256": scorer.sha256_bytes(request.query.encode("utf-8")),
        "positive_candidate_id": request.positive.candidate_id,
        "positive_score": positive_score,
        "relative_score_limit": relative_limit,
        "absolute_score_limit": absolute_limit,
        "effective_score_limit": effective_limit,
        "eligible_count": len(eligible),
        "top_pool_count": len(pool),
        "selected_pool_indices_zero_based": indices,
        "selected": [
            {
                "candidate_id": document.candidate_id,
                "score": score,
                "source_rank": source_rank,
            }
            for document, score, source_rank in selected_full
        ],
        "exclusions": dict(sorted(exclusions.items())),
    }
    return output, audit


def atomic_jsonl_pair(
    requests: Path,
    scores: Path,
    output: Path,
    audit: Path,
    policy: SelectionPolicy,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    output_fd, output_tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    audit_fd, audit_tmp_name = tempfile.mkstemp(prefix=f".{audit.name}.", dir=audit.parent)
    output_tmp, audit_tmp = Path(output_tmp_name), Path(audit_tmp_name)
    counters: Counter[str] = Counter()
    seen_ids: set[str] = set()
    try:
        with os.fdopen(output_fd, "w", encoding="utf-8") as output_handle, os.fdopen(
            audit_fd, "w", encoding="utf-8"
        ) as audit_handle:
            request_rows = strict_rows(requests)
            score_rows = strict_rows(scores)
            for row_index, (request_raw, score_raw) in enumerate(
                itertools.zip_longest(request_rows, score_rows), 1
            ):
                if request_raw is None or score_raw is None:
                    raise ValueError("request and score row counts differ")
                request = scorer.parse_input_row(
                    request_raw,
                    max_documents_per_row=201,
                    max_text_characters=1_000_000,
                )
                if request.generated_id in seen_ids:
                    raise ValueError("requests repeat generated_id")
                seen_ids.add(request.generated_id)
                compiled, row_audit = compile_row(request, score_raw, policy)
                row_audit = {"input_row": row_index, **row_audit}
                audit_handle.write(
                    json.dumps(row_audit, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
                counters["input_rows"] += 1
                if compiled is None:
                    counters[f"dropped:{row_audit['drop_reason']}"] += 1
                    continue
                output_handle.write(
                    json.dumps(compiled, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
                counters["output_rows"] += 1
            output_handle.flush()
            audit_handle.flush()
            os.fsync(output_handle.fileno())
            os.fsync(audit_handle.fileno())
        if counters["output_rows"] < 2:
            raise ValueError("compiler emitted fewer than two KD rows")
        os.replace(output_tmp, output)
        os.replace(audit_tmp, audit)
    finally:
        output_tmp.unlink(missing_ok=True)
        audit_tmp.unlink(missing_ok=True)
    validate_embedding_jsonl(output, require_teacher_scores=True)
    return dict(sorted(counters.items()))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--score-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Training-source rights manifest inherited by the derived KD dataset.",
    )
    parser.add_argument("--candidate-pool-size", type=int, default=200)
    parser.add_argument("--negatives-per-query", type=int, default=15)
    parser.add_argument("--positive-relative-ratio", type=float, default=0.95)
    parser.add_argument("--absolute-positive-margin", type=float, default=0.02)
    parser.add_argument("--minimum-positive-score", type=float, default=0.5)
    parser.add_argument("--minimum-negative-score", type=float, default=0.0)
    args = parser.parse_args()
    policy = SelectionPolicy(
        candidate_pool_size=args.candidate_pool_size,
        negatives_per_query=args.negatives_per_query,
        positive_relative_ratio=args.positive_relative_ratio,
        absolute_positive_margin=args.absolute_positive_margin,
        minimum_positive_score=args.minimum_positive_score,
        minimum_negative_score=args.minimum_negative_score,
    )
    options = scorer.CacheOptions(
        input_path=args.requests.resolve(), output_dir=args.score_cache_dir.resolve()
    )
    verified_state = scorer.verify_complete_artifacts(options)
    if not scorer.is_training_admissible(verified_state):
        raise ValueError("teacher cache is not admissible for training")
    score_path = args.score_cache_dir / scorer.SCORES_NAME
    counters = atomic_jsonl_pair(
        args.requests, score_path, args.output, args.audit, policy
    )
    source_rights: dict[str, Any] = {}
    if args.source_manifest is not None:
        source_rights = json.loads(args.source_manifest.read_text(encoding="utf-8"))
        if not isinstance(source_rights, dict):
            raise ValueError("source manifest must be a JSON object")
    release_eligible = bool(source_rights.get("release_eligible") is True)
    release_blockers = list(source_rights.get("release_blockers") or [])
    if not release_eligible and not release_blockers:
        release_blockers = ["source training manifest is not public-release eligible"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "qwen3_reranker_listwise_embedding_kd_dataset",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_eligible": release_eligible,
        "release_blockers": release_blockers,
        "visibility": "public" if release_eligible else "private",
        "source_training_manifest": (
            {
                "path": str(args.source_manifest),
                "sha256": sha256_file(args.source_manifest),
            }
            if args.source_manifest is not None
            else None
        ),
        "teacher": {
            "model": scorer.MODEL_ID,
            "revision": scorer.MODEL_REVISION,
            "run_fingerprint_sha256": verified_state["run_fingerprint_sha256"],
            "score_cache_manifest_sha256": sha256_file(
                args.score_cache_dir / scorer.MANIFEST_NAME
            ),
        },
        "selection": {
            **policy.__dict__,
            "strategy": "score_rank_quantiles",
            "teacher_score_semantics": "normalized yes-token probability",
        },
        "counters": counters,
        "files": {
            "requests": {"sha256": sha256_file(args.requests)},
            "scores": {"sha256": sha256_file(score_path)},
            "train": {"sha256": sha256_file(args.output)},
            "audit": {"sha256": sha256_file(args.audit)},
        },
    }
    atomic_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
