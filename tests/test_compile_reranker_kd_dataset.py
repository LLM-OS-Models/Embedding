from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import cache_qwen3_reranker_scores as scorer
from scripts.compile_reranker_kd_dataset import (
    SelectionPolicy,
    atomic_jsonl_pair,
    compile_row,
    evenly_spaced_rank_indices,
)
from scripts.publish_reranker_kd_dataset import validate_artifacts


def request_raw(index: int, candidates: int = 21) -> dict:
    return {
        "generated_id": f"request-{index}",
        "query": f"query {index}",
        "positive": {
            "candidate_id": f"positive-{index}",
            "text": f"positive text {index}",
            "retriever_score": 1.0,
        },
        "candidates": [
            {
                "candidate_id": f"candidate-{index}-{candidate}",
                "text": f"candidate text {index} {candidate}",
                "retriever_score": 0.9 - candidate * 0.01,
            }
            for candidate in range(candidates)
        ],
    }


def scored_row(raw: dict) -> dict:
    request = scorer.parse_input_row(
        raw, max_documents_per_row=201, max_text_characters=1_000_000
    )
    probabilities = [0.95] + [0.8 - index * 0.02 for index in range(len(request.candidates))]
    documents = []
    for index, (document, probability) in enumerate(
        zip(request.documents, probabilities, strict=True)
    ):
        yes_logit = math.log(probability / (1 - probability))
        documents.append(
            {
                "candidate_id": document.candidate_id,
                "role": "positive" if index == 0 else "candidate",
                "text_sha256": scorer.sha256_bytes(document.text.encode()),
                "raw_no_logit": 0.0,
                "raw_yes_logit": yes_logit,
                "reranker_score": scorer.normalized_yes_probability(0.0, yes_logit),
                "retriever_score": document.retriever_score,
            }
        )
    scorer_identity = {
        "model": scorer.MODEL_ID,
        "revision": scorer.MODEL_REVISION,
        "backend": "pinned-local-qwen3-reranker",
    }
    return {
        "generated_id": request.generated_id,
        "query_sha256": scorer.sha256_bytes(request.query.encode()),
        "score_field": "reranker_score",
        "scorer": scorer_identity,
        "documents": documents,
    }


def test_rank_quantiles_cover_both_pool_ends() -> None:
    assert evenly_spaced_rank_indices(20, 5) == [0, 4, 9, 14, 19]


def test_compiler_emits_aligned_teacher_scores_and_audit() -> None:
    raw = request_raw(0)
    request = scorer.parse_input_row(
        raw, max_documents_per_row=201, max_text_characters=1_000_000
    )
    output, audit = compile_row(
        request,
        scored_row(raw),
        SelectionPolicy(candidate_pool_size=20, negatives_per_query=5),
    )
    assert output is not None
    assert len(output["negative_messages"]) == 5
    assert len(output["teacher_scores"]) == 6
    assert output["teacher_scores"][0] > max(output["teacher_scores"][1:])
    assert audit["selected_pool_indices_zero_based"] == [0, 4, 9, 14, 19]
    assert audit["drop_reason"] is None


def test_compiler_rejects_identity_tamper() -> None:
    raw = request_raw(0)
    request = scorer.parse_input_row(
        raw, max_documents_per_row=201, max_text_characters=1_000_000
    )
    scores = scored_row(raw)
    scores["documents"][2]["text_sha256"] = "0" * 64
    with pytest.raises(scorer.ScoreCacheError, match="identity"):
        compile_row(request, scores, SelectionPolicy(20, 5))


def test_atomic_compiler_preserves_row_order_and_validates_output(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    scores = tmp_path / "scores.jsonl"
    output = tmp_path / "train.jsonl"
    audit = tmp_path / "audit.jsonl"
    raw_rows = [request_raw(0), request_raw(1)]
    requests.write_text(
        "".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8"
    )
    scores.write_text(
        "".join(json.dumps(scored_row(row)) + "\n" for row in raw_rows),
        encoding="utf-8",
    )
    counters = atomic_jsonl_pair(
        requests,
        scores,
        output,
        audit,
        SelectionPolicy(candidate_pool_size=20, negatives_per_query=5),
    )
    assert counters == {"input_rows": 2, "output_rows": 2}
    output_rows = [json.loads(line) for line in output.read_text().splitlines()]
    audit_rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert output_rows[0]["messages"][0]["content"] == "query 0"
    assert audit_rows[1]["generated_id"] == "request-1"

    score_dir = tmp_path / "score-cache"
    score_dir.mkdir()
    score_path = score_dir / scorer.SCORES_NAME
    score_path.write_bytes(scores.read_bytes())
    (score_dir / scorer.MANIFEST_NAME).write_text(
        json.dumps({"admissible_for_training": True}) + "\n", encoding="utf-8"
    )
    from scripts.compile_reranker_kd_dataset import sha256_file

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_type": "qwen3_reranker_listwise_embedding_kd_dataset",
                "teacher": {"model": scorer.MODEL_ID, "revision": scorer.MODEL_REVISION},
                "selection": {"candidate_pool_size": 20, "negatives_per_query": 5},
                "counters": {"input_rows": 2, "output_rows": 2},
                "files": {
                    "requests": {"sha256": sha256_file(requests)},
                    "scores": {"sha256": sha256_file(score_path)},
                    "train": {"sha256": sha256_file(output)},
                    "audit": {"sha256": sha256_file(audit)},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    validated = validate_artifacts(
        SimpleNamespace(
            repo_id="LLM-OS-Models2/kd-fixture",
            train=output,
            audit=audit,
            manifest=manifest,
            requests=requests,
            score_cache_dir=score_dir,
        )
    )
    assert validated["output_rows"] == 2


def test_private_publisher_rejects_old_namespace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="LLM-OS-Models2"):
        validate_artifacts(
            SimpleNamespace(
                repo_id="LLM-OS-Models/forbidden",
                train=tmp_path / "missing",
                audit=tmp_path / "missing",
                manifest=tmp_path / "missing",
                requests=tmp_path / "missing",
                score_cache_dir=tmp_path,
            )
        )
