from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_multidomain_selection_holdout import MANIFEST_SHA256
from scripts.evaluate_multidomain_selection import (
    DATASET_MANIFEST_SHA256,
    metrics_from_ranks,
    validate_dataset,
)
from scripts.publish_multidomain_selection_dataset import (
    DEFAULT_REPO,
    dataset_card,
    portable_provenance_path,
    publication_manifest,
)
from scripts.select_best_clean_model import MULTIDOMAIN_MANIFEST_SHA256


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return {"rows": len(rows), "sha256": sha(path)}


def fixture(root: Path) -> None:
    root.mkdir()
    queries = [
        {"_id": "f0", "text": "finance query", "domain": "finance"},
        {"_id": "k0", "text": "knowledge query", "domain": "knowledge"},
    ]
    corpus = [
        {"_id": "fd", "text": "finance doc", "domain": "finance"},
        {"_id": "kd", "text": "knowledge doc", "domain": "knowledge"},
    ]
    qrels = [
        {"query-id": "f0", "corpus-id": "fd", "score": 1},
        {"query-id": "k0", "corpus-id": "kd", "score": 1},
    ]
    files = {
        "queries.jsonl": write_jsonl(root / "queries.jsonl", queries),
        "corpus.jsonl": write_jsonl(root / "corpus.jsonl", corpus),
        "qrels.jsonl": write_jsonl(root / "qrels.jsonl", qrels),
        "provenance.jsonl": write_jsonl(root / "provenance.jsonl", [{}, {}]),
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "protocol_id": "multidomain-selection-heldout-v1",
                "domains": {"finance": {"queries": 1}, "knowledge": {"queries": 1}},
                "assertions": {
                    "all_selected_query_exact_training_text_overlap": 0,
                    "knowledge_query_and_corpus_exact_training_text_overlap": 0,
                    "all_selected_query_and_corpus_benchmark_blocklist_overlap": 0,
                    "public_benchmark_score_used_for_selection": False,
                },
                "files": files,
            }
        ),
        encoding="utf-8",
    )


def test_multidomain_manifest_and_cross_domain_qrel_contract(tmp_path: Path) -> None:
    root = tmp_path / "data"
    fixture(root)
    queries, corpus, qrels, _ = validate_dataset(
        root, expected_manifest_sha256=None
    )
    assert len(queries) == 2 and len(corpus) == 2 and qrels["f0"] == {"fd"}
    qrel_path = root / "qrels.jsonl"
    qrel_path.write_text(
        json.dumps({"query-id": "f0", "corpus-id": "kd", "score": 1}) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"]["qrels.jsonl"] = {"rows": 1, "sha256": sha(qrel_path)}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Cross-domain"):
        validate_dataset(root, expected_manifest_sha256=None)


def test_multi_positive_ndcg_and_recall_are_exact() -> None:
    metrics = metrics_from_ranks([[1, 3], [11]])
    # Directly assert the standard binary-relevance values without depending on
    # the helper implementation's intermediate representation.
    expected_first = (1.0 + 1.0 / 2.0) / (1.0 + 1.0 / 1.584962500721156)
    assert metrics["ndcg_at_10"] == pytest.approx(expected_first / 2.0)
    assert metrics["recall_at_10"] == pytest.approx(0.5)
    assert metrics["mrr_at_10"] == pytest.approx(0.5)


def test_private_selection_publication_contract_is_portable_and_hash_bound(
    tmp_path: Path,
) -> None:
    assert portable_provenance_path("outputs/data/train.jsonl")
    assert not portable_provenance_path("/home/ubuntu/data/train.jsonl")
    assert not portable_provenance_path("../outside.jsonl")
    manifest = {
        "domains": {
            "finance": {
                "queries": 900,
                "corpus_training_text_occurrences": 1373,
            },
            "knowledge": {"queries": 1000},
        },
        "files": {
            "corpus.jsonl": {"rows": 4795},
            "qrels.jsonl": {"rows": 2941},
        },
    }
    payload = tmp_path / "queries.jsonl"
    payload.write_text('{"_id":"q"}\n', encoding="utf-8")
    validated = {"manifest": manifest, "source_manifest_sha256": "a" * 64}
    card = dataset_card(validated)
    assert "selection-only" in card
    assert "public benchmark score used for selection: false" in card
    assert "1,373" in card
    publication = publication_manifest(
        validated=validated, sources={"queries.jsonl": payload}
    )
    assert publication["repo_id"] == DEFAULT_REPO
    assert publication["visibility"] == "private"
    assert publication["files_excluding_publication_manifest"]["queries.jsonl"][
        "sha256"
    ] == hashlib.sha256(payload.read_bytes()).hexdigest()


def test_fixed_manifest_identity_is_shared_by_every_consumer() -> None:
    assert MANIFEST_SHA256 == DATASET_MANIFEST_SHA256
    assert MANIFEST_SHA256 == MULTIDOMAIN_MANIFEST_SHA256
