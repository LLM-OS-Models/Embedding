from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.publish_verified_legal_dataset import dataset_card, validate


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_validation(root: Path) -> None:
    root.mkdir()
    rows = [json.dumps({"row": index}) + "\n" for index in range(512)]
    for name in ("validation.jsonl", "provenance.jsonl"):
        path = root / name
        path.write_text("".join(rows), encoding="utf-8")
    manifest = {
        "artifact_id": "fixture-validation",
        "status": "complete",
        "assertions": {
            "source_holdout_contract_verified": True,
            "selected_query_training_text_overlap": 0,
            "selected_positive_training_text_overlap": 0,
            "selected_negative_training_text_overlap": 0,
            "selected_source_document_training_provenance_overlap": 0,
        },
        "files": {
            name: {"rows": 512, "sha256": file_sha(root / name)}
            for name in ("validation.jsonl", "provenance.jsonl")
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_validation_publication_gate_and_card(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    make_validation(artifact)
    validated = validate(
        artifact, "trainer-validation", "LLM-OS-Models2/fixture-validation"
    )
    card = dataset_card(validated)
    assert "visibility: **private**" in card
    assert "independence: **I, not Z**" in card
    assert validated["files"]["validation.jsonl"]["rows"] == 512


def test_wrong_namespace_and_hash_drift_fail_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    make_validation(artifact)
    with pytest.raises(ValueError, match="LLM-OS-Models2"):
        validate(artifact, "trainer-validation", "gyung/fixture")
    with (artifact / "validation.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="Artifact drift"):
        validate(
            artifact, "trainer-validation", "LLM-OS-Models2/fixture-validation"
        )
