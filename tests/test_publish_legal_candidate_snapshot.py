from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.publish_legal_candidate_snapshot as module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_snapshot_validation_binds_all_reference_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    declared = []
    for index in range(16):
        path = candidates / f"part-{index}.jsonl"
        path.write_text(json.dumps({"row": index}) + "\n", encoding="utf-8")
        extractor = path.with_suffix(".manifest.json")
        extractor.write_text(
            json.dumps(
                {
                    "output_sha256": sha(path),
                    "summary": {"records_emitted": 1},
                    "parameters": {"shard_count": 16, "max_records": 25_000},
                }
            ),
            encoding="utf-8",
        )
        declared.append({"path": str(path.relative_to(tmp_path)), "rows": 1, "sha256": sha(path)})
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "status": "complete",
                "artifact_id": "korean-legal-public-source-heldout-retrieval-v2-text-strict",
                "inputs": {"candidate_sources": {"files": declared}},
            }
        ),
        encoding="utf-8",
    )
    validated = module.validate(candidates, reference, "LLM-OS-Models2/fixture")
    assert validated["snapshot_manifest"]["counts"] == {
        "candidate_files": 16,
        "extractor_manifests": 16,
        "candidate_rows": 16,
    }
    assert len(validated["evidence"]) == 32


def test_snapshot_rejects_hash_drift_and_wrong_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "status": "complete",
                "artifact_id": "korean-legal-public-source-heldout-retrieval-v2-text-strict",
                "inputs": {"candidate_sources": {"files": []}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="LLM-OS-Models2"):
        module.validate(candidates, reference, "gyung/fixture")
    with pytest.raises(ValueError, match="exactly 16"):
        module.validate(candidates, reference, "LLM-OS-Models2/fixture")
