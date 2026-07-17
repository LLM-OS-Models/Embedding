from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import publish_private_clean_candidate as publisher


def test_repo_id_is_strictly_models2() -> None:
    publisher.validate_repo_id("LLM-OS-Models2/clean-candidate-v1")
    for invalid in (
        "LLM-OS-Models/clean-candidate-v1",
        "gyung/clean-candidate-v1",
        "LLM-OS-Models2/nested/repo",
        "LLM-OS-Models2/-bad",
    ):
        with pytest.raises(ValueError):
            publisher.validate_repo_id(invalid)


def test_sensitive_file_gate_allows_tokenizer_but_rejects_credentials(
    tmp_path: Path,
) -> None:
    (tmp_path / "tokenizer_config.json").write_text("{}")
    publisher.validate_no_sensitive_files(tmp_path)
    (tmp_path / "optimizer.pt").write_bytes(b"state")
    with pytest.raises(ValueError, match="forbidden file"):
        publisher.validate_no_sensitive_files(tmp_path)


def test_candidate_is_bound_to_exact_clean_winner_and_model_shards(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    model = root / "artifacts/models/kd-winner"
    model.mkdir(parents=True)
    (model / "model-00001-of-00001.safetensors").write_bytes(b"model-weights")
    weights_sha = publisher.model_weights_sha256(model)
    revision = f"model-{weights_sha[:12]}"
    (model / "merge_report.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "model": {"weights_sha256": weights_sha},
                "sentence_transformers_contract": {
                    "pooling": "last_token",
                    "normalize": True,
                },
            }
        )
    )
    clean_path = root / "eval/clean/summary.json"
    robust_path = root / "eval/robust/summary.json"
    clean_path.parent.mkdir(parents=True)
    robust_path.parent.mkdir(parents=True)
    clean_path.write_text("{}")
    robust_path.write_text("{}")
    selection_path = root / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy_id": publisher.POLICY_ID,
                "public_benchmark_used_for_selection": False,
                "best": {
                    "model": "artifacts/models/kd-winner",
                    "revision": revision,
                    "weights_sha256": weights_sha,
                    "clean_summary": str(clean_path),
                    "robustness_summary": str(robust_path),
                    "clean_ndcg_at_10": 0.8,
                    "robustness_floor_ndcg_at_10": 0.7,
                    "max_noise_intrusion_at_10": 0.1,
                },
            }
        )
    )
    training_manifest = root / "training.json"
    training_manifest.write_text('{"rows":10000}')
    args = SimpleNamespace(
        model_dir=model,
        selection=selection_path,
        training_manifest=training_manifest,
        repo_id="LLM-OS-Models2/kd-clean-winner-v1-private",
    )
    clean = {
        "model": "artifacts/models/kd-winner",
        "revision": revision,
        "weights_sha256": weights_sha,
        "dataset_manifest_sha256": "a" * 64,
        "clean_ndcg_at_10": 0.8,
    }
    robustness = {
        "model": "artifacts/models/kd-winner",
        "revision": revision,
        "dataset_manifest_sha256": "a" * 64,
        "robustness_floor_ndcg_at_10": 0.7,
        "max_noise_intrusion_at_10": 0.1,
    }
    with (
        patch.object(publisher, "ROOT", root),
        patch.object(publisher, "load_clean_candidate", return_value=clean),
        patch.object(publisher, "load_robustness", return_value=robustness),
    ):
        validated = publisher.validate_candidate(args)
    assert validated["weights_sha256"] == weights_sha

    selection = json.loads(selection_path.read_text())
    selection["best"]["weights_sha256"] = "0" * 64
    selection_path.write_text(json.dumps(selection))
    with (
        patch.object(publisher, "ROOT", root),
        patch.object(publisher, "load_clean_candidate", return_value=clean),
        patch.object(publisher, "load_robustness", return_value=robustness),
        pytest.raises(ValueError, match="Selection weights"),
    ):
        publisher.validate_candidate(args)
