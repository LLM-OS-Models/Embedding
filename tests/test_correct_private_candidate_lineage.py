from __future__ import annotations

import copy

import pytest

from scripts.correct_private_candidate_lineage import build_corrected_manifest


OLD_SHA = "1" * 64
NEW_SHA = "2" * 64


def source_manifest() -> dict:
    return {
        "schema_version": 1,
        "artifact_kind": "peft-lora-checkpoint-candidate",
        "distribution": "private-candidate-only",
        "checkpoint": {"label": "checkpoint-250", "step": 250},
        "adapter": {
            "weights": {"sha256": "3" * 64, "size_bytes": 10},
            "config": {"sha256": "4" * 64, "size_bytes": 10},
        },
        "validation": {
            "completion_sentinel_observed": True,
            "same_step_eval_observed": True,
            "safetensors_full_payload_validation": "pass",
            "all_tensor_values_finite": True,
            "staged_snapshot_sha256_reverified": True,
            "eval_loss": 0.1,
        },
        "lineage": {"training_data_sha256": OLD_SHA},
        "remote_allowlist": [
            "adapter_model.safetensors",
            "adapter_config.json",
            "candidate_manifest.json",
        ],
    }


def test_lineage_correction_changes_only_declared_identity_and_adds_provenance() -> None:
    source = source_manifest()
    original = copy.deepcopy(source)
    corrected = build_corrected_manifest(
        source,
        source_repo="LLM-OS-Models2/source",
        source_revision="5" * 40,
        source_manifest_sha256="6" * 64,
        checkpoint_label="checkpoint-250",
        expected_source_sha256=OLD_SHA,
        corrected_sha256=NEW_SHA,
    )
    assert source == original
    assert corrected["lineage"]["training_data_sha256"] == NEW_SHA
    assert corrected["adapter"] == original["adapter"]
    assert corrected["validation"] == original["validation"]
    evidence = corrected["lineage_correction"]
    assert evidence["previous_value"] == OLD_SHA
    assert evidence["corrected_value"] == NEW_SHA
    assert evidence["source_revision"] == "5" * 40


def test_lineage_correction_rejects_unverified_source_candidate() -> None:
    source = source_manifest()
    source["validation"]["all_tensor_values_finite"] = False
    with pytest.raises(ValueError, match="all_tensor_values_finite"):
        build_corrected_manifest(
            source,
            source_repo="LLM-OS-Models2/source",
            source_revision="5" * 40,
            source_manifest_sha256="6" * 64,
            checkpoint_label="checkpoint-250",
            expected_source_sha256=OLD_SHA,
            corrected_sha256=NEW_SHA,
        )
