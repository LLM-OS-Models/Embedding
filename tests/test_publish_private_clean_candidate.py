from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import publish_private_clean_candidate as publisher


FAKE_HF_CREDENTIAL = "hf_" + "a" * 24


def _publication_fixture(tmp_path: Path) -> tuple[SimpleNamespace, dict, Path]:
    root = tmp_path / "workspace"
    model = root / "artifacts/models/kd-winner"
    model.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"model-weights")
    (model / "config.json").write_text(
        json.dumps({"_name_or_path": f"{root}/private/base", "hidden_size": 4096})
    )
    (model / "modules.json").write_text("[]")
    (model / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {
                    "vocab": {
                        "/home/is-a-vocabulary-token": 1,
                        FAKE_HF_CREDENTIAL: 2,
                    }
                }
            },
            separators=(",", ":"),
        )
    )
    (model / "1_Pooling").mkdir()
    (model / "1_Pooling/config.json").write_text('{"pooling_mode_lasttoken":true}')
    evidence = model / "full_tuning_report.json"
    evidence.write_text(
        json.dumps(
            {
                "status": "pass",
                "source_checkpoint": f"{root}/outputs/checkpoint-10",
                "hf_token": FAKE_HF_CREDENTIAL,
            }
        )
    )
    selection = root / "selection.json"
    training = root / "training.json"
    clean = root / "clean/summary.json"
    robust = root / "robust/summary.json"
    for summary in (clean, robust):
        summary.parent.mkdir(parents=True)
        summary.write_text(json.dumps({"local": f"{root}/eval/private"}))
        (summary.parent / "ranks.jsonl").write_text(
            json.dumps({"rank": 1, "path": f"{root}/corpus"}) + "\n"
        )
    selection.write_text(json.dumps({"path": f"{root}/selection"}))
    training.write_text(json.dumps({"password": "do-not-upload", "rows": 10}))
    weights_sha = publisher.model_weights_sha256(model)
    args = SimpleNamespace(repo_id="LLM-OS-Models2/kd-clean-private")
    validated = {
        "model_dir": model,
        "model_rel": "artifacts/models/kd-winner",
        "selection_path": selection,
        "training_manifest_path": training,
        "clean_path": clean,
        "robustness_path": robust,
        "clean": {"clean_ndcg_at_10": 0.8},
        "robustness": {
            "robustness_floor_ndcg_at_10": 0.7,
            "max_noise_intrusion_at_10": 0.1,
        },
        "evidence_path": evidence,
        "evidence_name": evidence.name,
        "weights_sha256": weights_sha,
        "revision": f"model-{weights_sha[:12]}",
    }
    return args, validated, root


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


def test_isolated_publication_sanitizes_evidence_without_mutating_model(
    tmp_path: Path,
) -> None:
    args, validated, root = _publication_fixture(tmp_path)
    model = validated["model_dir"]
    original = {
        path.relative_to(model).as_posix(): path.read_bytes()
        for path in model.rglob("*")
        if path.is_file()
    }
    staging = root / "staging"
    with patch.object(publisher, "ROOT", root):
        manifest = publisher.prepare_publication(args, validated, staging)
        publisher.validate_staged_text(staging)
    observed = {
        path.relative_to(model).as_posix(): path.read_bytes()
        for path in model.rglob("*")
        if path.is_file()
    }
    assert observed == original
    assert publisher.model_weights_sha256(staging) == validated["weights_sha256"]
    assert (staging / "tokenizer.json").read_bytes() == original["tokenizer.json"]
    staged_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in staging.rglob("*")
        if path.is_file()
        and path.suffix in {".json", ".jsonl", ".md"}
        and path.name not in {"tokenizer.json", "vocab.json"}
    )
    assert str(root) not in staged_text
    assert FAKE_HF_CREDENTIAL not in staged_text
    assert "do-not-upload" not in staged_text
    assert "[REDACTED]" in staged_text
    manifest_value = json.loads(manifest.read_text())
    assert manifest_value["files_excluding_manifest"]["tokenizer.json"]["sha256"] == (
        publisher.sha256(model / "tokenizer.json")
    )
    assert manifest_value["publication_safety"] == {
        "allowlisted_model_payload": True,
        "isolated_staging": True,
        "local_paths_removed": True,
        "recognized_credentials_removed": True,
        "source_model_mutated": False,
    }


def test_publication_rejects_unapproved_model_payload(tmp_path: Path) -> None:
    args, validated, root = _publication_fixture(tmp_path)
    (validated["model_dir"] / "train.log").write_text("must remain local")
    with (
        patch.object(publisher, "ROOT", root),
        pytest.raises(ValueError, match="unapproved payload"),
    ):
        publisher.prepare_publication(args, validated, root / "staging")


def test_remote_publication_requires_exact_files_and_lfs_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    metadata = tmp_path / "config.json"
    metadata.write_text('{"hidden_size":4096}\n')
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"weights")
    expected = {
        "config.json": {
            "sha256": publisher.sha256(metadata),
            "size_bytes": metadata.stat().st_size,
        },
        "model.safetensors": {
            "sha256": publisher.sha256(shard),
            "size_bytes": shard.stat().st_size,
        },
    }
    info = SimpleNamespace(
        private=True,
        siblings=[
            SimpleNamespace(rfilename="config.json", lfs=None),
            SimpleNamespace(
                rfilename="model.safetensors",
                lfs={
                    "sha256": expected["model.safetensors"]["sha256"],
                    "size": shard.stat().st_size,
                },
            ),
            SimpleNamespace(rfilename=".gitattributes", lfs=None),
        ],
    )
    api = SimpleNamespace(model_info=lambda **_: info)
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: str(metadata) if kwargs["filename"] == "config.json" else None,
    )
    publisher.verify_remote_publication(
        api=api,
        repo_id="LLM-OS-Models2/exact-private",
        revision="a" * 40,
        token="not-a-real-token",
        expected=expected,
    )
    info.siblings.append(SimpleNamespace(rfilename="train.log", lfs=None))
    with pytest.raises(RuntimeError, match="file-set mismatch"):
        publisher.verify_remote_publication(
            api=api,
            repo_id="LLM-OS-Models2/exact-private",
            revision="a" * 40,
            token="not-a-real-token",
            expected=expected,
        )


@pytest.mark.parametrize(
    "evidence_name",
    ("merge_report.json", "full_tuning_report.json", "soup_report.json"),
)
def test_candidate_is_bound_to_exact_clean_winner_and_model_shards(
    tmp_path: Path, evidence_name: str
) -> None:
    root = tmp_path / "workspace"
    model = root / "artifacts/models/kd-winner"
    model.mkdir(parents=True)
    (model / "model-00001-of-00001.safetensors").write_bytes(b"model-weights")
    weights_sha = publisher.model_weights_sha256(model)
    revision = f"model-{weights_sha[:12]}"
    (model / evidence_name).write_text(
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
    assert validated["evidence_name"] == evidence_name

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
