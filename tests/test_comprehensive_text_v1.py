from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluate_comprehensive_text_v1 import (
    ASSET_MANIFEST,
    DEFAULT_PROTOCOL,
    EXPECTED_EXCLUSIONS,
    ProtocolError,
    atomic_write_json,
    canonical_sha256,
    enforce_offline_environment,
    load_protocol,
    normalize_cached_results,
    ordered_sequence_sha256,
    public_model_reference,
    resolve_and_validate_tasks,
    resolve_offline_model,
    selected_hf_subsets,
    validate_asset_manifest,
    validate_existing_result_contract,
    verify_local_asset_snapshots,
)


class FakeMetadata:
    def __init__(self, payload: dict) -> None:
        self._payload = copy.deepcopy(payload)
        for key, value in payload.items():
            setattr(self, key, copy.deepcopy(value))
        self.hf_subsets_to_langscripts = {"default": ["kor-Hang"]}

    def model_dump(self, *, mode: str) -> dict:
        assert mode == "json"
        return copy.deepcopy(self._payload)


class FakeTask:
    def __init__(
        self,
        metadata_payload: dict,
        subsets: list[str],
        *,
        split: str = "test",
        instruction: str = "Classify user passages.",
    ) -> None:
        self.metadata = FakeMetadata(metadata_payload)
        self.hf_subsets = list(subsets)
        self.eval_splits = [split]
        self.abstask_prompt = instruction
        self.seed = 42


class FakeMTEB:
    def __init__(self, metadata_payload: dict, subsets: list[str]) -> None:
        self.metadata_payload = metadata_payload
        self.subsets = subsets
        self.calls: list[dict] = []

    def get_task(self, name: str, **kwargs):
        assert name == "FakeClassification"
        self.calls.append(kwargs)
        return FakeTask(
            self.metadata_payload,
            kwargs.get("hf_subsets", self.subsets),
            split=kwargs.get("eval_splits", ["test"])[0],
        )


def fake_protocol() -> dict:
    metadata_payload = {
        "name": "FakeClassification",
        "type": "Classification",
        "eval_splits": ["test"],
        "dataset": {"path": "example/fake", "revision": "a" * 40},
        "modalities": ["text"],
        "category": "t2c",
        "main_score": "accuracy",
        "license": "mit",
        "prompt": None,
        "description": "fixture",
    }
    subsets = ["default"]
    spec = {
        "asset_key": "fake",
        "name": "FakeClassification",
        "task_class": f"{FakeTask.__module__}.{FakeTask.__qualname__}",
        "type": "Classification",
        "split": "test",
        "available_splits": ["test"],
        "dataset": metadata_payload["dataset"],
        "modalities": ["text"],
        "category": "t2c",
        "main_score": "accuracy",
        "license": "mit",
        "task_prompt": None,
        "instruction_fallback": "Classify user passages.",
        "metadata_sha256": canonical_sha256(metadata_payload),
        "registry_hf_subsets": {
            "count": 1,
            "ordered_sha256": ordered_sequence_sha256(subsets),
        },
        "hf_subset_selection": {
            "mode": "exact",
            "values": subsets,
            "count": 1,
            "ordered_sha256": ordered_sequence_sha256(subsets),
        },
        "contamination_grade": "medium",
        "claim_policy": "diagnostic_regression_only",
    }
    return {
        "mteb_version": "2.18.0",
        "seed": 42,
        "tasks": [spec],
        "_metadata_payload": metadata_payload,
    }


def write_task_result(
    run_dir: Path,
    spec: dict,
    *,
    revision: str | None = None,
    score: float = 0.75,
    languages: list[str] | None = None,
) -> Path:
    path = (
        run_dir
        / "mteb_cache"
        / "results"
        / "example__model"
        / ("b" * 40)
        / f"{spec['name']}.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "dataset_revision": revision or spec["dataset"]["revision"],
                "task_name": spec["name"],
                "mteb_version": "2.18.0",
                "scores": {
                    spec["split"]: [
                        {
                            "accuracy": score,
                            "main_score": score,
                            "hf_subset": "default",
                            "languages": languages or ["kor-Hang"],
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def fake_resolved_tasks(protocol: dict) -> list[dict]:
    spec = protocol["tasks"][0]
    return [
        {
            "name": spec["name"],
            "selected_hf_subset_languages": {"default": ["kor-Hang"]},
        }
    ]


def test_committed_protocol_is_exactly_text_only_and_explicitly_excludes_gaps() -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    assert len(protocol["tasks"]) == 7
    assert all(task["modalities"] == ["text"] for task in protocol["tasks"])
    assert sum(task["hf_subset_selection"]["count"] for task in protocol["tasks"]) == 414
    assert {
        item["asset_key"] for item in protocol["explicit_exclusions"]
    } == EXPECTED_EXCLUSIONS
    assert any(item["asset_key"] == "k_haters" for item in protocol["explicit_exclusions"])
    assert all(
        "visual-document" in item["reason"].lower()
        for item in protocol["explicit_exclusions"]
        if item["asset_key"] != "k_haters"
    )


def test_asset_manifest_sha_revisions_and_all_13_assets_are_accounted_for() -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    evidence = validate_asset_manifest(protocol, ASSET_MANIFEST)
    assert evidence["manifest_sha256"] == hashlib.sha256(
        ASSET_MANIFEST.read_bytes()
    ).hexdigest()
    assert len(evidence["assets"]) == 7
    assert all(len(asset["revision"]) == 40 for asset in evidence["assets"])


def test_offline_environment_is_forced_and_credentials_are_removed() -> None:
    env = {
        "HF_TOKEN": "hf_secret_sentinel",
        "HUGGINGFACE_HUB_TOKEN": "second_secret",
        "GITHUB": "github_secret",
        "HF_HUB_OFFLINE": "0",
    }
    evidence = enforce_offline_environment(env)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["HF_DATASETS_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_HOME"].endswith("/.cache/huggingface")
    assert not ({"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "GITHUB"} & set(env))
    assert evidence["credentials_available_to_runner"] == "none"
    assert "secret" not in json.dumps(evidence)


def test_snapshot_verifier_uses_only_exact_local_revisions(tmp_path: Path) -> None:
    protocol = fake_protocol()
    calls = []

    def resolver(**kwargs):
        calls.append(kwargs)
        path = tmp_path / kwargs["revision"]
        path.mkdir(exist_ok=True)
        return str(path)

    evidence = verify_local_asset_snapshots(
        protocol, cache_dir=tmp_path / "hub", resolver=resolver
    )
    assert evidence[0]["local_files_only"] is True
    assert calls == [
        {
            "repo_id": "example/fake",
            "repo_type": "dataset",
            "revision": "a" * 40,
            "cache_dir": tmp_path / "hub",
            "local_files_only": True,
            "token": False,
        }
    ]


def test_task_resolution_validates_full_metadata_and_resumes_by_selector() -> None:
    protocol = fake_protocol()
    mteb = FakeMTEB(protocol["_metadata_payload"], ["default"])
    tasks, resolved = resolve_and_validate_tasks(mteb, protocol, ["fake"])
    assert len(tasks) == 1
    assert resolved[0]["selected"] is True
    assert resolved[0]["selected_hf_subsets"] == ["default"]
    assert mteb.calls == [
        {},
        {"eval_splits": ["test"], "hf_subsets": ["default"]},
    ]


def test_any_task_metadata_or_registry_subset_drift_fails_closed() -> None:
    protocol = fake_protocol()
    drifted = copy.deepcopy(protocol["_metadata_payload"])
    drifted["main_score"] = "f1"
    with pytest.raises(ProtocolError, match="metadata drifted"):
        resolve_and_validate_tasks(FakeMTEB(drifted, ["default"]), protocol, None)

    with pytest.raises(ProtocolError, match="Registry subset"):
        resolve_and_validate_tasks(
            FakeMTEB(protocol["_metadata_payload"], ["default", "new"]),
            protocol,
            None,
        )


def test_flores_component_selection_is_exact_ordered_and_not_all_languages() -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    flores = protocol["tasks"][1]
    selected = [f"lang{i}-kor_Hang" for i in range(406)]
    drifted = copy.deepcopy(flores)
    drifted["hf_subset_selection"]["ordered_sha256"] = ordered_sequence_sha256(selected)
    assert selected_hf_subsets(drifted, selected + ["eng_Latn-deu_Latn"]) == selected
    with pytest.raises(ProtocolError, match="order/content drifted"):
        selected_hf_subsets(drifted, list(reversed(selected)))


def test_cached_results_are_normalized_without_unrelated_metrics(tmp_path: Path) -> None:
    protocol = fake_protocol()
    spec = protocol["tasks"][0]
    write_task_result(tmp_path, spec)
    rows, inventory = normalize_cached_results(
        tmp_path, protocol, fake_resolved_tasks(protocol)
    )
    assert rows == [
        {
            "asset_key": "fake",
            "task_name": "FakeClassification",
            "task_type": "Classification",
            "split": "test",
            "dataset_revision": "a" * 40,
            "main_metric": "accuracy",
            "task_score": 0.75,
            "leaderboard_points": 75.0,
            "subset_count": 1,
            "subsets": [
                {
                    "hf_subset": "default",
                    "languages": ["kor-Hang"],
                    "score": 0.75,
                }
            ],
            "contamination_grade": "medium",
            "claim_policy": "diagnostic_regression_only",
        }
    ]
    assert len(inventory) == 1
    assert len(inventory[0]["sha256"]) == 64
    assert not Path(inventory[0]["path"]).is_absolute()


def test_cached_revision_and_nonfinite_main_score_are_rejected(tmp_path: Path) -> None:
    protocol = fake_protocol()
    spec = protocol["tasks"][0]
    write_task_result(tmp_path, spec, revision="c" * 40)
    with pytest.raises(ProtocolError, match="dataset revision drifted"):
        normalize_cached_results(tmp_path, protocol, fake_resolved_tasks(protocol))

    run_two = tmp_path / "second"
    write_task_result(run_two, spec, score=float("nan"))
    with pytest.raises(ProtocolError, match="Non-finite main_score"):
        normalize_cached_results(run_two, protocol, fake_resolved_tasks(protocol))

    run_three = tmp_path / "third"
    write_task_result(run_three, spec, languages=["eng-Latn"])
    with pytest.raises(ProtocolError, match="subset-language mapping drifted"):
        normalize_cached_results(run_three, protocol, fake_resolved_tasks(protocol))


def test_existing_partial_cache_requires_matching_resolved_contract(tmp_path: Path) -> None:
    protocol = fake_protocol()
    write_task_result(tmp_path, protocol["tasks"][0])
    resolved = {"resolved_task_contract_sha256": "d" * 64}
    with pytest.raises(ProtocolError, match="lack runtime/protocol evidence"):
        validate_existing_result_contract(tmp_path, resolved)

    (tmp_path / "runtime_contract.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "protocol_resolved.json").write_text(
        json.dumps(resolved) + "\n", encoding="utf-8"
    )
    validate_existing_result_contract(tmp_path, resolved)

    with pytest.raises(ProtocolError, match="different task contract"):
        validate_existing_result_contract(
            tmp_path, {"resolved_task_contract_sha256": "e" * 64}
        )


def test_atomic_json_writes_verifiable_sha_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    digest = atomic_write_json(output, {"complete": False, "tasks": []})
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.with_name("summary.json.sha256").read_text(encoding="ascii") == (
        f"{digest}  summary.json\n"
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_hub_model_resolution_is_immutable_local_only(tmp_path: Path) -> None:
    revision = "f" * 40
    calls = []

    def resolver(**kwargs):
        calls.append(kwargs)
        path = tmp_path / revision
        path.mkdir()
        return str(path)

    resolved, evidence = resolve_offline_model(
        "example/model", revision, resolver=resolver
    )
    assert resolved == revision
    assert evidence["local_files_only"] is True
    assert calls[0]["local_files_only"] is True
    assert calls[0]["token"] is False
    with pytest.raises(Exception, match="full immutable commit SHA"):
        resolve_offline_model("example/model", "main", resolver=resolver)


def test_local_model_reference_does_not_expose_shared_machine_path(tmp_path: Path) -> None:
    model = tmp_path / "reviewed-model"
    model.mkdir()
    assert public_model_reference(str(model)) == "local:reviewed-model"
    assert public_model_reference("example/model") == "example/model"
