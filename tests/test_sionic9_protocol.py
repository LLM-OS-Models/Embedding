from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluate_sionic9 import (
    DEFAULT_PROTOCOL,
    build_resolved_protocol,
    load_protocol,
    resolve_and_validate_tasks,
    validate_existing_result_contract,
    validate_mteb_checkout,
    validate_mteb_package,
)


TASK_SPEC = {
    "label": "PublicHealthQA",
    "name": "PublicHealthQA",
    "type": "Retrieval",
    "split": "test",
    "available_splits": ["test"],
    "hf_subsets": ["korean"],
    "dataset": {
        "path": "xhluca/publichealth-qa",
        "revision": "3b67b6b63ee464870fc21cdc888289c843204051",
    },
    "registry_dataset_revision": "main",
    "modalities": ["text"],
    "category": "t2t",
    "main_score": "ndcg_at_10",
    "license": "cc-by-nc-sa-3.0",
    "task_prompt": None,
    "instruction_fallback": "Retrieve text based on user query.",
}


def fake_task() -> SimpleNamespace:
    metadata = SimpleNamespace(
        name="PublicHealthQA",
        type="Retrieval",
        eval_splits=["test"],
        dataset={"path": "xhluca/publichealth-qa", "revision": "main"},
        modalities=["text"],
        category="t2t",
        main_score="ndcg_at_10",
        license="cc-by-nc-sa-3.0",
        prompt=None,
    )
    return SimpleNamespace(
        metadata=metadata,
        eval_splits=["test"],
        hf_subsets=["korean"],
        abstask_prompt="Retrieve text based on user query.",
    )


class FakeMTEB:
    __version__ = "2.18.0"

    def __init__(self, task: SimpleNamespace) -> None:
        self.task = task

    def get_task(self, name, *, eval_splits, hf_subsets):
        assert name == "PublicHealthQA"
        assert eval_splits == ["test"]
        assert hf_subsets == ["korean"]
        return self.task


def minimal_protocol() -> dict:
    committed = load_protocol(DEFAULT_PROTOCOL)
    return {**committed, "tasks": [copy.deepcopy(TASK_SPEC)]}


def resolved_row(*, selected: bool) -> dict:
    return {
        "label": "PublicHealthQA",
        "name": "PublicHealthQA",
        "type": "Retrieval",
        "selected_splits": ["test"],
        "available_splits": ["test"],
        "hf_subsets": ["korean"],
        "registry_dataset": {
            "path": "xhluca/publichealth-qa",
            "revision": "main",
        },
        "modalities": ["text"],
        "category": "t2t",
        "main_score": "ndcg_at_10",
        "license": "cc-by-nc-sa-3.0",
        "task_prompt": None,
        "instruction_fallback": "Retrieve text based on user query.",
        "dataset": {
            "path": "xhluca/publichealth-qa",
            "revision": "3b67b6b63ee464870fc21cdc888289c843204051",
        },
        "selected": selected,
    }


def add_cached_result(run_dir: Path) -> None:
    result = run_dir / "mteb_cache" / "results" / "AutoRAGRetrieval.json"
    result.parent.mkdir(parents=True)
    result.write_text("{}\n", encoding="utf-8")


def test_committed_protocol_pins_package_checkout_loader_and_task_metadata() -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    assert protocol["mteb_version"] == "2.18.0"
    assert len(protocol["mteb_git_revision"]) == 40
    assert protocol["loader_contract"]["id"].endswith("-v1")
    assert len(protocol["tasks"]) == 9
    assert all(len(spec["dataset"]["revision"]) == 40 for spec in protocol["tasks"])


def test_public_health_registry_main_is_validated_then_immutably_pinned() -> None:
    original = fake_task()
    shared_metadata = original.metadata
    tasks, resolved = resolve_and_validate_tasks(
        FakeMTEB(original), minimal_protocol(), ["PublicHealthQA"]
    )
    assert len(tasks) == 1
    assert tasks[0].metadata is not shared_metadata
    assert shared_metadata.dataset["revision"] == "main"
    assert tasks[0].metadata.dataset["revision"] == TASK_SPEC["dataset"]["revision"]
    assert resolved[0]["registry_dataset"]["revision"] == "main"
    assert resolved[0]["dataset"]["revision"] == TASK_SPEC["dataset"]["revision"]


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("type", "Classification"),
        ("main_score", "map_at_10"),
        ("license", "unknown"),
        ("prompt", "changed task prompt"),
        ("category", "s2s"),
    ],
)
def test_task_metadata_drift_is_rejected(attribute: str, value: object) -> None:
    task = fake_task()
    setattr(task.metadata, attribute, value)
    with pytest.raises(RuntimeError, match="metadata drifted for PublicHealthQA"):
        resolve_and_validate_tasks(FakeMTEB(task), minimal_protocol(), None)


def test_dataset_registry_revision_drift_is_rejected() -> None:
    task = fake_task()
    task.metadata.dataset["revision"] = "a" * 40
    with pytest.raises(RuntimeError, match="metadata drifted for PublicHealthQA"):
        resolve_and_validate_tasks(FakeMTEB(task), minimal_protocol(), None)


def test_mteb_package_version_drift_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="MTEB version mismatch"):
        validate_mteb_package(
            SimpleNamespace(__version__="2.19.0"), minimal_protocol()
        )


def test_mteb_checkout_drift_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.evaluate_sionic9.subprocess.check_output",
        lambda *args, **kwargs: "f" * 40 + "\n",
    )
    with pytest.raises(RuntimeError, match="MTEB git mismatch"):
        validate_mteb_checkout(minimal_protocol())


def test_loader_contract_drift_is_rejected(tmp_path: Path) -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    protocol["loader_contract"]["normalize_embeddings"] = False
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="loader contract changed"):
        load_protocol(path)


def test_legacy_cached_results_without_runtime_contract_are_rejected(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    add_cached_result(run_dir)
    with pytest.raises(RuntimeError, match="legacy canonical results cannot be reused"):
        validate_existing_result_contract(
            run_dir, {"resolved_task_contract_sha256": "a" * 64}
        )


def test_legacy_resolved_protocol_cannot_authorize_cache_reuse(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    add_cached_result(run_dir)
    (run_dir / "runtime_contract.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "protocol_resolved.json").write_text(
        json.dumps({"protocol_id": "sionic9-fixed-prompt-v1"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="legacy or different task/loader contract"):
        validate_existing_result_contract(
            run_dir, {"resolved_task_contract_sha256": "a" * 64}
        )


def test_partial_then_full_selection_keeps_same_task_contract(tmp_path: Path) -> None:
    protocol = minimal_protocol()
    partial = build_resolved_protocol(
        protocol,
        DEFAULT_PROTOCOL,
        [resolved_row(selected=True)],
        protocol["mteb_git_revision"],
    )
    full = build_resolved_protocol(
        protocol,
        DEFAULT_PROTOCOL,
        [resolved_row(selected=False)],
        protocol["mteb_git_revision"],
    )
    assert (
        partial["resolved_task_contract_sha256"]
        == full["resolved_task_contract_sha256"]
    )

    run_dir = tmp_path / "run"
    add_cached_result(run_dir)
    (run_dir / "runtime_contract.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "protocol_resolved.json").write_text(
        json.dumps(partial), encoding="utf-8"
    )
    validate_existing_result_contract(run_dir, full)
