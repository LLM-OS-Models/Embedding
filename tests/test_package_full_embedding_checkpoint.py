from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.package_full_embedding_checkpoint import (
    hash_model_files,
    sha256_file,
    validate_existing_package,
    validate_training_contract,
    write_sentence_transformers_contract,
)
from scripts.model_lineage import resolve_base_lineage


BASE_MODEL = "Qwen/Qwen3-Embedding-8B"
BASE_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"


def build_contract(tmp_path: Path) -> tuple[argparse.Namespace, Path, Path]:
    run = tmp_path / "run"
    checkpoint = run / "v0" / "checkpoint-3123"
    checkpoint.mkdir(parents=True)
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text('{"query":"q"}\n', encoding="utf-8")
    validation.write_text('{"query":"v"}\n', encoding="utf-8")
    train_log = run / "train.log"
    logging = run / "v0" / "logging.jsonl"
    train_log.write_text("End time of running main\n", encoding="utf-8")
    logging.write_text('{"global_step":"3123/3123"}\n', encoding="utf-8")
    contract = run / "capacity_run_manifest.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "embedding-capacity-training-contract",
                "status": "complete",
                "mode": "last4",
                "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION,
                "train": {
                    "path": str(train),
                    "sha256": sha256_file(train),
                    "size_bytes": train.stat().st_size,
                },
                "validation": {
                    "path": str(validation),
                    "sha256": sha256_file(validation),
                    "size_bytes": validation.stat().st_size,
                },
                "optimization": {
                    "max_steps": 3123,
                    "global_batch_size": 64,
                    "dataset_shuffle": False,
                    "train_dataloader_shuffle": False,
                },
                "completion": {
                    "expected_steps": 3123,
                    "train_log": {"path": str(train_log), "sha256": sha256_file(train_log)},
                    "logging_jsonl": {"path": str(logging), "sha256": sha256_file(logging)},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        checkpoint=checkpoint,
        training_contract=contract,
        base_model=BASE_MODEL,
        base_revision=BASE_REVISION,
    )
    return args, contract, train


def test_capacity_training_contract_binds_inputs_completion_and_checkpoint(
    tmp_path: Path,
) -> None:
    args, contract, _ = build_contract(tmp_path)
    evidence = validate_training_contract(args)
    assert evidence == {"path": str(contract), "sha256": sha256_file(contract)}


def test_capacity_training_contract_rejects_input_hash_drift(tmp_path: Path) -> None:
    args, _, train = build_contract(tmp_path)
    train.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file/size mismatch|hash mismatch"):
        validate_training_contract(args)


def test_capacity_training_contract_rejects_external_checkpoint(tmp_path: Path) -> None:
    args, _, _ = build_contract(tmp_path)
    external = tmp_path / "external-checkpoint"
    external.mkdir()
    args.checkpoint = external
    with pytest.raises(ValueError, match="outside the contracted run directory"):
        validate_training_contract(args)


def test_existing_full_package_is_bound_to_source_checkpoint_bytes(
    tmp_path: Path,
) -> None:
    args, _, _ = build_contract(tmp_path)
    checkpoint = args.checkpoint
    (checkpoint / "model.safetensors").write_bytes(b"full-model-weights")
    output = tmp_path / "packaged"
    output.mkdir()
    (output / "model.safetensors").write_bytes(b"full-model-weights")
    (output / "config.json").write_text(
        json.dumps(
            {"hidden_size": 16, "architectures": ["Qwen3ForCausalLM"]}
        ),
        encoding="utf-8",
    )
    (output / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "padding_side": "left",
                "eos_token": "<|im_end|>",
                "pad_token": "<|endoftext|>",
            }
        ),
        encoding="utf-8",
    )
    write_sentence_transformers_contract(output, 16)
    report = {
        "status": "pass",
        "training_method": "partial-full-parameter-update",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "upstream_base_models": resolve_base_lineage(BASE_MODEL, BASE_REVISION),
        "source_checkpoint": str(checkpoint.resolve()),
        "training_contract": validate_training_contract(args),
        "model": {"weights_sha256": hash_model_files(output)},
    }
    (output / "full_tuning_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    args.output_dir = output
    validated = validate_existing_package(args)
    assert validated["model_weights_sha256"] == report["model"]["weights_sha256"]
    (checkpoint / "model.safetensors").write_bytes(b"changed-source")
    with pytest.raises(ValueError, match="source checkpoint shards"):
        validate_existing_package(args)
