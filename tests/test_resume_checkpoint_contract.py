from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import select_best_checkpoint
from scripts import validate_resume_checkpoint


def make_checkpoint(root: Path, step: int, *, train: Path, validation: Path) -> Path:
    checkpoint = root / "v0" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    files = {
        "adapter_model.safetensors": b"adapter",
        "adapter_config.json": b"{}",
        "optimizer.pt": b"optimizer",
        "scheduler.pt": b"scheduler",
        "rng_state.pth": b"rng",
        "training_args.bin": b"args",
    }
    for name, payload in files.items():
        (checkpoint / name).write_bytes(payload)
    state = {
        "global_step": step,
        "max_steps": 100,
        "log_history": [{"step": step, "eval_loss": 0.1}],
    }
    (checkpoint / "trainer_state.json").write_text(json.dumps(state))
    training = {
        "model": "Qwen/Qwen3-Embedding-8B",
        "model_revision": "1" * 40,
        "dataset": [str(train.resolve())],
        "val_dataset": [str(validation.resolve())],
        "max_steps": 100,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 8,
        "max_length": 512,
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "learning_rate": 1e-5,
        "loss_type": "infonce",
        "dataset_shuffle": False,
        "train_dataloader_shuffle": False,
        "seed": 42,
    }
    (checkpoint / "args.json").write_text(json.dumps(training))
    return checkpoint


def resume_args(checkpoint: Path, run: Path, train: Path, validation: Path) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=checkpoint,
        run_dir=run,
        train_file=train,
        val_file=validation,
        base_model="Qwen/Qwen3-Embedding-8B",
        base_revision="1" * 40,
        max_steps=100,
        train_batch_size=8,
        grad_accum_steps=8,
        max_length=512,
        lora_rank=64,
        lora_alpha=128.0,
        lora_dropout=0.05,
        learning_rate=1e-5,
        loss_type="infonce",
        dataset_shuffle=False,
        train_dataloader_shuffle=False,
        seed=42,
    )


def test_resume_checkpoint_requires_complete_optimizer_and_same_step_eval(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text("{}\n")
    validation.write_text("{}\n")
    checkpoint = make_checkpoint(tmp_path / "run", 25, train=train, validation=validation)
    assert select_best_checkpoint.complete_resume_checkpoint(checkpoint, "adapter")

    (checkpoint / "optimizer.pt").unlink()
    assert not select_best_checkpoint.complete_resume_checkpoint(checkpoint, "adapter")


def test_resume_contract_binds_data_model_and_optimization(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text("{}\n")
    validation.write_text("{}\n")
    run = tmp_path / "run"
    checkpoint = make_checkpoint(run, 25, train=train, validation=validation)
    args = resume_args(checkpoint, run, train, validation)
    assert validate_resume_checkpoint.validate(args) == 25

    args.learning_rate = 2e-5
    with pytest.raises(ValueError, match="learning_rate"):
        validate_resume_checkpoint.validate(args)
    args.learning_rate = 1e-5
    other = tmp_path / "other.jsonl"
    other.write_text("{}\n")
    args.train_file = other
    with pytest.raises(ValueError, match="dataset"):
        validate_resume_checkpoint.validate(args)


def test_resume_selector_rejects_duplicate_steps_across_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text("{}\n")
    validation.write_text("{}\n")
    run = tmp_path / "run"
    first = make_checkpoint(run, 25, train=train, validation=validation)
    duplicate = run / "v1" / first.name
    duplicate.parent.mkdir(parents=True)
    duplicate.mkdir()
    for source in first.iterdir():
        (duplicate / source.name).write_bytes(source.read_bytes())
    monkeypatch.setattr(
        "sys.argv",
        [
            "select_best_checkpoint.py",
            str(run),
            "--checkpoint-kind",
            "adapter",
            "--latest-resume",
        ],
    )
    with pytest.raises(ValueError, match="ambiguous"):
        select_best_checkpoint.main()
