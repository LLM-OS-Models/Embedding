from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.select_nemotron3_public_checkpoint import (
    BASE_MODEL,
    BASE_REVISION,
    parse_steps,
    select,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture(root: Path, losses: dict[int, float]) -> SimpleNamespace:
    run = root / "run"
    run.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"release_eligible": True, "release_blockers": [], "visibility": "public"}
        )
    )
    (run / "run_contract.json").write_text(
        json.dumps(
            {
                "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION,
                "training_data": {"manifest_sha256": digest(manifest)},
            }
        )
    )
    (run / "training-complete.json").write_text(json.dumps({"status": "complete"}))
    for step, loss in losses.items():
        checkpoint = run / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
        (checkpoint / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": f"/snapshots/{BASE_REVISION}"})
        )
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(
                {
                    "global_step": step,
                    "log_history": [{"step": step, "eval_loss": loss}],
                }
            )
        )
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
        (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    return SimpleNamespace(
        run_dir=run,
        training_manifest=manifest,
        output=root / "selection.json",
        expected_steps=",".join(map(str, losses)),
    )


def test_selects_minimum_same_step_heldout_loss() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        args = make_fixture(Path(temporary), {50: 0.8, 100: 0.7, 150: 0.75})
        result = select(args)
        assert result["selected"]["step"] == 100
        assert result["public_benchmark_used_for_selection"] is False


def test_rejects_incomplete_or_nonfinite_candidates() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        args = make_fixture(Path(temporary), {50: float("nan")})
        with pytest.raises(ValueError, match="finite same-step"):
            select(args)


def test_expected_steps_are_unique_and_increasing() -> None:
    assert parse_steps("50,100") == (50, 100)
    with pytest.raises(ValueError, match="unique and increasing"):
        parse_steps("100,50")
