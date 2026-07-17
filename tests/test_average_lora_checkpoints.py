from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts import average_lora_checkpoints as average


def make_checkpoint(
    version: Path,
    step: int,
    value: float,
    *,
    config_rank: int = 2,
) -> Path:
    checkpoint = version / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    config = {
        "peft_type": "LORA",
        "r": config_rank,
        "lora_alpha": 4,
        "target_modules": ["q_proj"],
        "base_model_name_or_path": "Qwen/Qwen3-Embedding-8B",
    }
    (checkpoint / average.CONFIG_NAME).write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )
    save_file(
        {
            "layer.lora_A.weight": torch.full((2, 3), value, dtype=torch.bfloat16),
            "layer.lora_B.weight": torch.full((3, 2), value * 2, dtype=torch.bfloat16),
        },
        checkpoint / average.WEIGHTS_NAME,
    )
    return checkpoint


def test_fp32_average_uses_latest_same_trajectory_and_is_hash_bound(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    version = run / "v0"
    for step, value in ((10, 1.0), (20, 2.0), (30, 3.0), (40, 4.0)):
        make_checkpoint(version, step, value)
    # A retry directory must never be mixed into the anchored trajectory.
    make_checkpoint(run / "v1-retry", 100, 100.0)
    output = tmp_path / "average"

    report = average.build_average(
        run_dir=run,
        anchor_checkpoint=version / "checkpoint-20",
        output_dir=output,
        last_n=3,
        minimum_checkpoints=2,
    )

    assert report["selection"]["steps"] == [20, 30, 40]
    assert report["selection"]["checkpoint_count"] == 3
    assert report["averaging"]["accumulation_dtype"] == "float32"
    tensors = load_file(output / average.WEIGHTS_NAME)
    assert tensors["layer.lora_A.weight"].dtype == torch.float32
    assert torch.equal(
        tensors["layer.lora_A.weight"], torch.full((2, 3), 3.0)
    )
    assert torch.equal(
        tensors["layer.lora_B.weight"], torch.full((3, 2), 6.0)
    )
    saved = json.loads((output / average.REPORT_NAME).read_text(encoding="utf-8"))
    assert saved["output"]["weights_sha256"] == average.sha256_file(
        output / average.WEIGHTS_NAME
    )


def test_average_rejects_config_drift(tmp_path: Path) -> None:
    run = tmp_path / "run"
    version = run / "v0"
    make_checkpoint(version, 10, 1.0)
    anchor = make_checkpoint(version, 20, 2.0, config_rank=4)
    with pytest.raises(ValueError, match="configurations differ"):
        average.build_average(
            run_dir=run,
            anchor_checkpoint=anchor,
            output_dir=tmp_path / "average",
            last_n=5,
            minimum_checkpoints=2,
        )


def test_average_prefers_validated_archive_beyond_trainer_retention(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    version = run / "v0"
    anchor = make_checkpoint(version, 50, 5.0)
    make_checkpoint(version, 40, 4.0)
    archive_version = run / average.ARCHIVE_NAME / version.name
    for step, value in ((10, 1.0), (20, 2.0), (30, 3.0), (40, 4.0), (50, 5.0)):
        checkpoint = make_checkpoint(archive_version, step, value)
        weights = checkpoint / average.WEIGHTS_NAME
        config = checkpoint / average.CONFIG_NAME
        (checkpoint / average.ARCHIVE_MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "complete",
                    "checkpoint": {"label": checkpoint.name, "step": step},
                    "adapter": {
                        "weights": {
                            "sha256": average.sha256_file(weights),
                            "size_bytes": weights.stat().st_size,
                        },
                        "config": {
                            "sha256": average.sha256_file(config),
                            "size_bytes": config.stat().st_size,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
    output = tmp_path / "average"
    report = average.build_average(
        run_dir=run,
        anchor_checkpoint=anchor,
        output_dir=output,
        last_n=5,
        minimum_checkpoints=2,
    )
    assert report["selection"]["steps"] == [10, 20, 30, 40, 50]
    assert report["selection"]["source_pool"] == "validated_local_archive"
    tensors = load_file(output / average.WEIGHTS_NAME)
    assert torch.equal(
        tensors["layer.lora_A.weight"], torch.full((2, 3), 3.0)
    )
    with (archive_version / "checkpoint-10" / average.WEIGHTS_NAME).open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="integrity validation"):
        average.select_checkpoints(
            run_dir=run,
            anchor_checkpoint=anchor,
            last_n=5,
            minimum_checkpoints=2,
        )


def test_average_rejects_disqualified_or_insufficient_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    anchor = make_checkpoint(run / "v0", 10, 1.0)
    with pytest.raises(ValueError, match="Only 1 complete checkpoints"):
        average.select_checkpoints(
            run_dir=run,
            anchor_checkpoint=anchor,
            last_n=5,
            minimum_checkpoints=2,
        )
    make_checkpoint(run / "v0", 20, 2.0)
    (run / "DISQUALIFIED.json").write_text('{"status":"failed"}\n')
    with pytest.raises(RuntimeError, match="disqualified"):
        average.select_checkpoints(
            run_dir=run,
            anchor_checkpoint=anchor,
            last_n=5,
            minimum_checkpoints=2,
        )
