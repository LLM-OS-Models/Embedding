from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.list_validated_adapter_checkpoints import discover


def write_checkpoint(path: Path, *, archived: bool = False) -> None:
    path.mkdir(parents=True)
    weights = path / "adapter_model.safetensors"
    config = path / "adapter_config.json"
    weights.write_bytes(b"fixture weights")
    config.write_text('{"peft_type":"LORA","r":1}\n', encoding="utf-8")
    if archived:
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "checkpoint": {"label": path.name},
            "adapter": {
                "weights": {
                    "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                    "size_bytes": weights.stat().st_size,
                },
                "config": {
                    "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    "size_bytes": config.stat().st_size,
                },
            },
        }
        (path / "archive_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_archive_is_canonical_complete_history(tmp_path: Path) -> None:
    run = tmp_path / "run"
    version = run / "v0"
    write_checkpoint(version / "checkpoint-500")
    write_checkpoint(version / "checkpoint-750")
    write_checkpoint(run / ".adapter-checkpoint-archive/v0/checkpoint-250", archived=True)
    write_checkpoint(run / ".adapter-checkpoint-archive/v0/checkpoint-500", archived=True)
    checkpoints, archived = discover(run, version / "checkpoint-500")
    assert archived is True
    assert [path.name for path in checkpoints] == ["checkpoint-250", "checkpoint-500"]


def test_invalid_archive_fails_closed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    version = run / "v0"
    write_checkpoint(version / "checkpoint-250")
    archived = run / ".adapter-checkpoint-archive/v0/checkpoint-250"
    write_checkpoint(archived, archived=True)
    (archived / "adapter_model.safetensors").write_bytes(b"corrupt")
    try:
        discover(run, version / "checkpoint-250")
    except ValueError as error:
        assert "integrity" in str(error)
    else:
        raise AssertionError("corrupt archive was accepted")
