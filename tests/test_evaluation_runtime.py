import json

import pytest

from scripts.evaluation_runtime import (
    effective_attention,
    enforce_runtime_contract,
    runtime_contract,
)


def make_contract(protocol, **overrides):
    values = {
        "protocol_id": "test-v1",
        "protocol_path": protocol,
        "model": "org/model",
        "revision": "a" * 40,
        "batch_size": 192,
        "max_length": 8192,
        "requested_attention": "flash_attention_2",
        "attention": "flash_attention_2",
        "evaluation_dtype": "bfloat16",
        "loader_contract": "sentence-transformer",
    }
    values.update(overrides)
    return runtime_contract(**values)


def test_float32_flash_attention_falls_back_to_sdpa():
    assert effective_attention("flash_attention_2", "float32") == "sdpa"
    assert effective_attention("sdpa", "float32") == "sdpa"
    assert effective_attention("flash_attention_2", "bfloat16") == "flash_attention_2"


def test_runtime_may_change_before_any_task_completes(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    first = make_contract(protocol)
    second = make_contract(protocol, batch_size=96)
    enforce_runtime_contract(run_dir, first)
    enforce_runtime_contract(run_dir, second)
    assert json.loads((run_dir / "runtime_contract.json").read_text()) == second


def test_runtime_change_is_rejected_after_a_completed_task(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    first = make_contract(protocol)
    enforce_runtime_contract(run_dir, first)
    result = run_dir / "mteb_cache" / "results" / "task.json"
    result.parent.mkdir(parents=True)
    result.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="fresh --output-dir"):
        enforce_runtime_contract(run_dir, make_contract(protocol, batch_size=96))
