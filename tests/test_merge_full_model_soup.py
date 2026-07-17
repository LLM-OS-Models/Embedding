from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts import merge_embedding_adapter as adapter_merge
from scripts import merge_full_model_soup as soup


def make_model(
    root: Path, value: float, *, hidden_size: int = 4, sharded: bool = False
) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": hidden_size,
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
            }
        )
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "padding_side": "left",
                "eos_token": "<|im_end|>",
                "pad_token": "<|endoftext|>",
            }
        )
    )
    adapter_merge.write_sentence_transformers_contract(root, hidden_size)
    tensors = {
        "layer.weight": torch.full((hidden_size, hidden_size), value, dtype=torch.bfloat16),
        "norm.weight": torch.full((hidden_size,), value * 2, dtype=torch.bfloat16),
    }
    if sharded:
        shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
        save_file({"layer.weight": tensors["layer.weight"]}, root / shards[0], metadata={"format": "pt"})
        save_file({"norm.weight": tensors["norm.weight"]}, root / shards[1], metadata={"format": "pt"})
        (root / soup.INDEX_NAME).write_text(
            json.dumps(
                {
                    "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
                    "weight_map": {"layer.weight": shards[0], "norm.weight": shards[1]},
                }
            )
        )
    else:
        shards = (soup.SINGLE_WEIGHTS_NAME,)
        save_file(tensors, root / soup.SINGLE_WEIGHTS_NAME, metadata={"format": "pt"})
    weights_sha = soup.model_weights_sha256(root, shards)
    contract = adapter_merge.validate_sentence_transformers_contract(root)
    (root / "merge_report.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "model": {"weights_sha256": weights_sha},
                "sentence_transformers_contract": contract,
            }
        )
    )
    return root


def test_full_model_soup_averages_in_fp32_and_emits_bf16(tmp_path: Path) -> None:
    first = make_model(tmp_path / "first", 1.0)
    second = make_model(tmp_path / "second", 3.0)
    output = tmp_path / "output"
    report = soup.build_soup(
        SimpleNamespace(
            model=[first, second],
            weight=[0.25, 0.75],
            output_dir=output,
            output_dtype="bfloat16",
            torch_threads=1,
        )
    )
    tensors = load_file(output / soup.SINGLE_WEIGHTS_NAME)
    assert tensors["layer.weight"].dtype == torch.bfloat16
    assert torch.equal(
        tensors["layer.weight"], torch.full((4, 4), 2.5, dtype=torch.bfloat16)
    )
    assert torch.equal(
        tensors["norm.weight"], torch.full((4,), 5.0, dtype=torch.bfloat16)
    )
    assert report["status"] == "pass"
    assert report["soup"]["accumulation_dtype"] == "float32"
    assert report["model"]["weights_sha256"] == soup.model_weights_sha256(
        output, (soup.SINGLE_WEIGHTS_NAME,)
    )
    assert adapter_merge.validate_sentence_transformers_contract(output)["status"] == "pass"


def test_soup_rejects_weight_or_architecture_drift(tmp_path: Path) -> None:
    first = make_model(tmp_path / "first", 1.0)
    second = make_model(tmp_path / "second", 2.0, hidden_size=8)
    with pytest.raises(ValueError, match="configurations differ"):
        soup.validate_sources([first, second], [0.5, 0.5])
    with pytest.raises(ValueError, match="sum exactly"):
        soup.validate_sources([first, first], [0.6, 0.6])


def test_sharded_soup_preserves_index_contract(tmp_path: Path) -> None:
    first = make_model(tmp_path / "first", 2.0, sharded=True)
    second = make_model(tmp_path / "second", 4.0, sharded=True)
    output = tmp_path / "output"
    soup.build_soup(
        SimpleNamespace(
            model=[first, second],
            weight=[0.5, 0.5],
            output_dir=output,
            output_dtype="bfloat16",
            torch_threads=1,
        )
    )
    index = json.loads((output / soup.INDEX_NAME).read_text())
    assert set(index["weight_map"]) == {"layer.weight", "norm.weight"}
    assert set(index["weight_map"].values()) == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert index["metadata"]["total_size"] == 40
    assert torch.equal(
        load_file(output / "model-00001-of-00002.safetensors")["layer.weight"],
        torch.full((4, 4), 3.0, dtype=torch.bfloat16),
    )
