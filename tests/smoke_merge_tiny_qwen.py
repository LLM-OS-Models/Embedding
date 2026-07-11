#!/usr/bin/env python3
"""CPU-only end-to-end smoke test for the adapter merge CLI.

The test constructs a random one-layer Qwen3 and a non-zero LoRA in a temporary
directory. It does not download or load the 8B checkpoint.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)


ROOT = Path(__file__).resolve().parents[1]


def make_tokenizer() -> PreTrainedTokenizerFast:
    tokens = [
        "<|endoftext|>",
        "<|im_end|>",
        "<unk>",
        "Instruct",
        ":",
        "Given",
        "a",
        "web",
        "search",
        "query",
        "retrieve",
        "relevant",
        "passages",
        "that",
        "answer",
        "the",
        "Query",
    ]
    backend = Tokenizer(
        WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )
    tokenizer.padding_side = "left"
    return tokenizer


def main() -> None:
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory(prefix="tiny-qwen-merge-") as temp:
        root = Path(temp)
        base_dir = root / "base"
        adapter_dir = root / "adapter"
        output_dir = root / "merged"
        config = Qwen3Config(
            vocab_size=17,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=512,
            eos_token_id=1,
            pad_token_id=0,
            bos_token_id=0,
            tie_word_embeddings=False,
            use_cache=False,
        )
        model = Qwen3ForCausalLM(config)
        model.save_pretrained(base_dir, safe_serialization=True)
        make_tokenizer().save_pretrained(base_dir)

        peft_model = get_peft_model(
            model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                bias="none",
            ),
        )
        for name, parameter in peft_model.named_parameters():
            if parameter.requires_grad:
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
        peft_model.save_pretrained(adapter_dir, safe_serialization=True)
        adapter_config_path = adapter_dir / "adapter_config.json"
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        adapter_config["base_model_name_or_path"] = str(base_dir.resolve())
        adapter_config_path.write_text(
            json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8"
        )

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/merge_embedding_adapter.py"),
                "--adapter",
                str(adapter_dir),
                "--output-dir",
                str(output_dir),
                "--base-model",
                str(base_dir),
                "--base-revision",
                "tiny-local",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--local-files-only",
            ],
            cwd=ROOT,
            check=True,
        )
        report = json.loads(
            (output_dir / "merge_report.json").read_text(encoding="utf-8")
        )
        assert report["status"] == "pass"
        assert report["probe"]["metrics"]["dimensions"] == 16
        assert report["probe"]["metrics"]["minimum_row_cosine"] >= 0.999
        assert (output_dir / "model.safetensors").is_file()
        assert (output_dir / "modules.json").is_file()
        assert (output_dir / "1_Pooling" / "config.json").is_file()
        print("tiny Qwen3 LoRA merge smoke: PASS")


if __name__ == "__main__":
    main()
