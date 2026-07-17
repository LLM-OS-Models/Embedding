from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.train_nemotron3_public_lora import (
    QUERY_PROMPT,
    TRAINING_PROMPTS,
    convert_example,
    latest_complete_checkpoint,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrainNemotron3PublicLoraTest(unittest.TestCase):
    def test_training_query_prompt_matches_fixed_sionic_protocol(self) -> None:
        protocol = json.loads((ROOT / "configs/sionic9_protocol.json").read_text())
        self.assertEqual(QUERY_PROMPT, protocol["query_prompt"])

    def test_contract_requires_public_rights_and_exact_train_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            (model / "1_Pooling").mkdir(parents=True)
            (model / "config.json").write_text(
                json.dumps(
                    {"model_type": "ministral3", "architectures": ["Ministral3Model"]}
                )
            )
            (model / "modules.json").write_text(
                json.dumps(
                    [
                        {"type": "sentence_transformers.models.Transformer"},
                        {"type": "sentence_transformers.models.Pooling"},
                        {"type": "sentence_transformers.models.Normalize"},
                    ]
                )
            )
            (model / "1_Pooling/config.json").write_text(
                json.dumps({"pooling_mode_mean_tokens": True})
            )
            train = root / "train.jsonl"
            train.write_text("{}\n{}\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "release_eligible": True,
                        "release_blockers": [],
                        "visibility": "public",
                        "outputs": {
                            "train": {"sha256": digest(train), "rows": 2}
                        },
                    }
                )
            )
            args = SimpleNamespace(
                model=model,
                revision="a" * 40,
                train=train,
                eval=None,
                eval_manifest=None,
                training_manifest=manifest,
                output_dir=root / "out",
                max_steps=1,
                batch_size=2,
                mini_batch_size=1,
                max_length=64,
                learning_rate=2e-5,
                warmup_ratio=0.05,
                lora_rank=8,
                lora_alpha=16,
                save_steps=1,
                seed=42,
            )
            contract = validate_contract(args)
            self.assertEqual(contract["base_revision"], "a" * 40)
            self.assertTrue(contract["training_data"]["release_eligible"])
            self.assertEqual(contract["training_prompts"]["anchor"], QUERY_PROMPT)
            self.assertEqual(contract["training_prompts"]["positive"], "")
            self.assertEqual(TRAINING_PROMPTS, {"anchor": QUERY_PROMPT})
            self.assertIn("prepended by the collator", contract["input_contract"]["query"])
            self.assertIn("no prefix", contract["input_contract"]["document"])
            args.max_steps = 2
            with self.assertRaisesRegex(ValueError, "requires --eval"):
                validate_contract(args)
            args.max_steps = 1
            manifest_value = json.loads(manifest.read_text())
            manifest_value["release_eligible"] = False
            manifest.write_text(json.dumps(manifest_value))
            with self.assertRaisesRegex(ValueError, "rights-safe"):
                validate_contract(args)

    def test_strict_row_conversion_and_complete_checkpoint_selection(self) -> None:
        row = {
            "messages": [
                {"role": "user", "content": "Instruct: legal retrieval\nQuery: query"}
            ],
            "positive_messages": [[{"role": "user", "content": "positive"}]],
            "negative_messages": [
                [{"role": "user", "content": "negative one"}],
                [{"role": "user", "content": "negative two"}],
            ],
        }
        self.assertEqual(
            convert_example(row, 0),
            {
                "anchor": "query",
                "positive": "positive",
                "negative_1": "negative one",
                "negative_2": "negative two",
            },
        )
        row["messages"][0]["content"] = "query"
        with self.assertRaisesRegex(ValueError, "explicit stored"):
            convert_example(row, 0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step, complete in ((50, True), (100, False), (150, True)):
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                if complete:
                    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
                        (checkpoint / name).write_text("x")
            self.assertEqual(latest_complete_checkpoint(root), root / "checkpoint-150")


if __name__ == "__main__":
    unittest.main()
