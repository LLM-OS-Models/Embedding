from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/merge_embedding_adapter.py"
SPEC = importlib.util.spec_from_file_location("merge_embedding_adapter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
merge = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = merge
SPEC.loader.exec_module(merge)


class ContractTests(unittest.TestCase):
    def make_model_dir(self) -> Path:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "config.json").write_text(
            json.dumps({"hidden_size": 16, "architectures": ["Qwen3ForCausalLM"]}),
            encoding="utf-8",
        )
        (root / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "padding_side": "left",
                    "eos_token": "<|im_end|>",
                    "pad_token": "<|endoftext|>",
                }
            ),
            encoding="utf-8",
        )
        merge.write_sentence_transformers_contract(root, 16)
        return root

    def test_exact_contract_round_trip(self) -> None:
        result = merge.validate_sentence_transformers_contract(self.make_model_dir())
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["hidden_size"], 16)
        self.assertEqual(result["pooling"], "last_token")
        self.assertTrue(result["normalize"])
        self.assertEqual(result["padding_side"], "left")

    def test_contract_rejects_mean_pooling(self) -> None:
        root = self.make_model_dir()
        path = root / "1_Pooling" / "config.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["pooling_mode_lasttoken"] = False
        value["pooling_mode_mean_tokens"] = True
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pooling drift"):
            merge.validate_sentence_transformers_contract(root)

    def test_contract_rejects_missing_left_padding(self) -> None:
        root = self.make_model_dir()
        path = root / "tokenizer_config.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["padding_side"] = "right"
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "padding_side='left'"):
            merge.validate_sentence_transformers_contract(root)

    def test_query_prompt_has_no_accidental_separator(self) -> None:
        rows = merge.format_probe_rows((("query", "질문"), ("document", "문서")))
        self.assertEqual(rows[0], merge.QUERY_PROMPT + "질문")
        self.assertEqual(rows[1], "문서")

    def test_adapter_base_reference_rejects_wrong_revision(self) -> None:
        config = {
            "base_model_name_or_path": "/cache/models--Qwen--Qwen3/snapshots/wrong"
        }
        with self.assertRaisesRegex(ValueError, "Adapter/base mismatch"):
            merge.validate_adapter_base_reference(
                config,
                "Qwen/Qwen3-Embedding-8B",
                merge.DEFAULT_BASE_REVISION,
            )


class ParityTests(unittest.TestCase):
    def test_identical_embeddings_pass_exactly(self) -> None:
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        result = merge.parity_metrics(matrix, matrix)
        self.assertEqual(result.rows, 2)
        self.assertEqual(result.dimensions, 2)
        self.assertAlmostEqual(result.minimum_row_cosine, 1.0)
        self.assertAlmostEqual(result.maximum_absolute_difference, 0.0)
        self.assertAlmostEqual(result.maximum_pairwise_score_difference, 0.0)

    def test_parity_reports_score_and_element_drift(self) -> None:
        result = merge.parity_metrics(
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.8, 0.6], [0.0, 1.0]],
        )
        self.assertAlmostEqual(result.minimum_row_cosine, 0.8)
        self.assertAlmostEqual(result.maximum_absolute_difference, 0.6)
        self.assertGreater(result.maximum_pairwise_score_difference, 0.0)

    def test_parity_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape-compatible"):
            merge.parity_metrics([[1.0, 0.0]], [[1.0]])


if __name__ == "__main__":
    unittest.main()
