from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublishBestModelTests(unittest.TestCase):
    def test_card_requires_complete_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            (model / "1_Pooling").mkdir(parents=True)
            (model / "2_Normalize").mkdir()
            for name in ("config.json", "modules.json", "1_Pooling/config.json"):
                (model / name).write_text("{}")
            (model / "model.safetensors").write_bytes(b"fixture")
            (model / "merge_report.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "base_model": "Qwen/Qwen3-Embedding-8B",
                        "base_revision": "1" * 40,
                        "adapter": {"weights_sha256": "2" * 64},
                        "adapter_config": {
                            "r": 64,
                            "lora_alpha": 128,
                            "lora_dropout": 0.05,
                            "target_modules": ["q_proj"],
                        },
                        "probe": {
                            "metrics": {
                                "minimum_row_cosine": 1.0,
                                "maximum_pairwise_score_difference": 0.0,
                            }
                        },
                        "sentence_transformers_contract": {
                            "pooling": "last_token",
                            "normalize": True,
                        },
                    }
                )
            )
            sionic = root / "sionic.json"
            names = [
                "MIRACL",
                "MrTidy",
                "MLDR",
                "AutoRAG",
                "Ko-StrategyQA",
                "PublicHealthQA",
                "Belebele",
                "SQuADKorV1",
                "LawIRKo",
            ]
            sionic.write_text(
                json.dumps(
                    {
                        "protocol_id": "fixture",
                        "completed_tasks": 9,
                        "average": 0.8,
                        "scores": {name: 0.8 for name in names},
                    }
                )
            )
            official = root / "official.json"
            official.write_text(
                json.dumps(
                    {
                        "complete": True,
                        "completed_tasks": 6,
                        "protocol_id": "fixture-official",
                        "mean_task_leaderboard_points": 80.0,
                        "mean_task_type_leaderboard_points": 79.0,
                        "scores": {
                            f"task-{index}": {"score": 0.8} for index in range(6)
                        },
                    }
                )
            )
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"phase": "fixture", "built_rows": 10}))
            subprocess.check_call(
                [
                    "python",
                    str(ROOT / "scripts/publish_best_embedding_model.py"),
                    "--model-dir",
                    str(model),
                    "--sionic-summary",
                    str(sionic),
                    "--official-summary",
                    str(official),
                    "--training-manifest",
                    str(manifest),
                ]
            )
            card = (model / "README.md").read_text()
            self.assertIn("9-task average: **0.80000**", card)
            self.assertIn("SentenceTransformers", card)
            self.assertIn("zero-shot", card)
            publication = json.loads((model / "publication_manifest.json").read_text())
            self.assertEqual(set(publication["evidence"]), {
                "sionic9_summary.json",
                "mteb_korean_v1_summary.json",
                "training_manifest.json",
            })

            (model / "merge_report.json").unlink()
            (model / "full_tuning_report.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "training_method": "partial-full-parameter-update",
                        "base_model": "Qwen/Qwen3-Embedding-8B",
                        "base_revision": "1" * 40,
                        "model": {"weights_sha256": "3" * 64},
                        "probe": {
                            "metrics": {
                                "maximum_norm_error": 1e-7,
                                "positive_margin": 0.25,
                            }
                        },
                        "sentence_transformers_contract": {
                            "pooling": "last_token",
                            "normalize": True,
                        },
                    }
                )
            )
            subprocess.check_call(
                [
                    "python",
                    str(ROOT / "scripts/publish_best_embedding_model.py"),
                    "--model-dir",
                    str(model),
                    "--sionic-summary",
                    str(sionic),
                    "--official-summary",
                    str(official),
                    "--training-manifest",
                    str(manifest),
                ]
            )
            full_card = (model / "README.md").read_text()
            self.assertIn("부분 full-parameter update", full_card)
            self.assertNotIn("LoRA rank/alpha/dropout", full_card)
            full_publication = json.loads(
                (model / "publication_manifest.json").read_text()
            )
            self.assertEqual(
                full_publication["model_evidence"]["file"],
                "full_tuning_report.json",
            )


if __name__ == "__main__":
    unittest.main()
