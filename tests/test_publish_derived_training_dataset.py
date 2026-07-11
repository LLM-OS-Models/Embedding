from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.publish_derived_training_dataset import dataset_card, validate


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublishDerivedTrainingDatasetTest(unittest.TestCase):
    def test_exact_files_and_quantile_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            provenance = root / "provenance.jsonl"
            audit = root / "audit.jsonl"
            train.write_text("{}\n{}\n", encoding="utf-8")
            provenance.write_text("{}\n{}\n", encoding="utf-8")
            audit.write_text("{}\n{}\n", encoding="utf-8")
            manifest = root / "final.json"
            manifest.write_text(
                json.dumps(
                    {
                        "output_rows": 2,
                        "batch_size": 1,
                        "benchmark_adaptation": "target-adapted-fixture",
                        "outputs": {
                            "train": {"sha256": digest(train)},
                            "provenance": {"sha256": digest(provenance)},
                        },
                    }
                ),
                encoding="utf-8",
            )
            mining = root / "mining.json"
            mining.write_text(
                json.dumps(
                    {
                        "rows": 2,
                        "selection_strategy": "score_rank_quantiles",
                        "candidate_pool_size": 24,
                        "num_negatives": 7,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                train=train,
                provenance=provenance,
                manifest=manifest,
                mining_manifest=mining,
                mining_audit=audit,
                repo_id="org/fixture",
                title="Fixture",
                source_dataset=["org/source"],
            )
            result = validate(args)
            self.assertEqual(result["rows"], 2)
            card = dataset_card(args, result)
            self.assertIn("score-rank", card)
            self.assertIn("release eligible: **false**", card)

            mining_payload = json.loads(mining.read_text())
            mining_payload["selection_strategy"] = "top_k"
            mining.write_text(json.dumps(mining_payload))
            with self.assertRaisesRegex(ValueError, "score_rank_quantiles"):
                validate(args)


if __name__ == "__main__":
    unittest.main()
