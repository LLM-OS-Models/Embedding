from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.publish_performance_dataset import sha256, validate


class PublishPerformanceDatasetTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        data = root / "data"
        data.mkdir()
        train = data / "train.jsonl"
        provenance = data / "provenance.jsonl"
        train.write_text('{"row":1}\n', encoding="utf-8")
        provenance.write_text('{"row_index":0}\n', encoding="utf-8")
        manifest = {
            "phase": "fixture",
            "built_rows": 1,
            "files": {
                "train.jsonl": {"rows": 1, "sha256": sha256(train)},
                "provenance.jsonl": {
                    "rows": 1,
                    "sha256": sha256(provenance),
                },
            },
        }
        (data / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        card = root / "README.md"
        card.write_text(
            "release_eligible: false\nSionic 9\nMIRACL\n", encoding="utf-8"
        )
        audit = root / "audit.json"
        audit.write_text(
            json.dumps(
                {
                    "rows": 1,
                    "inputs": {
                        "train": {"sha256": sha256(train)},
                        "provenance": {"sha256": sha256(provenance)},
                    },
                    "contract_checks": {"status": "pass"},
                }
            ),
            encoding="utf-8",
        )
        return data, card, audit

    def test_accepts_matching_quality_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, card, audit = self._fixture(Path(directory))
            _manifest, paths = validate(
                data, card, "fixture", 1, "train.jsonl", "provenance.jsonl", audit
            )
            self.assertEqual(paths[-1], audit)

    def test_rejects_drifted_quality_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, card, audit = self._fixture(Path(directory))
            payload = json.loads(audit.read_text(encoding="utf-8"))
            payload["inputs"]["train"]["sha256"] = "0" * 64
            audit.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate(
                    data,
                    card,
                    "fixture",
                    1,
                    "train.jsonl",
                    "provenance.jsonl",
                    audit,
                )

    def test_accepts_zero_critical_benchmark_overlap_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, card, quality = self._fixture(Path(directory))
            manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
            overlap = Path(directory) / "overlap.json"
            overlap.write_text(
                json.dumps(
                    {
                        "rows": 1,
                        "inputs": {
                            "train": {
                                "sha256": manifest["files"]["train.jsonl"]["sha256"]
                            },
                            "provenance": {
                                "sha256": manifest["files"]["provenance.jsonl"][
                                    "sha256"
                                ]
                            },
                        },
                        "unique_critical_query_or_evaluation_matches": 0,
                        "unique_retrieval_corpus_matches": 1,
                    }
                ),
                encoding="utf-8",
            )
            _manifest, paths = validate(
                data,
                card,
                "fixture",
                1,
                "train.jsonl",
                "provenance.jsonl",
                quality,
                overlap,
            )
            self.assertEqual(paths[-1], overlap)

            payload = json.loads(overlap.read_text(encoding="utf-8"))
            payload["unique_critical_query_or_evaluation_matches"] = 1
            overlap.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "critical overlap"):
                validate(
                    data,
                    card,
                    "fixture",
                    1,
                    "train.jsonl",
                    "provenance.jsonl",
                    quality,
                    overlap,
                )


if __name__ == "__main__":
    unittest.main()
