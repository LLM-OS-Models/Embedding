from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.finalize_public_training_manifest import build
from scripts.publish_derived_training_dataset import validate_public_rights


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FinalizePublicTrainingManifestTest(unittest.TestCase):
    def test_finalizer_requires_exact_transform_overlap_and_row_rights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            provenance = root / "provenance.jsonl"
            train.write_text("{}\n{}\n", encoding="utf-8")
            rights = {
                "source": "org/source",
                "revision": "a" * 40,
                "license": "cc-by-4.0",
                "redistribution_allowed": True,
            }
            provenance.write_text(
                json.dumps(rights) + "\n" + json.dumps(rights) + "\n",
                encoding="utf-8",
            )
            source = root / "source.json"
            source.write_text(
                json.dumps(
                    {
                        "visibility": "public",
                        "release_eligible": True,
                        "release_blockers": [],
                        "dataset_license": "other",
                        "sources": [rights],
                    }
                )
            )
            transform = root / "transform.json"
            transform.write_text(
                json.dumps(
                    {
                        "batch_size": 2,
                        "outputs": {
                            "train": {"sha256": digest(train)},
                            "provenance": {"sha256": digest(provenance)},
                        },
                    }
                )
            )
            overlap = root / "overlap.json"
            overlap.write_text(
                json.dumps(
                    {
                        "rows": 2,
                        "inputs": {
                            "train": {"sha256": digest(train)},
                            "provenance": {"sha256": digest(provenance)},
                        },
                        "unique_critical_query_or_evaluation_matches": 0,
                        "unique_retrieval_corpus_matches": 0,
                    }
                )
            )
            output = root / "final.json"
            args = SimpleNamespace(
                train=train,
                provenance=provenance,
                source_manifest=source,
                transform_manifest=[transform],
                benchmark_overlap_audit=overlap,
                output=output,
                artifact_id="fixture",
                required_next_stage="ready",
            )
            manifest = build(args)
            self.assertEqual(manifest["output_rows"], 2)
            self.assertEqual(validate_public_rights(manifest, provenance), 2)
            broken = json.loads(overlap.read_text())
            broken["unique_retrieval_corpus_matches"] = 1
            overlap.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "retrieval corpus"):
                build(args)


if __name__ == "__main__":
    unittest.main()
