from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.publish_derived_training_dataset import validate_public_rights


ROOT = Path(__file__).resolve().parents[1]


def strict_row(query: str, positive: str, negative: str) -> dict:
    return {
        "messages": [{"role": "user", "content": f"Instruct: x\nQuery: {query}"}],
        "positive_messages": [[{"role": "user", "content": positive}]],
        "negative_messages": [[{"role": "user", "content": negative}]],
    }


class BuildPublicLegalTrainingDatasetTest(unittest.TestCase):
    def test_filters_blocked_text_and_adds_public_rights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "input.jsonl"
            provenance = root / "input-provenance.jsonl"
            rows = [
                strict_row("blocked query", "positive a", "negative a"),
                strict_row("clean query b", "positive b", "negative b"),
                strict_row("clean query c", "positive c", "negative c"),
            ]
            train.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            revision = "a" * 40
            provenance.write_text(
                "".join(
                    json.dumps(
                        {
                            "source": "org/legal",
                            "provenance": {
                                "repository": "org/legal",
                                "revision": revision,
                            },
                        }
                    )
                    + "\n"
                    for _ in rows
                ),
                encoding="utf-8",
            )
            rights = root / "rights.json"
            rights.write_text(
                json.dumps(
                    {
                        "dataset_license": "other",
                        "sources": {
                            "org/legal": {
                                "revision": revision,
                                "source_url": "https://example.test/legal",
                                "license": "public-domain",
                                "redistribution_allowed": True,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            blocklist = root / "blocklist" / "task"
            blocklist.mkdir(parents=True)
            digest = hashlib.sha256("blocked query".encode()).hexdigest()
            with gzip.open(blocklist / "query_text.sha256.gz", "wt", encoding="ascii") as handle:
                handle.write(digest + "\n")
            for name in ("corpus_text.sha256.gz", "evaluation_text.sha256.gz"):
                with gzip.open(blocklist / name, "wt", encoding="ascii") as handle:
                    handle.write(hashlib.sha256(name.encode()).hexdigest() + "\n")
            (root / "blocklist" / "manifest.json").write_text("{}\n", encoding="utf-8")
            output_train = root / "train.jsonl"
            output_provenance = root / "provenance.jsonl"
            manifest = root / "manifest.json"
            subprocess.check_call(
                [
                    "python",
                    str(ROOT / "scripts/build_public_legal_training_dataset.py"),
                    "--train",
                    str(train),
                    "--provenance",
                    str(provenance),
                    "--output-train",
                    str(output_train),
                    "--output-provenance",
                    str(output_provenance),
                    "--manifest-output",
                    str(manifest),
                    "--rights-config",
                    str(rights),
                    "--blocklist-root",
                    str(root / "blocklist"),
                ]
            )
            built = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(built["input_rows"], 3)
            self.assertEqual(built["output_rows"], 2)
            self.assertEqual(built["removed_benchmark_overlap_rows"], 1)
            self.assertTrue(built["release_eligible"])
            self.assertEqual(validate_public_rights(built, output_provenance), 2)


if __name__ == "__main__":
    unittest.main()
