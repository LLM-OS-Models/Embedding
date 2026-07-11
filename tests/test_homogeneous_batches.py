from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HomogeneousBatchTests(unittest.TestCase):
    def test_every_emitted_batch_has_one_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            provenance = root / "provenance.jsonl"
            with train.open("w") as train_handle, provenance.open("w") as prov_handle:
                for index in range(11):
                    row = {
                        "messages": [{"role": "user", "content": f"q{index}"}],
                        "positive_messages": [[{"role": "user", "content": f"p{index}"}]],
                        "negative_messages": [[{"role": "user", "content": f"n{index}"}]],
                    }
                    train_handle.write(json.dumps(row) + "\n")
                    prov_handle.write(
                        json.dumps({"source_id": "a" if index < 6 else "b"}) + "\n"
                    )
            output = root / "ordered.jsonl"
            audit = root / "ordered.provenance.jsonl"
            manifest = root / "manifest.json"
            subprocess.check_call(
                [
                    "python",
                    str(ROOT / "scripts/build_homogeneous_batches.py"),
                    "--train",
                    str(train),
                    "--provenance",
                    str(provenance),
                    "--output",
                    str(output),
                    "--provenance-output",
                    str(audit),
                    "--manifest-output",
                    str(manifest),
                    "--batch-size",
                    "2",
                    "--length-bucketed",
                ]
            )
            rows = [json.loads(line) for line in audit.read_text().splitlines()]
            self.assertEqual(len(rows), 10)
            for start in range(0, len(rows), 2):
                sources = {row["source_id"] for row in rows[start : start + 2]}
                self.assertEqual(len(sources), 1)
            report = json.loads(manifest.read_text())
            self.assertEqual(report["output_rows"], 10)
            self.assertEqual(sum(report["dropped_source_remainders"].values()), 1)
            self.assertTrue(report["length_bucketed"])
            for start in range(0, len(rows), 2):
                batch = rows[start : start + 2]
                self.assertEqual(
                    {row["homogeneous_batch"]["batch_length_proxy_min"] for row in batch},
                    {batch[0]["homogeneous_batch"]["batch_length_proxy_min"]},
                )


if __name__ == "__main__":
    unittest.main()
