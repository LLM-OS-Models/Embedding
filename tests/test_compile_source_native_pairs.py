from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/grounded_synthetic_query_factory/candidates.jsonl"


class CompileSourceNativePairsTests(unittest.TestCase):
    def test_compile_is_strict_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = []
            for suffix in ("a", "b"):
                train = root / f"train-{suffix}.jsonl"
                provenance = root / f"provenance-{suffix}.jsonl"
                manifest = root / f"manifest-{suffix}.json"
                subprocess.check_call(
                    [
                        "python",
                        str(ROOT / "scripts/compile_source_native_pairs.py"),
                        "--input",
                        str(FIXTURE),
                        "--output",
                        str(train),
                        "--provenance-output",
                        str(provenance),
                        "--manifest-output",
                        str(manifest),
                    ]
                )
                outputs.append((train.read_bytes(), provenance.read_bytes()))
                rows = [json.loads(line) for line in train.read_text().splitlines()]
                self.assertEqual(len(rows), 10)
                self.assertEqual(
                    set(rows[0]),
                    {"messages", "positive_messages", "negative_messages"},
                )
                self.assertEqual(len(rows[0]["negative_messages"]), 1)
            self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
