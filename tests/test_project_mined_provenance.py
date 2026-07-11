from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectMinedProvenanceTests(unittest.TestCase):
    def test_dropped_rows_preserve_contiguous_output_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provenance = root / "input.jsonl"
            audit = root / "audit.jsonl"
            output = root / "output.jsonl"
            manifest = root / "manifest.json"
            provenance.write_text(
                "".join(json.dumps({"source_id": "a", "id": i}) + "\n" for i in range(3))
            )
            audit.write_text(
                "\n".join(
                    [
                        json.dumps({"input_row_index": 0, "output_row_index": 0}),
                        json.dumps({"input_row_index": 1, "output_row_index": None}),
                        json.dumps({"input_row_index": 2, "output_row_index": 1}),
                    ]
                )
                + "\n"
            )
            subprocess.check_call(
                [
                    "python",
                    str(ROOT / "scripts/project_mined_provenance.py"),
                    "--input-provenance",
                    str(provenance),
                    "--mining-audit",
                    str(audit),
                    "--output",
                    str(output),
                    "--manifest-output",
                    str(manifest),
                ]
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["id"] for row in rows], [0, 2])
            self.assertEqual(json.loads(manifest.read_text())["output_rows"], 2)


if __name__ == "__main__":
    unittest.main()
