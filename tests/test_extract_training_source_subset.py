from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.extract_training_source_subset import extract, row_sha


class SourceSubsetTests(unittest.TestCase):
    def test_extracts_selected_sources_and_reindexes_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            provenance = root / "provenance.jsonl"
            rows = [{"value": index} for index in range(3)]
            train.write_text("".join(json.dumps(row) + "\n" for row in rows))
            provenance.write_text(
                "".join(
                    json.dumps(
                        {
                            "row_index": index,
                            "row_sha256": row_sha(row),
                            "source_id": ("keep" if index != 1 else "drop"),
                        }
                    )
                    + "\n"
                    for index, row in enumerate(rows)
                )
            )
            output = root / "out"
            report = extract(
                Namespace(
                    train=train,
                    provenance=provenance,
                    output_dir=output,
                    source=["keep"],
                    phase="fixture",
                    expected_rows=2,
                )
            )
            self.assertEqual(report["built_rows"], 2)
            projected = [json.loads(line) for line in (output / "provenance.jsonl").read_text().splitlines()]
            self.assertEqual([row["row_index"] for row in projected], [0, 1])
            self.assertEqual([row["parent_row_index"] for row in projected], [0, 2])
