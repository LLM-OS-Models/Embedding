from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.build_fa2_probe_subset import build, scan_aligned


def canonical_sha(row: dict) -> str:
    payload = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def strict_row(batch_index: int, row_index: int) -> dict:
    marker = f"batch-{batch_index}-row-{row_index}"
    return {
        "messages": [{"role": "user", "content": f"query {marker}"}],
        "positive_messages": [
            [{"role": "user", "content": f"positive {marker}"}]
        ],
        "negative_messages": [
            [{"role": "user", "content": f"negative {marker}"}]
        ],
    }


def write_fixture(root: Path) -> tuple[Path, Path]:
    train = root / "train.jsonl"
    provenance = root / "provenance.jsonl"
    # Every source spans both global length halves.  The 6:4:2 batch ratio has
    # an exact 2:1:1 Hamilton allocation when four probe batches are selected.
    specs = [
        ("a", 10),
        ("b", 15),
        ("a", 20),
        ("b", 25),
        ("a", 30),
        ("c", 35),
        ("c", 65),
        ("a", 70),
        ("b", 75),
        ("a", 80),
        ("b", 85),
        ("a", 90),
    ]
    output_row_index = 0
    with train.open("w", encoding="utf-8") as train_handle, provenance.open(
        "w", encoding="utf-8"
    ) as provenance_handle:
        for batch_index, (source, length) in enumerate(specs):
            for batch_row in range(2):
                row = strict_row(batch_index, batch_row)
                train_handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                provenance_handle.write(
                    json.dumps(
                        {
                            "row_index": output_row_index,
                            "row_sha256": canonical_sha(row),
                            "source_id": source,
                            "homogeneous_batch": {
                                "batch_index": batch_index,
                                "batch_size": 2,
                                "source_id": source,
                                "output_row_index": output_row_index,
                                "length_proxy": length,
                                "batch_length_proxy_min": length,
                                "batch_length_proxy_max": length,
                            },
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                output_row_index += 1
    return train, provenance


def arguments(train: Path, provenance: Path, output_dir: Path) -> Namespace:
    return Namespace(
        train=train,
        provenance=provenance,
        output_dir=output_dir,
        batch_size=2,
        probe_steps=2,
        gradient_accumulation_steps=2,
        training_max_length=512,
        seed=42,
    )


class Fa2ProbeSubsetTests(unittest.TestCase):
    def test_preserves_source_quota_and_balances_each_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, provenance = write_fixture(root)
            report = build(arguments(train, provenance, root / "out"))

            self.assertEqual(report["parameters"]["selected_batches"], 4)
            self.assertEqual(report["parameters"]["selected_rows"], 8)
            self.assertEqual(
                report["representativeness"]["source_selected_batches"],
                {"a": 2, "b": 1, "c": 1},
            )
            for step in report["optimizer_step_plan"]:
                self.assertEqual(set(step["length_strata"]), {0, 1})
                self.assertEqual(len(step["original_batch_indices"]), 2)

            projected = [
                json.loads(line)
                for line in (root / "out/provenance.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["fa2_probe_subset"]["subset_row_index"] for row in projected],
                list(range(8)),
            )
            for start in range(0, len(projected), 2):
                batch = projected[start : start + 2]
                self.assertEqual(len({row["source_id"] for row in batch}), 1)
                self.assertEqual(
                    len(
                        {
                            row["fa2_probe_subset"]["original_batch_index"]
                            for row in batch
                        }
                    ),
                    1,
                )

            original_lines = set(train.read_bytes().splitlines(keepends=True))
            selected_lines = (root / "out/train.jsonl").read_bytes().splitlines(
                keepends=True
            )
            self.assertEqual(len(selected_lines), 8)
            self.assertTrue(all(line in original_lines for line in selected_lines))

    def test_outputs_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, provenance = write_fixture(root)
            first = build(arguments(train, provenance, root / "first"))
            second = build(arguments(train, provenance, root / "second"))
            self.assertEqual(
                first["files"]["train.jsonl"]["sha256"],
                second["files"]["train.jsonl"]["sha256"],
            )
            self.assertEqual(
                first["files"]["provenance.jsonl"]["sha256"],
                second["files"]["provenance.jsonl"]["sha256"],
            )
            self.assertEqual(first["optimizer_step_plan"], second["optimizer_step_plan"])

    def test_rejects_incomplete_homogeneous_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, provenance = write_fixture(root)
            train.write_bytes(b"".join(train.read_bytes().splitlines(keepends=True)[:-1]))
            provenance.write_bytes(
                b"".join(provenance.read_bytes().splitlines(keepends=True)[:-1])
            )
            with self.assertRaisesRegex(ValueError, "has 1 rows, expected 2"):
                scan_aligned(train, provenance, 2)


if __name__ == "__main__":
    unittest.main()
