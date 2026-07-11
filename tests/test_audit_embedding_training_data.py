from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AuditEmbeddingTrainingDataTest(unittest.TestCase):
    def test_aligned_contract_and_style_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            provenance = root / "provenance.jsonl"
            output = root / "audit.json"
            train_lines = []
            provenance_lines = []
            for index in range(2):
                query = "가, 나, 다" if index == 0 else "무엇을 찾나요?"
                formatted = (
                    f"Task\nQuery:{query}"
                    if index == 0
                    else f"Instruct: Retrieve evidence. Query: {query}"
                )
                row = {
                    "messages": [{"role": "user", "content": formatted}],
                    "positive_messages": [
                        [{"role": "user", "content": f"positive {index}"}]
                    ],
                    "negative_messages": [
                        [{"role": "user", "content": f"negative {index}"}]
                    ],
                }
                compact = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                train_lines.append(compact)
                provenance_lines.append(
                    json.dumps(
                        {
                            "row_sha256": hashlib.sha256(compact.encode()).hexdigest(),
                            "source_id": "fixture",
                            "trained_on_tasks": [],
                            "homogeneous_batch": {
                                "batch_index": 0,
                                "batch_size": 2,
                                "source_id": "fixture",
                                "output_row_index": index,
                            },
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            train.write_text("\n".join(train_lines) + "\n")
            provenance.write_text("\n".join(provenance_lines) + "\n")
            subprocess.check_call(
                [
                    "python",
                    str(ROOT / "scripts/audit_embedding_training_data.py"),
                    "--train",
                    str(train),
                    "--provenance",
                    str(provenance),
                    "--output",
                    str(output),
                    "--expected-batch-size",
                    "2",
                ]
            )
            report = json.loads(output.read_text())
            self.assertEqual(report["rows"], 2)
            self.assertEqual(report["query_style_heuristic"]["comma_keyword_list"], 1)
            self.assertEqual(report["query_style_heuristic"]["natural_question"], 1)
            self.assertEqual(
                report["instruction_counts"]["Instruct: Retrieve evidence."], 1
            )
            self.assertEqual(report["contract_checks"]["status"], "pass")

    def test_latest_curriculum_batch_contract_overrides_stale_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            provenance = root / "provenance.jsonl"
            output = root / "audit.json"
            train_rows = []
            provenance_rows = []
            for index in range(2):
                row = {
                    "messages": [{"role": "user", "content": f"질문 {index} 내용"}],
                    "positive_messages": [
                        [{"role": "user", "content": f"정답 문서 {index}"}]
                    ],
                    "negative_messages": [
                        [{"role": "user", "content": f"오답 문서 {index}"}]
                    ],
                }
                train_rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                provenance_rows.append(
                    json.dumps(
                        {
                            "source_id": "fixture",
                            "homogeneous_batch": {
                                "batch_index": 99,
                                "batch_size": 2,
                                "source_id": "fixture",
                                "output_row_index": 99 + index,
                            },
                            "multidomain_curriculum_batch": {
                                "batch_index": 0,
                                "batch_size": 2,
                                "role": "general",
                                "source_id": "fixture",
                                "output_row_index": index,
                            },
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            train.write_text("\n".join(train_rows) + "\n", encoding="utf-8")
            provenance.write_text(
                "\n".join(provenance_rows) + "\n", encoding="utf-8"
            )
            subprocess.check_call(
                [
                    "python",
                    str(ROOT / "scripts/audit_embedding_training_data.py"),
                    "--train",
                    str(train),
                    "--provenance",
                    str(provenance),
                    "--output",
                    str(output),
                    "--expected-batch-size",
                    "2",
                ]
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["contract_checks"]["status"], "pass")
            self.assertEqual(
                report["contract_checks"]["batch_contract_field_counts"],
                {"multidomain_curriculum_batch": 2},
            )


if __name__ == "__main__":
    unittest.main()
