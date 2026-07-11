from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_training_benchmark_overlap import audit, text_digest


def training_row(query: str, positive: str, negative: str) -> dict:
    return {
        "messages": [{"role": "user", "content": query}],
        "positive_messages": [[{"role": "user", "content": positive}]],
        "negative_messages": [[{"role": "user", "content": negative}]],
    }


class AuditTrainingBenchmarkOverlapTest(unittest.TestCase):
    def write_hash(self, root: Path, task: str, name: str, text: str) -> None:
        path = root / task / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="ascii") as handle:
            handle.write(text_digest(text).hex() + "\n")

    def write_data(
        self, root: Path, rows: list[dict]
    ) -> tuple[Path, Path]:
        train = root / "train.jsonl"
        provenance = root / "provenance.jsonl"
        train.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        provenance.write_text(
            "".join(
                json.dumps({"source_id": "fixture", "row_index": index}) + "\n"
                for index in range(len(rows))
            ),
            encoding="utf-8",
        )
        return train, provenance

    def test_instruction_stripped_query_match_is_critical_without_raw_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blocklist = root / "blocklist"
            self.write_hash(
                blocklist, "suite/task/test", "query_text.sha256.gz", "평가 질문"
            )
            self.write_hash(
                blocklist, "suite/task/test", "corpus_text.sha256.gz", "평가 문서"
            )
            train, provenance = self.write_data(
                root,
                [
                    training_row(
                        "Instruct: 검색하세요\nQuery: 평가 질문",
                        "평가 문서",
                        "완전히 다른 문서",
                    )
                ],
            )
            report = audit(train, provenance, blocklist)

            self.assertEqual(report["rows"], 1)
            self.assertEqual(
                report["unique_critical_query_or_evaluation_matches"], 1
            )
            self.assertEqual(report["unique_retrieval_corpus_matches"], 1)
            self.assertEqual(
                report["status"], "critical_query_or_evaluation_text_overlap"
            )
            serialized = json.dumps(report, ensure_ascii=False)
            self.assertNotIn("평가 질문", serialized)
            self.assertNotIn("평가 문서", serialized)

    def test_corpus_only_overlap_is_disclosed_but_not_critical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blocklist = root / "blocklist"
            self.write_hash(
                blocklist, "suite/task/test", "corpus_text.sha256.gz", "공유 코퍼스"
            )
            train, provenance = self.write_data(
                root,
                [training_row("새 학습 질문", "공유 코퍼스", "다른 음성 문서")],
            )
            report = audit(train, provenance, blocklist)

            self.assertEqual(report["unique_critical_query_or_evaluation_matches"], 0)
            self.assertEqual(report["unique_retrieval_corpus_matches"], 1)
            self.assertEqual(report["status"], "pass_with_retrieval_corpus_exposure")


if __name__ == "__main__":
    unittest.main()
