from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

from scripts.build_performance_mix import CriticalTextBlocklist, benchmark_text_digest


class PerformanceMixBlocklistTest(unittest.TestCase):
    def write_hash(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="ascii") as handle:
            handle.write(benchmark_text_digest(text).hex() + "\n")

    def test_retrieval_query_is_always_critical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_hash(
                root / "suite" / "MIRACL" / "ko" / "dev" / "query_text.sha256.gz",
                "평가 검색 질문",
            )
            blocklist = CriticalTextBlocklist(root)
            reasons = blocklist.critical_reasons(
                ["평가 검색 질문"], trained_on_tasks=["MIRACL"]
            )
            self.assertEqual(reasons["retrieval_eval_query_text"], 1)

    def test_declared_nonretrieval_train_family_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_hash(
                root
                / "official"
                / "KLUE-STS"
                / "default"
                / "validation"
                / "evaluation_text.sha256.gz",
                "공식 데이터 문장",
            )
            blocklist = CriticalTextBlocklist(root)
            self.assertFalse(
                blocklist.critical_reasons(
                    ["공식 데이터 문장"], trained_on_tasks=["KLUE-STS"]
                )
            )
            reasons = blocklist.critical_reasons(
                ["공식 데이터 문장"], trained_on_tasks=[]
            )
            self.assertEqual(reasons["undeclared_nonretrieval_text:KLUE-STS"], 1)


if __name__ == "__main__":
    unittest.main()
