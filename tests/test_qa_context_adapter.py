from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.build_performance_mix import iter_qa_context


class QaContextAdapterTest(unittest.TestCase):
    def test_emits_deterministic_bootstrap_negatives_from_other_contexts(self) -> None:
        rows = [
            {"question": f"질문 {index} 내용", "context": f"정답 문맥 {index} 내용입니다"}
            for index in range(6)
        ]
        source = {
            "repo_id": "fixture/korquad",
            "schema": {
                "question_field": "question",
                "context_field": "context",
            },
            "bootstrap_negatives": 3,
        }

        def fake_loader(_source, _seed, shuffle=True):
            return list(reversed(rows)) if shuffle else list(rows)

        with patch(
            "scripts.build_performance_mix.load_hf_dataset_stream",
            side_effect=fake_loader,
        ):
            first = list(iter_qa_context(source, seed=42))
            second = list(iter_qa_context(source, seed=42))

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        for query, positive, negatives in first:
            self.assertTrue(query.startswith("질문"))
            self.assertEqual(len(negatives), 3)
            self.assertNotIn(positive, negatives)
            self.assertEqual(len(negatives), len(set(negatives)))

    def test_requires_more_contexts_than_requested_negatives(self) -> None:
        rows = [{"question": "충분한 질문", "context": "하나뿐인 정답 문맥"}]
        source = {
            "repo_id": "fixture/tiny",
            "schema": {
                "question_field": "question",
                "context_field": "context",
            },
            "bootstrap_negatives": 1,
        }
        with patch(
            "scripts.build_performance_mix.load_hf_dataset_stream",
            return_value=rows,
        ):
            with self.assertRaisesRegex(RuntimeError, "more than 1 unique contexts"):
                list(iter_qa_context(source, seed=42))


if __name__ == "__main__":
    unittest.main()
