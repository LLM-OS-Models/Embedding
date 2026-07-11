from __future__ import annotations

import unittest

from scripts.build_performance_mix import cleaned_example, semantic_query_body


class PerformanceMixQueryValidationTest(unittest.TestCase):
    def test_extracts_inline_and_multiline_instruction_query(self) -> None:
        self.assertEqual(
            semantic_query_body("Instruct: Find evidence. Query: 실제 질문"),
            "실제 질문",
        )
        self.assertEqual(
            semantic_query_body("Instruct: Find evidence\nQuery: 다른 질문"),
            "다른 질문",
        )

    def test_rejects_short_body_hidden_by_long_instruction(self) -> None:
        source = {"require_hangul": True}
        defaults = {
            "min_query_chars": 4,
            "min_document_chars": 8,
            "max_document_chars": 100,
            "require_hangul": True,
            "query_instruction": "default",
        }
        cleaned = cleaned_example(
            (
                "Instruct: A very long instruction. Query: 가",
                "충분히 긴 정답 문서",
                ["충분히 긴 오답 문서"],
            ),
            source,
            defaults,
            seed=42,
            negatives_per_row=1,
        )
        self.assertEqual(cleaned[0], None)
        self.assertEqual(cleaned[1], "missing_or_short")


if __name__ == "__main__":
    unittest.main()
