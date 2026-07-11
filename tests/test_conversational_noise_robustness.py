from __future__ import annotations

import unittest

from scripts.evaluate_conversational_noise_robustness import (
    build_noise_documents,
    rank_with_appended_noise,
    summarize_condition,
)


class ConversationalNoiseRobustnessTest(unittest.TestCase):
    def test_noise_documents_are_deterministic_and_sized(self) -> None:
        first = build_noise_documents(10_000)
        second = build_noise_documents(10_000)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 500)
        self.assertEqual(len({row[0] for row in first}), 500)
        self.assertEqual(len({row[1] for row in first}), 500)

    def test_condition_summary(self) -> None:
        summary = summarize_condition([1, 2, 11, 101], [2, 7, 12, 200])
        self.assertAlmostEqual(summary["recall_at_10"], 0.5)
        self.assertAlmostEqual(summary["noise_intrusion_at_1"], 0.0)
        self.assertAlmostEqual(summary["noise_intrusion_at_5"], 0.25)
        self.assertAlmostEqual(summary["noise_intrusion_at_10"], 0.5)
        self.assertAlmostEqual(summary["median_positive_rank"], 6.5)

    def test_exact_rank_and_clean_before_noise_tie_contract(self) -> None:
        import torch

        clean = torch.tensor([[0.9, 0.8, 0.7], [0.1, 0.5, 0.2]])
        noise = torch.tensor([[0.95, 0.9], [0.5, 0.4]])
        positives = torch.tensor([0, 1])
        positive_ranks, noise_ranks = rank_with_appended_noise(clean, noise, positives)
        self.assertEqual(positive_ranks.tolist(), [2, 1])
        self.assertEqual(noise_ranks.tolist(), [1, 2])

        clean_only_ranks, clean_noise_ranks = rank_with_appended_noise(
            clean, noise[:, :0], positives
        )
        self.assertEqual(clean_only_ranks.tolist(), [1, 1])
        self.assertIsNone(clean_noise_ranks)


if __name__ == "__main__":
    unittest.main()
