from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mine_faiss_hard_negatives", ROOT / "scripts/mine_faiss_hard_negatives.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FaissSelectionTests(unittest.TestCase):
    def test_teacher_request_sampling_is_seeded_and_without_replacement(self) -> None:
        first = MODULE.deterministic_sample_indices(1000, 50, 42)
        second = MODULE.deterministic_sample_indices(1000, 50, 42)
        different = MODULE.deterministic_sample_indices(1000, 50, 43)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 50)
        self.assertNotEqual(first, different)

    def test_ivf_index_is_persisted_and_resumed(self) -> None:
        rng = np.random.default_rng(42)
        corpus = rng.normal(size=(200, 8)).astype(np.float32)
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
        args = SimpleNamespace(
            faiss_threads=2,
            nlist=8,
            nprobe=2,
            seed=42,
            training_points=200,
            add_block_size=64,
        )
        namespace = {"fixture": True}
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            index, config, resumed = MODULE.build_or_resume_index(
                corpus, 8, args, work, namespace
            )
            self.assertFalse(resumed)
            self.assertEqual(index.ntotal, 200)
            _, ids = index.search(corpus[:2], 4)
            self.assertEqual(ids.shape, (2, 4))
            second, second_config, resumed = MODULE.build_or_resume_index(
                corpus, 8, args, work, namespace
            )
            self.assertTrue(resumed)
            self.assertEqual(second.ntotal, 200)
            self.assertEqual(config, second_config)

    def test_filters_own_query_match_and_above_relative_threshold(self) -> None:
        query = np.asarray([1.0, 0.0], dtype=np.float32)
        corpus = np.asarray(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.8, 0.6],
                [0.6, 0.8],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
        positive_score, threshold, selected, exclusions = MODULE.select_candidates(
            query=query,
            positive=corpus[0],
            indices=np.asarray([0, 1, 2, 3, 4]),
            corpus_embeddings=corpus,
            own_index=0,
            query_match_index=1,
            ratio=0.95,
            pool_size=3,
        )
        self.assertAlmostEqual(positive_score, 1.0)
        self.assertAlmostEqual(threshold, 0.95)
        self.assertEqual([index for index, _ in selected], [2, 3, 4])
        self.assertEqual(exclusions["own_positive"], 1)
        self.assertEqual(exclusions["query_document_exact_match"], 1)

    def test_teacher_pool_keeps_high_scoring_candidates_for_reranker_judgment(self) -> None:
        query = np.asarray([1.0, 0.0], dtype=np.float32)
        corpus = np.asarray(
            [[1.0, 0.0], [0.999, 0.001], [0.8, 0.6], [0.0, 1.0]],
            dtype=np.float32,
        )
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
        positive_score, selected, exclusions = MODULE.select_unfiltered_candidates(
            query=query,
            positive=corpus[0],
            indices=np.asarray([0, 1, 2, 3]),
            corpus_embeddings=corpus,
            own_index=0,
            query_match_index=-1,
            pool_size=3,
        )
        self.assertAlmostEqual(positive_score, 1.0)
        self.assertEqual([index for index, _ in selected], [1, 2, 3])
        self.assertEqual(exclusions["own_positive"], 1)

    def test_teacher_request_matches_strict_score_cache_input_contract(self) -> None:
        row = SimpleNamespace(
            query="질의",
            positive="정답 문서",
            positive_normalized="정답 문서",
        )
        corpus = [
            SimpleNamespace(text="후보 1", sha256="1" * 64),
            SimpleNamespace(text="후보 2", sha256="2" * 64),
        ]
        request = MODULE.teacher_request_row(
            7, row, corpus, 0.9, [(0, 0.8), (1, 0.7)]
        )
        from scripts.cache_qwen3_reranker_scores import parse_input_row

        parsed = parse_input_row(
            request, max_documents_per_row=201, max_text_characters=1_000_000
        )
        self.assertEqual(parsed.generated_id[:9], "faiss-kd-")
        self.assertEqual(len(parsed.candidates), 2)

    def test_pool_selection_spans_score_ranks_and_is_deterministic(self) -> None:
        pool = [(index, 1.0 - index / 100) for index in range(24)]
        selected, indices = MODULE.select_from_pool(
            pool, 7, "score_rank_quantiles", seed=42, row_index=3
        )
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 23)
        self.assertEqual(len(indices), 7)
        self.assertEqual(selected, [pool[index] for index in indices])

        first, first_indices = MODULE.select_from_pool(
            pool, 7, "hash_sample_from_top_pool", seed=42, row_index=3
        )
        second, second_indices = MODULE.select_from_pool(
            pool, 7, "hash_sample_from_top_pool", seed=42, row_index=3
        )
        self.assertEqual((first, first_indices), (second, second_indices))


if __name__ == "__main__":
    unittest.main()
