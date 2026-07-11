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


if __name__ == "__main__":
    unittest.main()
