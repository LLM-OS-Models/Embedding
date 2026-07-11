from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_mining_manifest import manifest_matches
from scripts.mine_faiss_hard_negatives import local_model_weights_sha256


class LocalModelFingerprintTest(unittest.TestCase):
    def test_fingerprint_and_manifest_follow_weight_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "model-00001-of-00001.safetensors"
            shard.write_bytes(b"first")
            first = local_model_weights_sha256(str(root))
            manifest = {
                "model": str(root),
                "revision": "",
                "model_weights_sha256": first,
                "selection_strategy": "score_rank_quantiles",
                "candidate_pool_size": 24,
                "num_negatives": 7,
            }
            self.assertTrue(manifest_matches(manifest, str(root), ""))
            self.assertTrue(
                manifest_matches(
                    manifest, str(root), "", "score_rank_quantiles", 24, 7
                )
            )
            self.assertFalse(
                manifest_matches(manifest, str(root), "", "top_k", 24, 7)
            )
            shard.write_bytes(b"second")
            self.assertFalse(manifest_matches(manifest, str(root), ""))


if __name__ == "__main__":
    unittest.main()
