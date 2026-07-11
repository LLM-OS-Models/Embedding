from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_mteb_korean_v1 import canonical_local_revision as official_revision
from scripts.evaluate_sionic9 import canonical_local_revision as sionic_revision


class LocalModelRevisionTests(unittest.TestCase):
    def test_final_weight_hash_overrides_adapter_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            model_sha = "a" * 64
            (model / "merge_report.json").write_text(
                json.dumps({"model": {"weights_sha256": model_sha}}),
                encoding="utf-8",
            )
            expected = "model-" + model_sha[:12]
            self.assertEqual(sionic_revision(str(model), "adapter-old"), expected)
            self.assertEqual(official_revision(str(model), "adapter-old"), expected)


if __name__ == "__main__":
    unittest.main()
