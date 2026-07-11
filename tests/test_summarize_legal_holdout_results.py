from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_legal_holdout_results import END, START, load_rows, markdown, update_readme


class SummarizeLegalHoldoutTests(unittest.TestCase):
    def test_latest_model_result_and_readme_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, score in enumerate((0.5, 0.6)):
                path = root / f"run-{index}"
                path.mkdir()
                (path / "summary.json").write_text(
                    json.dumps(
                        {
                            "protocol_id": "legal-source-document-heldout-i-v1",
                            "model": "Qwen/Qwen3-Embedding-8B",
                            "created_at_utc": f"2026-01-0{index + 1}T00:00:00Z",
                            "metrics": {
                                "ndcg_at_10": score,
                                "recall_at_10": score,
                                "mrr_at_10": score,
                                "recall_at_100": score,
                                "mean_positive_rank": 10.0,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            rows = load_rows(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["metrics"]["ndcg_at_10"], 0.6)
            rendered = markdown(rows)
            readme = root / "README.md"
            readme.write_text(f"before\n{START}\nold\n{END}\nafter\n", encoding="utf-8")
            update_readme(readme, rendered)
            self.assertIn("0.60000", readme.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
