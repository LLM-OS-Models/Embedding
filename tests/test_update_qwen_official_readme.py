from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.update_qwen_official_readme import (
    QWEN,
    result_row,
    update_readme,
    validate,
)


class UpdateQwenOfficialReadmeTest(unittest.TestCase):
    def test_validates_and_replaces_exact_row(self) -> None:
        summary = {
            "model": QWEN,
            "requested_revision": "4e423935c619ae4df87b646a3ce949610c66241c",
            "protocol_id": "mteb-korean-v1-mteb-2.18.0",
            "complete": True,
            "completed_tasks": 6,
            "mean_task_leaderboard_points": 75.1,
            "mean_task_type_leaderboard_points": 72.2,
            "means_by_type": {"Retrieval": 0.731},
            "environment": {"registered_loader": True},
        }
        comparison = {
            "local": {
                "model": QWEN,
                "revision": "4e423935c619ae4df87b646a3ce949610c66241c",
                "rank_borda_if_inserted": 4,
            },
            "official_rank_reproduction": {"matched": 137, "total": 137},
        }
        validate(summary, comparison)
        row = result_row(summary, comparison)
        with tempfile.TemporaryDirectory() as temporary:
            readme = Path(temporary) / "README.md"
            readme.write_text(
                f"before\n| 비교 | `{QWEN}` | — | — | — | — | — | old |\nafter\n"
            )
            update_readme(readme, row)
            text = readme.read_text()
            self.assertIn("**4 if inserted**", text)
            self.assertIn("**73.10**", text)


if __name__ == "__main__":
    unittest.main()
