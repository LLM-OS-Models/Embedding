import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RecordCampaignResultTests(unittest.TestCase):
    def test_registry_and_readme_are_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sionic = root / "sionic.json"
            official = root / "official.json"
            registry = root / "reports/campaign-results.json"
            readme = root / "README.md"
            sionic.write_text(
                json.dumps({"completed_tasks": 9, "average": 0.8, "scores": {}})
            )
            official.write_text(
                json.dumps(
                    {
                        "complete": True,
                        "completed_tasks": 6,
                        "mean_task_leaderboard_points": 80.1,
                        "mean_task_type_leaderboard_points": 79.2,
                    }
                )
            )
            readme.write_text(
                "before\n<!-- CAMPAIGN_RESULTS_START -->\nold\n"
                "<!-- CAMPAIGN_RESULTS_END -->\nafter\n"
            )
            subprocess.run(
                [
                    "python",
                    str(ROOT / "scripts/record_campaign_result.py"),
                    "--stage",
                    "pilot-best",
                    "--model",
                    "artifacts/models/fixture",
                    "--repo-id",
                    "LLM-OS-Models/fixture",
                    "--sionic-summary",
                    str(sionic),
                    "--official-summary",
                    str(official),
                    "--readme",
                    str(readme),
                    "--registry",
                    str(registry),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            saved = json.loads(registry.read_text())
            self.assertEqual(saved["results"]["pilot-best"]["sionic_average"], 0.8)
            rendered = readme.read_text()
            self.assertIn("| pilot-best |", rendered)
            self.assertIn("+0.00700", rendered)
            self.assertNotIn("\nold\n", rendered)


if __name__ == "__main__":
    unittest.main()
