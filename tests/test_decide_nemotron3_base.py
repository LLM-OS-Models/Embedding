from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.decide_nemotron3_base import REVISIONS, build


class DecideNemotron3BaseTest(unittest.TestCase):
    def test_adopts_only_complete_raw_win_with_clean_guards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sionic = root / "sionic"
            legal = root / "legal"
            multi = root / "multi"
            for directory in (sionic, legal, multi):
                directory.mkdir()
            (sionic / "summary.json").write_text(
                json.dumps(
                    {
                        "requested_revision": REVISIONS["nemotron3"],
                        "average": 0.8,
                        "completed_tasks": 9,
                        "total_protocol_tasks": 9,
                        "scores": {f"task-{index}": 0.8 for index in range(9)},
                    }
                )
            )
            clean = {
                "nemotron3": (0.80, 0.81, 0.80, 0.82),
                "qwen3": (0.805, 0.81, 0.805, 0.815),
                "comsat": (0.807, 0.812, 0.808, 0.816),
            }
            for label, (legal_score, macro, finance, knowledge) in clean.items():
                legal_dir = legal / label
                multi_dir = multi / label
                legal_dir.mkdir()
                multi_dir.mkdir()
                (legal_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "requested_revision": REVISIONS[label],
                            "metrics": {"ndcg_at_10": legal_score},
                        }
                    )
                )
                (multi_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "requested_revision": REVISIONS[label],
                            "metrics": {"macro_domain_ndcg_at_10": macro},
                            "domain_metrics": {
                                "finance": {"ndcg_at_10": finance},
                                "knowledge": {"ndcg_at_10": knowledge},
                            },
                        }
                    )
                )
            args = SimpleNamespace(
                sionic_dir=sionic,
                legal_dir=legal,
                multidomain_dir=multi,
                target=0.793,
                clean_guard=0.01,
                domain_guard=0.015,
                max_short_adaptation_deficit=0.02,
            )
            report = build(args)
            self.assertEqual(
                report["decision"], "adopt_nemotron3_raw_and_run_short_public_lora"
            )
            self.assertTrue(report["gates"]["clean_guard_pass"])
            partial = json.loads((sionic / "summary.json").read_text())
            partial["completed_tasks"] = 8
            (sionic / "summary.json").write_text(json.dumps(partial))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                build(args)


if __name__ == "__main__":
    unittest.main()
