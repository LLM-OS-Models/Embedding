from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CandidateSelectionTests(unittest.TestCase):
    def test_checkpoint_uses_minimum_eval_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for step in (40, 80):
                checkpoint = run / "v1" / f"checkpoint-{step}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
                (checkpoint / "adapter_config.json").write_text("{}")
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps(
                        {
                            "log_history": [
                                {"step": 40, "eval_loss": 0.5},
                                {"step": 80, "eval_loss": 0.7},
                            ]
                        }
                    )
                )
            output = subprocess.check_output(
                [
                    "python",
                    str(ROOT / "scripts/select_best_checkpoint.py"),
                    str(run),
                    "--print-path",
                ],
                text=True,
            ).strip()
            self.assertTrue(output.endswith("checkpoint-40"), output)

    def test_sionic_selection_requires_complete_nine_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, average, tasks in (("a", 0.8, 9), ("b", 0.9, 3)):
                folder = root / name
                folder.mkdir()
                (folder / "summary.json").write_text(
                    json.dumps(
                        {
                            "model": name,
                            "average": average if tasks == 9 else None,
                            "completed_tasks": tasks,
                            "scores": {str(i): average for i in range(tasks)},
                        }
                    )
                )
            output = subprocess.check_output(
                [
                    "python",
                    str(ROOT / "scripts/select_best_sionic_model.py"),
                    str(root),
                    "--print-model",
                ],
                text=True,
            ).strip()
            self.assertEqual(output, "a")

    def test_sionic_selection_excludes_run_level_disqualification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summaries = root / "summaries"
            outputs = root / "outputs"
            for run, average in (("clean-run", 0.80), ("leaked-run", 0.99)):
                folder = summaries / run
                folder.mkdir(parents=True)
                (folder / "summary.json").write_text(
                    json.dumps(
                        {
                            "model": f"artifacts/models/{run}-best-merged",
                            "average": average,
                            "completed_tasks": 9,
                            "scores": {str(i): average for i in range(9)},
                        }
                    )
                )
            marker = outputs / "leaked-run" / "DISQUALIFIED.json"
            marker.parent.mkdir(parents=True)
            marker.write_text('{"reason":"fixture"}\n')
            report_path = root / "selection.json"
            subprocess.run(
                [
                    "python",
                    str(ROOT / "scripts/select_best_sionic_model.py"),
                    str(summaries),
                    "--output",
                    str(report_path),
                    "--disqualification-root",
                    str(outputs),
                ],
                text=True,
                check=True,
                capture_output=True,
            )
            report = json.loads(report_path.read_text())
            self.assertTrue(report["best"]["model"].endswith("clean-run-best-merged"))
            self.assertEqual(report["excluded"][0]["model"], "artifacts/models/leaked-run-best-merged")


if __name__ == "__main__":
    unittest.main()
