from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.preserve_best_checkpoint import preserve_current_best


class PreserveBestCheckpointTest(unittest.TestCase):
    def test_preserves_minimum_loss_outside_trainer_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            version = root / "v1-fixture"
            version.mkdir(parents=True)
            (version / "logging.jsonl").write_text("{}\n")
            for step, loss in ((40, 0.4), (80, 0.2), (120, 0.3)):
                checkpoint = version / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "adapter_model.safetensors").write_bytes(
                    f"weight-{step}".encode()
                )
                (checkpoint / "adapter_config.json").write_text("{}")
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps({"log_history": [{"step": step, "eval_loss": loss}]})
                )
            report = preserve_current_best(root)
            self.assertEqual(report["step"], 80)
            destination = root / "v1-fixture-preserved" / "checkpoint-80"
            self.assertTrue((destination / "adapter_model.safetensors").is_file())
            self.assertTrue((destination.parent / "logging.jsonl").is_file())
            (destination.parent / "preservation.json").unlink()
            second = preserve_current_best(root)
            self.assertEqual(second["status"], "already_preserved")
            evidence = json.loads(
                (destination.parent / "preservation.json").read_text()
            )
            self.assertEqual(evidence["step"], 80)
            self.assertEqual(evidence["eval_loss"], 0.2)


if __name__ == "__main__":
    unittest.main()
