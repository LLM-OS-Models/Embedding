import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SelectBestCheckpointTests(unittest.TestCase):
    def test_adapter_and_full_checkpoint_kinds(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            adapter = run / "nested" / "checkpoint-10"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            (adapter / "adapter_config.json").write_text("{}")
            full = run / "nested" / "checkpoint-20"
            (full / "1_Pooling").mkdir(parents=True)
            (full / "model-00001-of-00001.safetensors").write_bytes(b"full")
            (full / "config.json").write_text("{}")
            (full / "modules.json").write_text("[]")
            (full / "1_Pooling/config.json").write_text("{}")
            state = {
                "log_history": [
                    {"step": 10, "eval_loss": 0.3},
                    {"step": 20, "eval_loss": 0.2},
                ]
            }
            (adapter / "trainer_state.json").write_text(json.dumps(state))
            (full / "trainer_state.json").write_text(json.dumps(state))
            better_adapter = run / "second-version" / "checkpoint-10"
            better_adapter.mkdir(parents=True)
            (better_adapter / "adapter_model.safetensors").write_bytes(b"adapter-2")
            (better_adapter / "adapter_config.json").write_text("{}")
            (better_adapter / "trainer_state.json").write_text(
                json.dumps({"log_history": [{"step": 10, "eval_loss": 0.1}]})
            )

            def select(kind: str) -> str:
                return subprocess.check_output(
                    [
                        "python",
                        str(ROOT / "scripts/select_best_checkpoint.py"),
                        str(run),
                        "--checkpoint-kind",
                        kind,
                        "--print-path",
                    ],
                    text=True,
                ).strip()

            self.assertEqual(select("adapter"), str(better_adapter))
            self.assertEqual(select("full"), str(full))
            self.assertEqual(select("auto"), str(better_adapter))


if __name__ == "__main__":
    unittest.main()
