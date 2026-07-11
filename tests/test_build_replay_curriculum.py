import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def training_row(value: str) -> dict:
    return {
        "messages": [{"role": "user", "content": value}],
        "positive_messages": [{"role": "assistant", "content": value + "+"}],
        "negative_messages": [[{"role": "assistant", "content": value + "-"}]],
    }


class ReplayCurriculumTest(unittest.TestCase):
    def test_exact_mix_and_homogeneous_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = {}
            for role, sources in {
                "primary": ["legal-a"] * 2 + ["legal-b"] * 2,
                "replay": ["general-a"] * 2 + ["general-b"] * 2,
            }.items():
                train = root / f"{role}.jsonl"
                provenance = root / f"{role}.provenance.jsonl"
                train.write_text(
                    "".join(
                        json.dumps(training_row(f"{role}-{index}")) + "\n"
                        for index in range(4)
                    )
                )
                provenance.write_text(
                    "".join(json.dumps({"source_id": source}) + "\n" for source in sources)
                )
                inputs[role] = (train, provenance)

            output = root / "out.jsonl"
            provenance_output = root / "out.provenance.jsonl"
            manifest = root / "manifest.json"
            subprocess.run(
                [
                    "python",
                    str(ROOT / "scripts/build_replay_curriculum.py"),
                    "--primary-train",
                    str(inputs["primary"][0]),
                    "--primary-provenance",
                    str(inputs["primary"][1]),
                    "--primary-rows",
                    "4",
                    "--replay-train",
                    str(inputs["replay"][0]),
                    "--replay-provenance",
                    str(inputs["replay"][1]),
                    "--replay-rows",
                    "4",
                    "--output",
                    str(output),
                    "--provenance-output",
                    str(provenance_output),
                    "--manifest-output",
                    str(manifest),
                    "--batch-size",
                    "2",
                    "--seed",
                    "7",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            audit = json.loads(manifest.read_text())
            self.assertEqual(audit["output_rows"], 8)
            self.assertEqual(audit["role_counts"], {"primary": 4, "replay": 4})
            rows = [json.loads(line) for line in provenance_output.read_text().splitlines()]
            for start in range(0, len(rows), 2):
                batch = rows[start : start + 2]
                self.assertEqual(len({row["curriculum_batch"]["role"] for row in batch}), 1)
                self.assertEqual(
                    len({row["curriculum_batch"]["source_id"] for row in batch}), 1
                )


if __name__ == "__main__":
    unittest.main()
