from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.build_multidomain_curriculum import build, parse_component


def row(value: str) -> str:
    return json.dumps(
        {
            "messages": [{"role": "user", "content": value}],
            "positive_messages": [[{"role": "user", "content": value + " 정답"}]],
            "negative_messages": [[{"role": "user", "content": value + " 오답"}]],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


class BuildMultidomainCurriculumTest(unittest.TestCase):
    def test_mixes_complete_homogeneous_batches_from_multiple_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            components = []
            for role in ("general", "health", "legal"):
                train = root / f"{role}.train.jsonl"
                provenance = root / f"{role}.provenance.jsonl"
                train.write_text(
                    row(f"{role}-0") + "\n" + row(f"{role}-1") + "\n",
                    encoding="utf-8",
                )
                provenance.write_text(
                    "".join(
                        json.dumps(
                            {"source_id": f"{role}-source", "row_index": index}
                        )
                        + "\n"
                        for index in range(2)
                    ),
                    encoding="utf-8",
                )
                components.append(f"{role}={train}={provenance}=2")
            args = SimpleNamespace(
                component=components,
                output=root / "out.train.jsonl",
                provenance_output=root / "out.provenance.jsonl",
                manifest_output=root / "out.manifest.json",
                batch_size=2,
                seed=42,
                adaptation_label="target-adapted-multidomain",
            )

            manifest = build(args)

            self.assertEqual(manifest["output_rows"], 6)
            self.assertEqual(
                manifest["role_counts"], {"general": 2, "health": 2, "legal": 2}
            )
            provenance_rows = [
                json.loads(line)
                for line in args.provenance_output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["multidomain_curriculum_batch"]["output_row_index"] for row in provenance_rows],
                list(range(6)),
            )
            for start in range(0, 6, 2):
                batch = provenance_rows[start : start + 2]
                self.assertEqual(
                    len({row["multidomain_curriculum_batch"]["role"] for row in batch}),
                    1,
                )

    def test_rejects_invalid_component(self) -> None:
        with self.assertRaisesRegex(ValueError, "ROLE=TRAIN"):
            parse_component("broken")


if __name__ == "__main__":
    unittest.main()
