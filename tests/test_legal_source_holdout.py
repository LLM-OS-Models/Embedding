from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import unicodedata
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_legal_source_holdout.py"
CONFIG = ROOT / "configs" / "legal_source_holdout_v1.json"
POLICY = ROOT / "configs" / "decontamination_policy.json"
FIXTURE = ROOT / "tests" / "fixtures" / "legal_source_holdout"
CANDIDATES = FIXTURE / "candidates.jsonl"
TRAINING = FIXTURE / "training_provenance.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.translate(str.maketrans("", "", "\u200b\u200c\u200d\u2060\ufeff"))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(value.split())


def text_hash(value: str) -> str:
    return hashlib.sha256(normalize(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_hash_file(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write("".join(f"{value}\n" for value in sorted(set(values))).encode("utf-8"))


class LegalSourceHoldoutTest(unittest.TestCase):
    maxDiff = None

    def run_cli(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--config", str(CONFIG), *arguments],
            cwd=ROOT,
            check=check,
            capture_output=True,
            text=True,
        )

    def make_blocklist(self, root: Path) -> None:
        candidates = {row["id"]: row for row in read_jsonl(CANDIDATES)}
        write_hash_file(
            root / "fixture-suite" / "task-a" / "test" / "query_text.sha256.gz",
            [text_hash(candidates["fixture-adm-002"]["query"])],
        )
        write_hash_file(
            root / "fixture-suite" / "task-b" / "test" / "corpus_text.sha256.gz",
            [text_hash(candidates["fixture-statute-002"]["positive"])],
        )
        write_hash_file(
            root / "fixture-suite" / "task-c" / "test" / "evaluation_text.sha256.gz",
            [hashlib.sha256("unrelated fixture benchmark text".encode()).hexdigest()],
        )
        (root / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "fixture": True}, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_build_verify_balance_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blocklist = root / "blocklist"
            self.make_blocklist(blocklist)
            first = root / "holdout-first"
            second = root / "holdout-second"
            common = (
                "--candidate",
                str(CANDIDATES),
                "--training-provenance",
                str(TRAINING),
                "--blocklist-root",
                str(blocklist),
                "--target-size",
                "4",
                "--work-dir",
                str(root),
            )
            self.run_cli("build", *common, "--output-dir", str(first))
            self.run_cli(
                "verify",
                "--training-provenance",
                str(TRAINING),
                "--blocklist-root",
                str(blocklist),
                "--output-dir",
                str(first),
            )
            manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["independence"]["grade"], "I")
            self.assertEqual(manifest["independence"]["not_grade"], "Z")
            self.assertEqual(
                manifest["selection"]["selected_repository_counts"],
                {
                    "legalize-kr/admrule-kr": 1,
                    "legalize-kr/legalize-kr": 1,
                    "legalize-kr/ordinance-kr": 1,
                    "legalize-kr/precedent-kr": 1,
                },
            )
            provenance = read_jsonl(first / "provenance.jsonl")
            selected_ids = {row["source_candidate_id"] for row in provenance}
            self.assertNotIn("fixture-adm-001", selected_ids)
            self.assertNotIn("fixture-adm-001-alt", selected_ids)
            self.assertNotIn("fixture-adm-002", selected_ids)
            self.assertNotIn("fixture-statute-001", selected_ids)
            self.assertNotIn("fixture-statute-002", selected_ids)
            self.assertIn("fixture-adm-003", selected_ids)
            self.assertIn("fixture-statute-003", selected_ids)
            self.assertEqual(
                manifest["inputs"]["candidate_sources"]["counters"][
                    "candidate_ids_present_in_training"
                ],
                4,
            )
            self.assertEqual(
                manifest["inputs"]["candidate_sources"]["counters"][
                    "excluded_entire_training_source_document"
                ],
                5,
            )
            self.assertTrue(
                all(row["independence_grade"] == "I" for row in provenance)
            )
            self.assertTrue(all(row["selection_reason"] for row in provenance))

            self.run_cli("build", *common, "--output-dir", str(second))
            for name in (
                "queries.jsonl",
                "corpus.jsonl",
                "qrels.jsonl",
                "provenance.jsonl",
                "manifest.json",
            ):
                self.assertEqual(file_hash(first / name), file_hash(second / name), name)

    def test_all_training_documents_produce_blocked_manifest(self) -> None:
        candidates = read_jsonl(CANDIDATES)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blocklist = root / "blocklist"
            self.make_blocklist(blocklist)
            all_training = root / "all-training.jsonl"
            with all_training.open("w", encoding="utf-8") as handle:
                for candidate in candidates:
                    handle.write(
                        json.dumps(
                            {
                                "source_candidate_id": candidate["id"],
                                "provenance": candidate["provenance"],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            output = root / "blocked"
            result = self.run_cli(
                "build",
                "--candidate",
                str(CANDIDATES),
                "--training-provenance",
                str(all_training),
                "--blocklist-root",
                str(blocklist),
                "--target-size",
                "4",
                "--work-dir",
                str(root),
                "--output-dir",
                str(output),
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["status"],
                "blocked_insufficient_source_document_heldout_candidates",
            )
            self.assertEqual(
                manifest["counts"]["eligible_before_benchmark_exact_exclusion"], 0
            )
            self.assertEqual(
                manifest["inputs"]["candidate_sources"]["counters"][
                    "excluded_entire_training_source_document"
                ],
                len(candidates),
            )
            self.assertFalse((output / "queries.jsonl").exists())

    def test_declared_training_text_is_excluded_from_selection(self) -> None:
        candidates = {row["id"]: row for row in read_jsonl(CANDIDATES)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blocklist = root / "blocklist"
            self.make_blocklist(blocklist)
            training_text = root / "training.jsonl"
            blocked = candidates["fixture-adm-003"]
            strict = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Instruct: retrieve\nQuery: " + blocked["query"],
                    }
                ],
                "positive_messages": [[{"role": "user", "content": "unrelated positive"}]],
                "negative_messages": [[{"role": "user", "content": "unrelated negative"}]],
            }
            training_text.write_text(
                json.dumps(strict, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            output = root / "blocked-text"
            result = self.run_cli(
                "build",
                "--candidate",
                str(CANDIDATES),
                "--training-provenance",
                str(TRAINING),
                "--training-data",
                str(training_text),
                "--blocklist-root",
                str(blocklist),
                "--target-size",
                "4",
                "--work-dir",
                str(root),
                "--output-dir",
                str(output),
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["status"], "blocked_insufficient_post_decontamination_candidates"
            )
            self.assertEqual(
                manifest["selection"]["selection_skips"]["training_exact_query_hash"], 1
            )
            self.assertEqual(
                manifest["inputs"]["training_text"]["candidate_hash_intersections"], 1
            )


if __name__ == "__main__":
    unittest.main()
