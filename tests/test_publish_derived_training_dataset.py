from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.publish_derived_training_dataset import (
    dataset_card,
    expected_publication,
    validate,
    validate_public_rights,
    verify_remote_dataset,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublishDerivedTrainingDatasetTest(unittest.TestCase):
    def test_public_rights_require_row_level_source_revision_and_license(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            provenance = Path(temporary) / "provenance.jsonl"
            provenance.write_text(
                json.dumps(
                    {
                        "source": "org/source",
                        "revision": "a" * 40,
                        "license": "cc-by-4.0",
                        "redistribution_allowed": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = {
                "release_eligible": True,
                "release_blockers": [],
                "visibility": "public",
            }
            self.assertEqual(validate_public_rights(manifest, provenance), 1)
            manifest["release_eligible"] = False
            with self.assertRaisesRegex(ValueError, "release_eligible"):
                validate_public_rights(manifest, provenance)

    def test_exact_files_and_quantile_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            provenance = root / "provenance.jsonl"
            audit = root / "audit.jsonl"
            train.write_text("{}\n{}\n", encoding="utf-8")
            provenance.write_text("{}\n{}\n", encoding="utf-8")
            audit.write_text("{}\n{}\n", encoding="utf-8")
            quality = root / "quality.json"
            quality.write_text(
                json.dumps(
                    {
                        "rows": 2,
                        "inputs": {
                            "train": {"sha256": digest(train)},
                            "provenance": {"sha256": digest(provenance)},
                        },
                        "contract_checks": {"status": "pass"},
                    }
                ),
                encoding="utf-8",
            )
            overlap = root / "overlap.json"
            overlap.write_text(
                json.dumps(
                    {
                        "rows": 2,
                        "inputs": {
                            "train": {"sha256": digest(train)},
                            "provenance": {"sha256": digest(provenance)},
                        },
                        "unique_critical_query_or_evaluation_matches": 0,
                        "unique_retrieval_corpus_matches": 1,
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "final.json"
            manifest.write_text(
                json.dumps(
                    {
                        "output_rows": 2,
                        "batch_size": 1,
                        "benchmark_adaptation": "target-adapted-fixture",
                        "outputs": {
                            "train": {"sha256": digest(train)},
                            "provenance": {"sha256": digest(provenance)},
                        },
                    }
                ),
                encoding="utf-8",
            )
            mining = root / "mining.json"
            mining.write_text(
                json.dumps(
                    {
                        "rows": 2,
                        "selection_strategy": "score_rank_quantiles",
                        "candidate_pool_size": 24,
                        "num_negatives": 7,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                train=train,
                provenance=provenance,
                manifest=manifest,
                mining_manifest=mining,
                mining_audit=audit,
                quality_audit=quality,
                benchmark_overlap_audit=overlap,
                repo_id="LLM-OS-Models2/fixture",
                title="Fixture",
                source_dataset=["org/source"],
            )
            result = validate(args)
            self.assertEqual(result["rows"], 2)
            self.assertIn("quality_audit", result["evidence"])
            self.assertIn("benchmark_overlap_audit", result["evidence"])
            card = dataset_card(args, result)
            self.assertIn("score-rank", card)
            self.assertIn("release eligible: **false**", card)
            self.assertIn("retrieval-corpus matches", card)

            mining_payload = json.loads(mining.read_text())
            mining_payload["selection_strategy"] = "top_k"
            mining.write_text(json.dumps(mining_payload))
            with self.assertRaisesRegex(ValueError, "score_rank_quantiles"):
                validate(args)

    def test_remote_commit_verifies_file_set_visibility_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            readme = root / "README.md"
            train = root / "train.jsonl"
            readme.write_text("fixture\n", encoding="utf-8")
            train.write_text("{}\n{}\n", encoding="utf-8")
            sources = {"README.md": readme, "data/train.jsonl": train}
            expected = expected_publication(sources)

            class FakeApi:
                def __init__(self) -> None:
                    self.extra = False
                    self.private = False
                    self.corrupt_lfs = False

                def dataset_info(self, **_kwargs):
                    train_sha = expected["data/train.jsonl"]["sha256"]
                    if self.corrupt_lfs:
                        train_sha = "0" * 64
                    siblings = [
                        SimpleNamespace(rfilename="README.md", lfs=None),
                        SimpleNamespace(
                            rfilename="data/train.jsonl",
                            lfs={
                                "sha256": train_sha,
                                "size": expected["data/train.jsonl"]["size_bytes"],
                            },
                        ),
                        SimpleNamespace(rfilename=".gitattributes", lfs=None),
                    ]
                    if self.extra:
                        siblings.append(SimpleNamespace(rfilename="stale.bin", lfs=None))
                    return SimpleNamespace(private=self.private, siblings=siblings)

                def hf_hub_download(self, **kwargs):
                    return str(sources[kwargs["filename"]])

            api = FakeApi()
            verify_remote_dataset(
                api=api,
                repo_id="LLM-OS-Models2/fixture",
                revision="a" * 40,
                expected=expected,
                public=True,
            )
            api.corrupt_lfs = True
            with self.assertRaisesRegex(RuntimeError, "LFS object mismatch"):
                verify_remote_dataset(
                    api=api,
                    repo_id="LLM-OS-Models2/fixture",
                    revision="a" * 40,
                    expected=expected,
                    public=True,
                )
            api.corrupt_lfs = False
            api.extra = True
            with self.assertRaisesRegex(RuntimeError, "file set"):
                verify_remote_dataset(
                    api=api,
                    repo_id="LLM-OS-Models2/fixture",
                    revision="a" * 40,
                    expected=expected,
                    public=True,
                )
            api.extra = False
            api.private = True
            with self.assertRaisesRegex(RuntimeError, "not exactly public"):
                verify_remote_dataset(
                    api=api,
                    repo_id="LLM-OS-Models2/fixture",
                    revision="a" * 40,
                    expected=expected,
                    public=True,
                )


if __name__ == "__main__":
    unittest.main()
