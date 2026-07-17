from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import publish_best_embedding_model as publisher
from scripts import publish_private_clean_candidate as private_publisher
from scripts.publish_best_embedding_model import (
    COMPREHENSIVE_TEXT_PROTOCOL_ID,
    COMPREHENSIVE_TEXT_TASK_SUBSETS,
    load_hf_token,
    require_remote_visibility,
    training_dataset_repos,
    upload_model_folder,
    validate_comprehensive_summary,
    validate_public_release_approval,
    validate_embedding_contract_for_evidence,
    is_model_soup,
    weights_sha,
)


ROOT = Path(__file__).resolve().parents[1]


def comprehensive_payload(model: Path, weights_sha256: str) -> dict:
    return {
        "schema_version": "1.0.0",
        "protocol_id": COMPREHENSIVE_TEXT_PROTOCOL_ID,
        "complete": True,
        "completed_tasks": 7,
        "total_tasks": 7,
        "completed_subsets": 414,
        "expected_subsets": 414,
        "model": {
            "name_or_path": f"local:{model.name}",
            "revision": f"model-{weights_sha256[:12]}",
            "evidence": {"weights_sha256": weights_sha256},
        },
        "aggregate": {"mean_task": 0.71, "mean_task_type": 0.70},
        "tasks": [
            {"task_name": name, "subset_count": subset_count, "task_score": 0.71}
            for name, subset_count in COMPREHENSIVE_TEXT_TASK_SUBSETS.items()
        ],
        "claim_status": {
            "diagnostic_only": True,
            "clean_claim_allowed": False,
            "visual_document_ready": False,
            "k_haters_ready": False,
        },
    }


class PublishBestModelTests(unittest.TestCase):
    def test_nemotron_publication_accepts_only_masked_mean_contract(self) -> None:
        evidence = {
            "upstream_base_models": [
                {
                    "model": "nvidia/Nemotron-3-Embed-8B-BF16",
                    "revision": "a" * 40,
                }
            ],
            "sentence_transformers_contract": {
                "architecture": "Ministral3Model",
                "pooling": "masked_mean",
                "normalize": True,
            },
        }
        validate_embedding_contract_for_evidence(evidence)
        evidence["sentence_transformers_contract"]["pooling"] = "last_token"
        with self.assertRaisesRegex(ValueError, "contract drifted"):
            validate_embedding_contract_for_evidence(evidence)

    def test_final_upload_staging_is_isolated_sanitized_and_fully_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "artifacts/models/final-winner"
            model.mkdir(parents=True)
            (model / "model.safetensors").write_bytes(b"final-weights")
            (model / "config.json").write_text(
                json.dumps({"_name_or_path": f"{root}/cache/base", "hidden_size": 4096})
            )
            (model / "modules.json").write_text("[]")
            (model / "1_Pooling").mkdir()
            (model / "1_Pooling/config.json").write_text("{}")
            evidence_name = "full_tuning_report.json"
            (model / evidence_name).write_text(
                json.dumps({"status": "pass", "source_checkpoint": f"{root}/checkpoint"})
            )
            (model / "README.md").write_text("# final winner\n")
            evaluation = model / "evaluation"
            evaluation.mkdir()
            (evaluation / "summary.json").write_text(
                json.dumps({"model": str(model), "score": 0.9})
            )
            (model / "publication_manifest.json").write_text(
                json.dumps(
                    {
                        "model_dir": str(model),
                        "model_evidence": {"file": evidence_name, "sha256": "stale"},
                        "evidence": {"summary.json": {"sha256": "stale"}},
                        "raw_evaluation_json": {},
                    }
                )
            )
            source_before = {
                path.relative_to(model).as_posix(): path.read_bytes()
                for path in model.rglob("*")
                if path.is_file()
            }
            staging = root / "staging"
            with (
                patch.object(publisher, "ROOT", root),
                patch.object(private_publisher, "ROOT", root),
            ):
                manifest = publisher.prepare_isolated_upload_staging(
                    model_dir=model,
                    evidence_name=evidence_name,
                    publication_dir=staging,
                )
                private_publisher.validate_staged_text(staging)
            source_after = {
                path.relative_to(model).as_posix(): path.read_bytes()
                for path in model.rglob("*")
                if path.is_file()
            }
            self.assertEqual(source_after, source_before)
            staged_manifest = json.loads(manifest.read_text())
            self.assertEqual(staged_manifest["model_dir"], "artifacts/models/final-winner")
            self.assertEqual(
                staged_manifest["model_evidence"]["sha256"],
                publisher.sha256(staging / evidence_name),
            )
            self.assertIn("evaluation/summary.json", staged_manifest["files_excluding_manifest"])
            self.assertNotIn(
                "publication_manifest.json", staged_manifest["files_excluding_manifest"]
            )
            staged_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in staging.rglob("*")
                if path.is_file() and path.suffix in {".json", ".md"}
            )
            self.assertNotIn(str(root), staged_text)

    def test_soup_uses_full_model_weight_evidence(self) -> None:
        evidence = {
            "artifact_type": "weighted-full-model-embedding-soup",
            "model": {"weights_sha256": "a" * 64},
        }
        self.assertTrue(is_model_soup(evidence))
        self.assertEqual(weights_sha(evidence), "a" * 64)

    def test_private_candidate_does_not_require_release_approval(self) -> None:
        args = SimpleNamespace(public=False)
        result = validate_public_release_approval(
            args=args,
            model_evidence={"model": {"weights_sha256": "a" * 64}},
            training={
                "training_track": "performance-research",
                "release_eligible": False,
                "use_policy": "research-noncommercial",
                "visibility": "private",
            },
            comprehensive=None,
            clean=None,
            robustness=None,
        )
        self.assertIsNone(result)

    def test_public_release_approval_is_rights_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_sha = "a" * 64
            repo_id = "LLM-OS-Models2/release-fixture"
            paths = {
                "training": root / "training.json",
                "sionic9": root / "sionic.json",
                "official_korean_v1": root / "official.json",
                "comprehensive_text_v1": root / "comprehensive.json",
                "clean": root / "clean.json",
                "robustness": root / "robustness.json",
                "approval": root / "approval.json",
            }
            training = {
                "training_track": "rights-safe-release",
                "release_eligible": True,
                "use_policy": "public-release",
                "visibility": "public",
            }
            summaries = {
                "sionic9": {"protocol_id": "sionic9-fixed-prompt-v1"},
                "official_korean_v1": {
                    "protocol_id": "mteb-korean-v1-mteb-2.18.0"
                },
                "comprehensive_text_v1": {
                    "protocol_id": COMPREHENSIVE_TEXT_PROTOCOL_ID
                },
                "clean": {
                    "protocol_id": "legal-source-document-heldout-i-v2-text-strict"
                },
                "robustness": {
                    "protocol_id": "legal-conversational-noise-i-v2-text-strict"
                },
            }
            paths["training"].write_text(json.dumps(training))
            for label, payload in summaries.items():
                paths[label].write_text(json.dumps(payload))

            def file_sha(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            approval = {
                "schema_version": 1,
                "artifact_type": "embedding-model-public-release-approval",
                "approval_id": "release-fixture-v1",
                "decision": "approved",
                "approved_by": "release-reviewer",
                "approved_at_utc": "2026-07-15T00:00:00+00:00",
                "target": {"repo_id": repo_id, "visibility": "public"},
                "model": {"weights_sha256": model_sha},
                "training": {
                    "track": "rights-safe-release",
                    "manifest_sha256": file_sha(paths["training"]),
                },
                "rights_review": {
                    "status": "approved",
                    "release_eligible": True,
                    "public_redistribution": True,
                    "unresolved_blockers": [],
                },
                "evaluations": {
                    label: {
                        "status": "pass",
                        "protocol_id": payload["protocol_id"],
                        "summary_sha256": file_sha(paths[label]),
                    }
                    for label, payload in summaries.items()
                },
            }
            paths["approval"].write_text(json.dumps(approval))
            args = SimpleNamespace(
                public=True,
                clean_summary=paths["clean"],
                robustness_summary=paths["robustness"],
                comprehensive_summary=paths["comprehensive_text_v1"],
                release_approval=paths["approval"],
                training_manifest=paths["training"],
                sionic_summary=paths["sionic9"],
                official_summary=paths["official_korean_v1"],
                repo_id=repo_id,
            )
            validated = validate_public_release_approval(
                args=args,
                model_evidence={"model": {"weights_sha256": model_sha}},
                training=training,
                comprehensive=summaries["comprehensive_text_v1"],
                clean=summaries["clean"],
                robustness=summaries["robustness"],
            )
            self.assertEqual(validated["approval_id"], "release-fixture-v1")

            performance_training = {
                **training,
                "training_track": "performance-research",
                "release_eligible": False,
                "use_policy": "research-noncommercial",
                "visibility": "private",
            }
            with self.assertRaisesRegex(ValueError, "training_manifest.training_track"):
                validate_public_release_approval(
                    args=args,
                    model_evidence={"model": {"weights_sha256": model_sha}},
                    training=performance_training,
                    comprehensive=summaries["comprehensive_text_v1"],
                    clean=summaries["clean"],
                    robustness=summaries["robustness"],
                )

            with self.assertRaisesRegex(ValueError, "unresolved public release blockers"):
                validate_public_release_approval(
                    args=args,
                    model_evidence={"model": {"weights_sha256": model_sha}},
                    training={**training, "release_blockers": ["pending review"]},
                    comprehensive=summaries["comprehensive_text_v1"],
                    clean=summaries["clean"],
                    robustness=summaries["robustness"],
                )

            original_comprehensive = paths["comprehensive_text_v1"].read_text()
            paths["comprehensive_text_v1"].write_text('{"tampered":true}')
            with self.assertRaisesRegex(
                ValueError, "does not match comprehensive_text_v1 summary"
            ):
                validate_public_release_approval(
                    args=args,
                    model_evidence={"model": {"weights_sha256": model_sha}},
                    training=training,
                    comprehensive=summaries["comprehensive_text_v1"],
                    clean=summaries["clean"],
                    robustness=summaries["robustness"],
                )
            paths["comprehensive_text_v1"].write_text(original_comprehensive)

            missing_comprehensive_args = SimpleNamespace(
                **{
                    **vars(args),
                    "comprehensive_summary": None,
                }
            )
            with self.assertRaisesRegex(ValueError, "--comprehensive-summary"):
                validate_public_release_approval(
                    args=missing_comprehensive_args,
                    model_evidence={"model": {"weights_sha256": model_sha}},
                    training=training,
                    comprehensive=None,
                    clean=summaries["clean"],
                    robustness=summaries["robustness"],
                )

            paths["clean"].write_text('{"tampered":true}')
            with self.assertRaisesRegex(ValueError, "does not match clean summary"):
                validate_public_release_approval(
                    args=args,
                    model_evidence={"model": {"weights_sha256": model_sha}},
                    training=training,
                    comprehensive=summaries["comprehensive_text_v1"],
                    clean=summaries["clean"],
                    robustness=summaries["robustness"],
                )

    def test_comprehensive_summary_is_exact_model_and_claim_bound(self) -> None:
        weights_sha = "a" * 64
        summary = comprehensive_payload(Path("fixture-model"), weights_sha)
        validate_comprehensive_summary(
            summary,
            expected_revision="model-" + weights_sha[:12],
            expected_weights_sha256=weights_sha,
        )

        invalid_claim = json.loads(json.dumps(summary))
        invalid_claim["claim_status"]["clean_claim_allowed"] = True
        with self.assertRaisesRegex(ValueError, "diagnostic claim contract"):
            validate_comprehensive_summary(
                invalid_claim,
                expected_revision="model-" + weights_sha[:12],
                expected_weights_sha256=weights_sha,
            )

        invalid_subsets = json.loads(json.dumps(summary))
        invalid_subsets["completed_subsets"] = 413
        with self.assertRaisesRegex(ValueError, "414 subsets"):
            validate_comprehensive_summary(
                invalid_subsets,
                expected_revision="model-" + weights_sha[:12],
                expected_weights_sha256=weights_sha,
            )

        wrong_weights = json.loads(json.dumps(summary))
        wrong_weights["model"]["evidence"]["weights_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "weights do not match"):
            validate_comprehensive_summary(
                wrong_weights,
                expected_revision="model-" + weights_sha[:12],
                expected_weights_sha256=weights_sha,
            )

    def test_hf_token_file_is_permission_checked_and_unambiguous(self) -> None:
        self.assertEqual(
            load_hf_token(None, environ={"HF_TOKEN": "environment-fixture"}),
            "environment-fixture",
        )
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            load_hf_token(None, environ={})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            token_file = root / "credentials.env"
            token_file.write_text(
                "GITHUB=ignored-fixture\nHF_TOKEN=file-fixture\n",
                encoding="utf-8",
            )
            token_file.chmod(0o600)
            self.assertEqual(
                load_hf_token(token_file, environ={}),
                "file-fixture",
            )
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                load_hf_token(
                    token_file,
                    environ={"HF_TOKEN": "environment-fixture"},
                )

            token_file.chmod(0o640)
            with self.assertRaises(RuntimeError) as insecure:
                load_hf_token(token_file, environ={})
            self.assertNotIn(str(token_file), str(insecure.exception))
            self.assertNotIn("file-fixture", str(insecure.exception))

            token_file.chmod(0o600)
            symlink = root / "token-link"
            symlink.symlink_to(token_file)
            with self.assertRaisesRegex(RuntimeError, "security validation") as linked:
                load_hf_token(symlink, environ={})
            self.assertNotIn(str(symlink), str(linked.exception))

            with self.assertRaisesRegex(RuntimeError, "security validation") as directory:
                load_hf_token(root, environ={})
            self.assertNotIn(str(root), str(directory.exception))

            duplicate = root / "duplicate.env"
            duplicate.write_text(
                "HF_TOKEN=first-fixture\nHF_TOKEN=second-fixture\n",
                encoding="utf-8",
            )
            duplicate.chmod(0o600)
            with self.assertRaises(RuntimeError) as duplicated:
                load_hf_token(duplicate, environ={})
            self.assertNotIn("first-fixture", str(duplicated.exception))
            self.assertNotIn(str(duplicate), str(duplicated.exception))

    def test_remote_visibility_must_match_before_and_after_upload(self) -> None:
        class FakeApi:
            def __init__(self, visibility: list[object]):
                self.visibility = iter(visibility)
                self.upload_calls = 0

            def create_repo(self, **_kwargs: object) -> None:
                return None

            def model_info(self, **_kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(private=next(self.visibility))

            def upload_large_folder(self, **_kwargs: object) -> None:
                self.upload_calls += 1

        require_remote_visibility(FakeApi([True]), "org/private", public=False)
        require_remote_visibility(FakeApi([False]), "org/public", public=True)
        with self.assertRaisesRegex(RuntimeError, "not confirmed private"):
            require_remote_visibility(FakeApi([False]), "org/public", public=False)
        with self.assertRaisesRegex(RuntimeError, "not confirmed public"):
            require_remote_visibility(FakeApi([True]), "org/private", public=True)
        with self.assertRaisesRegex(RuntimeError, "not confirmed private"):
            require_remote_visibility(FakeApi([None]), "org/unknown", public=False)

        private_api = FakeApi([True, True])
        upload_model_folder(
            private_api,
            repo_id="org/private",
            model_dir=Path("fixture"),
            public=False,
        )
        self.assertEqual(private_api.upload_calls, 1)

        public_before_upload = FakeApi([False])
        with self.assertRaisesRegex(RuntimeError, "not confirmed private"):
            upload_model_folder(
                public_before_upload,
                repo_id="org/existing-public",
                model_dir=Path("fixture"),
                public=False,
            )
        self.assertEqual(public_before_upload.upload_calls, 0)

        visibility_drift = FakeApi([True, False])
        with self.assertRaisesRegex(RuntimeError, "not confirmed private"):
            upload_model_folder(
                visibility_drift,
                repo_id="org/drifted",
                model_dir=Path("fixture"),
                public=False,
            )
        self.assertEqual(visibility_drift.upload_calls, 1)

    def test_only_final_queue_uses_rights_approved_public_model_publisher(self) -> None:
        queue_paths = [
            ROOT / "scripts/run_post_training_eval_queue.sh",
            ROOT / "scripts/run_scale_1m_queue.sh",
            ROOT / "scripts/run_legal_adaptation_queue.sh",
            ROOT / "scripts/run_sionic_combined_adaptation_queue.sh",
            ROOT / "scripts/run_sionic_squad_adaptation_queue.sh",
        ]
        for path in queue_paths:
            lines = path.read_text().splitlines()
            publish_indexes = [
                index
                for index, line in enumerate(lines)
                if "publish_best_embedding_model.py" in line
            ]
            if path.name != "run_post_training_eval_queue.sh":
                self.assertFalse(publish_indexes, path)
                continue
            self.assertTrue(publish_indexes, path)
            for index in publish_indexes:
                invocation = "\n".join(lines[index : index + 16])
                self.assertIn("--upload", invocation, path)
                self.assertIn("--public", invocation, path)
                self.assertIn("--release-approval", invocation, path)

    def test_pilot_card_links_published_hard_negative_dataset(self) -> None:
        self.assertEqual(
            training_dataset_repos(
                {"purpose": "training-only-dense-hard-negative-mining"}
            ),
            ["LLM-OS-Models/korean-embedding-ko-triplet-hn-pilot-10k"],
        )

    def test_target_adapted_cards_link_exact_derived_datasets(self) -> None:
        scale = training_dataset_repos(
            {"benchmark_adaptation": "target-adapted-performance1m-current-student"}
        )
        self.assertEqual(
            scale[0],
            "LLM-OS-Models2/korean-embedding-performance-1m-quantile-hn7-v1",
        )
        legal = training_dataset_repos(
            {"benchmark_adaptation": "target-adapted-legal25-general75"}
        )
        self.assertEqual(legal[0], "LLM-OS-Models2/korean-legal-quantile-hn7-replay-v1")
        squad = training_dataset_repos(
            {"benchmark_adaptation": "target-adapted-squad50-general50"}
        )
        self.assertEqual(
            squad[0],
            "LLM-OS-Models2/korean-embedding-sionic-squad-quantile-hn7-replay-v1",
        )
        health = training_dataset_repos(
            {"benchmark_adaptation": "target-adapted-health-domain50-general50"}
        )
        self.assertEqual(
            health[0],
            "LLM-OS-Models2/korean-embedding-sionic-health-quantile-hn7-replay-v1",
        )
        autorag = training_dataset_repos(
            {"benchmark_adaptation": "target-adapted-autorag-domain50-general50"}
        )
        self.assertEqual(
            autorag[0],
            "LLM-OS-Models2/korean-embedding-sionic-autorag-quantile-hn7-replay-v1",
        )
        retrieval = training_dataset_repos(
            {"benchmark_adaptation": "target-adapted-retrieval-family50-general50"}
        )
        self.assertEqual(
            retrieval[0],
            "LLM-OS-Models2/korean-embedding-sionic-retrieval-family-quantile-hn7-replay-v1",
        )
        combined = training_dataset_repos(
            {"benchmark_adaptation": "target-adapted-sionic-combined-v1"}
        )
        self.assertEqual(
            combined[0], "LLM-OS-Models2/korean-embedding-sionic-combined-replay-v1"
        )

    def test_card_requires_complete_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            (model / "1_Pooling").mkdir(parents=True)
            (model / "2_Normalize").mkdir()
            for name in ("config.json", "modules.json", "1_Pooling/config.json"):
                (model / name).write_text("{}")
            (model / "model.safetensors").write_bytes(b"fixture")
            model_digest = hashlib.sha256()
            model_digest.update(b"model.safetensors\0")
            model_digest.update(b"fixture")
            model_sha = model_digest.hexdigest()
            (model / "merge_report.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "base_model": "Qwen/Qwen3-Embedding-8B",
                        "base_revision": "1" * 40,
                        "adapter": {"weights_sha256": "2" * 64},
                        "model": {"weights_sha256": model_sha},
                        "adapter_config": {
                            "r": 64,
                            "lora_alpha": 128,
                            "lora_dropout": 0.05,
                            "target_modules": ["q_proj"],
                        },
                        "probe": {
                            "metrics": {
                                "minimum_row_cosine": 1.0,
                                "maximum_pairwise_score_difference": 0.0,
                            }
                        },
                        "sentence_transformers_contract": {
                            "pooling": "last_token",
                            "normalize": True,
                        },
                    }
                )
            )
            sionic = root / "sionic.json"
            names = [
                "MIRACL",
                "MrTidy",
                "MLDR",
                "AutoRAG",
                "Ko-StrategyQA",
                "PublicHealthQA",
                "Belebele",
                "SQuADKorV1",
                "LawIRKo",
            ]
            sionic.write_text(
                json.dumps(
                    {
                        "protocol_id": "sionic9-fixed-prompt-v1",
                        "model": str(model),
                        "requested_revision": "model-" + model_sha[:12],
                        "completed_tasks": 9,
                        "average": 0.8,
                        "scores": {name: 0.8 for name in names},
                    }
                )
            )
            official = root / "official.json"
            official.write_text(
                json.dumps(
                    {
                        "complete": True,
                        "completed_tasks": 6,
                        "protocol_id": "mteb-korean-v1-mteb-2.18.0",
                        "model": str(model),
                        "requested_revision": "model-" + model_sha[:12],
                        "mean_task_leaderboard_points": 80.0,
                        "mean_task_type_leaderboard_points": 79.0,
                        "means_by_type": {"Retrieval": 0.79},
                        "environment": {
                            "qwen3_instruction_loader": True,
                            "instruction_contract": "qwen3-task-instruction",
                        },
                        "scores": {
                            f"task-{index}": {"score": 0.8} for index in range(6)
                        },
                    }
                )
            )
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"phase": "fixture", "built_rows": 10}))
            public_attempt = subprocess.run(
                [
                    "python",
                    str(ROOT / "scripts/publish_best_embedding_model.py"),
                    "--model-dir",
                    str(model),
                    "--sionic-summary",
                    str(sionic),
                    "--official-summary",
                    str(official),
                    "--training-manifest",
                    str(manifest),
                    "--public",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(public_attempt.returncode, 0)
            self.assertIn("Public release requires explicit evidence", public_attempt.stderr)
            subprocess.check_call(
                [
                    "python",
                    str(ROOT / "scripts/publish_best_embedding_model.py"),
                    "--model-dir",
                    str(model),
                    "--sionic-summary",
                    str(sionic),
                    "--official-summary",
                    str(official),
                    "--training-manifest",
                    str(manifest),
                ]
            )
            card = (model / "README.md").read_text()
            self.assertIn("9-task average: **0.80000**", card)
            self.assertIn("SentenceTransformers", card)
            self.assertIn("zero-shot", card)
            publication = json.loads((model / "publication_manifest.json").read_text())
            self.assertEqual(
                set(publication["evidence"]),
                {
                    "sionic9_summary.json",
                    "mteb_korean_v1_summary.json",
                    "training_manifest.json",
                },
            )

            comprehensive_dir = root / "comprehensive"
            comprehensive_raw = (
                comprehensive_dir / "mteb_cache" / "results" / "fixture-model"
            )
            comprehensive_raw.mkdir(parents=True)
            (comprehensive_raw / "XPQARetrieval.json").write_text(
                json.dumps({"task_name": "XPQARetrieval", "main_score": 0.71}),
                encoding="utf-8",
            )
            comprehensive = comprehensive_dir / "summary.json"
            comprehensive.write_text(
                json.dumps(comprehensive_payload(model, model_sha)),
                encoding="utf-8",
            )
            comprehensive_attempt = subprocess.run(
                [
                    "python",
                    str(ROOT / "scripts/publish_best_embedding_model.py"),
                    "--model-dir",
                    str(model),
                    "--sionic-summary",
                    str(sionic),
                    "--official-summary",
                    str(official),
                    "--comprehensive-summary",
                    str(comprehensive),
                    "--training-manifest",
                    str(manifest),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            comprehensive_report = json.loads(comprehensive_attempt.stdout)
            self.assertEqual(
                comprehensive_report["comprehensive_text_v1"]["completed_subsets"],
                414,
            )
            comprehensive_card = (model / "README.md").read_text()
            self.assertIn("Comprehensive Korean text v1 진단", comprehensive_card)
            self.assertIn("diagnostic-only", comprehensive_card)
            comprehensive_publication = json.loads(
                (model / "publication_manifest.json").read_text()
            )
            self.assertEqual(
                comprehensive_publication["comprehensive_text_v1"]["status"],
                "complete_diagnostic",
            )
            self.assertEqual(
                comprehensive_publication["comprehensive_text_v1"]["summary_sha256"],
                hashlib.sha256(comprehensive.read_bytes()).hexdigest(),
            )
            self.assertIn(
                "comprehensive_text_v1_summary.json",
                comprehensive_publication["evidence"],
            )
            self.assertEqual(
                len(
                    comprehensive_publication["raw_evaluation_json"][
                        "comprehensive_text_v1"
                    ]
                ),
                1,
            )
            copied_raw = (
                model
                / "evaluation"
                / "raw"
                / "comprehensive_text_v1"
                / "results"
                / "fixture-model"
                / "XPQARetrieval.json"
            )
            self.assertTrue(copied_raw.is_file())

            clean_dir = root / "clean"
            clean_dir.mkdir()
            clean = clean_dir / "summary.json"
            clean.write_text(
                json.dumps(
                    {
                        "protocol_id": "legal-source-document-heldout-i-v2-text-strict",
                        "model": str(model),
                        "requested_revision": "model-" + model_sha[:12],
                        "dataset": {"independence_grade": "I", "not_grade": "Z"},
                        "metrics": {
                            "ndcg_at_10": 0.7,
                            "recall_at_10": 0.8,
                            "mrr_at_10": 0.6,
                            "recall_at_100": 0.9,
                        },
                    }
                )
            )
            (clean_dir / "ranks.jsonl").write_text('{"query_id":"q1"}\n')
            robust_dir = root / "robustness"
            robust_dir.mkdir()
            robustness = robust_dir / "summary.json"
            conditions = {}
            for state in ("on", "off"):
                for ratio in ("0.00", "0.01", "0.05"):
                    conditions[f"prompt_{state}/noise_{ratio}"] = {
                        "ndcg_at_10": 0.7,
                        "ndcg_retention_vs_same_prompt_clean": 1.0,
                        "noise_intrusion_at_10": 0.0,
                    }
            robustness.write_text(
                json.dumps(
                    {
                        "protocol_id": "legal-conversational-noise-i-v2-text-strict",
                        "model": str(model),
                        "requested_revision": "model-" + model_sha[:12],
                        "dataset": {"independence_grade": "I", "not_grade": "Z"},
                        "conditions": conditions,
                    }
                )
            )
            (robust_dir / "ranks.jsonl").write_text('{"query_id":"q1"}\n')
            subprocess.check_call(
                [
                    "python",
                    str(ROOT / "scripts/publish_best_embedding_model.py"),
                    "--model-dir",
                    str(model),
                    "--sionic-summary",
                    str(sionic),
                    "--official-summary",
                    str(official),
                    "--clean-summary",
                    str(clean),
                    "--robustness-summary",
                    str(robustness),
                    "--training-manifest",
                    str(manifest),
                ]
            )
            robust_card = (model / "README.md").read_text()
            self.assertIn("대화형 구조 노이즈 강건성", robust_card)
            robust_publication = json.loads(
                (model / "publication_manifest.json").read_text()
            )
            self.assertIn(
                "conversational_noise_ranks.jsonl", robust_publication["evidence"]
            )

            release_repo = "LLM-OS-Models2/release-fixture"
            manifest.write_text(
                json.dumps(
                    {
                        "phase": "fixture",
                        "built_rows": 10,
                        "training_track": "rights-safe-release",
                        "release_eligible": True,
                        "use_policy": "public-release",
                        "visibility": "public",
                    }
                )
            )

            def file_sha(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            release_approval = root / "release-approval.json"
            release_approval.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_type": "embedding-model-public-release-approval",
                        "approval_id": "release-fixture-v1",
                        "decision": "approved",
                        "approved_by": "release-reviewer",
                        "approved_at_utc": "2026-07-15T00:00:00+00:00",
                        "target": {
                            "repo_id": release_repo,
                            "visibility": "public",
                        },
                        "model": {"weights_sha256": model_sha},
                        "training": {
                            "track": "rights-safe-release",
                            "manifest_sha256": file_sha(manifest),
                        },
                        "rights_review": {
                            "status": "approved",
                            "release_eligible": True,
                            "public_redistribution": True,
                            "unresolved_blockers": [],
                        },
                        "evaluations": {
                            "sionic9": {
                                "status": "pass",
                                "protocol_id": "sionic9-fixed-prompt-v1",
                                "summary_sha256": file_sha(sionic),
                            },
                            "official_korean_v1": {
                                "status": "pass",
                                "protocol_id": "mteb-korean-v1-mteb-2.18.0",
                                "summary_sha256": file_sha(official),
                            },
                            "comprehensive_text_v1": {
                                "status": "pass",
                                "protocol_id": COMPREHENSIVE_TEXT_PROTOCOL_ID,
                                "summary_sha256": file_sha(comprehensive),
                            },
                            "clean": {
                                "status": "pass",
                                "protocol_id": "legal-source-document-heldout-i-v2-text-strict",
                                "summary_sha256": file_sha(clean),
                            },
                            "robustness": {
                                "status": "pass",
                                "protocol_id": "legal-conversational-noise-i-v2-text-strict",
                                "summary_sha256": file_sha(robustness),
                            },
                        },
                    }
                )
            )
            subprocess.check_call(
                [
                    "python",
                    str(ROOT / "scripts/publish_best_embedding_model.py"),
                    "--model-dir",
                    str(model),
                    "--sionic-summary",
                    str(sionic),
                    "--official-summary",
                    str(official),
                    "--comprehensive-summary",
                    str(comprehensive),
                    "--clean-summary",
                    str(clean),
                    "--robustness-summary",
                    str(robustness),
                    "--training-manifest",
                    str(manifest),
                    "--release-approval",
                    str(release_approval),
                    "--repo-id",
                    release_repo,
                    "--public",
                ]
            )
            public_manifest = json.loads(
                (model / "publication_manifest.json").read_text()
            )
            self.assertEqual(public_manifest["public_release_gate"]["status"], "pass")
            self.assertIn(
                "public_release_approval.json", public_manifest["evidence"]
            )

            (model / "merge_report.json").unlink()
            (model / "full_tuning_report.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "training_method": "partial-full-parameter-update",
                        "base_model": "Qwen/Qwen3-Embedding-8B",
                        "base_revision": "1" * 40,
                        "model": {"weights_sha256": model_sha},
                        "probe": {
                            "metrics": {
                                "maximum_norm_error": 1e-7,
                                "positive_margin": 0.25,
                            }
                        },
                        "sentence_transformers_contract": {
                            "pooling": "last_token",
                            "normalize": True,
                        },
                    }
                )
            )
            sionic_payload = json.loads(sionic.read_text())
            sionic_payload["requested_revision"] = "model-" + model_sha[:12]
            sionic.write_text(json.dumps(sionic_payload))
            official_payload = json.loads(official.read_text())
            official_payload["requested_revision"] = "model-" + model_sha[:12]
            official.write_text(json.dumps(official_payload))
            subprocess.check_call(
                [
                    "python",
                    str(ROOT / "scripts/publish_best_embedding_model.py"),
                    "--model-dir",
                    str(model),
                    "--sionic-summary",
                    str(sionic),
                    "--official-summary",
                    str(official),
                    "--training-manifest",
                    str(manifest),
                ]
            )
            full_card = (model / "README.md").read_text()
            self.assertIn("부분 full-parameter update", full_card)
            self.assertNotIn("LoRA rank/alpha/dropout", full_card)
            full_publication = json.loads(
                (model / "publication_manifest.json").read_text()
            )
            self.assertEqual(
                full_publication["model_evidence"]["file"],
                "full_tuning_report.json",
            )
            self.assertEqual(
                full_publication["comprehensive_text_v1"]["status"],
                "not_provided",
            )
            self.assertFalse(
                (model / "evaluation" / "comprehensive_text_v1_summary.json").exists()
            )
            self.assertFalse(
                (model / "evaluation" / "raw" / "comprehensive_text_v1").exists()
            )

            (model / "full_tuning_report.json").unlink()
            (model / "soup_report.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "artifact_type": "weighted-full-model-embedding-soup",
                        "training_method": "weighted-full-model-soup",
                        "upstream_base_models": [
                            {
                                "model": "Qwen/Qwen3-Embedding-8B",
                                "revision": "1" * 40,
                            },
                            {
                                "model": "sionic-ai/comsat-embed-ko-8b-preview",
                                "revision": "a" * 40,
                            },
                        ],
                        "sources": [
                            {"model": "/models/general", "weight": 0.5},
                            {"model": "/models/combined", "weight": 0.5},
                        ],
                        "soup": {
                            "accumulation_dtype": "float32",
                            "output_floating_dtype": "bfloat16",
                            "tensor_count": 2,
                        },
                        "model": {"weights_sha256": model_sha},
                        "sentence_transformers_contract": {
                            "pooling": "last_token",
                            "normalize": True,
                        },
                    }
                )
            )
            subprocess.check_call(
                [
                    "python",
                    str(ROOT / "scripts/publish_best_embedding_model.py"),
                    "--model-dir",
                    str(model),
                    "--sionic-summary",
                    str(sionic),
                    "--official-summary",
                    str(official),
                    "--training-manifest",
                    str(manifest),
                ]
            )
            soup_card = (model / "README.md").read_text()
            self.assertIn("safe-merged full transformer weight", soup_card)
            self.assertIn("base_model:\n- Qwen/Qwen3-Embedding-8B", soup_card)
            self.assertIn("- sionic-ai/comsat-embed-ko-8b-preview", soup_card)
            self.assertIn("CC-BY-NC-4.0", soup_card)
            soup_publication = json.loads(
                (model / "publication_manifest.json").read_text()
            )
            self.assertEqual(
                soup_publication["model_evidence"]["file"], "soup_report.json"
            )


if __name__ == "__main__":
    unittest.main()
