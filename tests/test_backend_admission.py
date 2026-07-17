from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from scripts.backend_admission import (
    SCHEMA_VERSION,
    build_workload_contract,
    canonical_sha256,
    validate_admission_report,
    validate_matched_sdpa_report,
)


QWEN = "Qwen/Qwen3-Embedding-8B"
REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"


def runtime_fixture() -> dict:
    return {
        "contract": "embedding-fa2-runtime-v1",
        "python_version": "3.10.0",
        "python_executable": "/repo/.venv-train-fa2/bin/python",
        "python_prefix": "/repo/.venv-train-fa2",
        "platform": "fixture",
        "packages": {
            "torch": "2.5.0",
            "torch_cuda": "12.6",
            "flash_attn": "2.4.2",
            "swift": "4.5.0.dev0",
            "transformers": "5.12.1",
        },
        "cuda_available": True,
        "cuda_device": {"name": "NVIDIA H100 80GB HBM3"},
        "runtime_environment": {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    }


def admitted_report(contract: dict, runtime: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "flash_attention_2",
        "real_8b_backward_probe": True,
        "process_status": 0,
        "matched_sdpa_process_status": 0,
        "baseline_source": "matched_subset_same_environment",
        "probe_steps": 5,
        "baseline_sdpa_seconds_per_step": 30.0,
        "measured_seconds_per_step": 25.0,
        "required_speedup": 1.05,
        "admission_threshold_seconds_per_step": 30.0 / 1.05,
        "admitted": True,
        "workload_contract": contract,
        "workload_contract_sha256": canonical_sha256(contract),
        "runtime_fingerprint": runtime,
        "runtime_fingerprint_sha256": canonical_sha256(runtime),
    }


def matched_sdpa_report(contract: dict, runtime: dict) -> dict:
    report = admitted_report(contract, runtime)
    report.update(
        {
            "admitted": False,
            "matched_sdpa_eligible": True,
            "selected_backend": "sdpa",
            "selected_environment": runtime["python_prefix"],
            "measured_seconds_per_step": 29.0,
        }
    )
    return report


class BackendAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.train = self.root / "train.jsonl"
        self.train.write_text('{"row":1}\n', encoding="utf-8")
        self.contract = build_workload_contract(
            train_file=self.train,
            backend="flash_attention_2",
            batch_size=16,
            gradient_accumulation_steps=4,
            max_length=512,
            lora_rank=64,
            lora_alpha=128,
            lora_dropout=0.05,
            dtype="bfloat16",
            base_model=QWEN,
            base_revision=REVISION,
            hard_negatives=4,
        )
        self.runtime = runtime_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_current_200k_contract_passes(self) -> None:
        report = admitted_report(self.contract, self.runtime)
        self.assertEqual(
            validate_admission_report(
                report,
                expected_contract=self.contract,
                current_runtime=self.runtime,
            ),
            [],
        )
        self.assertEqual(self.contract["per_device_train_batch_size"], 16)
        self.assertEqual(self.contract["gradient_accumulation_steps"], 4)
        self.assertEqual(self.contract["max_length"], 512)
        self.assertEqual(self.contract["lora_rank"], 64)
        self.assertEqual(self.contract["lora_alpha"], 128)
        self.assertFalse(self.contract["dataset_shuffle"])
        self.assertFalse(self.contract["train_dataloader_shuffle"])
        self.assertTrue(self.contract["strict"])

    def test_workload_fields_are_exact_not_compatibility_hints(self) -> None:
        report = admitted_report(self.contract, self.runtime)
        fields_and_values = {
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 8,
            "max_length": 2048,
            "lora_rank": 32,
            "lora_alpha": 64,
            "dtype": "float16",
            "infonce_hard_negatives": 7,
            "dataset_shuffle": True,
            "train_dataloader_shuffle": True,
            "strict": False,
            "lazy_tokenize": False,
        }
        for field, value in fields_and_values.items():
            with self.subTest(field=field):
                expected = copy.deepcopy(self.contract)
                expected[field] = value
                errors = validate_admission_report(
                    report,
                    expected_contract=expected,
                    current_runtime=self.runtime,
                )
                self.assertIn("workload contract mismatch", errors)

    def test_base_and_training_data_are_part_of_contract(self) -> None:
        report = admitted_report(self.contract, self.runtime)
        expected = copy.deepcopy(self.contract)
        expected["base"]["revision"] = "f" * 40
        self.assertIn(
            "workload contract mismatch",
            validate_admission_report(
                report,
                expected_contract=expected,
                current_runtime=self.runtime,
            ),
        )

        expected = copy.deepcopy(self.contract)
        expected["train_sha256"] = "0" * 64
        self.assertIn(
            "workload contract mismatch",
            validate_admission_report(
                report,
                expected_contract=expected,
                current_runtime=self.runtime,
            ),
        )

    def test_runtime_drift_fails_closed(self) -> None:
        report = admitted_report(self.contract, self.runtime)
        current = copy.deepcopy(self.runtime)
        current["packages"]["flash_attn"] = "2.5.0"
        errors = validate_admission_report(
            report,
            expected_contract=self.contract,
            current_runtime=current,
        )
        self.assertIn("runtime fingerprint mismatch", errors)
        self.assertIn("runtime fingerprint SHA256 mismatch", errors)

    def test_rejected_fa2_can_select_exact_matched_sdpa_runtime(self) -> None:
        report = matched_sdpa_report(self.contract, self.runtime)
        self.assertEqual(
            validate_matched_sdpa_report(
                report,
                expected_contract=self.contract,
                current_runtime=self.runtime,
            ),
            [],
        )

    def test_matched_sdpa_selection_fails_closed_on_policy_or_runtime_drift(self) -> None:
        for field, value, expected_error in (
            ("matched_sdpa_eligible", False, "matched SDPA runtime is not eligible"),
            ("selected_backend", "flash_attention_2", "report did not select matched SDPA"),
            ("matched_sdpa_process_status", 1, "matched SDPA probe process did not exit 0"),
        ):
            with self.subTest(field=field):
                report = matched_sdpa_report(self.contract, self.runtime)
                report[field] = value
                errors = validate_matched_sdpa_report(
                    report,
                    expected_contract=self.contract,
                    current_runtime=self.runtime,
                )
                self.assertIn(expected_error, errors)

        report = matched_sdpa_report(self.contract, self.runtime)
        current = copy.deepcopy(self.runtime)
        current["packages"]["torch"] = "different"
        errors = validate_matched_sdpa_report(
            report,
            expected_contract=self.contract,
            current_runtime=current,
        )
        self.assertIn("runtime fingerprint mismatch", errors)

    def test_boolean_only_and_tampered_reports_fail_closed(self) -> None:
        errors = validate_admission_report(
            {"admitted": True},
            expected_contract=self.contract,
            current_runtime=self.runtime,
        )
        self.assertIn("workload contract mismatch", errors)
        self.assertIn("runtime fingerprint mismatch", errors)

        report = admitted_report(self.contract, self.runtime)
        report["workload_contract_sha256"] = "0" * 64
        self.assertIn(
            "workload contract SHA256 mismatch",
            validate_admission_report(
                report,
                expected_contract=self.contract,
                current_runtime=self.runtime,
            ),
        )

    def test_old_dataset_shuffle_true_probe_cannot_be_upgraded(self) -> None:
        legacy = {
            "admitted": True,
            "real_8b_backward_probe": True,
            "process_status": 0,
            "matched_sdpa_process_status": 0,
            "backend": "flash_attention_2",
            "baseline_source": "matched_subset_same_environment",
            "workload": {
                "source_train_sha256": self.contract["train_sha256"],
                "dataset_shuffle": True,
                "strict": False,
            },
        }
        errors = validate_admission_report(
            legacy,
            expected_contract=self.contract,
            current_runtime=self.runtime,
        )
        self.assertTrue(any("schema_version" in error for error in errors))
        self.assertIn("workload contract mismatch", errors)
        self.assertIn("runtime fingerprint mismatch", errors)


class BackendAdmissionWiringTests(unittest.TestCase):
    def test_all_training_fallbacks_use_the_resolved_runtime(self) -> None:
        root = Path(__file__).resolve().parents[1]
        selector = (root / "scripts/backend_admission.sh").read_text(encoding="utf-8")
        trainer = (
            root / "experiments/020_hard_negative/train_pilot_lora_r64.sh"
        ).read_text(encoding="utf-8")
        quality = (
            root / "experiments/070_tuning_strategy/train_quality.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('EMBEDDING_TRAIN_ENV:-$root/.venv-train', selector)
        self.assertIn("embedding_resolve_train_runtime", trainer)
        self.assertIn('TRAIN_ENV="${TRAIN_ENV:-$EMBEDDING_TRAIN_ENV}"', trainer)
        self.assertIn("embedding_resolve_train_runtime", quality)
        self.assertNotIn('"$ROOT/.venv-train/bin/python"', trainer)
        self.assertNotIn('"$ROOT/.venv-train/bin/python"', quality)

    def test_queues_do_not_reuse_boolean_only_200k_report(self) -> None:
        root = Path(__file__).resolve().parents[1]
        queues = [
            "scripts/run_scale_1m_queue.sh",
            "scripts/run_legal_adaptation_queue.sh",
            "scripts/run_sionic_squad_adaptation_queue.sh",
            "scripts/run_sionic_combined_adaptation_queue.sh",
        ]
        for relative in queues:
            with self.subTest(queue=relative):
                text = (root / relative).read_text(encoding="utf-8")
                self.assertIn("embedding_select_fa2_backend", text)
                self.assertNotIn(
                    "outputs/backend-probes/performance200k-lora-r64/admission.json",
                    text,
                )
                self.assertNotIn(".admitted == true", text)
                self.assertIn("DATASET_SHUFFLE=false", text)
                self.assertIn("TRAIN_DATALOADER_SHUFFLE=false", text)

    def test_probe_and_trainer_disable_both_shuffle_layers_and_are_strict(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative in (
            "experiments/070_tuning_strategy/admit_fa2_lora_backend.sh",
            "experiments/020_hard_negative/train_pilot_lora_r64.sh",
        ):
            with self.subTest(script=relative):
                text = (root / relative).read_text(encoding="utf-8")
                self.assertIn("--dataset_shuffle", text)
                self.assertIn("--train_dataloader_shuffle", text)
                self.assertIn("--strict true", text)

    def test_fast_sdpa_requires_the_exact_matched_report(self) -> None:
        root = Path(__file__).resolve().parents[1]
        selector = (root / "scripts/backend_admission.sh").read_text(encoding="utf-8")
        trainer = (
            root / "experiments/020_hard_negative/train_pilot_lora_r64.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("embedding_check_matched_sdpa", selector)
        self.assertIn("BACKEND_SDPA_VERIFIED_REPORT", selector)
        self.assertIn("embedding_check_matched_sdpa", trainer)
        self.assertIn("unverified or contract-mismatched fast SDPA runtime", trainer)


if __name__ == "__main__":
    unittest.main()
