import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "restore_hf_assets", ROOT / "scripts" / "restore_hf_assets.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RestoreHfAssetsTests(unittest.TestCase):
    def test_dataset_contracts_are_unique_and_pinned(self) -> None:
        keys = [asset.key for asset in MODULE.DATASETS]
        destinations = [asset.destination for asset in MODULE.DATASETS]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(destinations), len(set(destinations)))
        self.assertNotIn("pilot50k", keys)
        for asset in MODULE.DATASETS:
            self.assertEqual(len(asset.revision), 40)
            for contract in asset.files:
                self.assertEqual(len(contract.sha256), 64)

    def test_queue_critical_aliases_exist(self) -> None:
        aliases = {
            asset.key: {contract.alias for contract in asset.files}
            for asset in MODULE.DATASETS
        }
        self.assertIn("validation.hn-qwen3-r095-n4.jsonl", aliases["pilot10k"])
        self.assertIn("train.homogeneous-b16.jsonl", aliases["performance200k"])
        self.assertIn("train.homogeneous-b16.jsonl", aliases["performance1m"])
        self.assertIn("train.bootstrap.jsonl", aliases["legal250k"])

    def test_next_stage_assets_are_exactly_pinned(self) -> None:
        datasets = {asset.key: asset for asset in MODULE.DATASETS}
        models = {asset.key: asset for asset in MODULE.MODELS}
        self.assertEqual(
            datasets["bcai-finance-triplet"].revision,
            "f63d59969dba9916bd34c86c82112331890b11da",
        )
        self.assertEqual(
            datasets["bcai-finance-pair"].revision,
            "e022cb013f2907e0716ebe40d13f30ed93ffa9b0",
        )
        self.assertEqual(
            datasets["kotsqa-v2"].revision,
            "ff9349df469a765b4561959e36ef1b3f377765cd",
        )
        teacher = models["qwen-reranker-teacher"]
        self.assertEqual(teacher.repo_id, "Qwen/Qwen3-Reranker-8B")
        self.assertEqual(
            teacher.revision,
            "77d193c791ed757ca307ee72715aa132723da912",
        )
        self.assertEqual(teacher.group, "teacher")

    def test_private_text_strict_assets_are_pinned_and_opt_in(self) -> None:
        datasets = {asset.key: asset for asset in MODULE.DATASETS}
        clean = datasets["cleanlegal10k-v2-text-strict"]
        validation = datasets["legal-validation-v2-text-strict-512"]
        self.assertTrue(clean.requires_token)
        self.assertTrue(validation.requires_token)
        self.assertEqual(clean.repo_id.split("/", 1)[0], "LLM-OS-Models2")
        self.assertEqual(validation.repo_id.split("/", 1)[0], "LLM-OS-Models2")
        self.assertEqual(clean.revision, "ce9d3bb57ca4dc5144753f6d0f8b4a2256851e97")
        self.assertEqual(validation.revision, "8fdd1cad0007a9bfadf328d1702dcf6973c3c03d")


if __name__ == "__main__":
    unittest.main()
