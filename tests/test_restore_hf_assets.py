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


if __name__ == "__main__":
    unittest.main()
