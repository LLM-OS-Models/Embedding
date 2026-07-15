import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "restore_comprehensive_eval_assets",
    ROOT / "scripts" / "restore_comprehensive_eval_assets.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


EXPECTED_REVISIONS = {
    "xpqa": "fc4624be978945a0ceebbc4b85737258fe26330b",
    "flores_bitext": "2144d16cc15edd22d4a9237d12bff5f31f5c07fc",
    "kor_sarcasm": "0e5e17b4dba569776e445f5639ba13dc406b2b0e",
    "kor_hate": "5d64e6dcbe9204c934e9a3852b1130a6f2d51ad4",
    "korfin_asc": "07cc4a29341ef26e8614ae1139847f4d4888727d",
    "kor_hate_speech_ml": "47cd2e61b64f2f11ccb006a579cda71318c6de9b",
    "kor_nli": "3e0e4626f66911b344c490c26e3cc07e6c3bb0f9",
    "k_haters": "67b979254f2f5874f3c0f649e0db8ab33d038e0d",
    "sds_kopub_vdr_t2it": "208fb1837d6be4178059af397e5e8497fe83d220",
    "kovidore_v2_cybersecurity": "577d7c45f79d8eb4e7584db3990f91daa7e47956",
    "kovidore_v2_economic": "0189c26211290a902cd9d41a0db932808a54c0a8",
    "kovidore_v2_energy": "8c09a3d22b1fa3a7f5e815e9521da9b048754211",
    "kovidore_v2_hr": "d9432c782a9a3e2eed064f6fac08b4c967d92b99",
}


class RestoreComprehensiveEvalAssetsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assets = MODULE.load_assets()

    def test_manifest_has_exact_assets_full_revisions_and_safety_fields(self) -> None:
        self.assertEqual(
            {asset.key: asset.revision for asset in self.assets},
            EXPECTED_REVISIONS,
        )
        self.assertEqual(len(self.assets), 13)
        self.assertEqual(
            MODULE.CACHE_DIR,
            ROOT / ".cache" / "huggingface" / "hub",
        )
        for asset in self.assets:
            self.assertTrue(asset.license)
            self.assertTrue(asset.purpose)
            self.assertTrue(asset.contamination_notes)
            self.assertIn(asset.contamination_grade, {"low", "medium", "high"})
            self.assertIn(asset.download_tier, {"small", "metadata_first"})
            self.assertIn("evaluation", asset.usage_policy)

    def test_small_and_metadata_plans_are_disjoint_and_can_be_combined(self) -> None:
        small = MODULE.build_plan(
            self.assets, small=True, metadata=False, all_assets=False
        )
        metadata = MODULE.build_plan(
            self.assets, small=False, metadata=True, all_assets=False
        )
        combined = MODULE.build_plan(
            self.assets, small=True, metadata=True, all_assets=False
        )
        default = MODULE.build_plan(
            self.assets, small=False, metadata=False, all_assets=False
        )

        self.assertEqual(len(small), 8)
        self.assertEqual(len(metadata), 5)
        self.assertEqual(len(combined), 13)
        self.assertEqual(default, small)
        self.assertTrue(all(not item.metadata_only for item in small))
        self.assertTrue(all(item.metadata_only for item in metadata))
        self.assertTrue(
            set(item.asset.key for item in small).isdisjoint(
                item.asset.key for item in metadata
            )
        )

    def test_all_requests_complete_snapshots_without_allow_patterns(self) -> None:
        plan = MODULE.build_plan(
            self.assets, small=False, metadata=False, all_assets=True
        )
        calls = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            return "/private/cache/result"

        with tempfile.TemporaryDirectory() as directory:
            for item in plan:
                MODULE.restore_asset(
                    item,
                    token="secret-token",
                    local_only=False,
                    max_workers=3,
                    cache_dir=Path(directory),
                    downloader=fake_snapshot_download,
                )
        self.assertEqual(len(calls), 13)
        for item, kwargs in zip(plan, calls):
            self.assertEqual(kwargs["repo_id"], item.asset.repo_id)
            self.assertEqual(kwargs["revision"], item.asset.revision)
            self.assertEqual(kwargs["repo_type"], "dataset")
            self.assertFalse(kwargs["local_files_only"])
            self.assertNotIn("allow_patterns", kwargs)

    def test_metadata_mode_passes_only_reviewed_allow_patterns(self) -> None:
        plan = MODULE.build_plan(
            self.assets, small=False, metadata=True, all_assets=False
        )
        calls = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            return "/private/cache/result"

        with tempfile.TemporaryDirectory() as directory:
            for item in plan:
                MODULE.restore_asset(
                    item,
                    token=None,
                    local_only=True,
                    max_workers=1,
                    cache_dir=Path(directory),
                    downloader=fake_snapshot_download,
                )
        self.assertEqual(len(calls), 5)
        for kwargs in calls:
            self.assertEqual(
                kwargs["allow_patterns"], list(MODULE.METADATA_ALLOW_PATTERNS)
            )
            self.assertTrue(kwargs["local_files_only"])
            self.assertNotIn("corpus/**", kwargs["allow_patterns"])

    def test_token_reader_prefers_environment_then_parses_dotenv_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "GITHUB_TOKEN=do-not-use\n"
                "export HUGGINGFACE_HUB_TOKEN='dotenv-secret'\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"HF_TOKEN": "env-secret"}, clear=True):
                self.assertEqual(MODULE.read_hf_token(env_file), "env-secret")
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(MODULE.read_hf_token(env_file), "dotenv-secret")

    def test_restore_output_and_failure_do_not_expose_token_or_paths(self) -> None:
        item = MODULE.build_plan(
            self.assets, small=True, metadata=False, all_assets=False
        )[0]
        token = "hf_sensitive_sentinel"
        sensitive_path = "/private/user/cache/sentinel"

        def successful_download(**_kwargs):
            return sensitive_path

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            MODULE.restore_asset(
                item,
                token=token,
                local_only=False,
                max_workers=1,
                cache_dir=Path(sensitive_path),
                downloader=successful_download,
            )
        self.assertNotIn(token, output.getvalue())
        self.assertNotIn(sensitive_path, output.getvalue())

        def failing_download(**_kwargs):
            raise RuntimeError(f"failure containing {token} and {sensitive_path}")

        with self.assertRaises(MODULE.RestoreError) as caught:
            MODULE.restore_asset(
                item,
                token=token,
                local_only=False,
                max_workers=1,
                cache_dir=Path(sensitive_path),
                downloader=failing_download,
            )
        self.assertNotIn(token, str(caught.exception))
        self.assertNotIn(sensitive_path, str(caught.exception))

    def test_all_rejects_selector_combinations(self) -> None:
        with self.assertRaises(MODULE.ManifestError):
            MODULE.build_plan(
                self.assets, small=True, metadata=False, all_assets=True
            )


if __name__ == "__main__":
    unittest.main()
