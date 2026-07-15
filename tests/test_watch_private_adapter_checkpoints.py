from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from safetensors.numpy import save_file

from scripts import watch_private_adapter_checkpoints as watcher


TOKEN_CANARY = "hf_" + "TEST_CANARY_NOT_A_CREDENTIAL_12345"


def make_checkpoint(root: Path, step: int = 250) -> Path:
    checkpoint = root / "v0-fixture" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": np.ones(
                (2, 4), dtype=np.float32
            ),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": np.ones(
                (4, 2), dtype=np.float32
            ),
        },
        checkpoint / watcher.WEIGHTS_NAME,
    )
    (checkpoint / watcher.CONFIG_NAME).write_text(
        json.dumps(
            {
                "base_model_name_or_path": "/private/local/model-cache",
                "bias": "none",
                "inference_mode": True,
                "lora_alpha": 128,
                "lora_dropout": 0.05,
                "peft_type": "LORA",
                "r": 64,
                "target_modules": ["q_proj"],
                "task_type": "FEATURE_EXTRACTION",
                # Modern PEFT emits these legitimate schema fields even for a
                # standard LoRA adapter.  Their names must not be confused with
                # credential-bearing arbitrary keys.
                "alora_invocation_tokens": None,
                "trainable_token_indices": None,
                "unknown_local_path": "/must/not/leak",
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / watcher.COMPLETION_SENTINEL).write_text(
        json.dumps(
            {
                "global_step": step,
                "best_model_checkpoint": "/private/output/checkpoint-250",
                "log_history": [
                    {"loss": 0.1, "step": step - 1},
                    {"eval_loss": 0.0125, "step": step},
                ],
            }
        ),
        encoding="utf-8",
    )
    # Representative forbidden neighbors.  The watcher must never read them
    # into an upload operation.
    for name in (
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "training_args.bin",
        "train.log",
        "raw-data.jsonl",
    ):
        (checkpoint / name).write_bytes(
            f"forbidden:{name}:{TOKEN_CANARY}".encode()
        )
    return checkpoint


def make_args(root: Path, *, upload: bool) -> Namespace:
    return Namespace(
        watch_dir=root,
        repo_id=watcher.DEFAULT_REPO_ID,
        state_file=None,
        env_file=root / ".env",
        base_model=watcher.DEFAULT_BASE_MODEL,
        base_revision=watcher.DEFAULT_BASE_REVISION,
        run_id=watcher.DEFAULT_RUN_ID,
        training_data_sha256=watcher.DEFAULT_TRAIN_SHA256,
        training_manifest_sha256="2" * 64,
        admission_report_sha256="3" * 64,
        poll_seconds=1.0,
        settle_seconds=0.0,
        once=True,
        upload=upload,
    )


class FakeApi:
    def __init__(
        self,
        *,
        private: bool = True,
        remote_manifest: Path | None = None,
        head_sha: str | None = "b" * 40,
    ):
        self.private = private
        self.remote_manifest = remote_manifest
        self.head_sha = head_sha
        self.create_repo_calls: list[dict] = []
        self.create_commit_calls: list[dict] = []
        self.uploaded: dict[str, bytes] = {}

    def create_repo(self, **kwargs):
        self.create_repo_calls.append(kwargs)

    def model_info(self, **kwargs):
        return SimpleNamespace(private=self.private, sha=self.head_sha)

    def file_exists(self, **kwargs):
        return self.remote_manifest is not None

    def hf_hub_download(self, **kwargs):
        if self.remote_manifest is None:
            raise AssertionError("no remote manifest configured")
        return str(self.remote_manifest)

    def create_commit(self, **kwargs):
        self.create_commit_calls.append(kwargs)
        for operation in kwargs["operations"]:
            source = operation.path_or_fileobj
            if isinstance(source, bytes):
                payload = source
            else:
                position = source.tell()
                source.seek(0)
                payload = source.read()
                source.seek(position)
            self.uploaded[operation.path_in_repo] = payload
        return SimpleNamespace(oid="a" * 40)


class FakeOperationAdd:
    def __init__(self, *, path_in_repo: str, path_or_fileobj):
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


class PrivateCheckpointWatcherTests(unittest.TestCase):
    def test_nested_sensitive_config_key_still_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_checkpoint(root)
            config_path = checkpoint / watcher.CONFIG_NAME
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["auto_mapping"] = {"token": TOKEN_CANARY}
            config_path.write_text(json.dumps(config), encoding="utf-8")
            args = make_args(root, upload=False)
            with self.assertRaises(watcher.WatcherError) as caught:
                watcher.validate_checkpoint(
                    checkpoint,
                    base_model=args.base_model,
                    base_revision=args.base_revision,
                    run_id=args.run_id,
                    training_data_sha256=args.training_data_sha256,
                    training_manifest_sha256=args.training_manifest_sha256,
                    admission_report_sha256=args.admission_report_sha256,
                    settle_seconds=0,
                    sleep=lambda _seconds: None,
                )
            self.assertEqual(caught.exception.code, "unsafe_config")
            self.assertNotIn(TOKEN_CANARY, str(caught.exception))

    def test_real_bfloat16_safetensors_payload_is_fully_validated(self) -> None:
        import torch
        from safetensors.torch import save_file as save_torch_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_checkpoint(root)
            weights = checkpoint / watcher.WEIGHTS_NAME
            save_torch_file(
                {
                    "layer.lora_A.default.weight": torch.ones(
                        (8, 16), dtype=torch.bfloat16
                    ),
                    "layer.lora_B.default.weight": torch.ones(
                        (16, 8), dtype=torch.bfloat16
                    ),
                },
                weights,
                metadata={"format": "pt"},
            )
            args = make_args(root, upload=False)
            validated = watcher.validate_checkpoint(
                checkpoint,
                staging_dir=root / watcher.STAGING_NAME,
                base_model=args.base_model,
                base_revision=args.base_revision,
                run_id=args.run_id,
                training_data_sha256=args.training_data_sha256,
                training_manifest_sha256=args.training_manifest_sha256,
                admission_report_sha256=args.admission_report_sha256,
                settle_seconds=0,
                sleep=lambda _seconds: None,
            )
            self.assertIsNotNone(validated)
            manifest = json.loads(validated.manifest_bytes)
            self.assertEqual(manifest["adapter"]["tensor_dtypes"], {"BF16": 2})
            self.assertEqual(manifest["adapter"]["parameter_count"], 256)
            self.assertEqual(validated.weights_sha256, watcher.sha256_file(weights))
            self.assertEqual(watcher.inspect_safetensors(validated.weights_path)["tensor_dtypes"], {"BF16": 2})
            validated.weights_path.unlink()

    def test_full_dry_run_makes_no_remote_or_state_and_prints_no_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_checkpoint(root)
            args = make_args(root, upload=False)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                watcher.run(args)
            output = stream.getvalue()
            self.assertIn('"event": "watcher_started"', output)
            self.assertIn('"event": "validated_dry_run"', output)
            self.assertNotIn(str(root), output)
            self.assertFalse((root / watcher.STATE_NAME).exists())
            self.assertEqual(list((root / watcher.STAGING_NAME).iterdir()), [])

    def test_upload_is_allowlist_only_private_and_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_checkpoint(root)
            args = make_args(root, upload=True)
            state_path = root / watcher.STATE_NAME
            api = FakeApi()
            remote = watcher.PrivateCandidateRemote(
                api=api, repo_id=args.repo_id, operation_add_cls=FakeOperationAdd
            )
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                first = watcher.scan_once(
                    args=args,
                    state_path=state_path,
                    remote=remote,
                    sleep=lambda _seconds: None,
                )
                second_api = FakeApi()
                second = watcher.scan_once(
                    args=args,
                    state_path=state_path,
                    remote=watcher.PrivateCandidateRemote(
                        api=second_api,
                        repo_id=args.repo_id,
                        operation_add_cls=FakeOperationAdd,
                    ),
                    sleep=lambda _seconds: None,
                )

            self.assertEqual(len(api.create_commit_calls), 1)
            self.assertEqual(api.create_commit_calls[0]["parent_commit"], "b" * 40)
            self.assertEqual(second_api.create_commit_calls, [])
            self.assertEqual(second_api.create_repo_calls, [])
            self.assertEqual(len(api.create_repo_calls), 1)
            self.assertTrue(api.create_repo_calls[0]["private"])
            self.assertTrue(api.create_repo_calls[0]["exist_ok"])
            self.assertEqual(set(first["checkpoints"]), {"checkpoint-250"})
            self.assertEqual(first, second)

            remote_names = {name.rsplit("/", 1)[-1] for name in api.uploaded}
            self.assertEqual(remote_names, watcher.REMOTE_ALLOWLIST)
            forbidden_names = {
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "training_args.bin",
                "train.log",
                "raw-data.jsonl",
                watcher.COMPLETION_SENTINEL,
            }
            self.assertTrue(forbidden_names.isdisjoint(remote_names))

            prefix = "checkpoints/checkpoint-250"
            sanitized_config = json.loads(
                api.uploaded[f"{prefix}/{watcher.CONFIG_NAME}"]
            )
            self.assertEqual(
                sanitized_config["base_model_name_or_path"],
                watcher.DEFAULT_BASE_MODEL,
            )
            self.assertEqual(
                sanitized_config["revision"], watcher.DEFAULT_BASE_REVISION
            )
            self.assertNotIn("unknown_local_path", sanitized_config)
            manifest_bytes = api.uploaded[f"{prefix}/{watcher.MANIFEST_NAME}"]
            manifest = json.loads(manifest_bytes)
            self.assertEqual(manifest["checkpoint"]["step"], 250)
            self.assertEqual(manifest["validation"]["eval_loss"], 0.0125)
            self.assertEqual(
                manifest["adapter"]["weights"]["sha256"],
                watcher.sha256_file(checkpoint / watcher.WEIGHTS_NAME),
            )
            all_uploaded = b"\n".join(api.uploaded.values())
            self.assertNotIn(str(root).encode(), all_uploaded)
            for forbidden in forbidden_names:
                self.assertNotIn(f"forbidden:{forbidden}".encode(), all_uploaded)
            self.assertNotIn(TOKEN_CANARY.encode(), all_uploaded)

            state_payload = state_path.read_text(encoding="utf-8")
            self.assertNotIn(str(root), state_payload)
            self.assertEqual(stat.S_IMODE(state_path.stat().st_mode), 0o600)
            staging = root / watcher.STAGING_NAME
            self.assertEqual(stat.S_IMODE(staging.stat().st_mode), 0o700)
            self.assertEqual(list(staging.iterdir()), [])
            output = stream.getvalue()
            self.assertNotIn(str(root), output)
            self.assertNotIn("optimizer", output)

    def test_remote_manifest_recovers_after_commit_before_state_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_checkpoint(root)
            args = make_args(root, upload=True)
            validated = watcher.validate_checkpoint(
                checkpoint,
                base_model=args.base_model,
                base_revision=args.base_revision,
                run_id=args.run_id,
                training_data_sha256=args.training_data_sha256,
                training_manifest_sha256=args.training_manifest_sha256,
                admission_report_sha256=args.admission_report_sha256,
                settle_seconds=0,
                sleep=lambda _seconds: None,
            )
            self.assertIsNotNone(validated)
            remote_manifest = root / "remote-manifest.json"
            remote_manifest.write_bytes(validated.manifest_bytes)
            api = FakeApi(remote_manifest=remote_manifest)
            remote = watcher.PrivateCandidateRemote(
                api=api, repo_id=args.repo_id, operation_add_cls=FakeOperationAdd
            )
            state = watcher.scan_once(
                args=args,
                state_path=root / watcher.STATE_NAME,
                remote=remote,
                sleep=lambda _seconds: None,
            )
            self.assertEqual(api.create_commit_calls, [])
            record = state["checkpoints"]["checkpoint-250"]
            self.assertTrue(record["recovered_existing_remote"])

    def test_incomplete_or_changing_checkpoint_is_not_uploaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_checkpoint(root)
            (checkpoint / watcher.COMPLETION_SENTINEL).unlink()
            args = make_args(root, upload=True)
            api = FakeApi()
            state = watcher.scan_once(
                args=args,
                state_path=root / watcher.STATE_NAME,
                remote=watcher.PrivateCandidateRemote(
                    api=api,
                    repo_id=args.repo_id,
                    operation_add_cls=FakeOperationAdd,
                ),
                sleep=lambda _seconds: None,
            )
            self.assertEqual(state["checkpoints"], {})
            self.assertEqual(api.create_repo_calls, [])

            make_checkpoint(root)

            def mutate(_seconds: float) -> None:
                with (checkpoint / watcher.WEIGHTS_NAME).open("ab") as handle:
                    handle.write(b"still-writing")

            validated = watcher.validate_checkpoint(
                checkpoint,
                base_model=args.base_model,
                base_revision=args.base_revision,
                run_id=args.run_id,
                training_data_sha256=args.training_data_sha256,
                training_manifest_sha256=args.training_manifest_sha256,
                admission_report_sha256=args.admission_report_sha256,
                settle_seconds=1,
                sleep=mutate,
            )
            self.assertIsNone(validated)

    def test_corrupt_safetensors_and_missing_eval_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_checkpoint(root)
            (checkpoint / watcher.WEIGHTS_NAME).write_bytes(b"not-safetensors")
            args = make_args(root, upload=False)
            with self.assertRaises(watcher.WatcherError) as caught:
                watcher.validate_checkpoint(
                    checkpoint,
                    base_model=args.base_model,
                    base_revision=args.base_revision,
                    run_id=args.run_id,
                    training_data_sha256=args.training_data_sha256,
                    training_manifest_sha256=None,
                    admission_report_sha256=None,
                    settle_seconds=0,
                    sleep=lambda _seconds: None,
                )
            self.assertEqual(caught.exception.code, "invalid_safetensors")

            checkpoint = make_checkpoint(root, step=500)
            (checkpoint / watcher.COMPLETION_SENTINEL).write_text(
                json.dumps({"global_step": 500, "log_history": []}),
                encoding="utf-8",
            )
            with self.assertRaises(watcher.WatcherError) as caught:
                watcher.validate_checkpoint(
                    checkpoint,
                    base_model=args.base_model,
                    base_revision=args.base_revision,
                    run_id=args.run_id,
                    training_data_sha256=args.training_data_sha256,
                    training_manifest_sha256=None,
                    admission_report_sha256=None,
                    settle_seconds=0,
                    sleep=lambda _seconds: None,
                )
            self.assertEqual(caught.exception.code, "incomplete_validation")

    def test_public_repository_is_refused_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_checkpoint(root)
            args = make_args(root, upload=True)
            api = FakeApi(private=False)
            with self.assertRaises(watcher.WatcherError) as caught:
                watcher.scan_once(
                    args=args,
                    state_path=root / watcher.STATE_NAME,
                    remote=watcher.PrivateCandidateRemote(
                        api=api,
                        repo_id=args.repo_id,
                        operation_add_cls=FakeOperationAdd,
                    ),
                    sleep=lambda _seconds: None,
                )
            self.assertEqual(caught.exception.code, "public_repo_refused")
            self.assertEqual(api.create_commit_calls, [])

    def test_nonfinite_tensor_symlink_and_source_race_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_checkpoint(root)
            save_file(
                {
                    "layer.lora_A.default.weight": np.array(
                        [[np.nan]], dtype=np.float32
                    ),
                    "layer.lora_B.default.weight": np.ones((1, 1), dtype=np.float32),
                },
                checkpoint / watcher.WEIGHTS_NAME,
            )
            args = make_args(root, upload=False)
            with self.assertRaises(watcher.WatcherError) as caught:
                watcher.validate_checkpoint(
                    checkpoint,
                    base_model=args.base_model,
                    base_revision=args.base_revision,
                    run_id=args.run_id,
                    training_data_sha256=args.training_data_sha256,
                    training_manifest_sha256=None,
                    admission_report_sha256=None,
                    settle_seconds=0,
                    sleep=lambda _seconds: None,
                )
            self.assertEqual(caught.exception.code, "invalid_safetensors")

            checkpoint = make_checkpoint(root, step=1000)
            save_file(
                {
                    "layer.lora_A.default.weight": np.ones((1, 1), dtype=np.float32),
                    "layer.lora_B.default.weight": np.ones((1, 1), dtype=np.float32),
                },
                checkpoint / watcher.WEIGHTS_NAME,
                metadata={"secret": TOKEN_CANARY},
            )
            with self.assertRaises(watcher.WatcherError) as caught:
                watcher.validate_checkpoint(
                    checkpoint,
                    base_model=args.base_model,
                    base_revision=args.base_revision,
                    run_id=args.run_id,
                    training_data_sha256=args.training_data_sha256,
                    training_manifest_sha256=None,
                    admission_report_sha256=None,
                    settle_seconds=0,
                    sleep=lambda _seconds: None,
                )
            self.assertEqual(caught.exception.code, "invalid_safetensors")

            # Restore valid bytes, then mutate source metadata immediately after
            # the private weights snapshot. The final fingerprint gate catches it.
            checkpoint = make_checkpoint(root)
            original_snapshot = watcher.snapshot_weights

            def snapshot_then_mutate(*snapshot_args, **snapshot_kwargs):
                result = original_snapshot(*snapshot_args, **snapshot_kwargs)
                config = checkpoint / watcher.CONFIG_NAME
                config.write_text(config.read_text() + " ", encoding="utf-8")
                return result

            with mock.patch.object(
                watcher, "snapshot_weights", side_effect=snapshot_then_mutate
            ):
                with self.assertRaises(watcher.WatcherError) as caught:
                    watcher.validate_checkpoint(
                        checkpoint,
                        staging_dir=root / watcher.STAGING_NAME,
                        base_model=args.base_model,
                        base_revision=args.base_revision,
                        run_id=args.run_id,
                        training_data_sha256=args.training_data_sha256,
                        training_manifest_sha256=None,
                        admission_report_sha256=None,
                        settle_seconds=0,
                        sleep=lambda _seconds: None,
                    )
            self.assertEqual(caught.exception.code, "checkpoint_changed")
            self.assertEqual(list((root / watcher.STAGING_NAME).iterdir()), [])

            symlink_root = root / "symlink-watch"
            target = root / "elsewhere/checkpoint-750"
            target.mkdir(parents=True)
            symlink_root.mkdir()
            (symlink_root / "checkpoint-750").symlink_to(target, target_is_directory=True)
            with self.assertRaises(watcher.WatcherError) as caught:
                watcher.discover_checkpoints(symlink_root)
            self.assertEqual(caught.exception.code, "unsafe_checkpoint")

    def test_remote_conflict_and_exception_text_never_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_checkpoint(root)
            args = make_args(root, upload=True)
            conflict_manifest = root / "remote.json"
            conflict_manifest.write_text(
                json.dumps(
                    {
                        "checkpoint": {"label": "checkpoint-250"},
                        "adapter": {
                            "weights": {"sha256": "0" * 64},
                            "config": {"sha256": "1" * 64},
                        },
                    }
                ),
                encoding="utf-8",
            )
            api = FakeApi(remote_manifest=conflict_manifest)
            with self.assertRaises(watcher.WatcherError) as caught:
                watcher.scan_once(
                    args=args,
                    state_path=root / watcher.STATE_NAME,
                    remote=watcher.PrivateCandidateRemote(
                        api=api,
                        repo_id=args.repo_id,
                        operation_add_cls=FakeOperationAdd,
                    ),
                    sleep=lambda _seconds: None,
                )
            self.assertEqual(caught.exception.code, "remote_conflict")
            self.assertEqual(api.create_commit_calls, [])

            secret = TOKEN_CANARY

            class ExplodingApi(FakeApi):
                def create_repo(self, **kwargs):
                    raise RuntimeError(f"{secret}:{root}")

            remote = watcher.PrivateCandidateRemote(
                api=ExplodingApi(),
                repo_id=args.repo_id,
                operation_add_cls=FakeOperationAdd,
            )
            with self.assertRaises(watcher.WatcherError) as caught:
                remote.ensure_private()
            rendered = str(caught.exception)
            self.assertNotIn(secret, rendered)
            self.assertNotIn(str(root), rendered)

    def test_corrupt_state_and_world_readable_dotenv_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / watcher.STATE_NAME
            state.write_text(
                json.dumps({"token": TOKEN_CANARY, "path": str(root)}),
                encoding="utf-8",
            )
            state.chmod(0o600)
            with self.assertRaises(watcher.WatcherError) as caught:
                watcher.load_state(state, watcher.DEFAULT_REPO_ID)
            self.assertEqual(caught.exception.code, "invalid_state")
            self.assertNotIn(TOKEN_CANARY, str(caught.exception))
            self.assertNotIn(str(root), str(caught.exception))

            env_file = root / ".env"
            env_file.write_text(f"HF_TOKEN={TOKEN_CANARY}\n", encoding="utf-8")
            env_file.chmod(0o644)
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(watcher.WatcherError) as caught:
                    watcher.read_hf_token(env_file)
            self.assertEqual(caught.exception.code, "unsafe_env")
            self.assertNotIn(TOKEN_CANARY, str(caught.exception))

    def test_token_reader_uses_memory_only_and_never_prints_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env"
            env_file.write_text(
                "OTHER=value\nexport HUGGINGFACE_HUB_TOKEN='dotenv-secret'\n",
                encoding="utf-8",
            )
            env_file.chmod(0o600)
            with mock.patch.dict(os.environ, {"HF_TOKEN": "env-secret"}, clear=True):
                self.assertEqual(watcher.read_hf_token(env_file), "env-secret")
                self.assertNotIn("HUGGINGFACE_HUB_TOKEN", os.environ)
            with mock.patch.dict(os.environ, {}, clear=True):
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    token = watcher.read_hf_token(env_file)
                self.assertEqual(token, "dotenv-secret")
                self.assertEqual(stream.getvalue(), "")
                self.assertEqual(dict(os.environ), {})


if __name__ == "__main__":
    unittest.main()
