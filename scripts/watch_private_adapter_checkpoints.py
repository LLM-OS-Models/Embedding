#!/usr/bin/env python3
"""Validate and incrementally publish completed LoRA checkpoints privately.

The remote commit is constructed from an explicit three-file allowlist.  Local
trainer state is used only as a completion/validation sentinel and is never
staged or sent to the Hub.  Uploading is opt-in via ``--upload``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import stat
import tempfile
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = (
    "LLM-OS-Models2/qwen3-embedding-8b-ko-performance200k-lora-r64-candidates"
)
DEFAULT_BASE_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_BASE_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
DEFAULT_TRAIN_SHA256 = (
    "8e2731ab25299ff558af675f067b253a6ce4375a850aa925acfe3b3117505e3c"
)
DEFAULT_RUN_ID = "qwen3-embedding-8b-ko-performance200k-lora-r64"
STATE_NAME = ".hf-candidate-upload-state.json"
LOCK_NAME = ".hf-candidate-upload-state.lock"
STAGING_NAME = ".hf-candidate-staging"
ARCHIVE_NAME = ".adapter-checkpoint-archive"
ARCHIVE_MANIFEST_NAME = "archive_manifest.json"
CHECKPOINT_RE = re.compile(r"checkpoint-([1-9][0-9]*)")
VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
REPO_ID_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_OID_RE = re.compile(r"[0-9a-f]{40,64}")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

WEIGHTS_NAME = "adapter_model.safetensors"
CONFIG_NAME = "adapter_config.json"
MANIFEST_NAME = "candidate_manifest.json"
COMPLETION_SENTINEL = "trainer_state.json"
REMOTE_ALLOWLIST = frozenset({WEIGHTS_NAME, CONFIG_NAME, MANIFEST_NAME})
ALLOWED_TENSOR_DTYPES = frozenset({"F16", "BF16", "F32"})
RETRYABLE_REMOTE_CODES = frozenset(
    {"remote_repo_failed", "remote_recovery_failed", "remote_upload_failed"}
)

# Pinned to the fields understood by the installed PEFT 0.19 line.  Unknown
# fields are omitted instead of risking accidental path or secret disclosure.
ADAPTER_CONFIG_ALLOWLIST = frozenset(
    {
        "alora_invocation_tokens",
        "alpha_pattern",
        "arrow_config",
        "auto_mapping",
        "base_model_name_or_path",
        "bias",
        "corda_config",
        "ensure_weight_tying",
        "eva_config",
        "exclude_modules",
        "fan_in_fan_out",
        "inference_mode",
        "init_lora_weights",
        "layer_replication",
        "layers_pattern",
        "layers_to_transform",
        "loftq_config",
        "lora_alpha",
        "lora_bias",
        "lora_dropout",
        "lora_ga_config",
        "megatron_config",
        "megatron_core",
        "modules_to_save",
        "peft_type",
        "peft_version",
        "qalora_group_size",
        "r",
        "rank_pattern",
        "revision",
        "target_modules",
        "target_parameters",
        "task_type",
        "trainable_token_indices",
        "use_bdlora",
        "use_dora",
        "use_qalora",
        "use_rslora",
    }
)
SENSITIVE_KEY_RE = re.compile(
    r"(?:^|_)(?:token|secret|password|passwd|authorization|cookie)(?:$|_)", re.I
)
TOKEN_VALUE_RES = (
    re.compile(r"\bhf_[A-Za-z0-9]{10,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{10,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{10,}", re.I),
)


class WatcherError(RuntimeError):
    """A deliberately path- and secret-free watcher failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ValidatedCheckpoint:
    label: str
    step: int
    weights_path: Path
    weights_sha256: str
    weights_size: int
    config_bytes: bytes
    config_sha256: str
    manifest_bytes: bytes
    manifest_sha256: str
    eval_loss: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch checkpoint-N directories and upload only verified LoRA adapter "
            "files to a private Hugging Face candidate repository."
        )
    )
    parser.add_argument("--watch-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base-revision", default=DEFAULT_BASE_REVISION)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--training-data-sha256", default=DEFAULT_TRAIN_SHA256)
    parser.add_argument("--training-manifest-sha256")
    parser.add_argument("--admission-report-sha256")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--settle-seconds", type=float, default=10.0)
    parser.add_argument("--remote-attempts", type=int, default=3)
    parser.add_argument("--remote-retry-seconds", type=float, default=15.0)
    parser.add_argument(
        "--once", action="store_true", help="Scan once instead of watching continuously"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Create/check the private repo and upload; omitted means local validation only",
    )
    parser.add_argument(
        "--no-local-archive",
        action="store_true",
        help=(
            "Do not retain validated adapter-only snapshots for later checkpoint "
            "averaging. Upload mode archives them by default."
        ),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(event: str, **fields: Any) -> None:
    """Print only caller-supplied safe identifiers; never paths or exceptions."""

    payload = {"event": event, **fields}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _parse_dotenv_value(raw_value: str) -> str | None:
    try:
        parts = shlex.split(raw_value, comments=False, posix=True)
    except ValueError:
        return None
    return parts[0] if parts else None


def read_hf_token(env_file: Path) -> str | None:
    """Return a token in memory without exporting, persisting, or logging it."""

    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    try:
        env_stat = env_file.lstat()
        if env_file.is_symlink() or not stat.S_ISREG(env_stat.st_mode):
            raise WatcherError("unsafe_env", "ignored .env must be a regular file")
        if env_stat.st_uid != os.geteuid():
            raise WatcherError("unsafe_env", "ignored .env must be owned by the current user")
        if stat.S_IMODE(env_stat.st_mode) & 0o077:
            raise WatcherError("unsafe_env", "ignored .env permissions must be 0600")
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    except WatcherError:
        raise
    except (OSError, UnicodeError) as error:
        raise WatcherError(
            "unsafe_env", f"ignored .env is unreadable ({type(error).__name__})"
        ) from None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        raw_key, raw_value = line.split("=", 1)
        key = raw_key.removeprefix("export ").strip()
        if key not in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"}:
            continue
        value = _parse_dotenv_value(raw_value)
        if value:
            return value
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def prepare_staging_dir(path: Path) -> None:
    try:
        if path.exists():
            if path.is_symlink() or not stat.S_ISDIR(path.lstat().st_mode):
                raise WatcherError("unsafe_staging", "staging root is not a regular directory")
            if path.lstat().st_uid != os.geteuid():
                raise WatcherError("unsafe_staging", "staging root has an unexpected owner")
        else:
            path.mkdir(parents=True, mode=0o700)
        os.chmod(path, 0o700)
    except WatcherError:
        raise
    except OSError as error:
        raise WatcherError(
            "staging_failed", f"staging setup failed ({type(error).__name__})"
        ) from None


def snapshot_weights(
    source: Path, *, staging_dir: Path, label: str
) -> tuple[Path, str, int]:
    """Copy a stable source to a private immutable upload snapshot."""

    prepare_staging_dir(staging_dir)
    before = _regular_signature(source)
    if before is None:
        raise WatcherError("checkpoint_changed", "adapter weights disappeared")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=staging_dir, prefix=f".{label}.", suffix=".safetensors.tmp"
    )
    temporary = Path(temporary_name)
    destination: Path | None = None
    digest = hashlib.sha256()
    size = 0
    try:
        os.fchmod(descriptor, 0o600)
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            for block in iter(lambda: input_handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
                output_handle.write(block)
                size += len(block)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if before != _regular_signature(source):
            raise WatcherError("checkpoint_changed", "adapter weights changed while staging")
        source_sha256 = digest.hexdigest()
        destination = staging_dir / f"{label}-{source_sha256}.safetensors"
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
        if _regular_signature(destination) is None or sha256_file(destination) != source_sha256:
            raise WatcherError("staging_checksum_failed", "staged adapter checksum mismatch")
        return destination, source_sha256, size
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise


def json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _prepare_private_directory(path: Path, *, code: str) -> None:
    try:
        if path.exists():
            path_stat = path.lstat()
            if path.is_symlink() or not stat.S_ISDIR(path_stat.st_mode):
                raise WatcherError(code, "archive path is not a regular directory")
            if path_stat.st_uid != os.geteuid():
                raise WatcherError(code, "archive path has an unexpected owner")
        else:
            path.mkdir(mode=0o700)
        os.chmod(path, 0o700)
    except WatcherError:
        raise
    except OSError as error:
        raise WatcherError(
            code, f"archive directory setup failed ({type(error).__name__})"
        ) from None


def archive_destination(checkpoint: Path, archive_root: Path) -> Path:
    version = checkpoint.parent.name
    if not VERSION_RE.fullmatch(version):
        raise WatcherError("unsafe_archive", "training version label is invalid")
    if not CHECKPOINT_RE.fullmatch(checkpoint.name):
        raise WatcherError("unsafe_archive", "checkpoint label is invalid")
    return archive_root / version / checkpoint.name


def archived_checkpoint_matches(
    checkpoint: Path,
    archive_root: Path,
    *,
    expected_weights_sha256: str,
    expected_config_sha256: str | None = None,
) -> bool:
    """Cheaply validate an immutable archive using its creation-time hashes."""

    destination = archive_destination(checkpoint, archive_root)
    if not destination.exists():
        return False
    try:
        destination_stat = destination.lstat()
        if destination.is_symlink() or not stat.S_ISDIR(destination_stat.st_mode):
            raise WatcherError("unsafe_archive", "archive checkpoint is not a directory")
        if destination_stat.st_uid != os.geteuid():
            raise WatcherError("unsafe_archive", "archive checkpoint has an unexpected owner")
        manifest = load_small_json(
            destination / ARCHIVE_MANIFEST_NAME,
            max_bytes=1024 * 1024,
            code="unsafe_archive",
        )
        weights = destination / WEIGHTS_NAME
        config = destination / CONFIG_NAME
        weights_signature = _regular_signature(weights)
        config_signature = _regular_signature(config)
        if weights_signature is None or config_signature is None:
            raise WatcherError("unsafe_archive", "archive payload is missing or unsafe")
        if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
            raise WatcherError("unsafe_archive", "archive manifest is incomplete")
        if manifest.get("checkpoint", {}).get("label") != checkpoint.name:
            raise WatcherError("unsafe_archive", "archive checkpoint label drift")
        adapter = manifest.get("adapter", {})
        weights_meta = adapter.get("weights", {})
        config_meta = adapter.get("config", {})
        if (
            weights_meta.get("sha256") != expected_weights_sha256
            or weights_meta.get("size_bytes") != weights_signature[2]
            or config_meta.get("size_bytes") != config_signature[2]
        ):
            raise WatcherError("unsafe_archive", "archive payload metadata drift")
        if expected_config_sha256 is not None and (
            config_meta.get("sha256") != expected_config_sha256
        ):
            raise WatcherError("unsafe_archive", "archive config metadata drift")
        return True
    except WatcherError:
        raise
    except (AttributeError, OSError, TypeError) as error:
        raise WatcherError(
            "unsafe_archive", f"archive validation failed ({type(error).__name__})"
        ) from None


def archive_validated_checkpoint(
    checkpoint: Path,
    validated: ValidatedCheckpoint,
    *,
    archive_root: Path,
    run_id: str,
) -> Path:
    """Atomically retain only validated adapter bytes and sanitized metadata."""

    _prepare_private_directory(archive_root, code="archive_failed")
    destination = archive_destination(checkpoint, archive_root)
    version_root = destination.parent
    _prepare_private_directory(version_root, code="archive_failed")
    if destination.exists():
        if archived_checkpoint_matches(
            checkpoint,
            archive_root,
            expected_weights_sha256=validated.weights_sha256,
            expected_config_sha256=validated.config_sha256,
        ):
            return destination
        raise WatcherError("archive_conflict", "existing archive does not match checkpoint")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{checkpoint.name}.archive-", dir=version_root)
    )
    try:
        os.chmod(temporary, 0o700)
        archived_weights = temporary / WEIGHTS_NAME
        os.replace(validated.weights_path, archived_weights)
        os.chmod(archived_weights, 0o600)
        archived_config = temporary / CONFIG_NAME
        archived_config.write_bytes(validated.config_bytes)
        os.chmod(archived_config, 0o600)
        manifest = {
            "schema_version": 1,
            "artifact_kind": "validated-local-lora-checkpoint-archive",
            "status": "complete",
            "created_at_utc": utc_now(),
            "run_id": run_id,
            "training_version": checkpoint.parent.name,
            "checkpoint": {"label": validated.label, "step": validated.step},
            "adapter": {
                "weights": {
                    "file": WEIGHTS_NAME,
                    "sha256": validated.weights_sha256,
                    "size_bytes": validated.weights_size,
                },
                "config": {
                    "file": CONFIG_NAME,
                    "sha256": validated.config_sha256,
                    "size_bytes": len(validated.config_bytes),
                },
            },
            "source_validation_manifest_sha256": validated.manifest_sha256,
            "contents_allowlist": sorted(
                {WEIGHTS_NAME, CONFIG_NAME, ARCHIVE_MANIFEST_NAME}
            ),
        }
        manifest_path = temporary / ARCHIVE_MANIFEST_NAME
        manifest_path.write_bytes(json_bytes(sanitize_json_value(manifest)))
        os.chmod(manifest_path, 0o600)
        if sha256_file(archived_weights) != validated.weights_sha256:
            raise WatcherError("archive_failed", "archived adapter checksum mismatch")
        if sha256_file(archived_config) != validated.config_sha256:
            raise WatcherError("archive_failed", "archived config checksum mismatch")
        os.replace(temporary, destination)
        return destination
    except WatcherError:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    except OSError as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise WatcherError(
            "archive_failed", f"checkpoint archive failed ({type(error).__name__})"
        ) from None


def load_small_json(path: Path, *, max_bytes: int, code: str) -> dict[str, Any]:
    try:
        file_stat = path.lstat()
        if not stat.S_ISREG(file_stat.st_mode) or path.is_symlink():
            raise WatcherError(code, "required metadata is not a regular file")
        if file_stat.st_uid != os.geteuid():
            raise WatcherError(code, "required metadata has an unexpected owner")
        if file_stat.st_size < 2 or file_stat.st_size > max_bytes:
            raise WatcherError(code, "required metadata has an invalid size")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except WatcherError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WatcherError(code, f"required metadata is unreadable ({type(error).__name__})") from None
    if not isinstance(payload, dict):
        raise WatcherError(code, "required metadata is not a JSON object")
    return payload


def _is_absolute_path_string(value: str) -> bool:
    return value.startswith(("/", "\\\\")) or bool(
        re.match(r"^[A-Za-z]:[\\/]", value)
    )


def _contains_absolute_path_string(value: str) -> bool:
    return _is_absolute_path_string(value) or bool(
        re.search(r"(?:^|[\s=:])(?:/[A-Za-z0-9_.-]|[A-Za-z]:[\\/])", value)
    )


def sanitize_json_value(value: Any, *, key: str = "") -> Any:
    if SENSITIVE_KEY_RE.search(key):
        raise WatcherError("unsafe_config", "adapter config contains a sensitive key")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WatcherError("unsafe_config", "adapter config contains a non-finite number")
        return value
    if isinstance(value, str):
        if _contains_absolute_path_string(value):
            raise WatcherError("unsafe_config", "adapter config contains an absolute path")
        if any(pattern.search(value) for pattern in TOKEN_VALUE_RES):
            raise WatcherError("unsafe_config", "adapter config contains a token-like value")
        if "\x00" in value:
            raise WatcherError("unsafe_config", "adapter config contains a NUL byte")
        return value
    if isinstance(value, list):
        return [sanitize_json_value(item, key=key) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for nested_key, nested_value in value.items():
            if not isinstance(nested_key, str):
                raise WatcherError("unsafe_config", "adapter config has a non-string key")
            result[nested_key] = sanitize_json_value(nested_value, key=nested_key)
        return result
    raise WatcherError("unsafe_config", "adapter config contains an unsupported value")


def sanitize_adapter_config(
    source: dict[str, Any], *, base_model: str, base_revision: str
) -> dict[str, Any]:
    if source.get("peft_type") != "LORA":
        raise WatcherError("invalid_config", "checkpoint is not a PEFT LoRA adapter")
    rank = source.get("r")
    alpha = source.get("lora_alpha")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise WatcherError("invalid_config", "LoRA rank is missing or invalid")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or alpha <= 0:
        raise WatcherError("invalid_config", "LoRA alpha is missing or invalid")
    target_modules = source.get("target_modules")
    if not isinstance(target_modules, (list, str)) or not target_modules:
        raise WatcherError("invalid_config", "LoRA target_modules is missing or invalid")

    sanitized = {
        # The top-level key is already selected by the exact PEFT schema
        # allowlist.  Sanitize its value directly so legitimate field names
        # such as ``trainable_token_indices`` are not mistaken for credentials.
        # Any keys nested inside a mapping are still checked by
        # ``sanitize_json_value``.
        key: sanitize_json_value(source[key])
        for key in sorted(source.keys() & ADAPTER_CONFIG_ALLOWLIST)
        if key not in {"base_model_name_or_path", "revision"}
    }
    # Validate caller-controlled lineage identifiers too.  The output key set
    # remains the exact allowlist plus these two fixed lineage keys.
    sanitized["base_model_name_or_path"] = sanitize_json_value(base_model)
    sanitized["revision"] = sanitize_json_value(base_revision)
    return sanitized


def _regular_signature(path: Path) -> tuple[int, int, int, int] | None:
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise WatcherError(
            "checkpoint_stat_failed", f"checkpoint stat failed ({type(error).__name__})"
        ) from None
    if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size < 1:
        raise WatcherError("unsafe_checkpoint", "checkpoint contains a non-regular required file")
    if file_stat.st_uid != os.geteuid():
        raise WatcherError("unsafe_checkpoint", "checkpoint required file has an unexpected owner")
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
    )


def checkpoint_is_settled(
    checkpoint: Path, *, settle_seconds: float, sleep: Callable[[float], None]
) -> bool:
    required = (
        checkpoint / WEIGHTS_NAME,
        checkpoint / CONFIG_NAME,
        checkpoint / COMPLETION_SENTINEL,
    )
    before = tuple(_regular_signature(path) for path in required)
    if any(signature is None for signature in before):
        return False
    if settle_seconds:
        sleep(settle_seconds)
    after = tuple(_regular_signature(path) for path in required)
    return before == after and all(signature is not None for signature in after)


def read_completed_eval(trainer_state: dict[str, Any], *, step: int) -> float:
    global_step = trainer_state.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step != step:
        raise WatcherError(
            "incomplete_validation", "trainer completion step does not match checkpoint"
        )
    history = trainer_state.get("log_history")
    if not isinstance(history, list):
        raise WatcherError("incomplete_validation", "trainer state has no validation history")
    losses: list[float] = []
    for entry in history:
        if not isinstance(entry, dict) or entry.get("step") != step:
            continue
        value = entry.get("eval_loss")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        if math.isfinite(value):
            losses.append(value)
    if not losses:
        raise WatcherError(
            "incomplete_validation", "checkpoint has no completed same-step eval_loss"
        )
    return losses[-1]


def inspect_safetensors(path: Path) -> dict[str, Any]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError:
        raise WatcherError(
            "missing_dependency", "torch and safetensors are required to validate weights"
        ) from None
    # Full-payload finite checks over a ~700MB r64 adapter otherwise let
    # PyTorch fan out across every CPU core and starve the active Trainer's
    # tokenization/dataloader workers.  One validation thread keeps the H100
    # supplied while the recovery upload proceeds in the background.  A small
    # explicit override is available for maintenance windows with no training.
    raw_threads = os.environ.get("EMBEDDING_WATCHER_TORCH_THREADS", "1")
    try:
        validation_threads = int(raw_threads)
    except ValueError:
        raise WatcherError(
            "invalid_argument", "watcher validation thread count must be an integer"
        ) from None
    if validation_threads < 1 or validation_threads > 8:
        raise WatcherError(
            "invalid_argument", "watcher validation thread count must be in [1, 8]"
        )
    torch.set_num_threads(validation_threads)
    try:
        # NumPy cannot materialize BF16 safetensors.  PyTorch's CPU backend
        # supports F16/BF16/F32 and lets us validate every payload byte/value.
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if not keys:
                raise WatcherError("invalid_safetensors", "adapter safetensors has no tensors")
            metadata = handle.metadata() or {}
            if set(metadata) - {"format"} or metadata.get("format") not in {
                None,
                "pt",
                "torch",
            }:
                raise WatcherError(
                    "invalid_safetensors", "adapter safetensors metadata is not allowlisted"
                )
            parameter_count = 0
            dtypes: Counter[str] = Counter()
            for key in keys:
                if (
                    not isinstance(key, str)
                    or "\x00" in key
                    or "\n" in key
                    or "/" in key
                    or "\\" in key
                    or _is_absolute_path_string(key)
                    or any(pattern.search(key) for pattern in TOKEN_VALUE_RES)
                ):
                    raise WatcherError(
                        "invalid_safetensors", "adapter safetensors has an unsafe tensor key"
                    )
                tensor_slice = handle.get_slice(key)
                shape = tensor_slice.get_shape()
                if not shape or any(not isinstance(dim, int) or dim < 0 for dim in shape):
                    raise WatcherError(
                        "invalid_safetensors", "adapter safetensors has an invalid shape"
                    )
                dtype = str(tensor_slice.get_dtype())
                if dtype not in ALLOWED_TENSOR_DTYPES:
                    raise WatcherError(
                        "invalid_safetensors", "adapter safetensors has a disallowed dtype"
                    )
                # Materializing each tensor verifies that every declared payload
                # range is readable, not merely that the safetensors header parses.
                tensor = handle.get_tensor(key)
                if not bool(torch.isfinite(tensor).all().item()):
                    raise WatcherError(
                        "invalid_safetensors", "adapter safetensors contains non-finite values"
                    )
                parameter_count += tensor.numel()
                dtypes[dtype] += 1
                del tensor
    except WatcherError:
        raise
    except Exception as error:
        raise WatcherError(
            "invalid_safetensors",
            f"adapter safetensors failed structural validation ({type(error).__name__})",
        ) from None
    if not any("lora_A" in key for key in keys) or not any("lora_B" in key for key in keys):
        raise WatcherError(
            "invalid_safetensors", "adapter safetensors lacks LoRA A/B tensor pairs"
        )
    return {
        "tensor_count": len(keys),
        "parameter_count": parameter_count,
        "tensor_dtypes": dict(sorted(dtypes.items())),
        "validation_cpu_threads": validation_threads,
    }


def validate_checkpoint(
    checkpoint: Path,
    *,
    staging_dir: Path | None = None,
    base_model: str,
    base_revision: str,
    run_id: str,
    training_data_sha256: str,
    training_manifest_sha256: str | None,
    admission_report_sha256: str | None,
    settle_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> ValidatedCheckpoint | None:
    match = CHECKPOINT_RE.fullmatch(checkpoint.name)
    if not match:
        raise WatcherError("invalid_checkpoint", "checkpoint label is invalid")
    label = checkpoint.name
    step = int(match.group(1))
    if not checkpoint_is_settled(
        checkpoint, settle_seconds=settle_seconds, sleep=sleep
    ):
        return None

    required_paths = (
        checkpoint / WEIGHTS_NAME,
        checkpoint / CONFIG_NAME,
        checkpoint / COMPLETION_SENTINEL,
    )
    settled_signatures = tuple(_regular_signature(path) for path in required_paths)

    trainer_state = load_small_json(
        checkpoint / COMPLETION_SENTINEL,
        max_bytes=64 * 1024 * 1024,
        code="invalid_trainer_state",
    )
    eval_loss = read_completed_eval(trainer_state, step=step)
    source_config = load_small_json(
        checkpoint / CONFIG_NAME,
        max_bytes=2 * 1024 * 1024,
        code="invalid_config",
    )
    config = sanitize_adapter_config(
        source_config, base_model=base_model, base_revision=base_revision
    )
    config_bytes = json_bytes(config)
    weights_source = checkpoint / WEIGHTS_NAME
    staging_dir = staging_dir or checkpoint.parent / STAGING_NAME
    staged_weights: Path | None = None
    try:
        staged_weights, weights_sha256, weights_size = snapshot_weights(
            weights_source, staging_dir=staging_dir, label=label
        )
        tensor_summary = inspect_safetensors(staged_weights)
        if settled_signatures != tuple(
            _regular_signature(path) for path in required_paths
        ):
            raise WatcherError("checkpoint_changed", "checkpoint changed during validation")

        config_sha256 = sha256_bytes(config_bytes)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "artifact_kind": "peft-lora-checkpoint-candidate",
            "distribution": "private-candidate-only",
            "checkpoint": {"label": label, "step": step},
            "adapter": {
                "weights": {
                    "file": WEIGHTS_NAME,
                    "sha256": weights_sha256,
                    "size_bytes": weights_size,
                },
                "config": {
                    "file": CONFIG_NAME,
                    "sha256": config_sha256,
                    "size_bytes": len(config_bytes),
                },
                **tensor_summary,
            },
            "validation": {
                "completion_sentinel_observed": True,
                "same_step_eval_observed": True,
                "eval_loss": eval_loss,
                "safetensors_full_payload_validation": "pass",
                "all_tensor_values_finite": True,
                "staged_snapshot_sha256_reverified": True,
            },
            "lineage": {
                "run_id": run_id,
                "base_model": {"id": base_model, "revision": base_revision},
                "training_data_sha256": training_data_sha256,
                "training_manifest_sha256": training_manifest_sha256,
                "fa2_admission_report_sha256": admission_report_sha256,
            },
            "remote_allowlist": sorted(REMOTE_ALLOWLIST),
            "release_eligible": False,
            "excluded": [
                "optimizer state",
                "scheduler state",
                "RNG state",
                "trainer state",
                "training arguments",
                "logs",
                "raw or processed training data",
                "local filesystem paths",
                "credentials",
            ],
        }
        # Validate every caller-derived string before materializing upload bytes.
        manifest = sanitize_json_value(manifest)
        manifest_bytes = json_bytes(manifest)
        return ValidatedCheckpoint(
            label=label,
            step=step,
            weights_path=staged_weights,
            weights_sha256=weights_sha256,
            weights_size=weights_size,
            config_bytes=config_bytes,
            config_sha256=config_sha256,
            manifest_bytes=manifest_bytes,
            manifest_sha256=sha256_bytes(manifest_bytes),
            eval_loss=eval_loss,
        )
    except BaseException:
        if staged_weights is not None:
            staged_weights.unlink(missing_ok=True)
        raise


def discover_checkpoints(watch_dir: Path) -> list[Path]:
    found: dict[str, Path] = {}
    try:
        root = watch_dir.resolve(strict=True)
    except OSError as error:
        raise WatcherError(
            "checkpoint_discovery_failed",
            f"watch root resolution failed ({type(error).__name__})",
        ) from None
    try:
        candidates: list[Path] = []
        for current, directories, _files in os.walk(root, followlinks=False):
            current_path = Path(current)
            safe_directories: list[str] = []
            for name in directories:
                if current_path == root and name in {STAGING_NAME, ARCHIVE_NAME}:
                    continue
                candidate = current_path / name
                if candidate.is_symlink():
                    if CHECKPOINT_RE.fullmatch(name):
                        raise WatcherError(
                            "unsafe_checkpoint", "checkpoint directory must not be a symlink"
                        )
                    continue
                safe_directories.append(name)
                if CHECKPOINT_RE.fullmatch(name):
                    candidates.append(candidate)
            directories[:] = safe_directories
    except WatcherError:
        raise
    except OSError as error:
        raise WatcherError(
            "checkpoint_discovery_failed",
            f"checkpoint discovery failed ({type(error).__name__})",
        ) from None
    for candidate in candidates:
        if not CHECKPOINT_RE.fullmatch(candidate.name):
            continue
        try:
            candidate_stat = candidate.lstat()
            mode = candidate_stat.st_mode
        except OSError:
            continue
        if candidate.is_symlink() or not stat.S_ISDIR(mode):
            raise WatcherError("unsafe_checkpoint", "checkpoint directory is not regular")
        if candidate_stat.st_uid != os.geteuid():
            raise WatcherError("unsafe_checkpoint", "checkpoint directory has an unexpected owner")
        try:
            candidate.resolve(strict=True).relative_to(root)
        except (OSError, ValueError):
            raise WatcherError("unsafe_checkpoint", "checkpoint escapes the watch root") from None
        if candidate.name in found:
            raise WatcherError(
                "duplicate_checkpoint", f"multiple {candidate.name} directories were found"
            )
        found[candidate.name] = candidate
    return sorted(found.values(), key=lambda path: int(CHECKPOINT_RE.fullmatch(path.name).group(1)))


def empty_state(repo_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "repo_id": repo_id,
        "checkpoints": {},
    }


def load_state(path: Path, repo_id: str) -> dict[str, Any]:
    if not path.exists():
        return empty_state(repo_id)
    if path.is_symlink():
        raise WatcherError("unsafe_state", "state file must not be a symlink")
    state = load_small_json(path, max_bytes=8 * 1024 * 1024, code="invalid_state")
    if state.get("schema_version") != 1 or state.get("repo_id") != repo_id:
        raise WatcherError("invalid_state", "state schema or repository does not match")
    checkpoints = state.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise WatcherError("invalid_state", "state checkpoint map is invalid")
    for label, record in checkpoints.items():
        if not isinstance(label, str) or not CHECKPOINT_RE.fullmatch(label):
            raise WatcherError("invalid_state", "state contains an invalid checkpoint label")
        if not isinstance(record, dict) or record.get("status") != "uploaded":
            raise WatcherError("invalid_state", "state contains an invalid checkpoint record")
        for field in (
            "adapter_sha256",
            "adapter_config_sha256",
            "candidate_manifest_sha256",
        ):
            if not SHA256_RE.fullmatch(str(record.get(field, ""))):
                raise WatcherError("invalid_state", "state contains an invalid checksum")
        step = int(CHECKPOINT_RE.fullmatch(label).group(1))
        if record.get("step") != step:
            raise WatcherError("invalid_state", "state checkpoint step is inconsistent")
        oid = record.get("commit_oid")
        if oid is not None and (
            not isinstance(oid, str) or not COMMIT_OID_RE.fullmatch(oid)
        ):
            raise WatcherError("invalid_state", "state contains an invalid commit ID")
    return state


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json_bytes(state))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def state_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        lock_stat = os.fstat(descriptor)
        if lock_stat.st_uid != os.geteuid() or not stat.S_ISREG(lock_stat.st_mode):
            raise OSError("unsafe lock owner or type")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise WatcherError(
            "state_lock_failed", f"state lock failed ({type(error).__name__})"
        ) from None
    try:
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class PrivateCandidateRemote:
    def __init__(self, *, api: Any, repo_id: str, operation_add_cls: Any = None):
        self.api = api
        self.repo_id = repo_id
        self.operation_add_cls = operation_add_cls
        self._private_checked = False
        self._head_sha: str | None = None

    def _check_private_info(self) -> str | None:
        try:
            info = self.api.model_info(repo_id=self.repo_id)
        except Exception as error:
            raise WatcherError(
                "remote_repo_failed",
                f"private repository check failed ({type(error).__name__})",
            ) from None
        if getattr(info, "private", None) is not True:
            raise WatcherError(
                "public_repo_refused", "refusing to upload because repository is not private"
            )
        sha = getattr(info, "sha", None)
        return sha if isinstance(sha, str) and COMMIT_OID_RE.fullmatch(sha) else None

    def ensure_private(self) -> None:
        if self._private_checked:
            return
        try:
            self.api.create_repo(
                repo_id=self.repo_id,
                repo_type="model",
                private=True,
                exist_ok=True,
            )
        except Exception as error:
            raise WatcherError(
                "remote_repo_failed",
                f"private repository check failed ({type(error).__name__})",
            ) from None
        self._head_sha = self._check_private_info()
        self._private_checked = True

    @staticmethod
    def remote_prefix(checkpoint: ValidatedCheckpoint) -> str:
        return f"checkpoints/{checkpoint.label}"

    def recover_existing(self, checkpoint: ValidatedCheckpoint) -> str | None:
        self.ensure_private()
        remote_manifest = f"{self.remote_prefix(checkpoint)}/{MANIFEST_NAME}"
        try:
            exists = self.api.file_exists(
                repo_id=self.repo_id,
                filename=remote_manifest,
                repo_type="model",
            )
            if not exists:
                return None
            downloaded = Path(
                self.api.hf_hub_download(
                    repo_id=self.repo_id,
                    filename=remote_manifest,
                    repo_type="model",
                )
            )
            payload = json.loads(downloaded.read_text(encoding="utf-8"))
        except Exception as error:
            raise WatcherError(
                "remote_recovery_failed",
                f"remote idempotency check failed ({type(error).__name__})",
            ) from None
        try:
            remote_label = payload["checkpoint"]["label"]
            remote_weights_sha = payload["adapter"]["weights"]["sha256"]
            remote_config_sha = payload["adapter"]["config"]["sha256"]
        except (KeyError, TypeError):
            raise WatcherError(
                "remote_conflict", "existing remote candidate manifest is malformed"
            ) from None
        if (
            remote_label != checkpoint.label
            or remote_weights_sha != checkpoint.weights_sha256
            or remote_config_sha != checkpoint.config_sha256
        ):
            raise WatcherError(
                "remote_conflict", "existing remote checkpoint has different checksums"
            )
        return sha256_file(downloaded)

    def upload(self, checkpoint: ValidatedCheckpoint) -> str | None:
        self.ensure_private()
        CommitOperationAdd = self.operation_add_cls
        if CommitOperationAdd is None:
            try:
                from huggingface_hub import CommitOperationAdd
            except ImportError:
                raise WatcherError(
                    "missing_dependency", "huggingface_hub is required for --upload"
                ) from None
        prefix = self.remote_prefix(checkpoint)
        try:
            # Observe HEAD immediately before committing and use it as a CAS
            # parent. A concurrent writer then causes a conflict, not overwrite.
            parent_commit = self._check_private_info()
            with ExitStack() as stack:
                weights_handle = stack.enter_context(checkpoint.weights_path.open("rb"))
                operations = [
                    CommitOperationAdd(
                        path_in_repo=f"{prefix}/{WEIGHTS_NAME}",
                        path_or_fileobj=weights_handle,
                    ),
                    CommitOperationAdd(
                        path_in_repo=f"{prefix}/{CONFIG_NAME}",
                        path_or_fileobj=checkpoint.config_bytes,
                    ),
                    CommitOperationAdd(
                        path_in_repo=f"{prefix}/{MANIFEST_NAME}",
                        path_or_fileobj=checkpoint.manifest_bytes,
                    ),
                ]
                uploaded_names = {operation.path_in_repo.rsplit("/", 1)[-1] for operation in operations}
                if uploaded_names != REMOTE_ALLOWLIST or len(operations) != 3:
                    raise WatcherError("allowlist_violation", "remote operation allowlist drift")
                commit_kwargs = {
                    "repo_id": self.repo_id,
                    "repo_type": "model",
                    "operations": operations,
                    "commit_message": f"Add verified private candidate {checkpoint.label}",
                    "commit_description": (
                        "Allowlist-only LoRA adapter checkpoint; no trainer or data artifacts"
                    ),
                    "num_threads": 1,
                }
                if parent_commit is not None:
                    commit_kwargs["parent_commit"] = parent_commit
                commit = self.api.create_commit(
                    **commit_kwargs,
                )
        except WatcherError:
            raise
        except Exception as error:
            raise WatcherError(
                "remote_upload_failed",
                f"private candidate upload failed ({type(error).__name__})",
            ) from None
        oid = getattr(commit, "oid", None)
        result = oid if isinstance(oid, str) and COMMIT_OID_RE.fullmatch(oid) else None
        # Visibility is checked again after the commit. A repository whose
        # visibility changed concurrently is never recorded as uploaded.
        self._head_sha = self._check_private_info()
        return result


def checkpoint_state_record(
    checkpoint: ValidatedCheckpoint,
    *,
    commit_oid: str | None,
    recovered: bool,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "status": "uploaded",
        "uploaded_at_utc": utc_now(),
        "step": checkpoint.step,
        "adapter_sha256": checkpoint.weights_sha256,
        "adapter_config_sha256": checkpoint.config_sha256,
        "candidate_manifest_sha256": manifest_sha256 or checkpoint.manifest_sha256,
        "commit_oid": commit_oid,
        "recovered_existing_remote": recovered,
    }


def publish_with_retry(
    remote: PrivateCandidateRemote,
    checkpoint: ValidatedCheckpoint,
    *,
    attempts: int,
    retry_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str | None, bool, str | None]:
    """Publish exactly once, retrying only transient remote failures."""

    for attempt in range(1, attempts + 1):
        try:
            recovered_manifest_sha = remote.recover_existing(checkpoint)
            recovered = recovered_manifest_sha is not None
            commit_oid = None
            if not recovered:
                try:
                    commit_oid = remote.upload(checkpoint)
                except WatcherError as error:
                    # A commit may have succeeded remotely while the response was
                    # lost. Reconcile the immutable remote manifest before retrying.
                    if error.code != "remote_upload_failed":
                        raise
                    recovered_manifest_sha = remote.recover_existing(checkpoint)
                    if recovered_manifest_sha is None:
                        raise
                    recovered = True
            return commit_oid, recovered, recovered_manifest_sha
        except WatcherError as error:
            if error.code not in RETRYABLE_REMOTE_CODES or attempt == attempts:
                raise
            emit(
                "remote_retry",
                checkpoint=checkpoint.label,
                step=checkpoint.step,
                attempt=attempt,
                max_attempts=attempts,
                code=error.code,
            )
            sleep(retry_seconds)
    raise WatcherError("internal_error", "remote retry loop exhausted unexpectedly")


def scan_once(
    *,
    args: argparse.Namespace,
    state_path: Path,
    remote: PrivateCandidateRemote | None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    state = load_state(state_path, args.repo_id)
    staging_dir = state_path.parent / STAGING_NAME
    archive_enabled = bool(args.upload and not getattr(args, "no_local_archive", False))
    archive_root = state_path.parent / ARCHIVE_NAME
    for checkpoint_path in discover_checkpoints(args.watch_dir):
        label = checkpoint_path.name
        if label in state["checkpoints"]:
            if archive_enabled:
                record = state["checkpoints"][label]
                if archived_checkpoint_matches(
                    checkpoint_path,
                    archive_root,
                    expected_weights_sha256=record["adapter_sha256"],
                    expected_config_sha256=record["adapter_config_sha256"],
                ):
                    continue
                validated = validate_checkpoint(
                    checkpoint_path,
                    staging_dir=staging_dir,
                    base_model=args.base_model,
                    base_revision=args.base_revision,
                    run_id=args.run_id,
                    training_data_sha256=args.training_data_sha256,
                    training_manifest_sha256=args.training_manifest_sha256,
                    admission_report_sha256=args.admission_report_sha256,
                    settle_seconds=args.settle_seconds,
                    sleep=sleep,
                )
                if validated is None:
                    continue
                try:
                    if (
                        validated.weights_sha256 != record["adapter_sha256"]
                        or validated.config_sha256 != record["adapter_config_sha256"]
                    ):
                        raise WatcherError(
                            "archive_conflict", "local checkpoint differs from upload state"
                        )
                    archive_validated_checkpoint(
                        checkpoint_path,
                        validated,
                        archive_root=archive_root,
                        run_id=args.run_id,
                    )
                    emit("archived_backfill", checkpoint=label, step=validated.step)
                finally:
                    validated.weights_path.unlink(missing_ok=True)
            continue
        validated = validate_checkpoint(
            checkpoint_path,
            staging_dir=staging_dir,
            base_model=args.base_model,
            base_revision=args.base_revision,
            run_id=args.run_id,
            training_data_sha256=args.training_data_sha256,
            training_manifest_sha256=args.training_manifest_sha256,
            admission_report_sha256=args.admission_report_sha256,
            settle_seconds=args.settle_seconds,
            sleep=sleep,
        )
        if validated is None:
            continue
        try:
            if not args.upload:
                emit(
                    "validated_dry_run",
                    checkpoint=validated.label,
                    step=validated.step,
                    adapter_sha256=validated.weights_sha256,
                )
                continue
            if remote is None:
                raise WatcherError("internal_error", "upload requested without a remote client")
            commit_oid, recovered, recovered_manifest_sha = publish_with_retry(
                remote,
                validated,
                attempts=getattr(args, "remote_attempts", 3),
                retry_seconds=getattr(args, "remote_retry_seconds", 15.0),
                sleep=sleep,
            )
            state["checkpoints"][validated.label] = checkpoint_state_record(
                validated,
                commit_oid=commit_oid,
                recovered=recovered,
                manifest_sha256=recovered_manifest_sha,
            )
            write_state(state_path, state)
            if archive_enabled:
                archive_validated_checkpoint(
                    checkpoint_path,
                    validated,
                    archive_root=archive_root,
                    run_id=args.run_id,
                )
            emit(
                "remote_recovered" if recovered else "uploaded",
                checkpoint=validated.label,
                step=validated.step,
                repo_id=args.repo_id,
                adapter_sha256=validated.weights_sha256,
            )
        finally:
            validated.weights_path.unlink(missing_ok=True)
    return state


def validate_cli(args: argparse.Namespace) -> None:
    if not REPO_ID_RE.fullmatch(args.repo_id) or not args.repo_id.startswith(
        "LLM-OS-Models2/"
    ):
        raise WatcherError(
            "invalid_argument", "repo ID must be under the LLM-OS-Models2 organization"
        )
    if any(pattern.search(args.repo_id) for pattern in TOKEN_VALUE_RES):
        raise WatcherError("invalid_argument", "repo ID contains a token-like value")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", args.base_model) or _is_absolute_path_string(
        args.base_model
    ):
        raise WatcherError("invalid_argument", "base model ID is invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", args.base_revision):
        raise WatcherError("invalid_argument", "base revision must be a pinned 40-hex SHA")
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise WatcherError("invalid_argument", "run ID is invalid")
    for name in (
        "training_data_sha256",
        "training_manifest_sha256",
        "admission_report_sha256",
    ):
        value = getattr(args, name)
        if value is not None and not SHA256_RE.fullmatch(value):
            raise WatcherError("invalid_argument", f"{name} must be a SHA-256")
    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        raise WatcherError("invalid_argument", "poll seconds must be in (0, 60]")
    if args.settle_seconds < 0 or args.settle_seconds > 60:
        raise WatcherError("invalid_argument", "settle seconds must be in [0, 60]")
    if args.remote_attempts < 1 or args.remote_attempts > 10:
        raise WatcherError("invalid_argument", "remote attempts must be in [1, 10]")
    if args.remote_retry_seconds <= 0 or args.remote_retry_seconds > 60:
        raise WatcherError(
            "invalid_argument", "remote retry seconds must be in (0, 60]"
        )


def make_remote(args: argparse.Namespace) -> PrivateCandidateRemote | None:
    if not args.upload:
        return None
    token = read_hf_token(args.env_file)
    if not token:
        raise WatcherError(
            "missing_token", "HF token is unavailable in the environment or ignored .env"
        )
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        raise WatcherError(
            "missing_dependency", "huggingface_hub is required for --upload"
        ) from None
    # The token remains an in-memory constructor argument and is never exported.
    return PrivateCandidateRemote(
        api=HfApi(token=token),
        repo_id=args.repo_id,
        operation_add_cls=CommitOperationAdd,
    )


def run(args: argparse.Namespace) -> None:
    validate_cli(args)
    try:
        args.watch_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise WatcherError(
            "watch_dir_failed", f"watch directory setup failed ({type(error).__name__})"
        ) from None
    args.watch_dir = args.watch_dir.resolve()
    state_path = (
        args.state_file.resolve()
        if args.state_file
        else args.watch_dir / STATE_NAME
    )
    lock_path = (
        state_path.with_name(LOCK_NAME)
        if state_path.name == STATE_NAME
        else state_path.with_name(f".{state_path.name}.lock")
    )
    remote = make_remote(args)
    emit(
        "watcher_started",
        mode="upload" if args.upload else "dry_run",
        repo_id=args.repo_id,
        once=bool(args.once),
    )
    with state_lock(lock_path):
        while True:
            scan_once(args=args, state_path=state_path, remote=remote)
            if args.once:
                return
            time.sleep(args.poll_seconds)


def main() -> None:
    try:
        run(parse_args())
    except WatcherError as error:
        emit("error", code=error.code, message=str(error))
        raise SystemExit(1)
    except KeyboardInterrupt:
        emit("stopped", reason="keyboard_interrupt")
    except Exception as error:
        # Never stringify an unexpected exception: third-party messages often
        # contain local paths, request headers, or credential-bearing URLs.
        emit("error", code="unexpected_error", error_type=type(error).__name__)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
