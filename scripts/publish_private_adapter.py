#!/usr/bin/env python3
"""Stage and optionally publish a verified LoRA adapter to a private HF repo.

The publisher deliberately uses an allowlist. It never stages raw data, optimizer
state, scheduler state, RNG state, trainer state, logs, or the original args files.
Uploading is opt-in via ``--upload`` and public repositories are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_BASE_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
CHECKPOINT_FILES = ("adapter_model.safetensors", "adapter_config.json")
EXCLUDED_ARTIFACTS = (
    "raw or processed training examples",
    "optimizer and scheduler state",
    "trainer and RNG state",
    "training logs",
    "unsanitized training arguments and local filesystem paths",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage a verified PEFT adapter and optionally upload it to a PRIVATE "
            "Hugging Face model repository."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="For example: org/model-name")
    parser.add_argument(
        "--verification",
        type=Path,
        help="Defaults to <checkpoint parent>/verification.json",
    )
    parser.add_argument(
        "--run-args",
        type=Path,
        help="Defaults to <checkpoint parent>/args.json; only allowlisted fields are used",
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        help="Optional source manifest; row counts, hashes, and licensing status only",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base-revision", default=DEFAULT_BASE_REVISION)
    parser.add_argument(
        "--stage-dir",
        type=Path,
        help="Must be new or empty; defaults under artifacts/hf-staging/",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Actually upload. Without this flag the command only creates the staging folder.",
    )
    return parser.parse_args()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require_nonempty_stage(stage_dir: Path) -> None:
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise FileExistsError(
            f"Staging directory is not empty; refusing to risk stale uploads: {stage_dir}"
        )
    stage_dir.mkdir(parents=True, exist_ok=True)


def sanitize_verification(
    source: dict[str, Any], *, expected_sha256: str, base_model: str
) -> dict[str, Any]:
    if source.get("status") != "pass":
        raise ValueError("Verification status is not 'pass'; refusing to publish")
    if source.get("adapter_sha256") != expected_sha256:
        raise ValueError("Verification SHA-256 does not match adapter_model.safetensors")
    if source.get("base_model") != base_model:
        raise ValueError(
            "Verification base model does not match --base-model; refusing to publish"
        )

    allowed = (
        "verified_at_utc",
        "base_model",
        "adapter_sha256",
        "embedding_shape",
        "norms",
        "query_positive",
        "query_negative",
        "margin",
        "torch",
        "swift",
        "status",
    )
    return {key: source[key] for key in allowed if key in source}


def sanitize_training_args(source: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "model_type",
        "task_type",
        "tuner_type",
        "torch_dtype",
        "attn_impl",
        "max_length",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "lr_scheduler_type",
        "warmup_ratio",
        "max_steps",
        "seed",
        "loss_type",
        "gradient_checkpointing",
    )
    return {key: source[key] for key in allowed if key in source}


def sanitize_data_manifest(source: dict[str, Any] | None) -> dict[str, Any]:
    if source is None:
        return {
            "documented": False,
            "release_eligible": False,
            "release_blocker": "No data manifest was supplied to the publisher",
        }

    src = source.get("source") if isinstance(source.get("source"), dict) else {}
    sampling = (
        source.get("sampling") if isinstance(source.get("sampling"), dict) else {}
    )
    files = source.get("files") if isinstance(source.get("files"), dict) else {}
    safe_files: dict[str, Any] = {}
    for name, metadata in files.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            continue
        # File names are metadata only; no dataset files are copied into staging.
        safe_files[Path(name).name] = {
            key: metadata[key] for key in ("sha256", "rows") if key in metadata
        }

    return {
        "documented": True,
        "purpose": source.get("purpose"),
        "release_eligible": bool(source.get("release_eligible", False)),
        "release_blocker": source.get("release_blocker"),
        "source": {
            key: src[key]
            for key in ("dataset", "revision", "split", "url", "declared_license")
            if key in src
        },
        "sampling": {
            key: sampling[key]
            for key in ("seed", "requested", "train_rows", "validation_rows")
            if key in sampling
        },
        "files": safe_files,
    }


def sanitize_additional_config(source: dict[str, Any]) -> dict[str, Any]:
    allowed = ("lora_dtype", "lorap_lr_ratio", "lorap_emb_lr")
    return {key: source[key] for key in allowed if key in source}


def yaml_quote(value: str) -> str:
    """Return a safe double-quoted YAML scalar without requiring PyYAML."""
    return json.dumps(value, ensure_ascii=False)


def build_model_card(
    *,
    repo_id: str,
    base_model: str,
    base_revision: str,
    adapter_sha256: str,
    adapter_size: int,
    adapter_config: dict[str, Any],
    training: dict[str, Any],
    data: dict[str, Any],
    verification: dict[str, Any],
) -> str:
    source = data.get("source") if isinstance(data.get("source"), dict) else {}
    dataset_name = source.get("dataset", "not documented")
    dataset_revision = source.get("revision", "not documented")
    train_rows = data.get("sampling", {}).get("train_rows", "not documented")
    validation_rows = data.get("sampling", {}).get("validation_rows", "not documented")
    blocker = data.get("release_blocker") or "No public release clearance is recorded"
    margin = verification.get("margin", "not recorded")
    verified_at = verification.get("verified_at_utc", "not recorded")

    return f"""---
base_model: {yaml_quote(base_model)}
library_name: peft
pipeline_tag: feature-extraction
language:
  - ko
license: other
tags:
  - peft
  - lora
  - embedding
  - retrieval
  - korean
  - pipeline-validation
---

# {repo_id.split('/', 1)[1]}

> **Private pipeline-validation artifact. This is not a performance release.**

This repository contains a LoRA adapter for `{base_model}`. It exists to verify
the end-to-end data, training, checkpoint, reload, and publishing pipeline. It
must not be presented as a public benchmark result or a production-ready model.

## Artifact identity

- Base model: `{base_model}`
- Pinned base revision: `{base_revision}`
- Adapter type: PEFT LoRA, rank {adapter_config.get('r', 'unknown')}, alpha {adapter_config.get('lora_alpha', 'unknown')}
- Adapter SHA-256: `{adapter_sha256}`
- Adapter size: {adapter_size:,} bytes
- Task: Korean text embedding/retrieval

The base weights are not included. Load this adapter only with the exact pinned
base revision above; behavior with a different revision has not been verified.

## Training

- Objective: `{training.get('loss_type', 'not documented')}` contrastive embedding training
- Steps: {training.get('max_steps', 'not documented')}
- Maximum input length: {training.get('max_length', 'not documented')}
- Precision: `{training.get('torch_dtype', 'not documented')}`
- Seed: {training.get('seed', 'not documented')}
- Source dataset: `{dataset_name}` at revision `{dataset_revision}`
- Rows: {train_rows} train / {validation_rows} validation

The source dataset card did not declare an explicit license at preparation time.
Consequently this repository is private and is **not cleared for public weight
redistribution**. Release blocker: {blocker}.

Raw examples, optimizer/scheduler state, trainer/RNG state, logs, and local paths
are intentionally excluded from this repository.

## Verification and evaluation status

The saved adapter was reloaded successfully on {verified_at}. A single smoke
triplet produced a positive-minus-negative cosine margin of `{margin}`. This is
only a functional integrity check; it is not evidence of retrieval quality or
generalization.

- Public benchmark claim: **none**
- MTEB Korean: **not evaluated for this artifact**
- Sionic nine-task retrieval suite: **not evaluated for this artifact**
- Comparison with `sionic-ai/comsat-embed-ko-8b-preview`: **not established**

See `verification.json` and `artifact_manifest.json` for machine-readable,
sanitized provenance.

## Intended use

Internal pipeline testing and reproducibility checks only. Do not use this smoke
adapter for production search, external model comparisons, or public claims.
"""


def stage_artifact(args: argparse.Namespace) -> Path:
    if not REPO_ID_RE.fullmatch(args.repo_id):
        raise ValueError("--repo-id must have the form owner/model-name")
    if not re.fullmatch(r"[0-9a-f]{40}", args.base_revision):
        raise ValueError("--base-revision must be a pinned 40-character Git SHA")

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise NotADirectoryError(f"Checkpoint directory does not exist: {checkpoint}")
    for name in CHECKPOINT_FILES:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"Required checkpoint file is missing: {name}")

    verification_path = (args.verification or checkpoint.parent / "verification.json").resolve()
    run_args_path = (args.run_args or checkpoint.parent / "args.json").resolve()
    source_verification = load_json(verification_path, "verification")
    source_run_args = load_json(run_args_path, "run args")
    source_data = (
        load_json(args.data_manifest.resolve(), "data manifest")
        if args.data_manifest
        else None
    )

    weights = checkpoint / "adapter_model.safetensors"
    weights_sha256 = sha256(weights)
    verification = sanitize_verification(
        source_verification,
        expected_sha256=weights_sha256,
        base_model=args.base_model,
    )
    training = sanitize_training_args(source_run_args)
    data = sanitize_data_manifest(source_data)

    adapter_config = load_json(checkpoint / "adapter_config.json", "adapter config")
    if adapter_config.get("peft_type") != "LORA":
        raise ValueError("Only PEFT LoRA adapters are supported by this publisher")
    adapter_config["base_model_name_or_path"] = args.base_model
    adapter_config["revision"] = args.base_revision

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_repo_name = args.repo_id.replace("/", "--")
    stage_dir = (
        args.stage_dir.resolve()
        if args.stage_dir
        else (Path.cwd() / "artifacts" / "hf-staging" / f"{safe_repo_name}-{timestamp}")
    )
    require_nonempty_stage(stage_dir)

    shutil.copy2(weights, stage_dir / weights.name)
    write_json(stage_dir / "adapter_config.json", adapter_config)

    additional_config_path = checkpoint / "additional_config.json"
    if additional_config_path.is_file():
        additional_config = sanitize_additional_config(
            load_json(additional_config_path, "additional config")
        )
        write_json(stage_dir / "additional_config.json", additional_config)

    write_json(stage_dir / "verification.json", verification)
    artifact_manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_kind": "peft-lora-adapter",
        "distribution": "private-pipeline-validation-only",
        "repository": {"id": args.repo_id, "required_visibility": "private"},
        "base_model": {"id": args.base_model, "revision": args.base_revision},
        "adapter": {
            "checkpoint_label": checkpoint.name,
            "file": weights.name,
            "sha256": weights_sha256,
            "size_bytes": weights.stat().st_size,
            "peft_type": adapter_config.get("peft_type"),
            "rank": adapter_config.get("r"),
            "alpha": adapter_config.get("lora_alpha"),
            "dropout": adapter_config.get("lora_dropout"),
            "target_modules": adapter_config.get("target_modules"),
        },
        "training": training,
        "data": data,
        "verification": verification,
        "public_benchmark_claim": None,
        "release_eligible": False,
        "excluded_artifacts": list(EXCLUDED_ARTIFACTS),
    }
    write_json(stage_dir / "artifact_manifest.json", artifact_manifest)
    (stage_dir / "README.md").write_text(
        build_model_card(
            repo_id=args.repo_id,
            base_model=args.base_model,
            base_revision=args.base_revision,
            adapter_sha256=weights_sha256,
            adapter_size=weights.stat().st_size,
            adapter_config=adapter_config,
            training=training,
            data=data,
            verification=verification,
        ),
        encoding="utf-8",
    )

    expected = {
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "artifact_manifest.json",
        "verification.json",
    }
    if additional_config_path.is_file():
        expected.add("additional_config.json")
    entries = list(stage_dir.iterdir())
    if any(not path.is_file() for path in entries):
        raise RuntimeError("Staging directory contains a non-file entry; refusing to upload")
    actual = {path.name for path in entries}
    if actual != expected:
        raise RuntimeError(f"Unexpected staging contents: {sorted(actual - expected)}")
    return stage_dir


def upload_private(stage_dir: Path, repo_id: str) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be present in the environment when --upload is used")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --upload; use the project training environment"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.model_info(repo_id=repo_id, token=token)
    if not info.private:
        raise RuntimeError(
            f"Refusing to upload because {repo_id} is not a private repository"
        )

    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(stage_dir),
        commit_message="Publish verified private pipeline-validation adapter",
        token=token,
    )
    return commit.commit_url


def main() -> None:
    args = parse_args()
    stage_dir = stage_artifact(args)
    print(f"Staged allowlisted private artifact: {stage_dir}")
    print("Included files:")
    for path in sorted(stage_dir.iterdir()):
        print(f"  - {path.name} ({path.stat().st_size:,} bytes)")

    if not args.upload:
        print("Dry run complete; nothing was uploaded. Add --upload to publish privately.")
        return

    commit_url = upload_private(stage_dir, args.repo_id)
    print(f"Private upload complete: {commit_url}")


if __name__ == "__main__":
    main()
