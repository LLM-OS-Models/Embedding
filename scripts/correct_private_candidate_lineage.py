#!/usr/bin/env python3
"""Migrate one verified private adapter candidate after a lineage-only correction."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

try:
    from scripts.watch_private_adapter_checkpoints import (
        ARCHIVE_MANIFEST_NAME,
        CONFIG_NAME,
        MANIFEST_NAME,
        REMOTE_ALLOWLIST,
        WEIGHTS_NAME,
        inspect_safetensors,
        read_hf_token,
        sanitize_json_value,
        sha256_file,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from watch_private_adapter_checkpoints import (
        ARCHIVE_MANIFEST_NAME,
        CONFIG_NAME,
        MANIFEST_NAME,
        REMOTE_ALLOWLIST,
        WEIGHTS_NAME,
        inspect_safetensors,
        read_hf_token,
        sanitize_json_value,
        sha256_file,
    )


ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
LABEL_RE = re.compile(r"checkpoint-([1-9][0-9]*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--expected-source-training-data-sha256", required=True)
    parser.add_argument("--corrected-training-data-sha256", required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def validate_args(args: argparse.Namespace) -> None:
    for repo in (args.source_repo, args.target_repo):
        if not re.fullmatch(r"LLM-OS-Models2/[A-Za-z0-9][A-Za-z0-9._-]*", repo):
            raise ValueError("Both repositories must be under LLM-OS-Models2")
    if args.source_repo == args.target_repo:
        raise ValueError("Source and target repositories must differ")
    if not COMMIT_RE.fullmatch(args.source_revision):
        raise ValueError("Source revision must be an exact 40-hex commit")
    if not LABEL_RE.fullmatch(args.checkpoint_label):
        raise ValueError("Checkpoint label is invalid")
    for value in (
        args.expected_source_training_data_sha256,
        args.corrected_training_data_sha256,
    ):
        if not SHA256_RE.fullmatch(value):
            raise ValueError("Training data identities must be SHA-256 values")
    if (
        args.expected_source_training_data_sha256
        == args.corrected_training_data_sha256
    ):
        raise ValueError("Lineage correction does not change the declared identity")


def load_archive(args: argparse.Namespace) -> dict[str, Any]:
    archive = args.archive_dir.expanduser().resolve()
    archive_root = (ROOT / "outputs").resolve()
    if (
        args.archive_dir.is_symlink()
        or not archive.is_dir()
        or not archive.is_relative_to(archive_root)
        or archive.name != args.checkpoint_label
    ):
        raise ValueError("Archive directory is missing or unsafe")
    paths = {
        "weights": archive / WEIGHTS_NAME,
        "config": archive / CONFIG_NAME,
        "manifest": archive / ARCHIVE_MANIFEST_NAME,
    }
    for path in paths.values():
        if path.is_symlink() or not path.is_file() or path.stat().st_size < 2:
            raise ValueError("Archive allowlist file is missing or unsafe")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("checkpoint", {}).get("label") != args.checkpoint_label
        or set(manifest.get("contents_allowlist", []))
        != {WEIGHTS_NAME, CONFIG_NAME, ARCHIVE_MANIFEST_NAME}
    ):
        raise ValueError("Archive manifest contract is invalid")
    adapter = manifest.get("adapter", {})
    if (
        adapter.get("weights", {}).get("sha256") != sha256_file(paths["weights"])
        or adapter.get("config", {}).get("sha256") != sha256_file(paths["config"])
    ):
        raise ValueError("Archive file checksum mismatch")
    return {"root": archive, "paths": paths, "manifest": manifest}


def build_corrected_manifest(
    source: dict[str, Any],
    *,
    source_repo: str,
    source_revision: str,
    source_manifest_sha256: str,
    checkpoint_label: str,
    expected_source_sha256: str,
    corrected_sha256: str,
) -> dict[str, Any]:
    if (
        source.get("schema_version") != 1
        or source.get("artifact_kind") != "peft-lora-checkpoint-candidate"
        or source.get("distribution") != "private-candidate-only"
        or source.get("checkpoint", {}).get("label") != checkpoint_label
        or set(source.get("remote_allowlist", [])) != REMOTE_ALLOWLIST
    ):
        raise ValueError("Source candidate manifest contract is invalid")
    validation = source.get("validation", {})
    for field in (
        "completion_sentinel_observed",
        "same_step_eval_observed",
        "safetensors_full_payload_validation",
        "all_tensor_values_finite",
        "staged_snapshot_sha256_reverified",
    ):
        expected: Any = "pass" if field == "safetensors_full_payload_validation" else True
        if validation.get(field) != expected:
            raise ValueError(f"Source validation evidence failed: {field}")
    lineage = source.get("lineage", {})
    if lineage.get("training_data_sha256") != expected_source_sha256:
        raise ValueError("Source manifest does not contain the expected incorrect identity")
    corrected = copy.deepcopy(source)
    corrected["lineage"]["training_data_sha256"] = corrected_sha256
    corrected["lineage_correction"] = {
        "schema_version": 1,
        "field": "lineage.training_data_sha256",
        "previous_value": expected_source_sha256,
        "corrected_value": corrected_sha256,
        "source_private_repo": source_repo,
        "source_revision": source_revision,
        "source_candidate_manifest_sha256": source_manifest_sha256,
        "method": "immutable-private-manifest-plus-local-archive-full-payload-revalidation",
    }
    return sanitize_json_value(corrected)


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    archive = load_archive(args)
    if not args.upload:
        raise ValueError("--upload is required for a lineage migration")
    token = read_hf_token(args.env_file)
    if not token:
        raise ValueError("A private Hub token is required")
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)
    source_info = api.model_info(
        repo_id=args.source_repo, revision=args.source_revision, files_metadata=True
    )
    if not source_info.private or source_info.sha != args.source_revision:
        raise ValueError("Source repository/revision is not exact and private")
    source_name = f"checkpoints/{args.checkpoint_label}/{MANIFEST_NAME}"
    source_path = Path(
        hf_hub_download(
            repo_id=args.source_repo,
            filename=source_name,
            revision=args.source_revision,
            token=token,
        )
    )
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    source_manifest_sha = sha256_bytes(source_bytes)
    corrected = build_corrected_manifest(
        source,
        source_repo=args.source_repo,
        source_revision=args.source_revision,
        source_manifest_sha256=source_manifest_sha,
        checkpoint_label=args.checkpoint_label,
        expected_source_sha256=args.expected_source_training_data_sha256,
        corrected_sha256=args.corrected_training_data_sha256,
    )
    weights = archive["paths"]["weights"]
    config = archive["paths"]["config"]
    adapter = source["adapter"]
    if (
        adapter["weights"]["sha256"] != sha256_file(weights)
        or adapter["weights"]["size_bytes"] != weights.stat().st_size
        or adapter["config"]["sha256"] != sha256_file(config)
        or adapter["config"]["size_bytes"] != config.stat().st_size
    ):
        raise ValueError("Source remote evidence and local archive differ")
    tensor_summary = inspect_safetensors(weights)
    for field in ("tensor_count", "parameter_count", "tensor_dtypes"):
        if tensor_summary[field] != adapter[field]:
            raise ValueError(f"Full-payload tensor evidence drifted: {field}")
    corrected_bytes = json_bytes(corrected)

    api.create_repo(
        repo_id=args.target_repo, repo_type="model", private=True, exist_ok=True
    )
    target_info = api.model_info(repo_id=args.target_repo, files_metadata=True)
    if not target_info.private:
        raise ValueError("Target repository is not private")
    prefix = f"checkpoints/{args.checkpoint_label}/"
    existing = {
        item.rfilename: item
        for item in target_info.siblings
        if item.rfilename.startswith(prefix)
    }
    expected_names = {
        prefix + WEIGHTS_NAME,
        prefix + CONFIG_NAME,
        prefix + MANIFEST_NAME,
    }
    recovered = False
    commit_sha = target_info.sha
    if existing:
        if set(existing) != expected_names:
            raise ValueError("Target checkpoint prefix has a non-allowlisted file set")
        remote_manifest = Path(
            hf_hub_download(
                repo_id=args.target_repo,
                filename=prefix + MANIFEST_NAME,
                revision=target_info.sha,
                token=token,
            )
        )
        if remote_manifest.read_bytes() != corrected_bytes:
            raise ValueError("Target checkpoint prefix contains conflicting evidence")
        recovered = True
    else:
        commit = api.create_commit(
            repo_id=args.target_repo,
            repo_type="model",
            parent_commit=target_info.sha,
            commit_message=f"Correct private lineage for {args.checkpoint_label}",
            commit_description="Allowlist-only migration; adapter/config bytes unchanged",
            operations=[
                CommitOperationAdd(
                    path_in_repo=prefix + WEIGHTS_NAME, path_or_fileobj=weights
                ),
                CommitOperationAdd(
                    path_in_repo=prefix + CONFIG_NAME, path_or_fileobj=config
                ),
                CommitOperationAdd(
                    path_in_repo=prefix + MANIFEST_NAME,
                    path_or_fileobj=corrected_bytes,
                ),
            ],
            num_threads=1,
        )
        commit_sha = commit.oid
    verified = api.model_info(
        repo_id=args.target_repo, revision=commit_sha, files_metadata=True
    )
    remote = {
        item.rfilename: item
        for item in verified.siblings
        if item.rfilename.startswith(prefix)
    }
    if not verified.private or set(remote) != expected_names:
        raise ValueError("Corrected remote allowlist/visibility verification failed")
    remote_manifest = Path(
        hf_hub_download(
            repo_id=args.target_repo,
            filename=prefix + MANIFEST_NAME,
            revision=commit_sha,
            token=token,
        )
    )
    if remote_manifest.read_bytes() != corrected_bytes:
        raise ValueError("Corrected remote manifest is not exact")
    remote_lfs = getattr(remote[prefix + WEIGHTS_NAME], "lfs", None)
    if getattr(remote_lfs, "sha256", None) != sha256_file(weights):
        raise ValueError("Corrected remote adapter LFS checksum mismatch")
    return {
        "status": "complete",
        "checkpoint": args.checkpoint_label,
        "source_repo": args.source_repo,
        "source_revision": args.source_revision,
        "target_repo": args.target_repo,
        "target_revision": verified.sha,
        "adapter_sha256": sha256_file(weights),
        "candidate_manifest_sha256": sha256_bytes(corrected_bytes),
        "corrected_training_data_sha256": args.corrected_training_data_sha256,
        "private": True,
        "allowlist_exact": True,
        "remote_manifest_exact": True,
        "recovered_existing_remote": recovered,
    }


def main() -> None:
    args = parse_args()
    report = run(args)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report_output:
        output = args.report_output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
