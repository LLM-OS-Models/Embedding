#!/usr/bin/env python3
"""Validate and publish the exact 10K/512 Qwen3 hard-negative pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/ko_triplet_pilot_10k"
REPORTS = ROOT / "reports"
DEFAULT_REPO = "LLM-OS-Models/korean-embedding-ko-triplet-hn-pilot-10k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--public", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def artifact_map() -> dict[str, Path]:
    return {
        "README.md": ROOT / "cards/korean-embedding-ko-triplet-hn-pilot-10k/README.md",
        "data/train.jsonl": DATA / "train.hn-qwen3-r095-n4.jsonl",
        "data/validation.jsonl": DATA / "validation.hn-qwen3-r095-n4.jsonl",
        "metadata/source_manifest.json": DATA / "manifest.json",
        "metadata/train_hn_manifest.json": DATA / "train.hn-qwen3-r095-n4.jsonl.manifest.json",
        "metadata/validation_hn_manifest.json": DATA / "validation.hn-qwen3-r095-n4.jsonl.manifest.json",
        "metadata/train_mining_audit.jsonl": DATA / "train.hn-qwen3-r095-n4.jsonl.audit.jsonl",
        "metadata/validation_mining_audit.jsonl": DATA / "validation.hn-qwen3-r095-n4.jsonl.audit.jsonl",
        "metadata/train_benchmark_overlap_audit.json": REPORTS / "ko-triplet-pilot-10k-benchmark-overlap-audit.json",
        "metadata/validation_benchmark_overlap_audit.json": REPORTS / "ko-triplet-validation-512-benchmark-overlap-audit.json",
    }


def validate(files: dict[str, Path]) -> dict:
    for path in files.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    train_manifest = read_json(files["metadata/train_hn_manifest.json"])
    validation_manifest = read_json(files["metadata/validation_hn_manifest.json"])
    train_audit = read_json(files["metadata/train_benchmark_overlap_audit.json"])
    validation_audit = read_json(
        files["metadata/validation_benchmark_overlap_audit.json"]
    )
    checks = (
        ("data/train.jsonl", train_manifest, 10_000),
        ("data/validation.jsonl", validation_manifest, 512),
    )
    for repo_path, manifest, expected_rows in checks:
        path = files[repo_path]
        declared = next(iter(manifest["files"].values()))
        if rows(path) != expected_rows or declared["rows"] != expected_rows:
            raise RuntimeError(f"Row count mismatch: {path}")
        if sha256(path) != declared["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {path}")
    for audit, expected_rows in ((train_audit, 10_000), (validation_audit, 512)):
        if audit["rows"] != expected_rows:
            raise RuntimeError("Benchmark audit row count mismatch")
        if audit["unique_critical_query_or_evaluation_matches"] != 0:
            raise RuntimeError("Critical benchmark overlap blocks publication")
        if audit["unique_matches"] != 0:
            raise RuntimeError("Unexpected benchmark text exposure")
    return {
        "status": "pass",
        "train_rows": 10_000,
        "validation_rows": 512,
        "train_sha256": sha256(files["data/train.jsonl"]),
        "validation_sha256": sha256(files["data/validation.jsonl"]),
        "critical_overlap": 0,
        "release_eligible": False,
    }


def main() -> None:
    args = parse_args()
    files = artifact_map()
    report = validate(files)
    if args.upload:
        from huggingface_hub import CommitOperationAdd, HfApi

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required with --upload")
        api = HfApi(token=token)
        api.create_repo(
            args.repo_id,
            repo_type="dataset",
            private=not args.public,
            exist_ok=True,
        )
        commit = api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
                for name, path in files.items()
            ],
            commit_message="Publish audited Qwen3 hard-negative pilot",
        )
        if args.public:
            api.update_repo_settings(args.repo_id, repo_type="dataset", private=False)
        info = api.repo_info(args.repo_id, repo_type="dataset")
        report.update(
            {
                "repo_id": info.id,
                "revision": info.sha,
                "private": info.private,
                "commit_url": commit.commit_url,
            }
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
