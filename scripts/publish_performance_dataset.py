#!/usr/bin/env python3
"""Validate and publish a pinned performance-training dataset to HF Hub."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "outputs/data/performance-v1/pilot-50k",
    )
    parser.add_argument(
        "--card",
        type=Path,
        default=ROOT / "cards/korean-embedding-performance-v1-pilot-50k/README.md",
    )
    parser.add_argument(
        "--repo-id",
        default="LLM-OS-Models/korean-embedding-performance-v1-pilot-50k",
    )
    parser.add_argument("--expected-phase", default="pilot_50k")
    parser.add_argument("--expected-rows", type=int, default=50_000)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            count += block.count(b"\n")
    return count


def validate(
    data_dir: Path, card: Path, expected_phase: str, expected_rows: int
) -> tuple[dict[str, Any], list[Path]]:
    manifest_path = data_dir / "manifest.json"
    train_path = data_dir / "train.jsonl"
    provenance_path = data_dir / "provenance.jsonl"
    required = [manifest_path, train_path, provenance_path, card]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing publication inputs: {missing}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("phase") != expected_phase
        or manifest.get("built_rows") != expected_rows
    ):
        raise ValueError(
            "Dataset does not match publication gate: "
            f"expected={expected_phase}:{expected_rows}, "
            f"found={manifest.get('phase')}:{manifest.get('built_rows')}"
        )
    for name, path in (("train.jsonl", train_path), ("provenance.jsonl", provenance_path)):
        expected = manifest["files"][name]
        actual_rows = line_count(path)
        actual_sha = sha256(path)
        if actual_rows != expected["rows"] or actual_sha != expected["sha256"]:
            raise ValueError(
                f"Artifact drift for {name}: rows={actual_rows}, sha256={actual_sha}"
            )

    card_text = card.read_text(encoding="utf-8")
    for disclosure in ("release_eligible: false", "Sionic 9", "MIRACL"):
        if disclosure not in card_text:
            raise ValueError(f"Dataset card is missing required disclosure: {disclosure}")
    return manifest, [train_path, provenance_path, manifest_path, card]


def main() -> None:
    args = parse_args()
    manifest, paths = validate(
        args.data_dir.resolve(),
        args.card.resolve(),
        args.expected_phase,
        args.expected_rows,
    )
    report = {
        "repo_id": args.repo_id,
        "visibility": "public" if args.public else "private",
        "phase": manifest["phase"],
        "rows": manifest["built_rows"],
        "validated": True,
        "upload_requested": args.upload,
    }
    if not args.upload:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be exported; token values are never read from CLI")
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    train_path, provenance_path, manifest_path, card_path = paths
    operations = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card_path),
        CommitOperationAdd(path_in_repo="data/train.jsonl", path_or_fileobj=train_path),
        CommitOperationAdd(
            path_in_repo="metadata/provenance.jsonl", path_or_fileobj=provenance_path
        ),
        CommitOperationAdd(
            path_in_repo="metadata/manifest.json", path_or_fileobj=manifest_path
        ),
    ]
    commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message="Publish validated Korean embedding pilot 50K",
    )
    report["commit_url"] = commit.commit_url
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
