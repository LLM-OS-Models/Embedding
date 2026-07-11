#!/usr/bin/env python3
"""Validate and publish the evaluation-only Korean benchmark hash blocklist."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blocklist-dir",
        type=Path,
        default=ROOT / "outputs/decontamination/benchmark_blocklist",
    )
    parser.add_argument(
        "--card",
        type=Path,
        default=ROOT / "cards/korean-embedding-benchmark-blocklist-v1/README.md",
    )
    parser.add_argument(
        "--policy", type=Path, default=ROOT / "configs/decontamination_policy.json"
    )
    parser.add_argument(
        "--repo-id",
        default="LLM-OS-Models/korean-embedding-benchmark-blocklist-v1",
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--public", action="store_true")
    return parser.parse_args()


def validate(blocklist_dir: Path, card: Path, policy: Path) -> dict:
    manifest_path = blocklist_dir / "manifest.json"
    for path in (manifest_path, card, policy):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks", [])
    if manifest.get("completed_tasks") != 15 or len(tasks) != 15:
        raise ValueError("Expected complete 15-task blocklist")
    artifact = manifest.get("artifact_policy", {})
    if artifact.get("contains_source_ids") is not False:
        raise ValueError("Blocklist must not contain raw source IDs")
    if artifact.get("contains_source_text") is not False:
        raise ValueError("Blocklist must not contain raw source text")
    missing = []
    for task in tasks:
        task_dir = blocklist_dir / task["path"]
        if not (task_dir / "_SUCCESS").is_file() or not (task_dir / "manifest.json").is_file():
            missing.append(task["path"])
    if missing:
        raise ValueError(f"Incomplete task artifacts: {missing}")
    building = list(blocklist_dir.rglob("*.building-*"))
    if building:
        raise ValueError(f"Temporary build artifacts remain: {building[:3]}")
    return manifest


def main() -> None:
    args = parse_args()
    blocklist_dir = args.blocklist_dir.resolve()
    card = args.card.resolve()
    policy = args.policy.resolve()
    manifest = validate(blocklist_dir, card, policy)
    report = {
        "repo_id": args.repo_id,
        "completed_tasks": manifest["completed_tasks"],
        "contains_source_ids": False,
        "contains_source_text": False,
        "visibility": "public" if args.public else "private",
        "upload_requested": args.upload,
    }
    if args.upload:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN must be exported")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=not args.public,
            exist_ok=True,
        )
        api.upload_file(
            path_or_fileobj=card,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=policy,
            path_in_repo="decontamination_policy.json",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=blocklist_dir,
            private=not args.public,
            num_workers=4,
            print_report_every=30,
        )
        report["url"] = f"https://huggingface.co/datasets/{args.repo_id}"
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
