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
        default="LLM-OS-Models2/korean-embedding-performance-v1-pilot-50k",
    )
    parser.add_argument("--expected-phase", default="pilot_50k")
    parser.add_argument("--expected-rows", type=int, default=50_000)
    parser.add_argument("--train-name", default="train.jsonl")
    parser.add_argument("--provenance-name", default="provenance.jsonl")
    parser.add_argument(
        "--quality-audit",
        type=Path,
        help="Optional exact training-data audit uploaded with the dataset",
    )
    parser.add_argument(
        "--benchmark-overlap-audit",
        type=Path,
        help="Optional text-only benchmark overlap audit uploaded with the dataset",
    )
    parser.add_argument("--ordered-train", type=Path)
    parser.add_argument("--ordered-provenance", type=Path)
    parser.add_argument("--ordered-manifest", type=Path)
    parser.add_argument("--ordered-quality-audit", type=Path)
    parser.add_argument("--ordered-benchmark-overlap-audit", type=Path)
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
    data_dir: Path,
    card: Path,
    expected_phase: str,
    expected_rows: int,
    train_name: str,
    provenance_name: str,
    quality_audit: Path | None = None,
    benchmark_overlap_audit: Path | None = None,
    ordered_train: Path | None = None,
    ordered_provenance: Path | None = None,
    ordered_manifest: Path | None = None,
    ordered_quality_audit: Path | None = None,
    ordered_benchmark_overlap_audit: Path | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    manifest_path = data_dir / "manifest.json"
    train_path = data_dir / train_name
    provenance_path = data_dir / provenance_name
    required = [manifest_path, train_path, provenance_path, card]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing publication inputs: {missing}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_rows = manifest.get("built_rows", manifest.get("rows"))
    if (expected_phase and manifest.get("phase") != expected_phase) or (
        actual_rows != expected_rows
    ):
        raise ValueError(
            "Dataset does not match publication gate: "
            f"expected={expected_phase}:{expected_rows}, "
            f"found={manifest.get('phase')}:{actual_rows}"
        )
    for name, path in ((train_name, train_path), (provenance_name, provenance_path)):
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
    paths = [train_path, provenance_path, manifest_path, card]
    if quality_audit is not None:
        if not quality_audit.is_file():
            raise FileNotFoundError(f"Missing quality audit: {quality_audit}")
        audit = json.loads(quality_audit.read_text(encoding="utf-8"))
        expected_train_sha = manifest["files"][train_name]["sha256"]
        expected_provenance_sha = manifest["files"][provenance_name]["sha256"]
        if (
            audit.get("rows") != expected_rows
            or audit.get("inputs", {}).get("train", {}).get("sha256")
            != expected_train_sha
            or audit.get("inputs", {}).get("provenance", {}).get("sha256")
            != expected_provenance_sha
            or audit.get("contract_checks", {}).get("status") != "pass"
        ):
            raise ValueError("Quality audit does not match the publication artifacts")
        paths.append(quality_audit)
    if benchmark_overlap_audit is not None:
        if not benchmark_overlap_audit.is_file():
            raise FileNotFoundError(
                f"Missing benchmark overlap audit: {benchmark_overlap_audit}"
            )
        overlap = json.loads(benchmark_overlap_audit.read_text(encoding="utf-8"))
        if (
            overlap.get("rows") != expected_rows
            or overlap.get("inputs", {}).get("train", {}).get("sha256")
            != manifest["files"][train_name]["sha256"]
            or overlap.get("inputs", {}).get("provenance", {}).get("sha256")
            != manifest["files"][provenance_name]["sha256"]
            or overlap.get("unique_critical_query_or_evaluation_matches") != 0
        ):
            raise ValueError(
                "Benchmark overlap audit does not match artifacts or has critical overlap"
            )
        paths.append(benchmark_overlap_audit)
    ordered_core = (ordered_train, ordered_provenance, ordered_manifest)
    if any(ordered_core) and not all(ordered_core):
        raise ValueError("Ordered train/provenance/manifest must be provided together")
    if all(ordered_core):
        assert ordered_train and ordered_provenance and ordered_manifest
        missing_ordered = [
            str(path)
            for path in ordered_core
            if path is not None and not path.is_file()
        ]
        if missing_ordered:
            raise FileNotFoundError(f"Missing ordered artifacts: {missing_ordered}")
        ordered = json.loads(ordered_manifest.read_text(encoding="utf-8"))
        ordered_rows = ordered.get("output_rows")
        ordered_evidence = ordered.get("outputs", {})
        for role, path in (
            ("train", ordered_train),
            ("provenance", ordered_provenance),
        ):
            declared = ordered_evidence.get(role, {})
            if (
                line_count(path) != ordered_rows
                or sha256(path) != declared.get("sha256")
            ):
                raise ValueError(f"Ordered {role} differs from its manifest")
        paths.extend([ordered_train, ordered_provenance, ordered_manifest])
        if ordered_quality_audit:
            quality = json.loads(ordered_quality_audit.read_text(encoding="utf-8"))
            if (
                quality.get("rows") != ordered_rows
                or quality.get("inputs", {}).get("train", {}).get("sha256")
                != ordered_evidence["train"]["sha256"]
                or quality.get("inputs", {}).get("provenance", {}).get("sha256")
                != ordered_evidence["provenance"]["sha256"]
                or quality.get("contract_checks", {}).get("status") != "pass"
            ):
                raise ValueError("Ordered quality audit differs from ordered artifacts")
            paths.append(ordered_quality_audit)
        if ordered_benchmark_overlap_audit:
            overlap = json.loads(
                ordered_benchmark_overlap_audit.read_text(encoding="utf-8")
            )
            if (
                overlap.get("rows") != ordered_rows
                or overlap.get("inputs", {}).get("train", {}).get("sha256")
                != ordered_evidence["train"]["sha256"]
                or overlap.get("inputs", {}).get("provenance", {}).get("sha256")
                != ordered_evidence["provenance"]["sha256"]
                or overlap.get("unique_critical_query_or_evaluation_matches") != 0
            ):
                raise ValueError(
                    "Ordered benchmark audit differs or has critical overlap"
                )
            paths.append(ordered_benchmark_overlap_audit)
    return manifest, paths


def main() -> None:
    args = parse_args()
    manifest, paths = validate(
        args.data_dir.resolve(),
        args.card.resolve(),
        args.expected_phase,
        args.expected_rows,
        args.train_name,
        args.provenance_name,
        args.quality_audit.resolve() if args.quality_audit else None,
        (
            args.benchmark_overlap_audit.resolve()
            if args.benchmark_overlap_audit
            else None
        ),
        args.ordered_train.resolve() if args.ordered_train else None,
        args.ordered_provenance.resolve() if args.ordered_provenance else None,
        args.ordered_manifest.resolve() if args.ordered_manifest else None,
        (
            args.ordered_quality_audit.resolve()
            if args.ordered_quality_audit
            else None
        ),
        (
            args.ordered_benchmark_overlap_audit.resolve()
            if args.ordered_benchmark_overlap_audit
            else None
        ),
    )
    report = {
        "repo_id": args.repo_id,
        "visibility": "public" if args.public else "private",
        "phase": manifest.get("phase", manifest.get("use_policy")),
        "rows": manifest.get("built_rows", manifest.get("rows")),
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
    train_path, provenance_path, manifest_path, card_path, *_optional_paths = paths
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
    if args.quality_audit:
        operations.append(
            CommitOperationAdd(
                path_in_repo="metadata/training_data_quality_audit.json",
                path_or_fileobj=args.quality_audit.resolve(),
            )
        )
    if args.benchmark_overlap_audit:
        operations.append(
            CommitOperationAdd(
                path_in_repo="metadata/benchmark_overlap_audit.json",
                path_or_fileobj=args.benchmark_overlap_audit.resolve(),
            )
        )
    if args.ordered_train:
        operations.extend(
            [
                CommitOperationAdd(
                    path_in_repo="data/train.homogeneous-b16-length-bucketed.jsonl",
                    path_or_fileobj=args.ordered_train.resolve(),
                ),
                CommitOperationAdd(
                    path_in_repo="metadata/provenance.homogeneous-b16-length-bucketed.jsonl",
                    path_or_fileobj=args.ordered_provenance.resolve(),
                ),
                CommitOperationAdd(
                    path_in_repo="metadata/homogeneous-b16-length-bucketed.manifest.json",
                    path_or_fileobj=args.ordered_manifest.resolve(),
                ),
            ]
        )
    if args.ordered_quality_audit:
        operations.append(
            CommitOperationAdd(
                path_in_repo="metadata/ordered_training_data_quality_audit.json",
                path_or_fileobj=args.ordered_quality_audit.resolve(),
            )
        )
    if args.ordered_benchmark_overlap_audit:
        operations.append(
            CommitOperationAdd(
                path_in_repo="metadata/ordered_benchmark_overlap_audit.json",
                path_or_fileobj=args.ordered_benchmark_overlap_audit.resolve(),
            )
        )
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
