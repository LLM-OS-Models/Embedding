#!/usr/bin/env python3
"""Validate and publish the exact derived JSONL consumed by a training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mining-manifest", type=Path)
    parser.add_argument("--mining-audit", type=Path)
    parser.add_argument("--quality-audit", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--source-dataset", action="append", default=[])
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def declared_output(manifest: dict[str, Any], role: str) -> dict[str, Any]:
    value = manifest.get("outputs", {}).get(role)
    if not isinstance(value, dict) or not value.get("sha256"):
        raise ValueError(f"Final manifest has no outputs.{role} SHA")
    return value


def validate(args: argparse.Namespace) -> dict[str, Any]:
    paths = [args.train, args.provenance, args.manifest]
    if args.mining_manifest:
        paths.append(args.mining_manifest)
    if args.mining_audit:
        paths.append(args.mining_audit)
    if args.quality_audit:
        paths.append(args.quality_audit)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing derived dataset publication inputs: {missing}"
        )

    manifest = read_json(args.manifest)
    rows = manifest.get("output_rows")
    if not isinstance(rows, int) or rows < 2:
        raise ValueError("Final manifest output_rows must be at least two")
    train_declared = declared_output(manifest, "train")
    provenance_declared = declared_output(manifest, "provenance")
    evidence = {}
    for role, path, declared in (
        ("train", args.train, train_declared),
        ("provenance", args.provenance, provenance_declared),
    ):
        actual_sha = sha256(path)
        actual_rows = line_count(path)
        if actual_sha != declared["sha256"] or actual_rows != rows:
            raise ValueError(
                f"{role} drift: rows={actual_rows}/{rows}, sha={actual_sha}/{declared['sha256']}"
            )
        evidence[role] = {"rows": actual_rows, "sha256": actual_sha}

    mining = None
    if args.mining_manifest:
        mining = read_json(args.mining_manifest)
        if mining.get("selection_strategy") != "score_rank_quantiles":
            raise ValueError("Only score_rank_quantiles derived data is admitted")
        if mining.get("candidate_pool_size") != 24 or mining.get("num_negatives") != 7:
            raise ValueError("Expected mining pool24/negative7 contract")
        if args.mining_audit:
            audit_rows = line_count(args.mining_audit)
            if audit_rows != mining.get("rows"):
                raise ValueError(
                    "Mining audit row count does not match mining manifest"
                )
            evidence["mining_audit"] = {
                "rows": audit_rows,
                "sha256": sha256(args.mining_audit),
            }
    evidence["manifest"] = {"sha256": sha256(args.manifest)}
    if args.mining_manifest:
        evidence["mining_manifest"] = {"sha256": sha256(args.mining_manifest)}
    if args.quality_audit:
        quality = read_json(args.quality_audit)
        if quality.get("rows") != rows:
            raise ValueError("Quality audit row count does not match final curriculum")
        checks = quality.get("contract_checks", {})
        if checks.get("status") != "pass":
            raise ValueError("Quality audit contract did not pass")
        quality_inputs = quality.get("inputs", {})
        if (
            quality_inputs.get("train", {}).get("sha256") != evidence["train"]["sha256"]
            or quality_inputs.get("provenance", {}).get("sha256")
            != evidence["provenance"]["sha256"]
        ):
            raise ValueError(
                "Quality audit belongs to different train/provenance files"
            )
        evidence["quality_audit"] = {"sha256": sha256(args.quality_audit)}
    return {"rows": rows, "manifest": manifest, "mining": mining, "evidence": evidence}


def dataset_card(args: argparse.Namespace, validated: dict[str, Any]) -> str:
    sources = "\n".join(
        f"- https://huggingface.co/datasets/{value}" for value in args.source_dataset
    )
    manifest = validated["manifest"]
    adaptation = manifest.get("benchmark_adaptation", manifest.get("adaptation_label"))
    return f"""---
language:
- ko
- en
license: other
task_categories:
- sentence-similarity
- text-retrieval
pretty_name: {args.title}
size_categories:
- 100K<n<1M
---

# {args.title}

실제 학습 queue가 소비하는 source-homogeneous, length-bucketed 파생 JSONL이다.
current-student FAISS 후보의 exact float32 score를 재계산하고, positive-relative `.95`
filter 뒤 top-24 score-rank에서 7개 quantile negative를 선택했다.

- rows: **{validated['rows']:,}**
- batch size: `{manifest.get('batch_size')}`
- benchmark adaptation: `{adaptation}`
- release eligible: **false**
- use: performance/non-commercial research

Sionic 9 및 MTEB task-family train source가 포함될 수 있다. 이 dataset으로 학습한 모델의
관련 점수는 clean zero-shot으로 해석하면 안 된다. 원 source의 license와 attribution은
`metadata/provenance.jsonl`에 row별로 보존되며 이 카드의 `other` 표기가 upstream 권리를
재허가하지 않는다.

## 파일

- `data/train.jsonl`: ms-swift query/positive/negative schema
- `metadata/provenance.jsonl`: source/revision/license/exposure lineage
- `metadata/final_manifest.json`: exact order, row count, SHA, source counts
- `metadata/mining_manifest.json`: model weight SHA, FAISS, filter와 selection 계약
- `metadata/mining_audit.jsonl`: per-input positive threshold와 selected document hashes
- `metadata/training_data_quality_audit.json`: 실제 final train/provenance의 source, query
  style, negative 수, 길이, 중복과 homogeneous-batch contract

## 원 dataset

{sources or '- final manifest의 input path와 SHA를 참조'}

코드: https://github.com/LLM-OS-Models/Embedding
"""


def main() -> None:
    args = parse_args()
    validated = validate(args)
    report = {
        "repo_id": args.repo_id,
        "rows": validated["rows"],
        "visibility": "public" if args.public else "private",
        "validated": True,
        "upload_requested": args.upload,
        "evidence": validated["evidence"],
    }
    if not args.upload:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be exported")
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    with tempfile.TemporaryDirectory() as temporary:
        card = Path(temporary) / "README.md"
        card.write_text(dataset_card(args, validated), encoding="utf-8")
        operations = [
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card),
            CommitOperationAdd(
                path_in_repo="data/train.jsonl", path_or_fileobj=args.train
            ),
            CommitOperationAdd(
                path_in_repo="metadata/provenance.jsonl",
                path_or_fileobj=args.provenance,
            ),
            CommitOperationAdd(
                path_in_repo="metadata/final_manifest.json",
                path_or_fileobj=args.manifest,
            ),
        ]
        if args.mining_manifest:
            operations.append(
                CommitOperationAdd(
                    path_in_repo="metadata/mining_manifest.json",
                    path_or_fileobj=args.mining_manifest,
                )
            )
        if args.mining_audit:
            operations.append(
                CommitOperationAdd(
                    path_in_repo="metadata/mining_audit.jsonl",
                    path_or_fileobj=args.mining_audit,
                )
            )
        if args.quality_audit:
            operations.append(
                CommitOperationAdd(
                    path_in_repo="metadata/training_data_quality_audit.json",
                    path_or_fileobj=args.quality_audit,
                )
            )
        commit = api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message="Publish exact quantile-hard-negative training curriculum",
        )
    report["commit_url"] = commit.commit_url
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
