#!/usr/bin/env python3
"""Validate and publish the exact derived JSONL consumed by a training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


COMMIT_RE = re.compile(r"[0-9a-f]{40}")
PLATFORM_FILES = {".gitattributes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mining-manifest", type=Path)
    parser.add_argument("--mining-audit", type=Path)
    parser.add_argument("--quality-audit", type=Path)
    parser.add_argument("--benchmark-overlap-audit", type=Path)
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


def validate_public_rights(manifest: dict[str, Any], provenance: Path) -> int:
    if manifest.get("release_eligible") is not True:
        raise ValueError("public dataset requires manifest.release_eligible=true")
    if manifest.get("release_blockers"):
        raise ValueError("public dataset manifest has unresolved release blockers")
    if manifest.get("visibility") != "public":
        raise ValueError("public dataset manifest visibility is not public")
    rows = 0
    with provenance.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"provenance row {line_number} is not an object")
            for field in ("source", "revision", "license"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(
                        f"public provenance row {line_number} has no {field}"
                    )
            if row.get("redistribution_allowed") is not True:
                raise ValueError(
                    f"public provenance row {line_number} is not redistribution-approved"
                )
            rows += 1
    return rows


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.repo_id.startswith("LLM-OS-Models2/"):
        raise ValueError("new derived datasets must use the LLM-OS-Models2 namespace")
    paths = [args.train, args.provenance, args.manifest]
    if args.mining_manifest:
        paths.append(args.mining_manifest)
    if args.mining_audit:
        paths.append(args.mining_audit)
    if args.quality_audit:
        paths.append(args.quality_audit)
    if args.benchmark_overlap_audit:
        paths.append(args.benchmark_overlap_audit)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing derived dataset publication inputs: {missing}"
        )

    manifest = read_json(args.manifest)
    if getattr(args, "public", False):
        rights_rows = validate_public_rights(manifest, args.provenance)
    else:
        rights_rows = None
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
    if rights_rows is not None:
        evidence["public_rights"] = {
            "rows": rights_rows,
            "all_rows_redistribution_approved": True,
        }
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
    overlap = None
    if args.benchmark_overlap_audit:
        overlap = read_json(args.benchmark_overlap_audit)
        overlap_inputs = overlap.get("inputs", {})
        if (
            overlap.get("rows") != rows
            or overlap_inputs.get("train", {}).get("sha256")
            != evidence["train"]["sha256"]
            or overlap_inputs.get("provenance", {}).get("sha256")
            != evidence["provenance"]["sha256"]
            or overlap.get("unique_critical_query_or_evaluation_matches") != 0
        ):
            raise ValueError(
                "Benchmark overlap audit differs from final curriculum or has critical overlap"
            )
        evidence["benchmark_overlap_audit"] = {
            "sha256": sha256(args.benchmark_overlap_audit)
        }
    return {
        "rows": rows,
        "manifest": manifest,
        "mining": mining,
        "overlap": overlap,
        "evidence": evidence,
    }


def require_dataset_visibility(info: Any, *, public: bool) -> None:
    expected_private = not public
    if getattr(info, "private", None) is not expected_private:
        expected = "public" if public else "private"
        raise RuntimeError(f"dataset repository is not exactly {expected}")


def remote_lfs_identity(item: Any) -> tuple[str | None, int | None]:
    lfs = getattr(item, "lfs", None)
    if isinstance(lfs, dict):
        return lfs.get("sha256"), lfs.get("size")
    return getattr(lfs, "sha256", None), getattr(lfs, "size", None)


def verify_remote_dataset(
    *,
    api: Any,
    repo_id: str,
    revision: str,
    expected: dict[str, dict[str, Any]],
    public: bool,
) -> None:
    """Verify one immutable dataset commit and every uploaded payload byte."""

    if not COMMIT_RE.fullmatch(revision):
        raise RuntimeError("dataset upload returned no immutable commit SHA")
    info = api.dataset_info(
        repo_id=repo_id, revision=revision, files_metadata=True
    )
    require_dataset_visibility(info, public=public)
    siblings = {item.rfilename: item for item in getattr(info, "siblings", [])}
    remote_files = set(siblings)
    expected_files = set(expected)
    if expected_files - remote_files or remote_files - expected_files - PLATFORM_FILES:
        raise RuntimeError("remote dataset file set differs from the upload allowlist")
    for name, evidence in expected.items():
        item = siblings[name]
        remote_sha, remote_size = remote_lfs_identity(item)
        if remote_sha is not None or remote_size is not None:
            if remote_sha != evidence["sha256"] or remote_size != evidence["size_bytes"]:
                raise RuntimeError(f"remote dataset LFS object mismatch: {name}")
            continue
        downloaded = Path(
            api.hf_hub_download(
                repo_id=repo_id,
                filename=name,
                repo_type="dataset",
                revision=revision,
            )
        )
        if (
            downloaded.stat().st_size != evidence["size_bytes"]
            or sha256(downloaded) != evidence["sha256"]
        ):
            raise RuntimeError(f"remote dataset metadata mismatch: {name}")


def expected_publication(sources: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        for name, path in sources.items()
    }


def dataset_card(args: argparse.Namespace, validated: dict[str, Any]) -> str:
    sources = "\n".join(
        f"- https://huggingface.co/datasets/{value}" for value in args.source_dataset
    )
    manifest = validated["manifest"]
    adaptation = manifest.get("benchmark_adaptation", manifest.get("adaptation_label"))
    overlap = validated.get("overlap")
    overlap_text = (
        f"- exact benchmark query/evaluation-text matches: "
        f"**{overlap['unique_critical_query_or_evaluation_matches']}**\n"
        f"- exact retrieval-corpus matches: "
        f"**{overlap['unique_retrieval_corpus_matches']:,} unique hashes**"
        if overlap is not None
        else "- benchmark text-overlap audit: not attached"
    )
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
{overlap_text}

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
- `metadata/benchmark_overlap_audit.json`: 원문 없는 15-task exact text-hash overlap,
  role/source count와 task 위치

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
        sources = {
            "README.md": card,
            "data/train.jsonl": args.train,
            "metadata/provenance.jsonl": args.provenance,
            "metadata/final_manifest.json": args.manifest,
        }
        if args.mining_manifest:
            sources["metadata/mining_manifest.json"] = args.mining_manifest
        if args.mining_audit:
            sources["metadata/mining_audit.jsonl"] = args.mining_audit
        if args.quality_audit:
            sources["metadata/training_data_quality_audit.json"] = args.quality_audit
        if args.benchmark_overlap_audit:
            sources["metadata/benchmark_overlap_audit.json"] = args.benchmark_overlap_audit
        expected = expected_publication(sources)
        before = api.dataset_info(repo_id=args.repo_id, files_metadata=True)
        require_dataset_visibility(before, public=args.public)
        before_files = {item.rfilename for item in getattr(before, "siblings", [])}
        if before_files - set(expected) - PLATFORM_FILES:
            raise RuntimeError("dataset repository contains unexpected pre-existing files")
        operations = [
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
            for name, path in sources.items()
        ]
        commit = api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message="Publish exact quantile-hard-negative training curriculum",
        )
        commit_sha = getattr(commit, "oid", None)
        if not isinstance(commit_sha, str) or not COMMIT_RE.fullmatch(commit_sha):
            raise RuntimeError("dataset upload returned no immutable commit SHA")
        verify_remote_dataset(
            api=api,
            repo_id=args.repo_id,
            revision=commit_sha,
            expected=expected,
            public=args.public,
        )
        if expected_publication(sources) != expected:
            raise RuntimeError("source dataset files changed during upload")
    report["commit_url"] = commit.commit_url
    report["commit_sha"] = commit_sha
    report["remote_file_set_exact"] = True
    report["remote_payload_hashes_exact"] = True
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
