#!/usr/bin/env python3
"""Validate and privately publish the exact reranker-KD training artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import cache_qwen3_reranker_scores as scorer
from scripts.compile_reranker_kd_dataset import sha256_file
from scripts.publish_derived_training_dataset import (
    COMMIT_RE,
    PLATFORM_FILES,
    expected_publication,
    require_dataset_visibility,
    verify_remote_dataset,
)
from scripts.validate_embedding_jsonl import validate as validate_embedding_jsonl


def line_count(path: Path) -> int:
    rows = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            rows += block.count(b"\n")
    return rows


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("publication manifest must be a JSON object")
    return value


def validate_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    if not args.repo_id.startswith("LLM-OS-Models2/"):
        raise ValueError("new KD datasets must use the LLM-OS-Models2 namespace")
    paths = (args.train, args.audit, args.manifest, args.requests)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("one or more KD publication artifacts are missing")
    score_manifest_path = args.score_cache_dir / scorer.MANIFEST_NAME
    score_path = args.score_cache_dir / scorer.SCORES_NAME
    if not score_manifest_path.is_file() or not score_path.is_file():
        raise FileNotFoundError("teacher score cache is incomplete")
    manifest = read_object(args.manifest)
    if manifest.get("artifact_type") != "qwen3_reranker_listwise_embedding_kd_dataset":
        raise ValueError("unexpected KD manifest type")
    score_manifest = read_object(score_manifest_path)
    if score_manifest.get("admissible_for_training") is not True:
        raise ValueError("teacher score cache is not admissible for training")
    declared = manifest.get("files", {})
    actual = {
        "requests": sha256_file(args.requests),
        "scores": sha256_file(score_path),
        "train": sha256_file(args.train),
        "audit": sha256_file(args.audit),
    }
    for role, digest in actual.items():
        if declared.get(role, {}).get("sha256") != digest:
            raise ValueError(f"KD {role} SHA differs from the compiler manifest")
    train_evidence = validate_embedding_jsonl(
        args.train, require_teacher_scores=True
    )
    output_rows = manifest.get("counters", {}).get("output_rows")
    input_rows = manifest.get("counters", {}).get("input_rows")
    if train_evidence["rows"] != output_rows or line_count(args.audit) != input_rows:
        raise ValueError("KD row counts differ from the compiler manifest")
    if line_count(args.requests) != input_rows or line_count(score_path) != input_rows:
        raise ValueError("request/score cache rows differ from the compiler manifest")
    return {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "sha256": actual,
        "teacher": manifest["teacher"],
        "selection": manifest["selection"],
    }


def dataset_card(validated: dict[str, Any]) -> str:
    selection = validated["selection"]
    return f"""---
language:
- ko
- en
license: other
task_categories:
- text-retrieval
pretty_name: Korean Qwen3 Reranker Listwise KD Pilot
---

# Korean Qwen3 Reranker Listwise KD Pilot

성능 우선/non-commercial 연구용 private dataset이다. current 1M student의 wide ANN
candidate를 pinned Qwen3-Reranker-8B로 점수화하고, false-negative gate 뒤 teacher score
rank-quantile을 보존했다.

- request rows: **{validated['input_rows']:,}**
- emitted train rows: **{validated['output_rows']:,}**
- candidates/query: `{selection['candidate_pool_size']}`
- selected negatives/query: `{selection['negatives_per_query']}`
- teacher: `{validated['teacher']['model']}@{validated['teacher']['revision']}`
- public benchmark score selection: **사용하지 않음**
- release eligible: **false**

`data/train.jsonl`은 실제 Swift KD 학습 입력이며 `teacher_scores` 순서는
positive, selected negatives 순서다. `metadata/requests.jsonl`과
`metadata/scores.jsonl`은 exact join 재현용이고, score cache에는 query/document 원문이
중복 저장되지 않는다. 원문은 request 파일에만 있다.

코드: https://github.com/LLM-OS-Models/Embedding
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--score-cache-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    validated = validate_artifacts(args)
    report = {
        "repo_id": args.repo_id,
        "visibility": "private",
        "validated": True,
        "input_rows": validated["input_rows"],
        "output_rows": validated["output_rows"],
        "upload_requested": args.upload,
    }
    if not args.upload:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be exported for upload")
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        card = Path(temporary) / "README.md"
        card.write_text(dataset_card(validated), encoding="utf-8")
        sources = {
            "README.md": card,
            "data/train.jsonl": args.train,
            "metadata/audit.jsonl": args.audit,
            "metadata/manifest.json": args.manifest,
            "metadata/requests.jsonl": args.requests,
            "metadata/scores.jsonl": args.score_cache_dir / scorer.SCORES_NAME,
            "metadata/score_cache_manifest.json": (
                args.score_cache_dir / scorer.MANIFEST_NAME
            ),
        }
        expected = expected_publication(sources)
        before = api.dataset_info(repo_id=args.repo_id, files_metadata=True)
        require_dataset_visibility(before, public=False)
        before_files = {item.rfilename for item in getattr(before, "siblings", [])}
        if before_files - set(expected) - PLATFORM_FILES:
            raise RuntimeError("KD repository contains unexpected pre-existing files")
        operations = [
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
            for name, path in sources.items()
        ]
        commit = api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message="Publish verified private Qwen3 reranker KD pilot",
        )
        commit_sha = getattr(commit, "oid", None)
        if not isinstance(commit_sha, str) or not COMMIT_RE.fullmatch(commit_sha):
            raise RuntimeError("KD upload returned no immutable commit SHA")
        verify_remote_dataset(
            api=api,
            repo_id=args.repo_id,
            revision=commit_sha,
            expected=expected,
            public=False,
        )
        if expected_publication(sources) != expected:
            raise RuntimeError("KD source files changed during upload")
    report["commit_url"] = commit.commit_url
    report["commit_sha"] = commit_sha
    report["remote_file_set_exact"] = True
    report["remote_payload_hashes_exact"] = True
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
