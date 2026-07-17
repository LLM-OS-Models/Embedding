#!/usr/bin/env python3
"""Validate and privately publish a text-strict legal evaluation/data artifact."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

try:
    from publish_best_embedding_model import load_hf_token
except ModuleNotFoundError:
    from scripts.publish_best_embedding_model import load_hf_token


ALLOWED_NAMESPACE = "LLM-OS-Models2/"
KINDS = {
    "retrieval": ("queries.jsonl", "corpus.jsonl", "qrels.jsonl", "provenance.jsonl"),
    "trainer-validation": ("validation.jsonl", "provenance.jsonl"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=sorted(KINDS), required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--hf-token-file", type=Path)
    parser.add_argument("--report", type=Path)
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


def validate(artifact_dir: Path, kind: str, repo_id: str) -> dict[str, Any]:
    artifact_dir = artifact_dir.expanduser().resolve()
    if not repo_id.startswith(ALLOWED_NAMESPACE) or repo_id.count("/") != 1:
        raise ValueError("Dataset repo must be directly under LLM-OS-Models2")
    manifest_path = artifact_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("Artifact manifest is not complete")
    expected_names = KINDS[kind]
    declared_files = manifest.get("files", {})
    if set(declared_files) != set(expected_names):
        raise ValueError("Manifest file allowlist differs from the publication contract")
    files: dict[str, dict[str, Any]] = {}
    for name in expected_names:
        path = artifact_dir / name
        declared = declared_files[name]
        actual_sha = sha256(path)
        actual_rows = line_count(path)
        if actual_sha != declared.get("sha256") or actual_rows != declared.get("rows"):
            raise ValueError(f"Artifact drift: {name}")
        files[name] = {
            "path": path,
            "sha256": actual_sha,
            "rows": actual_rows,
            "bytes": path.stat().st_size,
        }
    assertions = manifest.get("assertions", {})
    if kind == "retrieval":
        required_zero = (
            "selected_source_candidate_id_overlap_with_training",
            "selected_source_document_sha256_overlap_with_training",
            "selected_query_hash_overlap_with_training_text",
            "selected_positive_hash_overlap_with_training_text",
            "selected_query_hash_overlap_with_benchmark",
            "selected_positive_hash_overlap_with_benchmark",
        )
        if any(assertions.get(key) != 0 for key in required_zero):
            raise ValueError("Retrieval artifact did not pass all leakage assertions")
        if any(files[name]["rows"] != 10_000 for name in expected_names):
            raise ValueError("Retrieval artifact must contain exactly 10K rows per file")
    else:
        required_zero = (
            "selected_query_training_text_overlap",
            "selected_positive_training_text_overlap",
            "selected_negative_training_text_overlap",
            "selected_source_document_training_provenance_overlap",
        )
        if any(assertions.get(key) != 0 for key in required_zero):
            raise ValueError("Trainer validation did not pass all leakage assertions")
        if assertions.get("source_holdout_contract_verified") is not True:
            raise ValueError("Trainer validation source contract was not verified")
        if any(files[name]["rows"] != 512 for name in expected_names):
            raise ValueError("Trainer validation must contain exactly 512 rows per file")
    return {
        "artifact_dir": artifact_dir,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256(manifest_path),
        "files": files,
        "repo_id": repo_id,
        "kind": kind,
    }


def dataset_card(validated: dict[str, Any]) -> str:
    manifest = validated["manifest"]
    kind = validated["kind"]
    rows = next(iter(validated["files"].values()))["rows"]
    if kind == "retrieval":
        size_category = "10K<n<100K"
        purpose = (
            "모델 선택용 Korean legal/public retrieval board다. whole-source-document와 "
            "모든 선언 학습 역할의 normalized exact text를 함께 제외했다."
        )
        files = "- `queries.jsonl`, `corpus.jsonl`, `qrels.jsonl`: 10K retrieval\n- `provenance.jsonl`: pinned source와 exclusion evidence"
    else:
        size_category = "n<1K"
        purpose = (
            "train-time monitoring용 strict ms-swift validation이다. text-strict legal 10K에서 "
            "source-balanced 512행과 같은-repository lexical HN4를 고정했다."
        )
        files = "- `validation.jsonl`: 512 query/positive/HN4 rows\n- `provenance.jsonl`: row hash, source document, negative IDs"
    return f"""---
language:
- ko
license: other
task_categories:
- text-retrieval
pretty_name: {manifest.get('artifact_id')}
size_categories:
- {size_category}
---

# {manifest.get('artifact_id')}

{purpose}

- rows: **{rows:,}**
- visibility: **private**
- independence: **I, not Z**
- public benchmark used for model selection: **false**
- exact training query/positive overlap: **0**
- exact benchmark query/positive overlap: **0**

같은 Legalize-KR repository와 source-native schema를 사용하므로 unseen-source 또는 clean
zero-shot이라고 부르지 않는다. source-native positive 하나만 있어 relevance judgment도
exhaustive하지 않다. 원 source 권리는 `provenance.jsonl`과 manifest를 따른다.

## Files

{files}
- `manifest.json`: input/output SHA-256, exclusion counters와 assertions

Code: https://github.com/LLM-OS-Models/Embedding
"""


def require_private(api: Any, repo_id: str) -> Any:
    info = api.dataset_info(repo_id=repo_id, files_metadata=True)
    if getattr(info, "private", None) is not True:
        raise RuntimeError("Refusing upload because dataset is not confirmed private")
    return info


def upload(validated: dict[str, Any], token: str) -> dict[str, Any]:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    repo_id = validated["repo_id"]
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    require_private(api, repo_id)
    card_bytes = dataset_card(validated).encode("utf-8")
    operations = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=io.BytesIO(card_bytes)),
        CommitOperationAdd(
            path_in_repo="manifest.json", path_or_fileobj=validated["manifest_path"]
        ),
    ]
    for name, evidence in validated["files"].items():
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=evidence["path"]))
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Publish verified {validated['kind']} artifact",
    )
    info = require_private(api, repo_id)
    remote_files = {item.rfilename for item in info.siblings}
    expected = {"README.md", "manifest.json", *validated["files"]}
    unexpected = remote_files - expected - {".gitattributes"}
    if not expected <= remote_files or unexpected:
        raise RuntimeError("Remote dataset file allowlist verification failed")
    for name, expected_sha in (
        ("manifest.json", validated["manifest_sha256"]),
        *((name, evidence["sha256"]) for name, evidence in validated["files"].items()),
    ):
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=name,
                revision=info.sha,
                token=token,
            )
        )
        if sha256(downloaded) != expected_sha:
            raise RuntimeError(f"Remote hash mismatch: {name}")
    return {
        "commit_sha": info.sha,
        "commit_url": getattr(commit, "commit_url", None),
        "remote_private": True,
        "remote_file_allowlist_exact": True,
        "remote_hashes_exact": True,
    }


def main() -> None:
    args = parse_args()
    validated = validate(args.artifact_dir, args.kind, args.repo_id)
    report: dict[str, Any] = {
        "repo_id": args.repo_id,
        "kind": args.kind,
        "visibility": "private",
        "manifest_sha256": validated["manifest_sha256"],
        "files": {
            name: {key: value for key, value in evidence.items() if key != "path"}
            for name, evidence in validated["files"].items()
        },
        "validated": True,
        "upload_requested": args.upload,
    }
    if args.upload:
        report.update(upload(validated, load_hf_token(args.hf_token_file)))
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
