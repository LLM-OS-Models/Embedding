#!/usr/bin/env python3
"""Privately preserve the exact legal candidate snapshot behind holdout v2."""

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


ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "LLM-OS-Models2/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
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


def rows(path: Path) -> int:
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


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def validate(candidate_dir: Path, reference_manifest: Path, repo_id: str) -> dict[str, Any]:
    candidate_dir = candidate_dir.expanduser().resolve()
    reference_manifest = reference_manifest.expanduser().resolve()
    if not repo_id.startswith(NAMESPACE) or repo_id.count("/") != 1:
        raise ValueError("Snapshot repo must be directly under LLM-OS-Models2")
    reference = read_json(reference_manifest)
    if reference.get("status") != "complete" or reference.get("artifact_id") != "korean-legal-public-source-heldout-retrieval-v2-text-strict":
        raise ValueError("Reference is not the complete text-strict v2 manifest")
    declared = reference.get("inputs", {}).get("candidate_sources", {}).get("files")
    if not isinstance(declared, list) or len(declared) != 16:
        raise ValueError("Reference manifest must declare exactly 16 candidate files")
    evidence: dict[str, dict[str, Any]] = {}
    total_rows = 0
    for item in declared:
        source = (ROOT / str(item["path"])).resolve()
        try:
            relative = source.relative_to(candidate_dir).as_posix()
        except ValueError as error:
            raise ValueError("Referenced candidate escapes the candidate directory") from error
        if source.suffix != ".jsonl" or not source.is_file():
            raise ValueError(f"Candidate file is missing or invalid: {relative}")
        actual_rows = rows(source)
        actual_sha = sha256(source)
        if actual_rows != item.get("rows") or actual_sha != item.get("sha256"):
            raise ValueError(f"Candidate differs from the v2 reference: {relative}")
        extractor = source.with_suffix(".manifest.json")
        extractor_payload = read_json(extractor)
        if (
            extractor_payload.get("output_sha256") != actual_sha
            or extractor_payload.get("summary", {}).get("records_emitted") != actual_rows
            or extractor_payload.get("parameters", {}).get("shard_count") != 16
            or extractor_payload.get("parameters", {}).get("max_records") != 25_000
        ):
            raise ValueError(f"Extractor manifest differs: {relative}")
        extractor_relative = extractor.relative_to(candidate_dir).as_posix()
        evidence[relative] = {
            "path": source,
            "bytes": source.stat().st_size,
            "rows": actual_rows,
            "sha256": actual_sha,
        }
        evidence[extractor_relative] = {
            "path": extractor,
            "bytes": extractor.stat().st_size,
            "sha256": sha256(extractor),
        }
        total_rows += actual_rows
    snapshot_manifest = {
        "schema_version": 1,
        "artifact_id": "korean-legal-holdout-candidates-v1-shards12-15",
        "status": "complete",
        "visibility": "private",
        "source_revisions": {
            "legalize-kr/admrule-kr": "64a5a272909ab5bc077b0ad9519ef31de8febb46",
            "legalize-kr/legalize-kr": "db3cd760c14042ee04fd9166e1bdbb662fc999bc",
            "legalize-kr/ordinance-kr": "6443e5dd5833d863219064cd362111f516430bec",
            "legalize-kr/precedent-kr": "40cd00e54df19d98562abb170c8ff51fd6fe2c2e",
        },
        "parameters": {"shard_count": 16, "shard_indices": [12, 13, 14, 15], "max_records_per_source_shard": 25_000},
        "reference": {
            "artifact_id": reference["artifact_id"],
            "manifest_sha256": sha256(reference_manifest),
        },
        "counts": {"candidate_files": 16, "extractor_manifests": 16, "candidate_rows": total_rows},
        "files": {
            name: {key: value for key, value in item.items() if key != "path"}
            for name, item in sorted(evidence.items())
        },
    }
    snapshot_bytes = canonical_bytes(snapshot_manifest)
    return {
        "candidate_dir": candidate_dir,
        "reference_manifest": reference_manifest,
        "repo_id": repo_id,
        "evidence": evidence,
        "snapshot_manifest": snapshot_manifest,
        "snapshot_bytes": snapshot_bytes,
        "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
    }


def card(validated: dict[str, Any]) -> bytes:
    count = validated["snapshot_manifest"]["counts"]
    return f"""---
language:
- ko
license: other
task_categories:
- text-retrieval
pretty_name: Korean legal holdout candidates v1 shards 12-15
size_categories:
- 100K<n<1M
---

# Korean legal holdout candidates v1 shards 12-15

Text-strict legal retrieval v2를 재현한 pinned intermediate candidate snapshot이다.

- candidate rows: **{count['candidate_rows']:,}**
- JSONL files: **16**
- extractor manifests: **16**
- source file-hash shards: **12, 13, 14, 15 / 16**
- visibility: **private**

이 데이터는 평가 결과가 아니고 source-native structural pair 후보이다. 독립성·training
text·benchmark exclusion은 이 snapshot 자체가 아니라 v2 builder와 최종 manifest가
강제한다. 원 repository의 권리와 provenance를 그대로 따른다.

Code: https://github.com/LLM-OS-Models/Embedding
""".encode("utf-8")


def require_private(api: Any, repo_id: str) -> Any:
    info = api.dataset_info(repo_id=repo_id, files_metadata=True)
    if getattr(info, "private", None) is not True:
        raise RuntimeError("Snapshot repository is not confirmed private")
    return info


def upload(validated: dict[str, Any], token: str) -> dict[str, Any]:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)
    repo_id = validated["repo_id"]
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    require_private(api, repo_id)
    operations = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=io.BytesIO(card(validated))),
        CommitOperationAdd(path_in_repo="snapshot_manifest.json", path_or_fileobj=io.BytesIO(validated["snapshot_bytes"])),
    ]
    for name, item in sorted(validated["evidence"].items()):
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=item["path"]))
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message="Preserve exact legal holdout candidate snapshot",
    )
    info = require_private(api, repo_id)
    expected = {"README.md", "snapshot_manifest.json", *validated["evidence"]}
    remote = {item.rfilename for item in info.siblings}
    if not expected <= remote or remote - expected - {".gitattributes"}:
        raise RuntimeError("Remote snapshot allowlist verification failed")
    remote_manifest = Path(
        hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="snapshot_manifest.json", revision=info.sha, token=token)
    )
    if sha256(remote_manifest) != validated["snapshot_sha256"]:
        raise RuntimeError("Remote snapshot manifest hash mismatch")
    # files_metadata exposes the SHA-256 OID for Xet/LFS objects. Small files
    # are downloaded because their Git blob OID is not a content SHA-256.
    remote_by_name = {item.rfilename: item for item in info.siblings}
    for name, item in validated["evidence"].items():
        sibling = remote_by_name[name]
        lfs = getattr(sibling, "lfs", None)
        remote_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if remote_sha is not None:
            if remote_sha != item["sha256"]:
                raise RuntimeError(f"Remote snapshot LFS hash mismatch: {name}")
            continue
        downloaded = Path(
            hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=name, revision=info.sha, token=token)
        )
        if sha256(downloaded) != item["sha256"]:
            raise RuntimeError(f"Remote snapshot file hash mismatch: {name}")
    return {
        "commit_sha": info.sha,
        "commit_url": getattr(commit, "commit_url", None),
        "remote_private": True,
        "remote_allowlist_exact": True,
        "remote_hashes_exact": True,
    }


def main() -> None:
    args = parse_args()
    validated = validate(args.candidate_dir, args.reference_manifest, args.repo_id)
    report: dict[str, Any] = {
        "repo_id": args.repo_id,
        "visibility": "private",
        "snapshot_manifest_sha256": validated["snapshot_sha256"],
        "counts": validated["snapshot_manifest"]["counts"],
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
