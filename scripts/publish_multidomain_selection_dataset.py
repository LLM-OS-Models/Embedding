#!/usr/bin/env python3
"""Publish the fixed internal multidomain selector to one private HF dataset.

The publisher accepts only the verified finance/knowledge selection artifact,
copies a fixed allowlist into isolated staging, and re-downloads every metadata
file (or checks its LFS identity) at the immutable commit before reporting
success. It never changes repository visibility to public.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_multidomain_selection import PROTOCOL_ID, validate_dataset
    from scripts.publish_best_embedding_model import load_hf_token, sha256
    from scripts.publish_derived_training_dataset import (
        expected_publication,
        require_dataset_visibility,
        verify_remote_dataset,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from evaluate_multidomain_selection import PROTOCOL_ID, validate_dataset
    from publish_best_embedding_model import load_hf_token, sha256
    from publish_derived_training_dataset import (
        expected_publication,
        require_dataset_visibility,
        verify_remote_dataset,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "outputs/evaluation/multidomain-selection-heldout-v1"
DEFAULT_REPO = "LLM-OS-Models2/korean-embedding-multidomain-selection-heldout-v1"
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
PAYLOAD_NAMES = (
    "manifest.json",
    "queries.jsonl",
    "corpus.jsonl",
    "qrels.jsonl",
    "provenance.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--hf-token-file", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            count += block.count(b"\n")
    return count


def portable_provenance_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def validate_publication_source(root: Path, repo_id: str) -> dict[str, Any]:
    if repo_id != DEFAULT_REPO:
        raise ValueError(f"Selection dataset repo must be exactly {DEFAULT_REPO}")
    resolved = root.expanduser().resolve()
    if resolved != DEFAULT_DATASET.resolve():
        raise ValueError("Selection publication source must be the canonical artifact")
    queries, corpus, qrels, manifest = validate_dataset(resolved)
    if manifest.get("purpose") != (
        "fixed internal model selection; never training; not a public benchmark"
    ):
        raise ValueError("Selection dataset purpose drifted")
    if len(queries) != 1900 or len(corpus) != 4795:
        raise ValueError("Selection query/corpus cardinality drifted")
    if sum(len(values) for values in qrels.values()) != 2941:
        raise ValueError("Selection qrel cardinality drifted")
    domains = manifest.get("domains", {})
    if (
        domains.get("finance", {}).get("queries") != 900
        or domains.get("finance", {}).get("corpus_training_text_occurrences")
        != 1373
        or domains.get("knowledge", {}).get("queries") != 1000
    ):
        raise ValueError("Selection domain disclosure drifted")
    expected_sources = {
        (
            "BCCard/BCAI-Finance-Kor-Embedding-Triplet",
            "f63d59969dba9916bd34c86c82112331890b11da",
            "validation",
        ),
        (
            "etri-lirs/KoTSQA-v.2.0",
            "ff9349df469a765b4561959e36ef1b3f377765cd",
            "test-used-as-fixed-internal-selection",
        ),
    }
    provenance_path = resolved / "provenance.jsonl"
    observed_sources: set[tuple[str, str, str]] = set()
    with provenance_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            observed_sources.add(
                (str(row.get("source")), str(row.get("revision")), str(row.get("split")))
            )
    if observed_sources != expected_sources or line_count(provenance_path) != 1900:
        raise ValueError("Selection provenance source contract drifted")
    path_values = [row.get("path") for row in manifest.get("source_inputs", [])]
    path_values.extend(
        row.get("path") for row in manifest.get("training_roles_excluded", [])
    )
    path_values.append(manifest.get("benchmark_blocklist", {}).get("root"))
    if not path_values or not all(portable_provenance_path(value) for value in path_values):
        raise ValueError("Selection manifest contains non-portable provenance paths")
    payload = {name: resolved / name for name in PAYLOAD_NAMES}
    if any(not path.is_file() or path.is_symlink() for path in payload.values()):
        raise ValueError("Selection publication payload is incomplete or unsafe")
    return {
        "root": resolved,
        "manifest": manifest,
        "payload": payload,
        "source_manifest_sha256": sha256(resolved / "manifest.json"),
    }


def dataset_card(validated: dict[str, Any]) -> str:
    manifest = validated["manifest"]
    files = manifest["files"]
    return f"""---
language:
- ko
license: other
task_categories:
- text-retrieval
pretty_name: Korean Embedding Fixed Multidomain Selection Holdout v1
size_categories:
- 1K<n<10K
---

# Korean Embedding Fixed Multidomain Selection Holdout v1

한국어 embedding 후보를 공개 benchmark 점수 없이 고르기 위한 **비공개 selection-only**
finance/knowledge retrieval 보드다. 학습에 사용하지 않으며 공개 leaderboard가 아니다.

- finance: 900 queries, query exact-held-out; corpus training-text occurrence 1,373건 공개
- knowledge: 1,000 queries, query/corpus exact-text-held-out
- corpus: {files['corpus.jsonl']['rows']:,} rows
- qrels: {files['qrels.jsonl']['rows']:,} binary relevance rows
- selected query exact training overlap: 0
- knowledge query/corpus exact training overlap: 0
- selected text public benchmark blocklist overlap: 0
- public benchmark score used for selection: false

finance는 corpus 노출이 있으므로 target-dev이고, knowledge도 source-document Grade I 또는
unseen-source Grade Z가 아니다. KoTSQA relevance는 정규화한 정답 문자열이 passage에
실제로 포함된 경우에만 구성했다. 이 결과를 clean zero-shot 성능으로 부르지 않는다.

## Pinned sources

- [BCAI Finance Korean Embedding Triplet](https://huggingface.co/datasets/BCCard/BCAI-Finance-Kor-Embedding-Triplet/tree/f63d59969dba9916bd34c86c82112331890b11da), validation
- [KoTSQA v2](https://huggingface.co/datasets/etri-lirs/KoTSQA-v.2.0/tree/ff9349df469a765b4561959e36ef1b3f377765cd), sealed test used only for internal selection

각 source의 license와 이용 조건은 upstream을 따른다. 이 private 저장소의 `other` 표기가
upstream 권리를 재허가하지 않는다. exact source/training-role/blocklist SHA, exclusions,
한계와 모든 emitted file SHA는 `manifest.json`에 있다.

코드: https://github.com/LLM-OS-Models/Embedding
"""


def publication_manifest(
    *, validated: dict[str, Any], sources: dict[str, Path]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "private-fixed-multidomain-selection-dataset",
        "protocol_id": PROTOCOL_ID,
        "repo_id": DEFAULT_REPO,
        "visibility": "private",
        "selection_only": True,
        "public_benchmark": False,
        "source_manifest_sha256": validated["source_manifest_sha256"],
        "files_excluding_publication_manifest": expected_publication(sources),
    }


def write_atomic_report(path: Path, report: dict[str, Any]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    validated = validate_publication_source(args.dataset_dir, args.repo_id)
    report: dict[str, Any] = {
        "repo_id": args.repo_id,
        "visibility": "private",
        "protocol_id": PROTOCOL_ID,
        "source_manifest_sha256": validated["source_manifest_sha256"],
        "upload_requested": args.upload,
        "validated": True,
    }
    if not args.upload:
        if args.report_output:
            write_atomic_report(args.report_output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    token = load_hf_token(args.hf_token_file)
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id, repo_type="dataset", private=True, exist_ok=True
    )
    current = api.dataset_info(repo_id=args.repo_id, files_metadata=True)
    require_dataset_visibility(current, public=False)
    with tempfile.TemporaryDirectory(
        prefix=".multidomain-selection-publish-", dir=validated["root"].parent
    ) as temporary:
        staging = Path(temporary)
        sources: dict[str, Path] = {}
        card = staging / "README.md"
        card.write_text(dataset_card(validated), encoding="utf-8")
        sources["README.md"] = card
        for name, source in validated["payload"].items():
            destination = staging / name
            shutil.copy2(source, destination)
            sources[name] = destination
        publication_path = staging / "publication_manifest.json"
        publication_path.write_text(
            json.dumps(
                publication_manifest(validated=validated, sources=sources),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        sources["publication_manifest.json"] = publication_path
        expected = expected_publication(sources)
        current_files = {item.rfilename for item in getattr(current, "siblings", [])}
        if current_files - set(expected) - {".gitattributes"}:
            raise RuntimeError("Remote selector repo has unexpected pre-existing files")
        operations = [
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
            for name, path in sources.items()
        ]
        commit = api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message="Publish fixed private multidomain selection holdout v1",
        )
        commit_sha = getattr(commit, "oid", None)
        if not isinstance(commit_sha, str) or not COMMIT_RE.fullmatch(commit_sha):
            raise RuntimeError("Selection dataset upload returned no immutable commit")
        verify_remote_dataset(
            api=api,
            repo_id=args.repo_id,
            revision=commit_sha,
            expected=expected,
            public=False,
        )
        if expected_publication(sources) != expected:
            raise RuntimeError("Selection staging changed during upload")
        if sha256(validated["root"] / "manifest.json") != validated[
            "source_manifest_sha256"
        ]:
            raise RuntimeError("Selection source changed during upload")
    report.update(
        {
            "commit_sha": commit_sha,
            "commit_url": commit.commit_url,
            "remote_file_set_exact": True,
            "remote_payload_hashes_exact": True,
        }
    )
    if args.report_output:
        write_atomic_report(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
