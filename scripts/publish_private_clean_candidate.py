#!/usr/bin/env python3
"""Publish one clean-selected merged model as a strictly private candidate.

This publisher exists for intermediate KD/general winners that must be backed
up before public-benchmark final-once evaluation.  It accepts only the exact
winner of the clean-first Grade-I selector, binds model shards and evaluation
evidence by SHA-256, rejects trainer/credential artifacts, and verifies private
Hub visibility both before and after upload.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.publish_best_embedding_model import (
        load_hf_token,
        model_weights_sha256,
        require_remote_visibility,
        sha256,
    )
    from scripts.select_best_clean_model import (
        POLICY_ID,
        load_clean_candidate,
        load_robustness,
        read_json,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from publish_best_embedding_model import (
        load_hf_token,
        model_weights_sha256,
        require_remote_visibility,
        sha256,
    )
    from select_best_clean_model import (
        POLICY_ID,
        load_clean_candidate,
        load_robustness,
        read_json,
    )


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_TYPE = "private-clean-selected-embedding-candidate"
FORBIDDEN_FILE_NAMES = {
    ".env",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
}
FORBIDDEN_NAME_PARTS = (
    "credential",
    "secret",
    "apikey",
    "api_key",
    "access_token",
    "hf_token",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--hf-token-file", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def validate_repo_id(repo_id: str) -> None:
    if not repo_id.startswith("LLM-OS-Models2/") or repo_id.count("/") != 1:
        raise ValueError("Private candidate repo must be under LLM-OS-Models2")
    name = repo_id.split("/", 1)[1]
    if (
        not name
        or name.startswith((".", "-"))
        or name.endswith((".", "-"))
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in name)
    ):
        raise ValueError("Private candidate repo ID is invalid")


def workspace_model_identity(model_dir: Path) -> tuple[Path, str]:
    workspace = ROOT.resolve()
    expected_root = (workspace / "artifacts/models").resolve()
    resolved = model_dir.expanduser().resolve()
    try:
        relative = resolved.relative_to(workspace)
        resolved.relative_to(expected_root)
    except ValueError as error:
        raise ValueError("Model must be a local artifacts/models candidate") from error
    if resolved.is_symlink() or not resolved.is_dir():
        raise ValueError("Model directory is missing or unsafe")
    return resolved, relative.as_posix()


def validate_no_sensitive_files(model_dir: Path) -> None:
    for path in model_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError("Private candidate contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Private candidate contains a non-regular file")
        lower = path.name.lower()
        if lower in FORBIDDEN_FILE_NAMES or any(part in lower for part in FORBIDDEN_NAME_PARTS):
            raise ValueError(f"Private candidate contains forbidden file: {path.name}")


def validate_candidate(args: argparse.Namespace) -> dict[str, Any]:
    validate_repo_id(args.repo_id)
    model_dir, model_rel = workspace_model_identity(args.model_dir)
    selection_path = args.selection.expanduser().resolve()
    training_manifest_path = args.training_manifest.expanduser().resolve()
    if not selection_path.is_file() or not training_manifest_path.is_file():
        raise FileNotFoundError("Selection or training manifest is missing")
    selection = read_json(selection_path)
    training_manifest = read_json(training_manifest_path)
    if selection.get("schema_version") != 1 or selection.get("policy_id") != POLICY_ID:
        raise ValueError("Unexpected clean-selection policy")
    if selection.get("public_benchmark_used_for_selection") is not False:
        raise ValueError("Public benchmark was used for intermediate selection")
    best = selection.get("best")
    if not isinstance(best, dict) or best.get("model") != model_rel:
        raise ValueError("Model is not the exact clean-selected winner")
    clean_path = Path(str(best.get("clean_summary", ""))).expanduser().resolve()
    robustness_path = Path(str(best.get("robustness_summary", ""))).expanduser().resolve()
    if not clean_path.is_file() or not robustness_path.is_file():
        raise FileNotFoundError("Clean-selected evaluation evidence is missing")
    clean = load_clean_candidate(clean_path, ROOT)
    robustness = load_robustness(robustness_path)
    if clean["model"] != model_rel or robustness["model"] != model_rel:
        raise ValueError("Evaluation evidence belongs to a different model")
    if clean["revision"] != robustness["revision"]:
        raise ValueError("Clean and robustness revisions differ")
    if clean["dataset_manifest_sha256"] != robustness["dataset_manifest_sha256"]:
        raise ValueError("Clean and robustness datasets differ")
    if clean["weights_sha256"] != best.get("weights_sha256"):
        raise ValueError("Selection weights do not match clean evidence")
    for key in (
        "clean_ndcg_at_10",
        "robustness_floor_ndcg_at_10",
        "max_noise_intrusion_at_10",
    ):
        source = clean if key == "clean_ndcg_at_10" else robustness
        if float(source[key]) != float(best.get(key)):
            raise ValueError(f"Selection metric drift: {key}")

    evidence_paths = [
        model_dir / name
        for name in ("merge_report.json", "full_tuning_report.json", "soup_report.json")
        if (model_dir / name).is_file()
    ]
    if len(evidence_paths) != 1:
        raise FileNotFoundError(
            "Private intermediate candidate must have exactly one merge/full/soup report"
        )
    evidence_path = evidence_paths[0]
    evidence = read_json(evidence_path)
    if evidence.get("status") != "pass":
        raise ValueError("Model safe-merge evidence did not pass")
    contract = evidence.get("sentence_transformers_contract", {})
    if contract.get("pooling") != "last_token" or contract.get("normalize") is not True:
        raise ValueError("SentenceTransformers contract drifted")
    expected_sha = evidence.get("model", {}).get("weights_sha256")
    actual_sha = model_weights_sha256(model_dir)
    if expected_sha != actual_sha or clean["weights_sha256"] != actual_sha:
        raise ValueError("Model shards do not match selection/evaluation evidence")
    expected_revision = f"model-{actual_sha[:12]}"
    if clean["revision"] != expected_revision or best.get("revision") != expected_revision:
        raise ValueError("Immutable local revision drifted")
    validate_no_sensitive_files(model_dir)
    return {
        "model_dir": model_dir,
        "model_rel": model_rel,
        "selection_path": selection_path,
        "training_manifest_path": training_manifest_path,
        "clean_path": clean_path,
        "robustness_path": robustness_path,
        "selection": selection,
        "training_manifest": training_manifest,
        "clean": clean,
        "robustness": robustness,
        "evidence_path": evidence_path,
        "evidence_name": evidence_path.name,
        "weights_sha256": actual_sha,
        "revision": expected_revision,
    }


def build_card(args: argparse.Namespace, validated: dict[str, Any]) -> str:
    clean = validated["clean"]
    robustness = validated["robustness"]
    return f"""---
library_name: sentence-transformers
pipeline_tag: sentence-similarity
private_candidate: true
---

# {args.repo_id.split('/', 1)[1]}

비공개 연구용 중간 후보입니다. public leaderboard 점수로 선택하지 않았고, Grade-I
source-document-held-out retrieval과 대화형 noise robustness만으로 선택했습니다.

- selection policy: `{POLICY_ID}`
- immutable revision: `{validated['revision']}`
- model weights SHA-256: `{validated['weights_sha256']}`
- clean NDCG@10: `{clean['clean_ndcg_at_10']:.8f}`
- robustness floor NDCG@10: `{robustness['robustness_floor_ndcg_at_10']:.8f}`
- maximum noise intrusion@10: `{robustness['max_noise_intrusion_at_10']:.8f}`

이 artifact는 intermediate backup이며 Sionic 9/공식 Korean v1 최종 성능 주장이 아닙니다.
학습 manifest, clean selection, summary와 rank evidence는 `evaluation/clean_selection/`에
포함됩니다. optimizer state, raw training data, credential은 포함하지 않습니다.
"""


def prepare_publication(args: argparse.Namespace, validated: dict[str, Any]) -> Path:
    model_dir: Path = validated["model_dir"]
    evidence_dir = model_dir / "evaluation/clean_selection"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "selection.json": validated["selection_path"],
        "training_manifest.json": validated["training_manifest_path"],
        "clean_summary.json": validated["clean_path"],
        "clean_ranks.jsonl": validated["clean_path"].parent / "ranks.jsonl",
        "robustness_summary.json": validated["robustness_path"],
        "robustness_ranks.jsonl": validated["robustness_path"].parent / "ranks.jsonl",
    }
    for name, source in files.items():
        if not source.is_file():
            raise FileNotFoundError(f"Missing clean-selection evidence: {name}")
        shutil.copy2(source, evidence_dir / name)
    card_path = model_dir / "README.md"
    card_path.write_text(build_card(args, validated), encoding="utf-8")
    model_shards = {
        shard.name: {"sha256": sha256(shard), "size_bytes": shard.stat().st_size}
        for shard in sorted(model_dir.glob("model*.safetensors"))
    }
    if not model_shards:
        raise FileNotFoundError("Merged model has no safetensors shards")
    manifest = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "visibility": "private",
        "repo_id": args.repo_id,
        "model": {
            "path": validated["model_rel"],
            "revision": validated["revision"],
            "weights_sha256": validated["weights_sha256"],
            "evidence": {
                "file": validated["evidence_name"],
                "sha256": sha256(validated["evidence_path"]),
            },
            "shards": model_shards,
        },
        "selection": {
            "policy_id": POLICY_ID,
            "public_benchmark_used": False,
            "sha256": sha256(evidence_dir / "selection.json"),
        },
        "training_manifest_sha256": sha256(evidence_dir / "training_manifest.json"),
        "evidence": {
            name: {"sha256": sha256(evidence_dir / name)} for name in files
        },
        "card_sha256": sha256(card_path),
        "excluded_artifacts": [
            "optimizer state",
            "scheduler state",
            "raw training data",
            "credentials",
        ],
    }
    manifest_path = model_dir / "private_candidate_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_no_sensitive_files(model_dir)
    return manifest_path


def main() -> None:
    args = parse_args()
    validated = validate_candidate(args)
    manifest_path = prepare_publication(args, validated)
    report: dict[str, Any] = {
        "repo_id": args.repo_id,
        "visibility": "private",
        "model": validated["model_rel"],
        "revision": validated["revision"],
        "weights_sha256": validated["weights_sha256"],
        "private_candidate_manifest_sha256": sha256(manifest_path),
        "upload_requested": bool(args.upload),
        "validated": True,
    }
    if args.upload:
        token = load_hf_token(args.hf_token_file)
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=token)
        # Explicit checks surround upload_large_folder; a pre-existing public
        # repo with this name is rejected instead of silently changing it.
        api.create_repo(
            repo_id=args.repo_id, repo_type="model", private=True, exist_ok=True
        )
        require_remote_visibility(api, args.repo_id, public=False)
        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=validated["model_dir"],
            private=True,
            num_workers=1,
            print_report_every=60,
        )
        require_remote_visibility(api, args.repo_id, public=False)
        info = api.model_info(repo_id=args.repo_id, files_metadata=True)
        remote_files = {item.rfilename for item in info.siblings}
        expected_shards = set(
            read_json(manifest_path).get("model", {}).get("shards", {})
        )
        if not expected_shards.issubset(remote_files):
            raise RuntimeError("Remote private candidate is missing model shards")
        remote_manifest = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=manifest_path.name,
                revision=info.sha,
                token=token,
            )
        )
        if sha256(remote_manifest) != sha256(manifest_path):
            raise RuntimeError("Remote private candidate manifest hash mismatch")
        report["commit_sha"] = info.sha
        report["remote_manifest_exact"] = True
        report["url"] = f"https://huggingface.co/{args.repo_id}"
    if args.report_output:
        report_output = args.report_output.expanduser().resolve()
        report_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_output.with_name(f".{report_output.name}.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(report_output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
