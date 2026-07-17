#!/usr/bin/env python3
"""Publish one clean-selected merged model with exact visibility verification.

This publisher exists for intermediate KD/general winners that must be backed
up before public-benchmark final-once evaluation.  It accepts only the exact
winner of the clean-first Grade-I selector, binds model shards and evaluation
evidence by SHA-256, rejects trainer/credential artifacts, and verifies the
requested Hub visibility both before and after upload.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
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
KNOWN_BASE_LICENSES = {
    "Qwen/Qwen3-Embedding-8B": "apache-2.0",
    "sionic-ai/comsat-embed-ko-8b-preview": "cc-by-nc-4.0",
    "nvidia/Nemotron-3-Embed-8B-BF16": "OpenMDW-1.1",
}
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
MODEL_METADATA_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "config_sentence_transformers.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "modules.json",
    "preprocessor_config.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}
MODEL_CONTRACT_FILES = {
    "1_Pooling/config.json",
    "2_Normalize/config.json",
}
BYTE_EXACT_MODEL_METADATA = {
    "merges.txt",
    "model.safetensors.index.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
}
SENSITIVE_JSON_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "credential",
    "credentials",
    "github_token",
    "hf_token",
    "huggingface_hub_token",
    "password",
    "secret",
}
SECRET_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])gh[oprsu]_[A-Za-z0-9]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])glpat-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
)
LOCAL_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:/home/|/root/|/tmp/|/workspace/)[^\s\"'`<>]*"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--hf-token-file", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument(
        "--public",
        action="store_true",
        help="Publish a public candidate; private remains available for holdout-sensitive runs.",
    )
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


def sanitize_string(value: str) -> str:
    """Remove host-specific paths and recognized credentials from evidence text."""
    workspace = str(ROOT.resolve())
    sanitized = value.replace(workspace, ".")
    sanitized = LOCAL_PATH_PATTERN.sub("[local-path]", sanitized)
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, child in value.items():
            if str(key).lower() in SENSITIVE_JSON_KEYS:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = sanitize_json_value(child)
        return sanitized
    if isinstance(value, list):
        return [sanitize_json_value(child) for child in value]
    if isinstance(value, str):
        return sanitize_string(value)
    return value


def copy_sanitized_text(source: Path, destination: Path) -> None:
    suffix = source.suffix.lower()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        destination.write_text(
            json.dumps(
                sanitize_json_value(value), ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        return
    if suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as reader, destination.open(
            "w", encoding="utf-8"
        ) as writer:
            for line_number, line in enumerate(reader, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSONL evidence at line {line_number}: {source.name}"
                    ) from error
                writer.write(
                    json.dumps(
                        sanitize_json_value(value),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        return
    destination.write_text(
        sanitize_string(source.read_text(encoding="utf-8")), encoding="utf-8"
    )


def link_or_copy_immutable(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stage_model_payload(
    model_dir: Path,
    publication_dir: Path,
    evidence_name: str,
    *,
    ignore_existing_publication: bool = False,
) -> None:
    """Stage only load-bearing model files; never upload the mutable source tree."""
    allowed_relative = set(MODEL_CONTRACT_FILES)
    for source in sorted(model_dir.rglob("*")):
        if source.is_dir():
            continue
        relative = source.relative_to(model_dir).as_posix()
        if ignore_existing_publication and (
            relative in {"README.md", "publication_manifest.json"}
            or relative.startswith("evaluation/")
        ):
            continue
        allowed = (
            relative == evidence_name
            or relative in allowed_relative
            or ("/" not in relative and relative in MODEL_METADATA_FILES)
            or (
                "/" not in relative
                and source.name.startswith("model")
                and source.suffix == ".safetensors"
            )
        )
        if not allowed:
            raise ValueError(f"Private candidate contains an unapproved payload: {relative}")
        destination = publication_dir / relative
        if "/" not in relative and source.name in BYTE_EXACT_MODEL_METADATA:
            link_or_copy_immutable(source, destination)
        elif relative == evidence_name or source.suffix.lower() in {".json", ".jsonl"}:
            copy_sanitized_text(source, destination)
        elif source.suffix.lower() in {".md", ".txt", ".jinja"}:
            copy_sanitized_text(source, destination)
        else:
            link_or_copy_immutable(source, destination)


def validate_staged_text(publication_dir: Path) -> None:
    """Fail closed if normalized publication metadata still names this host/secrets."""
    excluded_large_text = {"tokenizer.json", "vocab.json", "merges.txt"}
    for path in publication_dir.rglob("*"):
        if not path.is_file() or path.name in excluded_large_text:
            continue
        if path.suffix.lower() not in {".json", ".jsonl", ".md", ".txt", ".jinja"}:
            continue
        text = path.read_text(encoding="utf-8")
        if str(ROOT.resolve()) in text or LOCAL_PATH_PATTERN.search(text):
            raise ValueError(f"Staged publication still contains a local path: {path.name}")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            raise ValueError(f"Staged publication still contains a credential: {path.name}")


def validate_candidate(args: argparse.Namespace) -> dict[str, Any]:
    validate_repo_id(args.repo_id)
    model_dir, model_rel = workspace_model_identity(args.model_dir)
    selection_path = args.selection.expanduser().resolve()
    training_manifest_path = args.training_manifest.expanduser().resolve()
    if not selection_path.is_file() or not training_manifest_path.is_file():
        raise FileNotFoundError("Selection or training manifest is missing")
    selection = read_json(selection_path)
    training_manifest = read_json(training_manifest_path)
    if getattr(args, "public", False):
        if training_manifest.get("release_eligible") is not True:
            raise ValueError(
                "Public candidate requires training_manifest.release_eligible=true"
            )
        if training_manifest.get("release_blockers"):
            raise ValueError("Public candidate training manifest has release blockers")
        if training_manifest.get("visibility") in {
            "private",
            "private-noncommercial-performance-track",
        }:
            raise ValueError("Private training data lineage cannot be published publicly")
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
    if getattr(args, "public", False):
        upstream = evidence.get("upstream_base_models")
        if not isinstance(upstream, list) or not upstream:
            raise ValueError("Public candidate has no upstream base-model lineage")
        for item in upstream:
            if not isinstance(item, dict) or any(
                not isinstance(item.get(key), str) or not item[key].strip()
                for key in ("model", "revision")
            ):
                raise ValueError(
                    "Public candidate upstream base requires model and revision"
                )
            if item["model"] not in KNOWN_BASE_LICENSES:
                raise ValueError("Public candidate upstream base license is not audited")
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
        "model_evidence": evidence,
        "weights_sha256": actual_sha,
        "revision": expected_revision,
    }


def build_card(args: argparse.Namespace, validated: dict[str, Any]) -> str:
    clean = validated["clean"]
    robustness = validated["robustness"]
    public = bool(getattr(args, "public", False))
    visibility = "public" if public else "private"
    visibility_ko = "공개" if public else "비공개"
    training = validated.get("training_manifest")
    if not isinstance(training, dict):
        training = read_json(validated["training_manifest_path"])
    model_evidence = validated.get("model_evidence")
    if not isinstance(model_evidence, dict):
        model_evidence = read_json(validated["evidence_path"])
    upstream = [
        {**row, "license": KNOWN_BASE_LICENSES.get(row.get("model"), "unknown")}
        for row in model_evidence.get("upstream_base_models", [])
    ]
    licenses = training.get("source_licenses", training.get("licenses", training.get("license", "see training manifest")))
    exposure = training.get("benchmark_exposure", training.get("benchmark_adaptation", "see training manifest"))
    return f"""---
library_name: sentence-transformers
pipeline_tag: sentence-similarity
candidate_visibility: {visibility}
license: other
---

# {args.repo_id.split('/', 1)[1]}

{visibility_ko} 연구용 중간 후보입니다. public leaderboard 점수로 선택하지 않았고, Grade-I
source-document-held-out retrieval과 대화형 noise robustness만으로 선택했습니다.

- selection policy: `{POLICY_ID}`
- immutable revision: `{validated['revision']}`
- model weights SHA-256: `{validated['weights_sha256']}`
- clean NDCG@10: `{clean['clean_ndcg_at_10']:.8f}`
- robustness floor NDCG@10: `{robustness['robustness_floor_ndcg_at_10']:.8f}`
- maximum noise intrusion@10: `{robustness['max_noise_intrusion_at_10']:.8f}`
- upstream base lineage: `{json.dumps(upstream, ensure_ascii=False, sort_keys=True)}`
- training data licenses: `{json.dumps(licenses, ensure_ascii=False, sort_keys=True)}`
- training use policy: `{training.get('use_policy', 'see training manifest')}`
- benchmark exposure: `{json.dumps(exposure, ensure_ascii=False, sort_keys=True)}`
- public redistribution review: `{training.get('release_eligible') is True}`

이 artifact는 intermediate backup이며 Sionic 9/공식 Korean v1 최종 성능 주장이 아닙니다.
학습 manifest, clean selection, summary와 rank evidence는 `evaluation/clean_selection/`에
포함됩니다. optimizer state, raw training data, credential은 포함하지 않습니다.
"""


def prepare_publication(
    args: argparse.Namespace,
    validated: dict[str, Any],
    publication_dir: Path,
) -> Path:
    model_dir: Path = validated["model_dir"]
    if publication_dir.exists() and any(publication_dir.iterdir()):
        raise FileExistsError("Private publication staging directory is not empty")
    publication_dir.mkdir(parents=True, exist_ok=True)
    stage_model_payload(model_dir, publication_dir, validated["evidence_name"])
    evidence_dir = publication_dir / "evaluation/clean_selection"
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
        copy_sanitized_text(source, evidence_dir / name)
    card_path = publication_dir / "README.md"
    card_path.write_text(build_card(args, validated), encoding="utf-8")
    model_shards = {
        shard.name: {"sha256": sha256(shard), "size_bytes": shard.stat().st_size}
        for shard in sorted(publication_dir.glob("model*.safetensors"))
    }
    if not model_shards:
        raise FileNotFoundError("Merged model has no safetensors shards")
    bound_files = publication_files(publication_dir)
    manifest = {
        "schema_version": 1,
        "artifact_type": (
            "public-clean-selected-embedding-candidate"
            if getattr(args, "public", False)
            else ARTIFACT_TYPE
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "visibility": "public" if getattr(args, "public", False) else "private",
        "repo_id": args.repo_id,
        "model": {
            "path": validated["model_rel"],
            "revision": validated["revision"],
            "weights_sha256": validated["weights_sha256"],
            "upstream_base_models": [
                {**row, "license": KNOWN_BASE_LICENSES.get(row.get("model"), "unknown")}
                for row in (
                    validated.get("model_evidence")
                    or read_json(validated["evidence_path"])
                ).get("upstream_base_models", [])
            ],
            "evidence": {
                "file": validated["evidence_name"],
                "sha256": sha256(publication_dir / validated["evidence_name"]),
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
        "files_excluding_manifest": bound_files,
        "publication_safety": {
            "isolated_staging": True,
            "source_model_mutated": False,
            "allowlisted_model_payload": True,
            "local_paths_removed": True,
            "recognized_credentials_removed": True,
        },
        "excluded_artifacts": [
            "optimizer state",
            "scheduler state",
            "raw training data",
            "credentials",
        ],
    }
    manifest_path = publication_dir / "private_candidate_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_no_sensitive_files(publication_dir)
    validate_staged_text(publication_dir)
    if model_weights_sha256(publication_dir) != validated["weights_sha256"]:
        raise RuntimeError("Staged model shards drifted from clean-selected weights")
    return manifest_path


def publication_files(publication_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(publication_dir.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        relative = path.relative_to(publication_dir).as_posix()
        result[relative] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    return result


def verify_remote_publication(
    *,
    api: Any,
    repo_id: str,
    revision: str,
    token: str,
    expected: dict[str, dict[str, Any]],
    expected_private: bool = True,
) -> None:
    from huggingface_hub import hf_hub_download

    info = api.model_info(repo_id=repo_id, revision=revision, files_metadata=True)
    if getattr(info, "private", None) is not expected_private:
        visibility = "private" if expected_private else "public"
        raise RuntimeError(f"Remote publication is not confirmed {visibility}")
    siblings = {item.rfilename: item for item in info.siblings}
    remote_files = set(siblings)
    missing = set(expected) - remote_files
    unexpected = remote_files - set(expected) - {".gitattributes"}
    if missing or unexpected:
        raise RuntimeError(
            f"Remote private candidate file-set mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    for name, evidence in expected.items():
        sibling = siblings[name]
        if name.endswith(".safetensors"):
            lfs = getattr(sibling, "lfs", None)
            if isinstance(lfs, dict):
                remote_sha = lfs.get("sha256")
                remote_size = lfs.get("size")
            else:
                remote_sha = getattr(lfs, "sha256", None) if lfs is not None else None
                remote_size = getattr(lfs, "size", None) if lfs is not None else None
            if remote_sha != evidence["sha256"] or remote_size != evidence["size_bytes"]:
                raise RuntimeError(f"Remote LFS object mismatch: {name}")
            continue
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=name,
                revision=revision,
                token=token,
            )
        )
        if sha256(downloaded) != evidence["sha256"]:
            raise RuntimeError(f"Remote metadata hash mismatch: {name}")


def main() -> None:
    args = parse_args()
    validated = validate_candidate(args)
    staging_parent = validated["model_dir"].parent
    with tempfile.TemporaryDirectory(
        prefix=f".{validated['model_dir'].name}.candidate-publish-", dir=staging_parent
    ) as temporary:
        publication_dir = Path(temporary)
        manifest_path = prepare_publication(args, validated, publication_dir)
        expected_files = publication_files(publication_dir)
        report: dict[str, Any] = {
            "repo_id": args.repo_id,
            "visibility": "public" if args.public else "private",
            "model": validated["model_rel"],
            "revision": validated["revision"],
            "weights_sha256": validated["weights_sha256"],
            "private_candidate_manifest_sha256": sha256(manifest_path),
            "publication_file_count": len(expected_files),
            "isolated_staging": True,
            "upload_requested": bool(args.upload),
            "validated": True,
        }
        if args.upload:
            token = load_hf_token(args.hf_token_file)
            from huggingface_hub import HfApi

            api = HfApi(token=token)
            # Explicit checks surround upload_large_folder; a pre-existing public
            # repo with this name is rejected instead of silently changing it.
            api.create_repo(
                repo_id=args.repo_id,
                repo_type="model",
                private=not args.public,
                exist_ok=True,
            )
            require_remote_visibility(api, args.repo_id, public=args.public)
            pre_info = api.model_info(repo_id=args.repo_id, files_metadata=True)
            pre_files = {item.rfilename for item in pre_info.siblings}
            unexpected_preexisting = pre_files - set(expected_files) - {".gitattributes"}
            if unexpected_preexisting:
                raise RuntimeError(
                    "Remote private candidate contains unexpected pre-existing files: "
                    f"{sorted(unexpected_preexisting)}"
                )
            api.upload_large_folder(
                repo_id=args.repo_id,
                repo_type="model",
                folder_path=publication_dir,
                private=not args.public,
                num_workers=1,
                print_report_every=60,
            )
            require_remote_visibility(api, args.repo_id, public=args.public)
            info = api.model_info(repo_id=args.repo_id, files_metadata=True)
            verify_remote_publication(
                api=api,
                repo_id=args.repo_id,
                revision=info.sha,
                token=token,
                expected=expected_files,
                expected_private=not args.public,
            )
            if model_weights_sha256(validated["model_dir"]) != validated["weights_sha256"]:
                raise RuntimeError("Source model changed during private upload")
            report["commit_sha"] = info.sha
            report["remote_manifest_exact"] = True
            report["remote_file_set_exact"] = True
            report["remote_files_verified"] = len(expected_files)
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
