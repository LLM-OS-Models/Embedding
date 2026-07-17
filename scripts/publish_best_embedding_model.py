#!/usr/bin/env python3
"""Validate, card, and resumably publish the selected merged embedding model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from statistics import fmean
from pathlib import Path
from typing import Any, Mapping

try:
    from scripts.model_lineage import lineage_from_evidence
    from scripts.select_best_clean_model import load_multidomain_candidate
except ImportError:  # pragma: no cover - direct script execution fallback
    from model_lineage import lineage_from_evidence
    from select_best_clean_model import load_multidomain_candidate


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RELEASE_APPROVAL_TYPE = "embedding-model-public-release-approval"
RIGHTS_SAFE_TRAINING_TRACK = "rights-safe-release"
COMPREHENSIVE_TEXT_PROTOCOL_ID = "comprehensive-korean-text-v1-mteb-2.18.0"
COMPREHENSIVE_TEXT_TASK_SUBSETS = {
    "XPQARetrieval": 3,
    "FloresBitextMining": 406,
    "KorSarcasmClassification.v2": 1,
    "KorHateClassification.v2": 1,
    "KorFin": 1,
    "KorHateSpeechMLClassification": 1,
    "KorNLI": 1,
}
COMPREHENSIVE_TEXT_SUBSETS = 414
SIONIC_ORDER = [
    "MIRACL",
    "MrTidy",
    "MLDR",
    "AutoRAG",
    "Ko-StrategyQA",
    "PublicHealthQA",
    "Belebele",
    "SQuADKorV1",
    "LawIRKo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--sionic-summary", type=Path, required=True)
    parser.add_argument("--official-summary", type=Path, required=True)
    parser.add_argument(
        "--comprehensive-summary",
        type=Path,
        help="Optional complete comprehensive Korean text diagnostic summary.",
    )
    parser.add_argument("--clean-summary", type=Path)
    parser.add_argument("--robustness-summary", type=Path)
    parser.add_argument(
        "--multidomain-summary",
        type=Path,
        help="Optional fixed finance/knowledge internal selection summary.",
    )
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument(
        "--release-approval",
        type=Path,
        help="Machine-readable approval required for every public release.",
    )
    parser.add_argument(
        "--repo-id", default="LLM-OS-Models2/qwen3-embedding-8b-ko-performance-v1"
    )
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--hf-token-file",
        type=Path,
        help="Secure file containing one HF_TOKEN= entry; only used with --upload.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        help="Atomic machine-readable publication completion report.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_json_tree(source_root: Path, destination_root: Path) -> list[dict[str, Any]]:
    try:
        from scripts.publish_private_clean_candidate import copy_sanitized_text
    except ImportError:  # pragma: no cover - direct script execution fallback
        from publish_private_clean_candidate import copy_sanitized_text

    copied: list[dict[str, Any]] = []
    if not source_root.is_dir():
        return copied
    for source in sorted(source_root.rglob("*.json")):
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_sanitized_text(source, destination)
        copied.append(
            {
                "path": str(destination.relative_to(destination_root.parent.parent)),
                "sha256": sha256(destination),
            }
        )
    return copied


def model_weights_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    shards = sorted(root.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No model safetensors under {root}")
    for shard in shards:
        digest.update(shard.name.encode() + b"\0")
        with shard.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def resolved_local_model(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Evaluation summary has no model path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise ValueError(f"Evaluation summary model is not a local artifact: {value}")
    return path.resolve()


def load_hf_token(
    token_file: Path | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Load one HF token without disclosing either the path or secret in errors.

    Environment-based authentication remains supported. A token file is opened
    without following its final symlink and accepted only when it is a regular
    file owned by this uid with no group/other permission bits. Other dotenv
    entries are ignored rather than evaluated.
    """

    environment = os.environ if environ is None else environ
    environment_has_token = "HF_TOKEN" in environment
    if token_file is not None and environment_has_token:
        raise RuntimeError("Hugging Face token source is ambiguous")
    if token_file is None:
        token = environment.get("HF_TOKEN")
        if not token:
            raise RuntimeError("Hugging Face token is unavailable")
        return token

    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Hugging Face token file failed security validation")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            os.fspath(token_file),
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
            or metadata.st_size > 1024 * 1024
        ):
            raise RuntimeError(
                "Hugging Face token file failed security validation"
            )
        content = bytearray()
        while True:
            block = os.read(descriptor, 64 * 1024)
            if not block:
                break
            content.extend(block)
            if len(content) > 1024 * 1024:
                raise RuntimeError(
                    "Hugging Face token file failed security validation"
                )
        text = bytes(content).decode("utf-8")
    except RuntimeError:
        raise
    except (OSError, UnicodeError):
        raise RuntimeError(
            "Hugging Face token file failed security validation"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)

    tokens = [
        line.removeprefix("HF_TOKEN=")
        for line in text.splitlines()
        if line.startswith("HF_TOKEN=")
    ]
    if (
        len(tokens) != 1
        or not tokens[0]
        or tokens[0] != tokens[0].strip()
        or any(character.isspace() for character in tokens[0])
    ):
        raise RuntimeError("Hugging Face token file has no unique valid HF_TOKEN entry")
    return tokens[0]


def validate_comprehensive_summary(
    summary: dict[str, Any],
    *,
    expected_revision: str,
    expected_weights_sha256: str,
) -> None:
    """Validate the complete text-only diagnostic contract for publication."""

    if summary.get("protocol_id") != COMPREHENSIVE_TEXT_PROTOCOL_ID:
        raise ValueError("Unexpected comprehensive Korean text protocol")
    if summary.get("complete") is not True:
        raise ValueError("Comprehensive Korean text summary is incomplete")
    tasks = summary.get("tasks")
    if (
        summary.get("completed_tasks") != len(COMPREHENSIVE_TEXT_TASK_SUBSETS)
        or summary.get("total_tasks") != len(COMPREHENSIVE_TEXT_TASK_SUBSETS)
        or not isinstance(tasks, list)
        or len(tasks) != len(COMPREHENSIVE_TEXT_TASK_SUBSETS)
    ):
        raise ValueError("Comprehensive Korean text summary does not cover 7 tasks")
    subset_counts: dict[str, int] = {}
    for row in tasks:
        if not isinstance(row, dict) or not isinstance(row.get("task_name"), str):
            raise ValueError("Comprehensive Korean text task evidence is malformed")
        task_name = row["task_name"]
        subset_count = row.get("subset_count")
        if (
            task_name in subset_counts
            or isinstance(subset_count, bool)
            or not isinstance(subset_count, int)
        ):
            raise ValueError("Comprehensive Korean text task evidence is malformed")
        subset_counts[task_name] = subset_count
    if subset_counts != COMPREHENSIVE_TEXT_TASK_SUBSETS:
        raise ValueError("Comprehensive Korean text task/subset contract drifted")
    if (
        summary.get("completed_subsets") != COMPREHENSIVE_TEXT_SUBSETS
        or summary.get("expected_subsets") != COMPREHENSIVE_TEXT_SUBSETS
        or sum(subset_counts.values()) != COMPREHENSIVE_TEXT_SUBSETS
    ):
        raise ValueError("Comprehensive Korean text summary does not cover 414 subsets")

    claim_status = summary.get("claim_status", {})
    expected_claims = {
        "diagnostic_only": True,
        "clean_claim_allowed": False,
        "visual_document_ready": False,
        "k_haters_ready": False,
    }
    if not isinstance(claim_status, dict) or any(
        claim_status.get(key) is not value for key, value in expected_claims.items()
    ):
        raise ValueError("Comprehensive Korean text diagnostic claim contract drifted")

    model = summary.get("model", {})
    if not isinstance(model, dict) or not isinstance(model.get("evidence"), dict):
        raise ValueError("Comprehensive Korean text model evidence is malformed")
    if model.get("revision") != expected_revision:
        raise ValueError("Comprehensive summary revision does not match model evidence")
    if model.get("evidence", {}).get("weights_sha256") != expected_weights_sha256:
        raise ValueError("Comprehensive summary weights do not match model evidence")
    aggregate = summary.get("aggregate", {})
    if not isinstance(aggregate, dict):
        raise ValueError("Comprehensive Korean text aggregate is malformed")
    for key in ("mean_task", "mean_task_type"):
        value = aggregate.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Comprehensive Korean text aggregate is malformed")
        if not math.isfinite(float(value)):
            raise ValueError("Comprehensive Korean text aggregate is non-finite")


def validate_public_release_approval(
    *,
    args: argparse.Namespace,
    model_evidence: dict[str, Any],
    training: dict[str, Any],
    comprehensive: dict[str, Any] | None,
    clean: dict[str, Any] | None,
    robustness: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Fail closed unless a public release is explicitly rights-approved.

    Private candidate publication intentionally keeps the previous evidence
    contract. Public visibility additionally requires a rights-safe training
    manifest, complete clean/robustness results, and an approval artifact bound
    to the exact model, target repo, manifest, and evaluation summaries.
    """

    if not args.public:
        return None

    missing_flags = [
        flag
        for flag, value in (
            ("--comprehensive-summary", args.comprehensive_summary),
            ("--clean-summary", args.clean_summary),
            ("--robustness-summary", args.robustness_summary),
            ("--release-approval", args.release_approval),
        )
        if value is None
    ]
    if missing_flags:
        raise ValueError(
            "Public release requires explicit evidence: " + ", ".join(missing_flags)
        )
    if comprehensive is None or clean is None or robustness is None:
        raise ValueError(
            "Public release requires validated comprehensive, clean, and robustness results"
        )

    training_track = training.get("training_track")
    if training_track != RIGHTS_SAFE_TRAINING_TRACK:
        raise ValueError(
            "Public release requires training_manifest.training_track="
            f"{RIGHTS_SAFE_TRAINING_TRACK!r}"
        )
    if training.get("release_eligible") is not True:
        raise ValueError(
            "Public release requires training_manifest.release_eligible=true"
        )
    if training.get("release_blockers"):
        raise ValueError("Training manifest has unresolved public release blockers")
    if training.get("use_policy") in {
        "research-noncommercial",
        "noncommercial",
        "research-only",
    }:
        raise ValueError("Non-commercial/research-only training manifests cannot be public")
    if training.get("visibility") in {
        "private",
        "private-noncommercial-performance-track",
    }:
        raise ValueError("Private training manifests cannot be promoted to public")

    approval_path = args.release_approval.resolve()
    approval = read_json(approval_path)
    if approval.get("schema_version") != 1:
        raise ValueError("Unsupported public release approval schema")
    if approval.get("artifact_type") != PUBLIC_RELEASE_APPROVAL_TYPE:
        raise ValueError("Unexpected public release approval artifact type")
    if approval.get("decision") != "approved":
        raise ValueError("Public release approval decision is not approved")
    for key in ("approval_id", "approved_by", "approved_at_utc"):
        if not isinstance(approval.get(key), str) or not approval[key].strip():
            raise ValueError(f"Public release approval has no non-empty {key}")

    target = approval.get("target", {})
    if target.get("repo_id") != args.repo_id or target.get("visibility") != "public":
        raise ValueError("Public release approval target does not match the public repo")
    model = approval.get("model", {})
    if model.get("weights_sha256") != model_evidence.get("model", {}).get(
        "weights_sha256"
    ):
        raise ValueError("Public release approval does not match model weights")
    approved_training = approval.get("training", {})
    if approved_training.get("track") != RIGHTS_SAFE_TRAINING_TRACK:
        raise ValueError("Public release approval does not approve the rights-safe track")
    if approved_training.get("manifest_sha256") != sha256(
        args.training_manifest.resolve()
    ):
        raise ValueError("Public release approval does not match training manifest")
    rights = approval.get("rights_review", {})
    if (
        rights.get("status") != "approved"
        or rights.get("release_eligible") is not True
        or rights.get("public_redistribution") is not True
        or rights.get("unresolved_blockers") != []
    ):
        raise ValueError("Public release rights review did not approve redistribution")

    evaluations = approval.get("evaluations", {})
    expected_evaluations = {
        "sionic9": (
            "sionic9-fixed-prompt-v1",
            args.sionic_summary.resolve(),
        ),
        "official_korean_v1": (
            "mteb-korean-v1-mteb-2.18.0",
            args.official_summary.resolve(),
        ),
        "comprehensive_text_v1": (
            COMPREHENSIVE_TEXT_PROTOCOL_ID,
            args.comprehensive_summary.resolve(),
        ),
        "clean": (
            "legal-source-document-heldout-i-v2-text-strict",
            args.clean_summary.resolve(),
        ),
        "robustness": (
            "legal-conversational-noise-i-v2-text-strict",
            args.robustness_summary.resolve(),
        ),
    }
    for label, (protocol_id, summary_path) in expected_evaluations.items():
        item = evaluations.get(label, {})
        if item.get("status") != "pass":
            raise ValueError(f"Public release approval has no passing {label} result")
        if item.get("protocol_id") != protocol_id:
            raise ValueError(f"Public release approval has unexpected {label} protocol")
        if item.get("summary_sha256") != sha256(summary_path):
            raise ValueError(
                f"Public release approval does not match {label} summary"
            )
    return approval


def require_remote_visibility(api: Any, repo_id: str, *, public: bool) -> None:
    """Refuse uploads when the existing HF repository visibility drifts."""

    try:
        info = api.model_info(repo_id=repo_id)
    except Exception as error:
        raise RuntimeError(
            f"Could not verify Hugging Face repository visibility ({type(error).__name__})"
        ) from None
    expected_private = not public
    if getattr(info, "private", None) is not expected_private:
        expected = "public" if public else "private"
        raise RuntimeError(
            f"Refusing upload because {repo_id} is not confirmed {expected}"
        )


def upload_model_folder(
    api: Any, *, repo_id: str, model_dir: Path, public: bool
) -> None:
    """Create and upload only while the requested visibility is observed."""

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=not public,
        exist_ok=True,
    )
    require_remote_visibility(api, repo_id, public=public)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=model_dir,
        private=not public,
        num_workers=4,
        print_report_every=60,
    )
    require_remote_visibility(api, repo_id, public=public)


def validate_embedding_contract_for_evidence(evidence: dict[str, Any]) -> None:
    contract = evidence.get("sentence_transformers_contract", {})
    bases = lineage_from_evidence(evidence, context="publication evidence")
    nemotron = any(row.get("model") == "nvidia/Nemotron-3-Embed-8B-BF16" for row in bases)
    expected_pooling = "masked_mean" if nemotron else "last_token"
    if contract.get("pooling") != expected_pooling or contract.get("normalize") is not True:
        raise ValueError("Merged SentenceTransformers contract drifted")
    if nemotron and contract.get("architecture") != "Ministral3Model":
        raise ValueError("Merged Nemotron architecture contract drifted")


def validate(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    model_dir = args.model_dir.resolve()
    evidence_paths = [
        path
        for path in (
            model_dir / "merge_report.json",
            model_dir / "full_tuning_report.json",
            model_dir / "soup_report.json",
        )
        if path.is_file()
    ]
    if len(evidence_paths) != 1:
        raise ValueError("Model must have exactly one merge/full/soup evidence report")
    model_evidence_path = evidence_paths[0]
    required = [
        model_dir / "config.json",
        model_dir / "modules.json",
        model_dir / "1_Pooling/config.json",
        model_dir / "2_Normalize",
        model_evidence_path,
        args.sionic_summary.resolve(),
        args.official_summary.resolve(),
        args.training_manifest.resolve(),
    ]
    if args.comprehensive_summary:
        required.append(args.comprehensive_summary.resolve())
    if args.clean_summary:
        required.append(args.clean_summary.resolve())
    if args.robustness_summary:
        required.append(args.robustness_summary.resolve())
    if args.multidomain_summary:
        required.append(args.multidomain_summary.resolve())
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing publication evidence: {missing}")
    if not list(model_dir.glob("model*.safetensors")):
        raise FileNotFoundError(f"No merged safetensors weights under {model_dir}")
    model_evidence = read_json(model_evidence_path)
    sionic = read_json(args.sionic_summary.resolve())
    official = read_json(args.official_summary.resolve())
    comprehensive = (
        read_json(args.comprehensive_summary.resolve())
        if args.comprehensive_summary
        else None
    )
    clean = read_json(args.clean_summary.resolve()) if args.clean_summary else None
    robustness = (
        read_json(args.robustness_summary.resolve())
        if args.robustness_summary
        else None
    )
    multidomain = (
        read_json(args.multidomain_summary.resolve())
        if args.multidomain_summary
        else None
    )
    training = read_json(args.training_manifest.resolve())
    if model_evidence.get("status") != "pass":
        raise ValueError("Model packaging/parity evidence did not pass")
    lineage_from_evidence(
        model_evidence,
        evidence_dir=model_dir,
        context=str(model_evidence_path),
    )
    validate_embedding_contract_for_evidence(model_evidence)
    if sionic.get("completed_tasks") != 9 or set(sionic.get("scores", {})) != set(
        SIONIC_ORDER
    ):
        raise ValueError("Sionic-9 summary is incomplete")
    if official.get("complete") is not True or official.get("completed_tasks") != 6:
        raise ValueError("Official Korean v1 summary is incomplete")
    if sionic.get("protocol_id") != "sionic9-fixed-prompt-v1":
        raise ValueError("Unexpected Sionic protocol")
    if official.get("protocol_id") != "mteb-korean-v1-mteb-2.18.0":
        raise ValueError("Unexpected official Korean protocol")
    official_environment = official.get("environment", {})
    if (
        official_environment.get("qwen3_instruction_loader") is not True
        or official_environment.get("instruction_contract") != "qwen3-task-instruction"
    ):
        raise ValueError(
            "Official Korean candidate result did not use Qwen3 task instructions"
        )
    if resolved_local_model(sionic.get("model")) != model_dir:
        raise ValueError("Sionic summary belongs to a different model artifact")
    if resolved_local_model(official.get("model")) != model_dir:
        raise ValueError("Official summary belongs to a different model artifact")
    expected_revision = f"model-{model_evidence['model']['weights_sha256'][:12]}"
    for label, summary in (("Sionic", sionic), ("official", official)):
        if summary.get("requested_revision") != expected_revision:
            raise ValueError(f"{label} summary revision does not match model evidence")
    if comprehensive is not None:
        validate_comprehensive_summary(
            comprehensive,
            expected_revision=expected_revision,
            expected_weights_sha256=model_evidence["model"]["weights_sha256"],
        )
    if clean is not None:
        if clean.get("protocol_id") != "legal-source-document-heldout-i-v2-text-strict":
            raise ValueError("Unexpected clean legal protocol")
        if resolved_local_model(clean.get("model")) != model_dir:
            raise ValueError(
                "Clean legal summary belongs to a different model artifact"
            )
        if clean.get("requested_revision") != expected_revision:
            raise ValueError(
                "Clean legal summary revision does not match model evidence"
            )
        dataset = clean.get("dataset", {})
        if dataset.get("independence_grade") != "I" or dataset.get("not_grade") != "Z":
            raise ValueError("Clean legal independence evidence is invalid")
    if robustness is not None:
        if robustness.get("protocol_id") != "legal-conversational-noise-i-v2-text-strict":
            raise ValueError("Unexpected conversational noise protocol")
        if resolved_local_model(robustness.get("model")) != model_dir:
            raise ValueError("Robustness summary belongs to a different model artifact")
        if robustness.get("requested_revision") != expected_revision:
            raise ValueError(
                "Robustness summary revision does not match model evidence"
            )
        robustness_dataset = robustness.get("dataset", {})
        if (
            robustness_dataset.get("independence_grade") != "I"
            or robustness_dataset.get("not_grade") != "Z"
        ):
            raise ValueError("Robustness independence evidence is invalid")
        expected_conditions = {
            f"prompt_{state}/noise_{ratio}"
            for state in ("on", "off")
            for ratio in ("0.00", "0.01", "0.05")
        }
        if set(robustness.get("conditions", {})) != expected_conditions:
            raise ValueError("Robustness summary has incomplete conditions")
        if clean is not None:
            clean_ndcg = float(clean["metrics"]["ndcg_at_10"])
            robustness_clean_ndcg = float(
                robustness["conditions"]["prompt_on/noise_0.00"]["ndcg_at_10"]
            )
            if abs(clean_ndcg - robustness_clean_ndcg) > 1e-12:
                raise ValueError("Clean and robustness prompt-on baselines disagree")
    if multidomain is not None:
        multidomain_evidence = load_multidomain_candidate(
            args.multidomain_summary.resolve(), ROOT
        )
        if resolved_local_model(multidomain_evidence["model"]) != model_dir:
            raise ValueError(
                "Multidomain summary belongs to a different model artifact"
            )
        if multidomain_evidence["revision"] != expected_revision:
            raise ValueError(
                "Multidomain summary revision does not match model evidence"
            )
    recomputed_sionic = fmean(float(value) for value in sionic["scores"].values())
    if abs(float(sionic["average"]) - recomputed_sionic) > 1e-12:
        raise ValueError("Sionic average is inconsistent with task scores")
    official_task_mean = fmean(
        float(row["score"]) for row in official["scores"].values()
    )
    if (
        abs(float(official["mean_task_leaderboard_points"]) - 100 * official_task_mean)
        > 1e-9
    ):
        raise ValueError("Official Mean(Task) is inconsistent with task scores")
    means_by_type = official.get("means_by_type", {})
    if not means_by_type:
        raise ValueError("Official summary has no task-type means")
    official_type_mean = fmean(float(value) for value in means_by_type.values())
    if (
        abs(
            float(official["mean_task_type_leaderboard_points"])
            - 100 * official_type_mean
        )
        > 1e-9
    ):
        raise ValueError("Official Mean(Type) is inconsistent with type means")
    actual_model_sha = model_weights_sha256(model_dir)
    if model_evidence.get("model", {}).get("weights_sha256") != actual_model_sha:
        raise ValueError("Published model shards do not match model evidence")
    approval = validate_public_release_approval(
        args=args,
        model_evidence=model_evidence,
        training=training,
        comprehensive=comprehensive,
        clean=clean,
        robustness=robustness,
    )
    return (
        model_evidence,
        sionic,
        official,
        comprehensive,
        training,
        clean,
        robustness,
        multidomain,
        approval,
    )


def is_full_update(evidence: dict[str, Any]) -> bool:
    return str(evidence.get("training_method", "")).startswith("partial-full") or (
        evidence.get("artifact_type") == "weighted-full-model-embedding-soup"
    )


def is_model_soup(evidence: dict[str, Any]) -> bool:
    return evidence.get("artifact_type") == "weighted-full-model-embedding-soup"


def weights_sha(evidence: dict[str, Any]) -> str:
    if is_full_update(evidence):
        return str(evidence["model"]["weights_sha256"])
    return str(evidence["adapter"]["weights_sha256"])


def score_table(scores: dict[str, float], order: list[str]) -> str:
    lines = ["| Task | Score |", "|---|---:|"]
    lines.extend(f"| {name} | {float(scores[name]):.5f} |" for name in order)
    return "\n".join(lines)


def training_rows(manifest: dict[str, Any]) -> str:
    for key in ("built_rows", "rows", "output_rows", "configured_target_rows"):
        if key in manifest:
            return str(manifest[key])
    files = manifest.get("files", {})
    values = [value.get("rows") for value in files.values() if isinstance(value, dict)]
    return str(
        max((value for value in values if isinstance(value, int)), default="unknown")
    )


def training_dataset_repos(manifest: dict[str, Any]) -> list[str]:
    adaptation = str(manifest.get("benchmark_adaptation", ""))
    if adaptation.startswith("target-adapted") and "combined" in adaptation:
        return [
            "LLM-OS-Models2/korean-embedding-sionic-combined-replay-v1",
            "LLM-OS-Models2/korean-embedding-sionic-retrieval-family-quantile-hn7-replay-v1",
            "LLM-OS-Models2/korean-embedding-sionic-squad-quantile-hn7-replay-v1",
            "LLM-OS-Models2/korean-embedding-sionic-health-quantile-hn7-replay-v1",
            "LLM-OS-Models2/korean-embedding-sionic-autorag-quantile-hn7-replay-v1",
            "LLM-OS-Models2/korean-legal-quantile-hn7-replay-v1",
            "LLM-OS-Models/korean-embedding-performance-v1-performance-1m",
        ]
    if adaptation.startswith("target-adapted") and "legal" in adaptation:
        return [
            "LLM-OS-Models2/korean-legal-quantile-hn7-replay-v1",
            "LLM-OS-Models/korean-legal-retrieval-source-native-250k",
            "LLM-OS-Models/korean-embedding-performance-v1-performance-1m",
        ]
    if adaptation.startswith("target-adapted") and "squad" in adaptation:
        return [
            "LLM-OS-Models2/korean-embedding-sionic-squad-quantile-hn7-replay-v1",
            "LLM-OS-Models/korean-embedding-performance-v1-sionic-squad-train-60k",
            "LLM-OS-Models/korean-embedding-performance-v1-performance-1m",
        ]
    if adaptation.startswith("target-adapted") and "health" in adaptation:
        return [
            "LLM-OS-Models2/korean-embedding-sionic-health-quantile-hn7-replay-v1",
            "LLM-OS-Models/korean-embedding-performance-v1-sionic-health-100k",
            "LLM-OS-Models/korean-embedding-performance-v1-performance-1m",
        ]
    if adaptation.startswith("target-adapted") and "autorag" in adaptation:
        return [
            "LLM-OS-Models2/korean-embedding-sionic-autorag-quantile-hn7-replay-v1",
            "LLM-OS-Models/korean-embedding-performance-v1-sionic-autorag-100k",
            "LLM-OS-Models/korean-embedding-performance-v1-performance-1m",
        ]
    if adaptation.startswith("target-adapted") and "retrieval-family" in adaptation:
        return [
            "LLM-OS-Models2/korean-embedding-sionic-retrieval-family-quantile-hn7-replay-v1",
            "LLM-OS-Models/korean-embedding-performance-v1-sionic-retrieval-train-family-4146",
            "LLM-OS-Models/korean-embedding-performance-v1-performance-1m",
        ]
    if adaptation.startswith("target-adapted"):
        return [
            "LLM-OS-Models2/korean-embedding-performance-1m-quantile-hn7-v1",
            "LLM-OS-Models/korean-embedding-performance-v1-performance-1m",
        ]
    if manifest.get("purpose") == "training-only-dense-hard-negative-mining":
        return ["LLM-OS-Models/korean-embedding-ko-triplet-hn-pilot-10k"]
    repo = {
        "pilot_50k": "LLM-OS-Models/korean-embedding-performance-v1-pilot-50k",
        "ablation_200k": "LLM-OS-Models/korean-embedding-performance-v1-ablation-200k",
        "performance_1m": "LLM-OS-Models/korean-embedding-performance-v1-performance-1m",
    }.get(manifest.get("phase"))
    if repo is None:
        train_path = str(manifest.get("inputs", {}).get("train", {}).get("path", ""))
        if not train_path:
            train_path = str(manifest.get("input", {}).get("path", ""))
        if "pilot-50k" in train_path:
            repo = "LLM-OS-Models/korean-embedding-performance-v1-pilot-50k"
        elif "ablation-200k" in train_path:
            repo = "LLM-OS-Models/korean-embedding-performance-v1-ablation-200k"
        elif "performance-1m" in train_path:
            repo = "LLM-OS-Models/korean-embedding-performance-v1-performance-1m"
        elif "ko_triplet_pilot_10k" in train_path:
            repo = "LLM-OS-Models/korean-embedding-ko-triplet-hn-pilot-10k"
    return [repo] if repo else []


def build_card(
    repo_id: str,
    evidence: dict[str, Any],
    sionic: dict[str, Any],
    official: dict[str, Any],
    comprehensive: dict[str, Any] | None,
    training: dict[str, Any],
    clean: dict[str, Any] | None,
    robustness: dict[str, Any] | None,
    multidomain: dict[str, Any] | None,
) -> str:
    delta = float(sionic["average"]) - 0.793
    full_update = is_full_update(evidence)
    adapter = evidence.get("adapter_config", {})
    merge_dtype = str(evidence.get("merge", {}).get("dtype", "bfloat16"))
    torch_dtype = "torch.float32" if merge_dtype == "float32" else "torch.bfloat16"
    official_order = list(official["scores"])
    dataset_repos = training_dataset_repos(training)
    dataset_yaml = (
        "datasets:\n" + "".join(f"- {repo}\n" for repo in dataset_repos)
        if dataset_repos
        else ""
    )
    dataset_link = (
        ", ".join(f"https://huggingface.co/datasets/{repo}" for repo in dataset_repos)
        if dataset_repos
        else "Training manifest is preserved with the model evaluation artifacts."
    )
    adaptation = str(training.get("benchmark_adaptation", ""))
    target_adapted = adaptation.startswith("target-adapted")
    upstream_bases = lineage_from_evidence(evidence, context="publication evidence")
    upstream_ids = [row["model"] for row in upstream_bases]
    upstream_rows = "; ".join(
        f"{row['model']}@{row['revision']}" for row in upstream_bases
    )
    upstream_label = " + ".join(upstream_ids)
    embedding_contract = evidence.get("sentence_transformers_contract", {})
    pooling_label = str(embedding_contract.get("pooling", "last_token"))
    if len(upstream_ids) == 1:
        base_model_yaml = f"base_model: {upstream_ids[0]}"
    else:
        base_model_yaml = "base_model:\n" + "\n".join(
            f"- {model_id}" for model_id in upstream_ids
        )
    base_model_relation = "merge" if is_model_soup(evidence) else "finetune"
    upstream_links = ", ".join(
        f"https://huggingface.co/{row['model']}/tree/{row['revision']}"
        for row in upstream_bases
    )
    if "sionic-ai/comsat-embed-ko-8b-preview" in upstream_ids:
        upstream_notice = (
            "**이 모델은 `sionic-ai/comsat-embed-ko-8b-preview` 계보를 포함하므로 "
            "해당 upstream의 CC-BY-NC-4.0 비상업 조건을 승계한다.**"
        )
    else:
        upstream_notice = "정확한 pinned upstream 계보는 아래 학습 절에 공개한다."
    if target_adapted and "legal" in adaptation:
        adaptation_notice = (
            "**이 모델은 법률/공공 target-adapted 모델이다. LawIRKo와 AutoRAG "
            "legal/public 점수를 clean zero-shot으로 해석하면 안 된다.**"
        )
    elif target_adapted:
        adaptation_notice = (
            "**이 모델은 공개 train/task-family와 current-student hard-negative를 사용한 "
            "performance target-adapted 모델이다. 관련 MTEB/Sionic 점수를 완전한 clean "
            "zero-shot으로 해석하면 안 된다.**"
        )
    else:
        adaptation_notice = "이 모델의 task-family 학습 노출은 아래와 같이 공개한다."
    if is_model_soup(evidence):
        method_intro = (
            "독립 LoRA factor를 직접 평균하지 않고 safe-merged full transformer weight를 "
            "FP32로 가중 평균한 한국어 embedding soup 후보다. 고정 coefficient와 source "
            f"hash는 soup_report.json에 기록했고 {pooling_label}/L2 계약을 검증했다."
        )
    else:
        method_intro = (
            f"{upstream_label} 계보 모델의 상위 transformer block을 부분 full-parameter update한 "
            "한국어 retrieval 성능 후보다. optimizer state를 제외한 SentenceTransformers "
            f"artifact를 만들고 {pooling_label}/L2 계약과 실제 embedding probe를 검증했다."
            if full_update
            else f"{upstream_label} 계보를 한국어 retrieval용 contrastive fine-tuning한 연구·비상업 "
            "성능 후보다. PEFT adapter를 base에 safe-merge하고 병합 전후 embedding parity와 "
            f"SentenceTransformers {pooling_label}/L2/prompt 계약을 검증했다."
        )
    clean_section = ""
    if clean is not None:
        clean_metrics = clean["metrics"]
        clean_section = f"""
### Clean 법률 source-document-held-out 10K

- NDCG@10: **{float(clean_metrics['ndcg_at_10']):.5f}**
- Recall@10: **{float(clean_metrics['recall_at_10']):.5f}**
- MRR@10: **{float(clean_metrics['mrr_at_10']):.5f}**
- Recall@100: **{float(clean_metrics['recall_at_100']):.5f}**
- independence: `I` (same-repository whole-source-document-held-out), **not Z**

각 query에 source-native positive qrel 하나만 있어 relevance judgment는 exhaustive하지 않다.
"""
    robustness_section = ""
    if robustness is not None:
        conditions = robustness["conditions"]
        on_5 = conditions["prompt_on/noise_0.05"]
        off_5 = conditions["prompt_off/noise_0.05"]
        robustness_section = f"""
### 대화형 구조 노이즈 강건성

| Query | Noise ratio | NDCG@10 | Clean 대비 유지율 | Noise intrusion@10 |
|---|---:|---:|---:|---:|
| prompt on | 5% | {float(on_5['ndcg_at_10']):.5f} | {float(on_5['ndcg_retention_vs_same_prompt_clean']):.5f} | {float(on_5['noise_intrusion_at_10']):.5f} |
| prompt off | 5% | {float(off_5['ndcg_at_10']):.5f} | {float(off_5['ndcg_retention_vs_same_prompt_clean']):.5f} | {float(off_5['noise_intrusion_at_10']):.5f} |

고정된 filler/system/assistant artifact를 clean corpus의 5%만큼 추가한 paired test다.
0/1/5% 전체 condition과 per-query rank는 `evaluation/`에 동봉한다.
"""
    multidomain_section = ""
    if multidomain is not None:
        finance = multidomain["domain_metrics"]["finance"]
        knowledge = multidomain["domain_metrics"]["knowledge"]
        macro = multidomain["metrics"]["macro_domain_ndcg_at_10"]
        multidomain_section = f"""
### 고정 비공개 다영역 선택 평가

| Domain | Queries | NDCG@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|
| finance | 900 | {float(finance['ndcg_at_10']):.5f} | {float(finance['recall_at_10']):.5f} | {float(finance['mrr_at_10']):.5f} |
| knowledge | 1,000 | {float(knowledge['ndcg_at_10']):.5f} | {float(knowledge['recall_at_10']):.5f} | {float(knowledge['mrr_at_10']):.5f} |

- domain-macro NDCG@10: **{float(macro):.5f}**
- protocol: `{multidomain['protocol_id']}`
- public benchmark score used for selection: **false**

finance는 query exact-held-out이지만 corpus는 학습 text 노출 가능성이 있는 target-dev이며,
knowledge는 query와 corpus 모두 exact-held-out이다. 이 보드는 모델 선택 전용이고 공개
leaderboard 점수나 완전한 zero-shot 성능으로 해석하지 않는다. 요약과 1,900개 per-query
rank를 `evaluation/`에 동봉한다.
"""
    comprehensive_section = ""
    if comprehensive is not None:
        aggregate = comprehensive["aggregate"]
        comprehensive_section = f"""
### Comprehensive Korean text v1 진단

- Mean(Task): **{100 * float(aggregate['mean_task']):.3f}**
- Mean(Type): **{100 * float(aggregate['mean_task_type']):.3f}**
- coverage: **{int(comprehensive['completed_tasks'])} tasks / {int(comprehensive['completed_subsets'])} subsets**
- protocol: `{comprehensive['protocol_id']}`

이 평가는 text-only **diagnostic-only** 결과다. clean 성능, visual-document
retrieval, K-HATERS 지원 또는 해당 영역의 우위를 주장하는 근거가 아니다. 요약과
원본 MTEB result JSON은 `evaluation/`에 동봉한다.
"""
    if is_model_soup(evidence):
        source_rows = "; ".join(
            f"{Path(str(row.get('model', 'unknown'))).name}={float(row.get('weight', 0.0)):.4f}"
            for row in evidence.get("sources", [])
            if isinstance(row, dict)
        )
        method_rows = f"""- method: basis-safe weighted arithmetic mean of safe-merged full model weights
- pinned upstream base lineage: `{upstream_rows}`
- fixed source weights: `{source_rows}`
- FP32 accumulation / emitted dtype: `{evidence.get('soup', {}).get('accumulation_dtype')}` / `{evidence.get('soup', {}).get('output_floating_dtype')}`
- model weight SHA-256: `{weights_sha(evidence)}`
- source count: `{len(evidence.get('sources', []))}`
- tensor count: `{evidence.get('soup', {}).get('tensor_count')}`"""
    elif full_update:
        method_rows = f"""- pinned upstream base lineage: `{upstream_rows}`
- method: partial full-parameter contrastive fine-tuning, InfoNCE/explicit negatives
- packaged model weight SHA-256: `{weights_sha(evidence)}`
- packaged probe maximum norm error: `{evidence['probe']['metrics']['maximum_norm_error']}`
- packaged probe positive margin: `{evidence['probe']['metrics']['positive_margin']}`"""
    else:
        training_arguments = evidence.get("adapter", {}).get("training", {}).get(
            "arguments", {}
        )
        method_rows = f"""- pinned upstream base lineage: `{upstream_rows}`
- method: LoRA continued contrastive fine-tuning, InfoNCE/explicit negatives
- LoRA rank/alpha/dropout: `{adapter.get('r')}` / `{adapter.get('lora_alpha')}` / `{adapter.get('lora_dropout')}`
- target modules: `{', '.join(adapter.get('target_modules') or [])}`
- adapter weight SHA-256: `{weights_sha(evidence)}`
- merge requested/effective dtype: `{evidence.get('merge', {}).get('requested_dtype', merge_dtype)}` / `{merge_dtype}`
- training attention/dtype: `{training_arguments.get('attn_impl', 'not recorded')}` / `{training_arguments.get('torch_dtype', 'not recorded')}`
- actual trainer rows after tokenization/filtering: `{evidence.get('adapter', {}).get('training', {}).get('actual_train_rows', 'not recorded')}`
- merge minimum probe cosine: `{evidence['probe']['metrics']['minimum_row_cosine']}`
- merge maximum pairwise score delta: `{evidence['probe']['metrics']['maximum_pairwise_score_difference']}`"""
    return f"""---
language:
- ko
- en
license: other
library_name: sentence-transformers
pipeline_tag: feature-extraction
{base_model_yaml}
base_model_relation: {base_model_relation}
{dataset_yaml.rstrip()}
tags:
- sentence-transformers
- text-embeddings-inference
- vllm
- korean
- retrieval
---

# {repo_id.split('/')[-1]}

{method_intro}

{adaptation_notice}

{upstream_notice}

## 결과

### Sionic Korean retrieval 9종

동일한 고정 query prompt, 각 task NDCG@10, 9개 macro average다.

{score_table(sionic['scores'], SIONIC_ORDER)}

- 9-task average: **{float(sionic['average']):.5f}**
- Comsat 카드의 0.7930 대비: **{delta:+.5f}**
- protocol: `{sionic['protocol_id']}`
- model revision evidence SHA: `{evidence['model']['weights_sha256']}`

### 공식 MTEB Korean v1 로컬 재현

{score_table({name: row['score'] for name, row in official['scores'].items()}, official_order)}

- Mean(Task): **{float(official['mean_task_leaderboard_points']):.3f}**
- Mean(Type): **{float(official['mean_task_type_leaderboard_points']):.3f}**
- protocol: `{official['protocol_id']}`
- instruction contract: `qwen3-task-instruction` (pinned MTEB task metadata/fallback,
  query에만 Qwen3 template 적용, passage 무지시문)

이 결과는 pinned MTEB protocol의 로컬 실행이며 MTEB leaderboard 제출 행 자체는
아니다. task별 MTEB raw result JSON은 이 model repository의 `evaluation/raw/`에,
실행 코드는 프로젝트 repository에 보존한다.

{comprehensive_section}
{clean_section}
{robustness_section}
{multidomain_section}

## 학습

{method_rows}
- training manifest phase: `{training.get('phase', training.get('purpose', 'documented in manifest'))}`
- manifest rows: `{training_rows(training)}`

학습 데이터에는 official train/task-family source가 포함될 수 있다. Sionic 9에서는
MIRACL, MrTidy, MLDR, Ko-StrategyQA 계열 노출을 명시하며, official Korean v1 결과를
완전한 zero-shot이라고 주장하지 않는다. 데이터 source의 license가 혼재하므로 이
모델 카드의 `other`는 upstream 권리를 재허가하지 않는다.

## SentenceTransformers 사용법

```python
import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "{repo_id}",
    model_kwargs={{
        "attn_implementation": "flash_attention_2",
        "torch_dtype": {torch_dtype},
    }},
    tokenizer_kwargs={{"padding_side": "left"}},
)
queries = model.encode(
    ["대한민국의 수도는 어디인가?"],
    prompt_name="query",
    normalize_embeddings=True,
)
documents = model.encode(
    ["대한민국의 수도는 서울특별시이다."],
    normalize_embeddings=True,
)
scores = queries @ documents.T
```

query에는 model의 `query` prompt를 적용하고 document에는 instruction을 붙이지 않는다.
출력은 4,096차원 L2-normalized vector이므로 dot product가 cosine similarity다.

## vLLM API 서빙

```bash
MODEL_ID={repo_id} \\
SERVED_MODEL_NAME=ko-embedding-8b \\
MAX_MODEL_LEN=8192 \\
DTYPE={merge_dtype} \\
scripts/serve_vllm_embedding.sh
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
result = client.embeddings.create(
    model="ko-embedding-8b",
    input=[
        "Instruct: Given a Korean web search query, retrieve relevant passages that answer the query\\nQuery: 질문",
        "검색할 문서",
    ],
)
```

공개 점수 재현에는 model card의 evaluation dtype
(`{sionic.get('environment', {}).get('torch_dtype', merge_dtype)}`)을 유지한다. 다른
dtype은 별도 parity/회귀 측정 없이 같은 점수라고 간주하지 않는다. vLLM pooling은
동시 요청 API에 적합하지만 offline 고정 대량 corpus에서는 항상
SentenceTransformers+FlashAttention 2보다 빠르지 않다. 실제 traffic으로 두 경로를
benchmark한다.

## 재현

- code: https://github.com/LLM-OS-Models/Embedding
- data: {dataset_link}
- pinned upstream base(s): {upstream_links}
- comparison: https://huggingface.co/sionic-ai/comsat-embed-ko-8b-preview

모델 선택·평가·데이터 노출과 exact command는 repository의 README와 docs에 기록돼
있다. 평가 test row를 학습 또는 hard-negative mining에 되먹이지 않는다.

## 제한

- 한국어 retrieval specialist이며 모든 언어·task에서 base보다 낫다고 보장하지 않는다.
- legal/public target-like 데이터가 포함된 후속 버전은 LawIRKo/AutoRAG에서 반드시
  target-adapted로 별도 표시한다.
- 8B/4096-d vector는 품질은 높지만 serving·storage 비용이 크다. MRL 차원 축소는 해당
  dimension의 회귀 평가 후 사용한다.
"""


def prepare_isolated_upload_staging(
    *, model_dir: Path, evidence_name: str, publication_dir: Path
) -> Path:
    try:
        from scripts.publish_private_clean_candidate import (
            copy_sanitized_text,
            publication_files,
            stage_model_payload,
            validate_no_sensitive_files,
            validate_staged_text,
        )
    except ImportError:  # pragma: no cover - direct script execution fallback
        from publish_private_clean_candidate import (
            copy_sanitized_text,
            publication_files,
            stage_model_payload,
            validate_no_sensitive_files,
            validate_staged_text,
        )

    model_dir = model_dir.resolve()
    expected_root = (ROOT / "artifacts/models").resolve()
    try:
        model_rel = model_dir.relative_to(ROOT.resolve()).as_posix()
        model_dir.relative_to(expected_root)
    except ValueError as error:
        raise ValueError("Uploaded model must be under workspace artifacts/models") from error
    if publication_dir.exists() and any(publication_dir.iterdir()):
        raise FileExistsError("Final publication staging directory is not empty")
    publication_dir.mkdir(parents=True, exist_ok=True)
    stage_model_payload(
        model_dir,
        publication_dir,
        evidence_name,
        ignore_existing_publication=True,
    )
    for name in ("README.md", "publication_manifest.json"):
        source = model_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Final publication source is missing {name}")
        copy_sanitized_text(source, publication_dir / name)
    evaluation_root = model_dir / "evaluation"
    if not evaluation_root.is_dir():
        raise FileNotFoundError("Final publication evaluation evidence is missing")
    source_manifest = read_json(model_dir / "publication_manifest.json")
    declared_evaluation_files = {
        f"evaluation/{name}" for name in source_manifest.get("evidence", {})
    }
    for records in source_manifest.get("raw_evaluation_json", {}).values():
        if not isinstance(records, list):
            raise ValueError("Final publication raw evaluation manifest is invalid")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Final publication raw evaluation record is invalid")
            declared_evaluation_files.add(str(record.get("path", "")))
    for relative in sorted(declared_evaluation_files):
        relative_path = Path(relative)
        if (
            not relative.startswith("evaluation/")
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise ValueError("Final publication evaluation path is unsafe")
        source = model_dir / relative_path
        if (
            not source.is_file()
            or source.is_symlink()
            or source.suffix.lower() not in {".json", ".jsonl"}
        ):
            raise ValueError("Final publication contains unsafe evaluation payload")
        destination = publication_dir / relative_path
        copy_sanitized_text(source, destination)

    staged_manifest_path = publication_dir / "publication_manifest.json"
    staged_manifest = read_json(staged_manifest_path)
    staged_manifest["model_dir"] = model_rel
    staged_manifest["model_evidence"]["sha256"] = sha256(
        publication_dir / evidence_name
    )
    files = publication_files(publication_dir)
    files.pop("publication_manifest.json", None)
    staged_manifest["files_excluding_manifest"] = files
    staged_manifest["publication_safety"] = {
        "isolated_staging": True,
        "source_model_mutated_during_upload": False,
        "allowlisted_model_payload": True,
        "local_paths_removed": True,
        "recognized_credentials_removed": True,
        "remote_exact_verification_required": True,
    }
    staged_manifest_path.write_text(
        json.dumps(staged_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_no_sensitive_files(publication_dir)
    validate_staged_text(publication_dir)
    if model_weights_sha256(publication_dir) != model_weights_sha256(model_dir):
        raise RuntimeError("Final publication staging changed model shards")
    return staged_manifest_path


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
    try:
        from scripts.publish_private_clean_candidate import (
            copy_sanitized_text,
            publication_files,
            verify_remote_publication,
        )
    except ImportError:  # pragma: no cover - direct script execution fallback
        from publish_private_clean_candidate import (
            copy_sanitized_text,
            publication_files,
            verify_remote_publication,
        )
    (
        evidence,
        sionic,
        official,
        comprehensive,
        training,
        clean,
        robustness,
        multidomain,
        approval,
    ) = validate(args)
    model_dir = args.model_dir.resolve()
    card = build_card(
        args.repo_id,
        evidence,
        sionic,
        official,
        comprehensive,
        training,
        clean,
        robustness,
        multidomain,
    )
    card_path = model_dir / "README.md"
    card_path.write_text(card, encoding="utf-8")
    evidence_dir = model_dir / "evaluation"
    evidence_dir.mkdir(exist_ok=True)
    if not args.comprehensive_summary:
        (evidence_dir / "comprehensive_text_v1_summary.json").unlink(
            missing_ok=True
        )
        stale_comprehensive_raw = evidence_dir / "raw" / "comprehensive_text_v1"
        if stale_comprehensive_raw.is_dir():
            shutil.rmtree(stale_comprehensive_raw)
    evidence_files = {
        "sionic9_summary.json": args.sionic_summary.resolve(),
        "mteb_korean_v1_summary.json": args.official_summary.resolve(),
        "training_manifest.json": args.training_manifest.resolve(),
    }
    if args.comprehensive_summary:
        evidence_files["comprehensive_text_v1_summary.json"] = (
            args.comprehensive_summary.resolve()
        )
    if args.clean_summary:
        evidence_files["legal_source_heldout_summary.json"] = (
            args.clean_summary.resolve()
        )
        clean_ranks = args.clean_summary.resolve().parent / "ranks.jsonl"
        if clean_ranks.is_file():
            evidence_files["legal_source_heldout_ranks.jsonl"] = clean_ranks
    if args.robustness_summary:
        evidence_files["conversational_noise_summary.json"] = (
            args.robustness_summary.resolve()
        )
        robustness_ranks = args.robustness_summary.resolve().parent / "ranks.jsonl"
        if robustness_ranks.is_file():
            evidence_files["conversational_noise_ranks.jsonl"] = robustness_ranks
    if args.multidomain_summary:
        evidence_files["multidomain_selection_summary.json"] = (
            args.multidomain_summary.resolve()
        )
        multidomain_ranks = args.multidomain_summary.resolve().parent / "ranks.jsonl"
        if multidomain_ranks.is_file():
            evidence_files["multidomain_selection_ranks.jsonl"] = multidomain_ranks
    if approval is not None:
        evidence_files["public_release_approval.json"] = (
            args.release_approval.resolve()
        )
    for name, source in evidence_files.items():
        copy_sanitized_text(source, evidence_dir / name)
    raw_evidence = {
        "sionic9": copy_json_tree(
            args.sionic_summary.resolve().parent / "mteb_cache",
            evidence_dir / "raw" / "sionic9",
        ),
        "mteb_korean_v1": copy_json_tree(
            args.official_summary.resolve().parent / "mteb_cache",
            evidence_dir / "raw" / "mteb_korean_v1",
        ),
    }
    if args.comprehensive_summary:
        comprehensive_raw_root = evidence_dir / "raw" / "comprehensive_text_v1"
        if comprehensive_raw_root.is_dir():
            shutil.rmtree(comprehensive_raw_root)
        raw_evidence["comprehensive_text_v1"] = copy_json_tree(
            args.comprehensive_summary.resolve().parent / "mteb_cache",
            comprehensive_raw_root,
        )
    evidence_name = (
        "soup_report.json"
        if is_model_soup(evidence)
        else ("full_tuning_report.json" if is_full_update(evidence) else "merge_report.json")
    )
    try:
        recorded_model_dir = model_dir.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        recorded_model_dir = f"local:{model_dir.name}"
    publication_manifest = {
        "schema_version": 1,
        "repo_id": args.repo_id,
        "model_dir": recorded_model_dir,
        "model_evidence": {
            "file": evidence_name,
            "sha256": sha256(model_dir / evidence_name),
        },
        "card_sha256": sha256(card_path),
        "evidence": {
            name: {"sha256": sha256(evidence_dir / name)} for name in evidence_files
        },
        "raw_evaluation_json": raw_evidence,
        "comprehensive_text_v1": (
            {
                "status": "complete_diagnostic",
                "protocol_id": comprehensive["protocol_id"],
                "summary_sha256": sha256(args.comprehensive_summary.resolve()),
                "completed_tasks": comprehensive["completed_tasks"],
                "completed_subsets": comprehensive["completed_subsets"],
                "claim_status": comprehensive["claim_status"],
            }
            if comprehensive is not None
            else {"status": "not_provided"}
        ),
        "multidomain_selection": (
            {
                "status": "complete_selection_only",
                "protocol_id": multidomain["protocol_id"],
                "summary_sha256": sha256(args.multidomain_summary.resolve()),
                "manifest_sha256": multidomain["dataset"]["manifest_sha256"],
                "macro_domain_ndcg_at_10": multidomain["metrics"][
                    "macro_domain_ndcg_at_10"
                ],
                "public_benchmark_score_used_for_selection": False,
            }
            if multidomain is not None
            else {"status": "not_provided"}
        ),
        "model_weights_evidence_sha256": evidence["model"]["weights_sha256"],
        "public_release_gate": (
            {
                "status": "pass",
                "approval_id": approval["approval_id"],
                "training_track": RIGHTS_SAFE_TRAINING_TRACK,
                "approval_sha256": sha256(args.release_approval.resolve()),
            }
            if approval is not None
            else {"status": "not_requested", "visibility": "private"}
        ),
    }
    (model_dir / "publication_manifest.json").write_text(
        json.dumps(publication_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "repo_id": args.repo_id,
        "model_dir": recorded_model_dir,
        "card": "README.md",
        "publication_manifest": "publication_manifest.json",
        "weights_sha256": evidence["model"]["weights_sha256"],
        "sionic9_average": sionic["average"],
        "official_mean_task": official["mean_task_leaderboard_points"],
        "visibility": "public" if args.public else "private",
        "upload_requested": args.upload,
        "validated": True,
        "public_release_approved": approval is not None,
        "comprehensive_text_v1": (
            {
                "protocol_id": comprehensive["protocol_id"],
                "summary_sha256": sha256(args.comprehensive_summary.resolve()),
                "completed_tasks": comprehensive["completed_tasks"],
                "completed_subsets": comprehensive["completed_subsets"],
                "diagnostic_only": True,
            }
            if comprehensive is not None
            else None
        ),
        "multidomain_selection": (
            {
                "protocol_id": multidomain["protocol_id"],
                "summary_sha256": sha256(args.multidomain_summary.resolve()),
                "manifest_sha256": multidomain["dataset"]["manifest_sha256"],
                "macro_domain_ndcg_at_10": multidomain["metrics"][
                    "macro_domain_ndcg_at_10"
                ],
                "selection_only": True,
            }
            if multidomain is not None
            else None
        ),
    }
    if args.upload:
        token = load_hf_token(args.hf_token_file)
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        with tempfile.TemporaryDirectory(
            prefix=f".{model_dir.name}.final-publish-", dir=model_dir.parent
        ) as temporary:
            publication_dir = Path(temporary)
            staged_manifest = prepare_isolated_upload_staging(
                model_dir=model_dir,
                evidence_name=evidence_name,
                publication_dir=publication_dir,
            )
            expected_files = publication_files(publication_dir)
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
                    "Remote final publication contains unexpected pre-existing files: "
                    f"{sorted(unexpected_preexisting)}"
                )
            source_weights_sha = model_weights_sha256(model_dir)
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
            if model_weights_sha256(model_dir) != source_weights_sha:
                raise RuntimeError("Source model changed during final publication upload")
            report["commit_sha"] = info.sha
            report["publication_manifest_sha256"] = sha256(staged_manifest)
            report["remote_manifest_exact"] = True
            report["remote_file_set_exact"] = True
            report["remote_files_verified"] = len(expected_files)
            report["isolated_staging"] = True
            report["url"] = f"https://huggingface.co/{args.repo_id}"
    if args.report_output:
        write_atomic_report(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
