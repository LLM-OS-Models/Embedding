#!/usr/bin/env python3
"""Select a local candidate without consulting public benchmark scores.

The verified Grade-I legal source holdout defines an admissible guard band.
The fixed non-public finance/knowledge macro then ranks broad quality inside
that band, followed by paired-noise robustness and deterministic fallbacks.
Sionic9 and official MTEB results are intentionally not accepted as inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = "clean-first-grade-i-near-tie-robustness-v1"
MULTIDOMAIN_POLICY_ID = "clean-guard-multidomain-near-tie-robustness-v2"
CLEAN_PROTOCOL_ID = "legal-source-document-heldout-i-v2-text-strict"
ROBUSTNESS_PROTOCOL_ID = "legal-conversational-noise-i-v2-text-strict"
MULTIDOMAIN_PROTOCOL_ID = "multidomain-selection-heldout-v1"
MULTIDOMAIN_MANIFEST_SHA256 = (
    "86fea553c6652388b1f67160c0e2e6b7626acf8929f86c1a2708156bd89b3c46"
)
EXPECTED_SCORE_CONTRACT = (
    "exact float32 normalized dot; TF32 disabled; rank ties by corpus ID ascending"
)
EXPECTED_ROBUST_SCORE_CONTRACT = (
    "exact float32 normalized dot; TF32 disabled; ties by combined corpus ordering"
)
EXPECTED_MULTIDOMAIN_SCORE_CONTRACT = (
    "exact float32 normalized dot; TF32 disabled; per-domain corpus; "
    "rank ties by corpus ID ascending"
)
EXPECTED_CONDITIONS = {
    f"{prompt}/noise_{ratio}"
    for prompt in ("prompt_on", "prompt_off")
    for ratio in ("0.00", "0.01", "0.05")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("clean_root", type=Path)
    parser.add_argument("robustness_root", type=Path)
    parser.add_argument("--multidomain-root", type=Path)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-model", action="store_true")
    parser.add_argument("--disqualification-root", type=Path)
    parser.add_argument(
        "--candidate-model",
        action="append",
        help="Exact local model path eligible for this campaign; repeat as needed.",
    )
    parser.add_argument("--clean-epsilon", type=float, default=0.005)
    parser.add_argument("--multidomain-epsilon", type=float, default=0.002)
    parser.add_argument("--robustness-epsilon", type=float, default=0.002)
    parser.add_argument("--intrusion-epsilon", type=float, default=0.001)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} is outside [0, 1]")
    return number


def valid_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verified_ranks(summary_path: Path, summary: dict[str, Any]) -> str:
    descriptor = summary.get("files", {}).get("ranks.jsonl", {})
    if descriptor.get("rows") != 10000 or not valid_sha(descriptor.get("sha256")):
        raise ValueError("ranks.jsonl evidence is incomplete")
    ranks = summary_path.parent / "ranks.jsonl"
    if not ranks.is_file() or sha256(ranks) != descriptor["sha256"]:
        raise ValueError("ranks.jsonl does not match its recorded SHA-256")
    with ranks.open("rb") as handle:
        rows = sum(1 for _ in handle)
    if rows != 10000:
        raise ValueError("ranks.jsonl must contain exactly 10K rows")
    return descriptor["sha256"]


def verified_multidomain_ranks(summary_path: Path, summary: dict[str, Any]) -> str:
    descriptor = summary.get("files", {}).get("ranks.jsonl", {})
    if descriptor.get("rows") != 1900 or not valid_sha(descriptor.get("sha256")):
        raise ValueError("multidomain ranks.jsonl evidence is incomplete")
    ranks = summary_path.parent / "ranks.jsonl"
    if not ranks.is_file() or sha256(ranks) != descriptor["sha256"]:
        raise ValueError("multidomain ranks.jsonl does not match recorded SHA-256")
    with ranks.open("rb") as handle:
        rows = sum(1 for _ in handle)
    if rows != 1900:
        raise ValueError("multidomain ranks.jsonl must contain exactly 1,900 rows")
    return descriptor["sha256"]


def model_run_name(model: str) -> str | None:
    name = Path(model).name
    for suffix in ("-best-merged", "-best-full"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    checkpoint = re.fullmatch(r"(.+)-checkpoint-[1-9][0-9]*-clean-candidate-merged", name)
    if checkpoint:
        return checkpoint.group(1)
    return None


def disqualification_marker(model: str, root: Path | None) -> Path | None:
    if root is None:
        return None
    run_name = model_run_name(model)
    if not run_name:
        return None
    marker = root.expanduser().resolve() / run_name / "DISQUALIFIED.json"
    return marker if marker.is_file() and marker.stat().st_size > 0 else None


def validate_model_evidence(
    workspace_root: Path, model: Any, revision: Any
) -> tuple[Path, str]:
    if not isinstance(model, str) or not model.startswith("artifacts/models/"):
        raise ValueError("candidate is not a local artifacts/models model")
    if not isinstance(revision, str):
        raise ValueError("candidate has no immutable local revision")
    relative = Path(model)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("candidate model path is unsafe")
    workspace = workspace_root.expanduser().resolve()
    model_dir = (workspace / relative).resolve()
    try:
        model_dir.relative_to(workspace)
    except ValueError as error:
        raise ValueError("candidate model escapes the workspace") from error
    reports = [
        path
        for path in (
            model_dir / "merge_report.json",
            model_dir / "full_tuning_report.json",
            model_dir / "soup_report.json",
        )
        if path.is_file()
    ]
    if len(reports) != 1:
        raise ValueError("candidate must have exactly one model evidence report")
    evidence = read_json(reports[0])
    weights_sha = evidence.get("model", {}).get("weights_sha256")
    if not valid_sha(weights_sha):
        raise ValueError("model evidence has no valid weights SHA-256")
    expected_revision = f"model-{weights_sha[:12]}"
    if revision != expected_revision:
        raise ValueError(
            f"summary revision {revision!r} does not match {expected_revision!r}"
        )
    return model_dir, weights_sha


def validate_common_summary(
    path: Path,
    summary: dict[str, Any],
    *,
    protocol_id: str,
    score_contract: str,
) -> tuple[str, str, str, str]:
    if summary.get("protocol_id") != protocol_id:
        raise ValueError(f"unexpected protocol_id for {path}")
    model = summary.get("model")
    revision = summary.get("requested_revision")
    if not isinstance(model, str) or not isinstance(revision, str):
        raise ValueError("summary has no model/revision identity")
    dataset = summary.get("dataset", {})
    manifest_sha = dataset.get("manifest_sha256")
    if not valid_sha(manifest_sha):
        raise ValueError("summary has no valid dataset manifest SHA-256")
    if dataset.get("independence_grade") != "I" or dataset.get("not_grade") != "Z":
        raise ValueError("summary is not the disclosed Grade-I/not-Grade-Z holdout")
    if summary.get("score_contract") != score_contract:
        raise ValueError("summary score contract drifted")
    environment = summary.get("environment", {})
    if environment.get("torch_dtype") != "bfloat16":
        raise ValueError("only canonical bfloat16 candidate evaluations are comparable")
    if environment.get("max_length") != 8192:
        raise ValueError("candidate evaluation max_length must be 8192")
    if environment.get("attention") != "flash_attention_2":
        raise ValueError("candidate evaluation must use flash_attention_2")
    ranks_sha = verified_ranks(path, summary)
    return model, revision, manifest_sha, ranks_sha


def load_clean_candidate(
    path: Path, workspace_root: Path
) -> dict[str, Any]:
    summary = read_json(path)
    model, revision, manifest_sha, ranks_sha = validate_common_summary(
        path,
        summary,
        protocol_id=CLEAN_PROTOCOL_ID,
        score_contract=EXPECTED_SCORE_CONTRACT,
    )
    model_dir, weights_sha = validate_model_evidence(workspace_root, model, revision)
    metrics = summary.get("metrics", {})
    clean_ndcg = finite_unit(metrics.get("ndcg_at_10"), "clean ndcg_at_10")
    finite_unit(metrics.get("recall_at_10"), "clean recall_at_10")
    finite_unit(metrics.get("mrr_at_10"), "clean mrr_at_10")
    finite_unit(metrics.get("recall_at_100"), "clean recall_at_100")
    return {
        "model": model,
        "revision": revision,
        "model_dir": str(model_dir),
        "weights_sha256": weights_sha,
        "dataset_manifest_sha256": manifest_sha,
        "clean_ndcg_at_10": clean_ndcg,
        "clean_summary": str(path.resolve()),
        "clean_ranks_sha256": ranks_sha,
        "query_prompt": summary.get("query_prompt"),
    }


def load_robustness(path: Path) -> dict[str, Any]:
    summary = read_json(path)
    model, revision, manifest_sha, ranks_sha = validate_common_summary(
        path,
        summary,
        protocol_id=ROBUSTNESS_PROTOCOL_ID,
        score_contract=EXPECTED_ROBUST_SCORE_CONTRACT,
    )
    conditions = summary.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != EXPECTED_CONDITIONS:
        raise ValueError("robustness conditions are incomplete or drifted")
    ndcgs: list[float] = []
    intrusions: list[float] = []
    for name in sorted(EXPECTED_CONDITIONS):
        metrics = conditions[name]
        if not isinstance(metrics, dict):
            raise ValueError(f"invalid robustness condition {name}")
        ndcgs.append(finite_unit(metrics.get("ndcg_at_10"), f"{name} ndcg_at_10"))
        finite_unit(metrics.get("recall_at_10"), f"{name} recall_at_10")
        retention = metrics.get("ndcg_retention_vs_same_prompt_clean")
        if retention is not None:
            if isinstance(retention, bool) or not math.isfinite(float(retention)):
                raise ValueError(f"{name} has invalid NDCG retention")
            if float(retention) < 0.0:
                raise ValueError(f"{name} has negative NDCG retention")
        if not name.endswith("noise_0.00"):
            intrusions.append(
                finite_unit(
                    metrics.get("noise_intrusion_at_10"),
                    f"{name} noise_intrusion_at_10",
                )
            )
    return {
        "model": model,
        "revision": revision,
        "dataset_manifest_sha256": manifest_sha,
        "robustness_floor_ndcg_at_10": min(ndcgs),
        "max_noise_intrusion_at_10": max(intrusions),
        "prompt_on_clean_ndcg_at_10": finite_unit(
            conditions["prompt_on/noise_0.00"].get("ndcg_at_10"),
            "prompt_on/noise_0.00 ndcg_at_10",
        ),
        "robustness_summary": str(path.resolve()),
        "robustness_ranks_sha256": ranks_sha,
        "query_prompt": summary.get("query_prompt"),
    }


def load_multidomain_candidate(
    path: Path, workspace_root: Path
) -> dict[str, Any]:
    summary = read_json(path)
    if summary.get("protocol_id") != MULTIDOMAIN_PROTOCOL_ID:
        raise ValueError("unexpected multidomain protocol")
    model = summary.get("model")
    revision = summary.get("requested_revision")
    if not isinstance(model, str) or not isinstance(revision, str):
        raise ValueError("multidomain summary has no model/revision")
    _, weights_sha = validate_model_evidence(workspace_root, model, revision)
    dataset = summary.get("dataset", {})
    manifest_sha = dataset.get("manifest_sha256")
    if (
        manifest_sha != MULTIDOMAIN_MANIFEST_SHA256
        or dataset.get("selection_only") is not True
        or dataset.get("public_benchmark") is not False
        or set(dataset.get("domains", {})) != {"finance", "knowledge"}
    ):
        raise ValueError("multidomain dataset disclosure is incomplete")
    domain_contract = dataset["domains"]
    if not all(isinstance(domain_contract[name], dict) for name in domain_contract):
        raise ValueError("multidomain domain descriptors are malformed")
    if (
        domain_contract["finance"].get("queries") != 900
        or domain_contract["finance"].get("independence")
        != "query-heldout; corpus exposure disclosed"
        or domain_contract["finance"].get("corpus_training_text_occurrences")
        != 1373
        or domain_contract["knowledge"].get("queries") != 1000
        or domain_contract["knowledge"].get("independence")
        != "query-and-corpus exact-text-heldout"
    ):
        raise ValueError("multidomain domain holdout contract drifted")
    if summary.get("score_contract") != EXPECTED_MULTIDOMAIN_SCORE_CONTRACT:
        raise ValueError("multidomain score contract drifted")
    environment = summary.get("environment", {})
    if (
        environment.get("torch_dtype") != "bfloat16"
        or environment.get("max_length") != 8192
        or environment.get("attention") != "flash_attention_2"
    ):
        raise ValueError("multidomain canonical evaluation environment drifted")
    metrics = summary.get("metrics", {})
    macro = finite_unit(
        metrics.get("macro_domain_ndcg_at_10"), "multidomain macro NDCG@10"
    )
    domains = summary.get("domain_metrics", {})
    if set(domains) != {"finance", "knowledge"}:
        raise ValueError("multidomain task metrics are incomplete")
    domain_scores = {
        domain: finite_unit(row.get("ndcg_at_10"), f"{domain} NDCG@10")
        for domain, row in domains.items()
        if isinstance(row, dict)
    }
    if len(domain_scores) != 2 or not math.isclose(
        macro, sum(domain_scores.values()) / 2.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("multidomain macro is inconsistent with domain metrics")
    ranks_sha = verified_multidomain_ranks(path, summary)
    return {
        "model": model,
        "revision": revision,
        "weights_sha256": weights_sha,
        "multidomain_manifest_sha256": manifest_sha,
        "multidomain_macro_ndcg_at_10": macro,
        "multidomain_domain_ndcg_at_10": domain_scores,
        "multidomain_summary": str(path.resolve()),
        "multidomain_ranks_sha256": ranks_sha,
    }


def summary_paths(root: Path) -> list[Path]:
    return sorted(root.expanduser().resolve().glob("*/*/summary.json"))


def select_candidates(
    *,
    clean_root: Path,
    robustness_root: Path,
    workspace_root: Path,
    disqualification_root: Path | None,
    candidate_models: set[str] | None,
    clean_epsilon: float,
    robustness_epsilon: float,
    intrusion_epsilon: float,
    multidomain_root: Path | None = None,
    multidomain_epsilon: float = 0.002,
) -> dict[str, Any]:
    for label, value in (
        ("clean_epsilon", clean_epsilon),
        ("multidomain_epsilon", multidomain_epsilon),
        ("robustness_epsilon", robustness_epsilon),
        ("intrusion_epsilon", intrusion_epsilon),
    ):
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{label} must be finite and within [0, 1]")

    if candidate_models is not None and not candidate_models:
        raise ValueError("candidate_models cannot be an empty set")
    clean: dict[tuple[str, str], dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []
    for path in summary_paths(clean_root):
        try:
            row = load_clean_candidate(path, workspace_root)
            if candidate_models is not None and row["model"] not in candidate_models:
                raise ValueError("model is not in the explicit campaign candidate allowlist")
            key = (row["model"], row["revision"])
            if key in clean:
                raise ValueError("duplicate clean summary for model/revision")
            marker = disqualification_marker(row["model"], disqualification_root)
            if marker is not None:
                raise ValueError(f"run-level disqualification marker: {marker}")
            clean[key] = row
        except (KeyError, TypeError, ValueError) as error:
            excluded.append({"summary": str(path.resolve()), "reason": str(error)})

    robust: dict[tuple[str, str], dict[str, Any]] = {}
    for path in summary_paths(robustness_root):
        try:
            row = load_robustness(path)
            key = (row["model"], row["revision"])
            if key in robust:
                raise ValueError("duplicate robustness summary for model/revision")
            robust[key] = row
        except (KeyError, TypeError, ValueError) as error:
            excluded.append({"summary": str(path.resolve()), "reason": str(error)})

    multidomain: dict[tuple[str, str], dict[str, Any]] = {}
    if multidomain_root is not None:
        for path in summary_paths(multidomain_root):
            try:
                row = load_multidomain_candidate(path, workspace_root)
                key = (row["model"], row["revision"])
                if key in multidomain:
                    raise ValueError("duplicate multidomain summary for model/revision")
                multidomain[key] = row
            except (KeyError, TypeError, ValueError) as error:
                excluded.append({"summary": str(path.resolve()), "reason": str(error)})

    candidates: list[dict[str, Any]] = []
    for key, clean_row in sorted(clean.items()):
        robust_row = robust.get(key)
        if robust_row is None:
            excluded.append(
                {
                    "model": clean_row["model"],
                    "revision": clean_row["revision"],
                    "summary": clean_row["clean_summary"],
                    "reason": "missing complete matching robustness summary",
                }
            )
            continue
        if robust_row["dataset_manifest_sha256"] != clean_row["dataset_manifest_sha256"]:
            excluded.append(
                {
                    "model": clean_row["model"],
                    "revision": clean_row["revision"],
                    "reason": "clean/robustness dataset manifest SHA mismatch",
                }
            )
            continue
        if robust_row["query_prompt"] != clean_row["query_prompt"]:
            excluded.append(
                {
                    "model": clean_row["model"],
                    "revision": clean_row["revision"],
                    "reason": "clean/robustness query prompt mismatch",
                }
            )
            continue
        if not math.isclose(
            robust_row["prompt_on_clean_ndcg_at_10"],
            clean_row["clean_ndcg_at_10"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            excluded.append(
                {
                    "model": clean_row["model"],
                    "revision": clean_row["revision"],
                    "reason": "clean NDCG does not reproduce in robustness evaluation",
                }
            )
            continue
        multidomain_row = multidomain.get(key) if multidomain_root is not None else None
        if multidomain_root is not None and multidomain_row is None:
            excluded.append(
                {
                    "model": clean_row["model"],
                    "revision": clean_row["revision"],
                    "reason": "missing complete matching multidomain summary",
                }
            )
            continue
        if multidomain_row is not None and multidomain_row["weights_sha256"] != clean_row["weights_sha256"]:
            excluded.append(
                {
                    "model": clean_row["model"],
                    "revision": clean_row["revision"],
                    "reason": "multidomain model weight evidence mismatch",
                }
            )
            continue
        candidates.append(
            {**clean_row, **robust_row, **(multidomain_row or {})}
        )

    if not candidates:
        raise RuntimeError(
            "No complete verified local candidate has matching clean and robustness results"
        )
    manifests = {row["dataset_manifest_sha256"] for row in candidates}
    if len(manifests) != 1:
        raise RuntimeError("Candidate summaries use different clean dataset manifests")
    multidomain_manifests = {
        row["multidomain_manifest_sha256"]
        for row in candidates
        if "multidomain_manifest_sha256" in row
    }
    if multidomain_root is not None and len(multidomain_manifests) != 1:
        raise RuntimeError("Candidate summaries use different multidomain manifests")

    best_clean = max(row["clean_ndcg_at_10"] for row in candidates)
    clean_shortlist = [
        row
        for row in candidates
        if best_clean - row["clean_ndcg_at_10"] <= clean_epsilon
    ]
    if multidomain_root is not None:
        best_multidomain = max(
            row["multidomain_macro_ndcg_at_10"] for row in clean_shortlist
        )
        multidomain_shortlist = [
            row
            for row in clean_shortlist
            if best_multidomain - row["multidomain_macro_ndcg_at_10"]
            <= multidomain_epsilon
        ]
    else:
        best_multidomain = None
        multidomain_shortlist = clean_shortlist
    best_robust = max(
        row["robustness_floor_ndcg_at_10"] for row in multidomain_shortlist
    )
    robust_shortlist = [
        row
        for row in multidomain_shortlist
        if best_robust - row["robustness_floor_ndcg_at_10"] <= robustness_epsilon
    ]
    best_intrusion = min(row["max_noise_intrusion_at_10"] for row in robust_shortlist)
    intrusion_shortlist = [
        row
        for row in robust_shortlist
        if row["max_noise_intrusion_at_10"] - best_intrusion <= intrusion_epsilon
    ]
    intrusion_shortlist.sort(
        key=lambda row: (
            -row.get("multidomain_macro_ndcg_at_10", 0.0),
            -row["clean_ndcg_at_10"],
            row["model"],
            row["revision"],
        )
    )
    best = intrusion_shortlist[0]

    candidates.sort(
        key=lambda row: (
            -(row["clean_ndcg_at_10"] >= best_clean - clean_epsilon),
            -(
                multidomain_root is not None
                and row in multidomain_shortlist
            ),
            -row.get("multidomain_macro_ndcg_at_10", 0.0),
            -row["clean_ndcg_at_10"],
            -row["robustness_floor_ndcg_at_10"],
            row["max_noise_intrusion_at_10"],
            row["model"],
            row["revision"],
        )
    )
    selected_key = (best["model"], best["revision"])
    ranking = [
        {
            **row,
            "within_clean_near_tie": row in clean_shortlist,
            "within_multidomain_near_tie": row in multidomain_shortlist,
            "within_robustness_near_tie": row in robust_shortlist,
            "within_intrusion_near_tie": row in intrusion_shortlist,
            "selected": (row["model"], row["revision"]) == selected_key,
        }
        for row in candidates
    ]
    return {
        "schema_version": 1,
        "policy_id": MULTIDOMAIN_POLICY_ID if multidomain_root is not None else POLICY_ID,
        "selection_order": [
            "verified Grade-I clean NDCG@10 near-tie band",
            *(
                ["fixed non-public finance/knowledge domain-macro NDCG@10 near-tie band"]
                if multidomain_root is not None
                else []
            ),
            "worst-condition robustness NDCG@10 near-tie band",
            "maximum synthetic-noise intrusion@10 near-tie band",
            (
                "multidomain macro, clean NDCG@10, then deterministic model/revision fallback"
                if multidomain_root is not None
                else "clean NDCG@10 then deterministic model/revision fallback"
            ),
        ],
        "public_benchmark_used_for_selection": False,
        "clean_independence": {"grade": "I", "not_grade": "Z"},
        "dataset_manifest_sha256": next(iter(manifests)),
        "multidomain_manifest_sha256": (
            next(iter(multidomain_manifests)) if multidomain_manifests else None
        ),
        "epsilon": {
            "clean_ndcg_at_10": clean_epsilon,
            "multidomain_macro_ndcg_at_10": multidomain_epsilon,
            "robustness_floor_ndcg_at_10": robustness_epsilon,
            "max_noise_intrusion_at_10": intrusion_epsilon,
        },
        "candidate_allowlist": sorted(candidate_models) if candidate_models else None,
        "best": best,
        "ranking": ranking,
        "excluded": excluded,
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(descriptor)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> None:
    args = parse_args()
    report = select_candidates(
        clean_root=args.clean_root,
        robustness_root=args.robustness_root,
        workspace_root=args.workspace_root,
        disqualification_root=args.disqualification_root,
        candidate_models=set(args.candidate_model) if args.candidate_model else None,
        clean_epsilon=args.clean_epsilon,
        robustness_epsilon=args.robustness_epsilon,
        intrusion_epsilon=args.intrusion_epsilon,
        multidomain_root=args.multidomain_root,
        multidomain_epsilon=args.multidomain_epsilon,
    )
    if args.output:
        atomic_write_json(args.output, report)
    if args.print_model:
        print(report["best"]["model"])
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
