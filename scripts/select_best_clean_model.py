#!/usr/bin/env python3
"""Select a local candidate without consulting public benchmark scores.

The primary signal is the verified Grade-I legal source holdout.  Differences
smaller than the configured absolute epsilon form a near-tie.  Robustness is
then used to resolve that near-tie, again with an epsilon, before deterministic
fallbacks are applied.  Sionic9 and official MTEB results are intentionally not
accepted as inputs to this selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = "clean-first-grade-i-near-tie-robustness-v1"
CLEAN_PROTOCOL_ID = "legal-source-document-heldout-i-v1"
ROBUSTNESS_PROTOCOL_ID = "legal-conversational-noise-i-v1"
EXPECTED_SCORE_CONTRACT = (
    "exact float32 normalized dot; TF32 disabled; rank ties by corpus ID ascending"
)
EXPECTED_ROBUST_SCORE_CONTRACT = (
    "exact float32 normalized dot; TF32 disabled; ties by combined corpus ordering"
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
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-model", action="store_true")
    parser.add_argument("--disqualification-root", type=Path)
    parser.add_argument("--clean-epsilon", type=float, default=0.002)
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


def model_run_name(model: str) -> str | None:
    name = Path(model).name
    for suffix in ("-best-merged", "-best-full"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
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


def summary_paths(root: Path) -> list[Path]:
    return sorted(root.expanduser().resolve().glob("*/*/summary.json"))


def select_candidates(
    *,
    clean_root: Path,
    robustness_root: Path,
    workspace_root: Path,
    disqualification_root: Path | None,
    clean_epsilon: float,
    robustness_epsilon: float,
    intrusion_epsilon: float,
) -> dict[str, Any]:
    for label, value in (
        ("clean_epsilon", clean_epsilon),
        ("robustness_epsilon", robustness_epsilon),
        ("intrusion_epsilon", intrusion_epsilon),
    ):
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{label} must be finite and within [0, 1]")

    clean: dict[tuple[str, str], dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []
    for path in summary_paths(clean_root):
        try:
            row = load_clean_candidate(path, workspace_root)
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
        candidates.append({**clean_row, **robust_row})

    if not candidates:
        raise RuntimeError(
            "No complete verified local candidate has matching clean and robustness results"
        )
    manifests = {row["dataset_manifest_sha256"] for row in candidates}
    if len(manifests) != 1:
        raise RuntimeError("Candidate summaries use different clean dataset manifests")

    best_clean = max(row["clean_ndcg_at_10"] for row in candidates)
    clean_shortlist = [
        row
        for row in candidates
        if best_clean - row["clean_ndcg_at_10"] <= clean_epsilon
    ]
    best_robust = max(row["robustness_floor_ndcg_at_10"] for row in clean_shortlist)
    robust_shortlist = [
        row
        for row in clean_shortlist
        if best_robust - row["robustness_floor_ndcg_at_10"] <= robustness_epsilon
    ]
    best_intrusion = min(row["max_noise_intrusion_at_10"] for row in robust_shortlist)
    intrusion_shortlist = [
        row
        for row in robust_shortlist
        if row["max_noise_intrusion_at_10"] - best_intrusion <= intrusion_epsilon
    ]
    intrusion_shortlist.sort(
        key=lambda row: (-row["clean_ndcg_at_10"], row["model"], row["revision"])
    )
    best = intrusion_shortlist[0]

    candidates.sort(
        key=lambda row: (
            -(row["clean_ndcg_at_10"] >= best_clean - clean_epsilon),
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
            "within_robustness_near_tie": row in robust_shortlist,
            "within_intrusion_near_tie": row in intrusion_shortlist,
            "selected": (row["model"], row["revision"]) == selected_key,
        }
        for row in candidates
    ]
    return {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "selection_order": [
            "verified Grade-I clean NDCG@10 near-tie band",
            "worst-condition robustness NDCG@10 near-tie band",
            "maximum synthetic-noise intrusion@10 near-tie band",
            "clean NDCG@10 then deterministic model/revision fallback",
        ],
        "public_benchmark_used_for_selection": False,
        "clean_independence": {"grade": "I", "not_grade": "Z"},
        "dataset_manifest_sha256": next(iter(manifests)),
        "epsilon": {
            "clean_ndcg_at_10": clean_epsilon,
            "robustness_floor_ndcg_at_10": robustness_epsilon,
            "max_noise_intrusion_at_10": intrusion_epsilon,
        },
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
        clean_epsilon=args.clean_epsilon,
        robustness_epsilon=args.robustness_epsilon,
        intrusion_epsilon=args.intrusion_epsilon,
    )
    if args.output:
        atomic_write_json(args.output, report)
    if args.print_model:
        print(report["best"]["model"])
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
