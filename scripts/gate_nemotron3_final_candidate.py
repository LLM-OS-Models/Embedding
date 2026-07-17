#!/usr/bin/env python3
"""Apply clean guards and the strict Sionic target to one merged Nemotron model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-decision", type=Path, required=True)
    parser.add_argument("--legal-summary", type=Path, required=True)
    parser.add_argument("--multidomain-summary", type=Path, required=True)
    parser.add_argument("--sionic-summary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric metric: {label}")
    return float(value)


def build(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.model_dir.resolve()
    merge_report_path = model_dir / "merge_report.json"
    for path in (
        args.base_decision,
        args.legal_summary,
        args.multidomain_summary,
        args.sionic_summary,
        merge_report_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    decision = read_object(args.base_decision)
    legal = read_object(args.legal_summary)
    multi = read_object(args.multidomain_summary)
    sionic = read_object(args.sionic_summary)
    merge = read_object(merge_report_path)
    if decision.get("decision") not in {
        "adopt_nemotron3_raw_and_run_short_public_lora",
        "short_public_nemotron3_lora_then_retest",
    }:
        raise ValueError("Base decision did not authorize a Nemotron final candidate")
    weights_sha = merge.get("model", {}).get("weights_sha256")
    if not isinstance(weights_sha, str) or len(weights_sha) != 64:
        raise ValueError("Merged model has no weights SHA")
    expected_revision = f"model-{weights_sha[:12]}"
    for label, summary in (("legal", legal), ("multidomain", multi), ("Sionic", sionic)):
        if Path(str(summary.get("model", ""))).resolve() != model_dir:
            raise ValueError(f"{label} summary belongs to a different model")
        if summary.get("requested_revision") != expected_revision:
            raise ValueError(f"{label} summary revision drifted")
    if legal.get("protocol_id") != "legal-source-document-heldout-i-v2-text-strict":
        raise ValueError("Unexpected legal protocol")
    if multi.get("protocol_id") != "multidomain-selection-heldout-v1":
        raise ValueError("Unexpected multidomain protocol")
    if sionic.get("protocol_id") != "sionic9-fixed-prompt-v1":
        raise ValueError("Unexpected Sionic protocol")
    if sionic.get("completed_tasks") != 9 or len(sionic.get("scores", {})) != 9:
        raise ValueError("Sionic summary is incomplete")

    thresholds = decision["thresholds"]
    clean_guard = number(thresholds.get("clean_absolute_ndcg"), "clean guard")
    domain_guard = number(thresholds.get("domain_absolute_ndcg"), "domain guard")
    target = number(decision.get("target"), "target")
    baseline_legal = decision["scores"]["legal_ndcg_at_10"]
    baseline_macro = decision["scores"]["multidomain_macro_ndcg_at_10"]
    baseline_domains = decision["scores"]["multidomain_domain_ndcg_at_10"]
    references = ("qwen3", "comsat")
    reference_legal = max(number(baseline_legal[name], name) for name in references)
    reference_macro = max(number(baseline_macro[name], name) for name in references)
    reference_domains = {
        domain: max(
            number(baseline_domains[name][domain], f"{name} {domain}")
            for name in references
        )
        for domain in ("finance", "knowledge")
    }
    candidate_legal = number(legal["metrics"].get("ndcg_at_10"), "candidate legal")
    candidate_macro = number(
        multi["metrics"].get("macro_domain_ndcg_at_10"), "candidate multidomain"
    )
    candidate_domains = {
        domain: number(
            multi["domain_metrics"][domain].get("ndcg_at_10"),
            f"candidate {domain}",
        )
        for domain in reference_domains
    }
    sionic_macro = number(sionic.get("average"), "candidate Sionic")
    gates = {
        "legal_within_absolute_guard": candidate_legal >= reference_legal - clean_guard,
        "multidomain_macro_within_absolute_guard": candidate_macro
        >= reference_macro - clean_guard,
        "multidomain_each_domain_within_absolute_guard": {
            domain: candidate_domains[domain] >= reference_domains[domain] - domain_guard
            for domain in reference_domains
        },
        "sionic_strictly_above_target": sionic_macro > target,
    }
    passed = (
        gates["legal_within_absolute_guard"]
        and gates["multidomain_macro_within_absolute_guard"]
        and all(gates["multidomain_each_domain_within_absolute_guard"].values())
        and gates["sionic_strictly_above_target"]
    )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "model": {
            "path": str(model_dir),
            "revision": expected_revision,
            "weights_sha256": weights_sha,
        },
        "target": target,
        "thresholds": {"clean_absolute_ndcg": clean_guard, "domain_absolute_ndcg": domain_guard},
        "gates": gates,
        "scores": {
            "candidate": {
                "sionic9": sionic_macro,
                "legal": candidate_legal,
                "multidomain_macro": candidate_macro,
                "multidomain_domains": candidate_domains,
            },
            "reference_best_qwen_or_comsat": {
                "legal": reference_legal,
                "multidomain_macro": reference_macro,
                "multidomain_domains": reference_domains,
            },
        },
        "public_benchmark_used_for_checkpoint_selection": False,
        "upstream_train_family_exposure": ["MIRACL", "MLDR"],
    }


def main() -> None:
    args = parse_args()
    report = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
