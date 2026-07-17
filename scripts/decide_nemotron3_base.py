#!/usr/bin/env python3
"""Combine completed Sionic and clean-selector summaries into one base decision."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


REVISIONS = {
    "nemotron3": "2b29550c4ab0646bb6bb47032dda54ea11f6dfe2",
    "qwen3": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
    "comsat": "a5cc22b651c1b2e51cdd8bf671774ae93584f0ab",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sionic-dir", type=Path, required=True)
    parser.add_argument("--legal-dir", type=Path, required=True)
    parser.add_argument("--multidomain-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=float, default=0.7930)
    parser.add_argument("--clean-guard", type=float, default=0.01)
    parser.add_argument("--domain-guard", type=float, default=0.015)
    parser.add_argument("--max-short-adaptation-deficit", type=float, default=0.02)
    return parser.parse_args()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def summaries(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    found = [(path, read_object(path)) for path in sorted(directory.rglob("summary.json"))]
    if not found:
        raise FileNotFoundError(f"No summary.json under {directory}")
    return found


def by_revision(directory: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    indexed: dict[str, tuple[Path, dict[str, Any]]] = {}
    revision_to_label = {revision: label for label, revision in REVISIONS.items()}
    for path, summary in summaries(directory):
        revision = summary.get("requested_revision")
        label = revision_to_label.get(revision)
        if label is None:
            continue
        if label in indexed:
            raise ValueError(f"Duplicate {label} summary under {directory}")
        indexed[label] = (path, summary)
    missing = set(REVISIONS) - set(indexed)
    if missing:
        raise ValueError(f"Missing pinned summaries under {directory}: {sorted(missing)}")
    return indexed


def require_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric metric: {name}")
    return float(value)


def build(args: argparse.Namespace) -> dict[str, Any]:
    sionic_candidates = [
        (path, summary)
        for path, summary in summaries(args.sionic_dir)
        if summary.get("requested_revision") == REVISIONS["nemotron3"]
    ]
    if len(sionic_candidates) != 1:
        raise ValueError("Expected exactly one pinned Nemotron Sionic summary")
    sionic_path, sionic = sionic_candidates[0]
    if sionic.get("completed_tasks") != 9 or sionic.get("total_protocol_tasks") != 9:
        raise ValueError("Nemotron Sionic summary is incomplete")
    raw_macro = require_number(sionic.get("average"), "Sionic average")
    scores = sionic.get("scores")
    if not isinstance(scores, dict) or len(scores) != 9:
        raise ValueError("Nemotron Sionic task scores are incomplete")

    legal = by_revision(args.legal_dir)
    multidomain = by_revision(args.multidomain_dir)
    legal_scores = {
        label: require_number(summary["metrics"].get("ndcg_at_10"), f"legal {label}")
        for label, (_, summary) in legal.items()
    }
    multi_scores = {
        label: require_number(
            summary["metrics"].get("macro_domain_ndcg_at_10"), f"multidomain {label}"
        )
        for label, (_, summary) in multidomain.items()
    }
    domains = ("finance", "knowledge")
    domain_scores = {
        label: {
            domain: require_number(
                summary["domain_metrics"][domain].get("ndcg_at_10"),
                f"multidomain {label} {domain}",
            )
            for domain in domains
        }
        for label, (_, summary) in multidomain.items()
    }
    references = ("qwen3", "comsat")
    legal_reference = max(legal_scores[label] for label in references)
    multi_reference = max(multi_scores[label] for label in references)
    domain_reference = {
        domain: max(domain_scores[label][domain] for label in references)
        for domain in domains
    }
    raw_pass = raw_macro > args.target
    legal_pass = legal_scores["nemotron3"] >= legal_reference - args.clean_guard
    multi_pass = multi_scores["nemotron3"] >= multi_reference - args.clean_guard
    domain_pass = {
        domain: domain_scores["nemotron3"][domain]
        >= domain_reference[domain] - args.domain_guard
        for domain in domains
    }
    clean_pass = legal_pass and multi_pass and all(domain_pass.values())
    deficit = args.target - raw_macro
    if raw_pass and clean_pass:
        decision = "adopt_nemotron3_raw_and_run_short_public_lora"
    elif raw_pass:
        decision = "nemotron3_teacher_only_due_to_clean_regression"
    elif clean_pass and 0 <= deficit <= args.max_short_adaptation_deficit:
        decision = "short_public_nemotron3_lora_then_retest"
    else:
        decision = "resume_qwen_checkpoint_1750_and_reselect"

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "decision": decision,
        "gates": {
            "raw_sionic_strictly_above_target": raw_pass,
            "legal_within_absolute_guard": legal_pass,
            "multidomain_macro_within_absolute_guard": multi_pass,
            "multidomain_each_domain_within_absolute_guard": domain_pass,
            "clean_guard_pass": clean_pass,
        },
        "thresholds": {
            "clean_absolute_ndcg": args.clean_guard,
            "domain_absolute_ndcg": args.domain_guard,
            "max_short_adaptation_deficit": args.max_short_adaptation_deficit,
        },
        "scores": {
            "sionic9": {
                "nemotron3_macro_ndcg_at_10": raw_macro,
                "target_delta": raw_macro - args.target,
                "tasks": scores,
                "upstream_train_family_exposure": ["MIRACL", "MLDR"],
            },
            "legal_ndcg_at_10": legal_scores,
            "multidomain_macro_ndcg_at_10": multi_scores,
            "multidomain_domain_ndcg_at_10": domain_scores,
            "clean_selector_mean": {
                label: fmean((legal_scores[label], multi_scores[label]))
                for label in REVISIONS
            },
        },
        "deltas_from_best_qwen_or_comsat": {
            "legal": legal_scores["nemotron3"] - legal_reference,
            "multidomain_macro": multi_scores["nemotron3"] - multi_reference,
            "multidomain_domains": {
                domain: domain_scores["nemotron3"][domain] - domain_reference[domain]
                for domain in domains
            },
        },
        "evidence": {
            "sionic": str(sionic_path.resolve()),
            "legal": {label: str(path.resolve()) for label, (path, _) in legal.items()},
            "multidomain": {
                label: str(path.resolve()) for label, (path, _) in multidomain.items()
            },
            "revisions": REVISIONS,
        },
    }


def main() -> None:
    args = parse_args()
    report = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
