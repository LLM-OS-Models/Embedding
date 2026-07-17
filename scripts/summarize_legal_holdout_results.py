#!/usr/bin/env python3
"""Aggregate clean legal holdout summaries and refresh the README result block."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


START = "<!-- CLEAN_LEGAL_RESULTS_START -->"
END = "<!-- CLEAN_LEGAL_RESULTS_END -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readme", type=Path)
    parser.add_argument("--robustness-root", type=Path)
    return parser.parse_args()


def load_rows(root: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for path in root.rglob("summary.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("protocol_id") != "legal-source-document-heldout-i-v2-text-strict":
            continue
        model = str(value.get("model", ""))
        if not model or not isinstance(value.get("metrics"), dict):
            continue
        row = {**value, "summary_path": str(path.resolve())}
        if model not in latest or str(row.get("created_at_utc", "")) > str(
            latest[model].get("created_at_utc", "")
        ):
            latest[model] = row
    preferred = {
        "Qwen/Qwen3-Embedding-8B": 0,
        "sionic-ai/comsat-embed-ko-8b-preview": 1,
    }
    return sorted(
        latest.values(), key=lambda row: (preferred.get(row["model"], 2), row["model"])
    )


def load_robustness_rows(root: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return latest
    for path in root.rglob("summary.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("protocol_id") != "legal-conversational-noise-i-v2-text-strict":
            continue
        model = str(value.get("model", ""))
        conditions = value.get("conditions", {})
        if not model or not isinstance(conditions, dict):
            continue
        row = {**value, "summary_path": str(path.resolve())}
        if model not in latest or str(row.get("created_at_utc", "")) > str(
            latest[model].get("created_at_utc", "")
        ):
            latest[model] = row
    return latest


def markdown(
    rows: list[dict[str, Any]], robustness: dict[str, dict[str, Any]] | None = None
) -> str:
    if not rows:
        return "아직 완료된 clean 법률 baseline이 없습니다."
    robustness = robustness or {}
    lines = [
        "| Model | NDCG@10 | Recall@10 | MRR@10 | Recall@100 | Mean rank | Prompt-on 5% retention | Prompt-off 5% retention | Noise@10 on/off |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        robust = robustness.get(row["model"], {}).get("conditions", {})
        on = robust.get("prompt_on/noise_0.05")
        off = robust.get("prompt_off/noise_0.05")
        on_retention = (
            f"{float(on['ndcg_retention_vs_same_prompt_clean']):.5f}" if on else "—"
        )
        off_retention = (
            f"{float(off['ndcg_retention_vs_same_prompt_clean']):.5f}" if off else "—"
        )
        noise_intrusion = (
            f"{float(on['noise_intrusion_at_10']):.5f}/{float(off['noise_intrusion_at_10']):.5f}"
            if on and off
            else "—"
        )
        lines.append(
            f"| `{row['model']}` | {metrics['ndcg_at_10']:.5f} | "
            f"{metrics['recall_at_10']:.5f} | {metrics['mrr_at_10']:.5f} | "
            f"{metrics['recall_at_100']:.5f} | {metrics['mean_positive_rank']:.2f} | "
            f"{on_retention} | {off_retention} | {noise_intrusion} |"
        )
    lines.extend(
        [
            "",
            "10K same-repository whole-source-document-held-out(I-not-Z) 결과다. "
            "각 query의 source-native positive 하나만 qrel이므로 relevance는 exhaustive하지 않다.",
        ]
    )
    return "\n".join(lines)


def update_readme(path: Path, rendered: str) -> None:
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError("README clean legal result markers are missing")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    path.write_text(f"{before}{START}\n{rendered}\n{END}{after}", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.result_root.resolve())
    robustness = (
        load_robustness_rows(args.robustness_root.resolve())
        if args.robustness_root
        else {}
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": "legal-source-document-heldout-i-v2-text-strict",
        "models": rows,
        "robustness_protocol_id": "legal-conversational-noise-i-v2-text-strict",
        "robustness_models": list(robustness.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.readme:
        update_readme(args.readme, markdown(rows, robustness))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
