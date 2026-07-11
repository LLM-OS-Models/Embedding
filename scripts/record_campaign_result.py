#!/usr/bin/env python3
"""Persist completed campaign summaries and refresh the generated README table."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = "<!-- CAMPAIGN_RESULTS_START -->"
END = "<!-- CAMPAIGN_RESULTS_END -->"
ORDER = {"pilot-best": 1, "scale-1m": 2, "legal-replay": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--sionic-summary", type=Path, required=True)
    parser.add_argument("--official-summary", type=Path, required=True)
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument(
        "--registry", type=Path, default=ROOT / "reports/campaign-results.json"
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def render(results: dict[str, dict]) -> str:
    lines = [
        "| Stage | Model | Sionic 9 Avg | Comsat 대비 | Official Mean(Task) | Mean(Type) | 공개 모델 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for stage, row in sorted(results.items(), key=lambda item: (ORDER.get(item[0], 99), item[0])):
        average = float(row["sionic_average"])
        repo = row["repo_id"]
        lines.append(
            f"| {stage} | `{row['model']}` | **{average:.5f}** | "
            f"{average - 0.793:+.5f} | {float(row['official_mean_task']):.3f} | "
            f"{float(row['official_mean_type']):.3f} | "
            f"[`{repo}`](https://huggingface.co/{repo}) |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    sionic = read_json(args.sionic_summary.resolve())
    official = read_json(args.official_summary.resolve())
    if sionic.get("completed_tasks") != 9 or sionic.get("average") is None:
        raise ValueError("Sionic summary is incomplete")
    if official.get("complete") is not True or official.get("completed_tasks") != 6:
        raise ValueError("Official Korean summary is incomplete")

    registry_path = args.registry.resolve()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = read_json(registry_path) if registry_path.is_file() else {"schema_version": 1, "results": {}}
    evidence_dir = registry_path.parent / "evidence" / args.stage
    evidence_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.sionic_summary.resolve(), evidence_dir / "sionic9_summary.json")
    shutil.copy2(args.official_summary.resolve(), evidence_dir / "mteb_korean_v1_summary.json")
    registry["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    registry["results"][args.stage] = {
        "model": args.model,
        "repo_id": args.repo_id,
        "sionic_average": sionic["average"],
        "official_mean_task": official["mean_task_leaderboard_points"],
        "official_mean_type": official["mean_task_type_leaderboard_points"],
        "sionic_evidence": display_path(evidence_dir / "sionic9_summary.json"),
        "official_evidence": display_path(evidence_dir / "mteb_korean_v1_summary.json"),
    }
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    readme_path = args.readme.resolve()
    readme = readme_path.read_text(encoding="utf-8")
    if readme.count(START) != 1 or readme.count(END) != 1:
        raise ValueError("README generated-result markers are missing or duplicated")
    prefix, remainder = readme.split(START, 1)
    _, suffix = remainder.split(END, 1)
    readme_path.write_text(
        prefix + START + "\n" + render(registry["results"]) + "\n" + END + suffix,
        encoding="utf-8",
    )
    print(json.dumps(registry["results"][args.stage], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
