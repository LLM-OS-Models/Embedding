#!/usr/bin/env python3
"""Materialize the user's explicit public-release instruction for one exact winner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOLS = {
    "sionic9": "sionic9-fixed-prompt-v1",
    "official_korean_v1": "mteb-korean-v1-mteb-2.18.0",
    "comprehensive_text_v1": "comprehensive-korean-text-v1-mteb-2.18.0",
    "clean": "legal-source-document-heldout-i-v2-text-strict",
    "robustness": "legal-conversational-noise-i-v2-text-strict",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--final-gate", type=Path, required=True)
    parser.add_argument("--sionic-summary", type=Path, required=True)
    parser.add_argument("--official-summary", type=Path, required=True)
    parser.add_argument("--comprehensive-summary", type=Path, required=True)
    parser.add_argument("--clean-summary", type=Path, required=True)
    parser.add_argument("--robustness-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--approved-by",
        default="workspace-owner-explicit-public-release-instruction-2026-07-17",
    )
    return parser.parse_args()


def read_object(path: Path) -> dict[str, Any]:
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


def build(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.model_dir.resolve()
    merge_path = model_dir / "merge_report.json"
    paths = {
        "training": args.training_manifest.resolve(),
        "gate": args.final_gate.resolve(),
        "sionic9": args.sionic_summary.resolve(),
        "official_korean_v1": args.official_summary.resolve(),
        "comprehensive_text_v1": args.comprehensive_summary.resolve(),
        "clean": args.clean_summary.resolve(),
        "robustness": args.robustness_summary.resolve(),
    }
    for path in (merge_path, *paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    merge = read_object(merge_path)
    gate = read_object(paths["gate"])
    training = read_object(paths["training"])
    weights_sha = merge.get("model", {}).get("weights_sha256")
    if gate.get("status") != "pass" or gate.get("model", {}).get("weights_sha256") != weights_sha:
        raise ValueError("Final performance/clean gate did not approve these model weights")
    if training.get("training_track") != "rights-safe-release":
        raise ValueError("Training manifest is not on the rights-safe release track")
    if (
        training.get("release_eligible") is not True
        or training.get("release_blockers")
        or training.get("visibility") != "public"
    ):
        raise ValueError("Training rights do not approve public redistribution")
    evaluations: dict[str, Any] = {}
    for label, protocol_id in PROTOCOLS.items():
        summary = read_object(paths[label])
        if summary.get("protocol_id") != protocol_id:
            raise ValueError(f"Unexpected {label} protocol")
        evaluations[label] = {
            "status": "pass",
            "protocol_id": protocol_id,
            "summary_sha256": sha256(paths[label]),
        }
    binding = "\0".join(
        (args.repo_id, str(weights_sha), sha256(paths["training"]), sha256(paths["gate"]))
    )
    return {
        "schema_version": 1,
        "artifact_type": "embedding-model-public-release-approval",
        "approval_id": "nemotron3-public-" + hashlib.sha256(binding.encode()).hexdigest()[:16],
        "decision": "approved",
        "approved_by": args.approved_by,
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_basis": "User explicitly instructed that models and redistributable datasets be public on 2026-07-17.",
        "target": {"repo_id": args.repo_id, "visibility": "public"},
        "model": {"weights_sha256": weights_sha},
        "training": {
            "track": "rights-safe-release",
            "manifest_sha256": sha256(paths["training"]),
        },
        "rights_review": {
            "status": "approved",
            "release_eligible": True,
            "public_redistribution": True,
            "unresolved_blockers": [],
        },
        "evaluations": evaluations,
        "final_gate_sha256": sha256(paths["gate"]),
    }


def main() -> None:
    args = parse_args()
    report = build(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
