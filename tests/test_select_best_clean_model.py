from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.select_best_clean_model import (
    CLEAN_PROTOCOL_ID,
    EXPECTED_ROBUST_SCORE_CONTRACT,
    EXPECTED_SCORE_CONTRACT,
    ROBUSTNESS_PROTOCOL_ID,
    atomic_write_json,
    select_candidates,
)


QUERY_PROMPT = "fixed fixture query prompt"
MANIFEST_SHA = "a" * 64


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_ranks(folder: Path, *, robust: bool = False) -> dict:
    path = folder / "ranks.jsonl"
    row = '{"query_id":"q","conditions":{}}\n' if robust else '{"query_id":"q","positive_rank":1}\n'
    path.write_text(row * 10000, encoding="utf-8")
    return {"rows": 10000, "sha256": file_sha(path)}


def make_candidate(
    root: Path,
    name: str,
    *,
    clean_ndcg: float,
    robust_floor: float,
    intrusion: float,
) -> tuple[str, str]:
    model = f"artifacts/models/{name}-best-merged"
    weights_sha = hashlib.sha256(name.encode()).hexdigest()
    revision = f"model-{weights_sha[:12]}"
    model_dir = root / model
    model_dir.mkdir(parents=True)
    (model_dir / "merge_report.json").write_text(
        json.dumps({"model": {"weights_sha256": weights_sha}}), encoding="utf-8"
    )
    safe = model.replace("/", "__")
    clean_dir = root / "clean" / safe / revision
    robust_dir = root / "robust" / safe / revision
    clean_dir.mkdir(parents=True)
    robust_dir.mkdir(parents=True)
    common = {
        "model": model,
        "requested_revision": revision,
        "dataset": {
            "manifest_sha256": MANIFEST_SHA,
            "independence_grade": "I",
            "not_grade": "Z",
        },
        "query_prompt": QUERY_PROMPT,
        "environment": {
            "torch_dtype": "bfloat16",
            "max_length": 8192,
            "attention": "flash_attention_2",
        },
    }
    clean = {
        **common,
        "protocol_id": CLEAN_PROTOCOL_ID,
        "score_contract": EXPECTED_SCORE_CONTRACT,
        "metrics": {
            "ndcg_at_10": clean_ndcg,
            "recall_at_10": clean_ndcg,
            "mrr_at_10": clean_ndcg,
            "recall_at_100": 1.0,
        },
        "files": {"ranks.jsonl": write_ranks(clean_dir)},
    }
    conditions = {}
    for prompt in ("prompt_on", "prompt_off"):
        for ratio in ("0.00", "0.01", "0.05"):
            ndcg = clean_ndcg if prompt == "prompt_on" and ratio == "0.00" else robust_floor
            metrics = {
                "ndcg_at_10": ndcg,
                "recall_at_10": ndcg,
                "ndcg_retention_vs_same_prompt_clean": 1.0,
            }
            if ratio != "0.00":
                metrics["noise_intrusion_at_10"] = intrusion
            conditions[f"{prompt}/noise_{ratio}"] = metrics
    robust = {
        **common,
        "protocol_id": ROBUSTNESS_PROTOCOL_ID,
        "score_contract": EXPECTED_ROBUST_SCORE_CONTRACT,
        "conditions": conditions,
        "files": {"ranks.jsonl": write_ranks(robust_dir, robust=True)},
    }
    (clean_dir / "summary.json").write_text(json.dumps(clean), encoding="utf-8")
    (robust_dir / "summary.json").write_text(json.dumps(robust), encoding="utf-8")
    return model, revision


def select(root: Path) -> dict:
    return select_candidates(
        clean_root=root / "clean",
        robustness_root=root / "robust",
        workspace_root=root,
        disqualification_root=root / "outputs",
        clean_epsilon=0.002,
        robustness_epsilon=0.002,
        intrusion_epsilon=0.001,
    )


def test_clean_near_tie_uses_robustness_but_not_distant_candidate(tmp_path: Path) -> None:
    make_candidate(tmp_path, "clean-leader", clean_ndcg=0.900, robust_floor=0.800, intrusion=0.02)
    expected, _ = make_candidate(
        tmp_path, "near-tie-robust", clean_ndcg=0.899, robust_floor=0.850, intrusion=0.02
    )
    make_candidate(tmp_path, "distant", clean_ndcg=0.895, robust_floor=0.990, intrusion=0.0)
    report = select(tmp_path)
    assert report["best"]["model"] == expected
    assert report["public_benchmark_used_for_selection"] is False
    distant = next(row for row in report["ranking"] if "distant" in row["model"])
    assert distant["within_clean_near_tie"] is False


def test_tiny_robustness_difference_uses_noise_intrusion(tmp_path: Path) -> None:
    make_candidate(tmp_path, "high-intrusion", clean_ndcg=0.900, robust_floor=0.850, intrusion=0.10)
    expected, _ = make_candidate(
        tmp_path, "low-intrusion", clean_ndcg=0.899, robust_floor=0.849, intrusion=0.01
    )
    assert select(tmp_path)["best"]["model"] == expected


def test_tampered_ranks_and_disqualified_runs_are_excluded(tmp_path: Path) -> None:
    tampered_model, tampered_revision = make_candidate(
        tmp_path, "tampered", clean_ndcg=0.99, robust_floor=0.99, intrusion=0.0
    )
    tampered_safe = tampered_model.replace("/", "__")
    (tmp_path / "clean" / tampered_safe / tampered_revision / "ranks.jsonl").write_text(
        "tampered\n", encoding="utf-8"
    )
    make_candidate(tmp_path, "disqualified", clean_ndcg=0.98, robust_floor=0.98, intrusion=0.0)
    marker = tmp_path / "outputs" / "disqualified" / "DISQUALIFIED.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"reason":"fixture"}\n', encoding="utf-8")
    expected, _ = make_candidate(
        tmp_path, "valid", clean_ndcg=0.80, robust_floor=0.80, intrusion=0.01
    )
    report = select(tmp_path)
    assert report["best"]["model"] == expected
    reasons = "\n".join(row["reason"] for row in report["excluded"])
    assert "recorded SHA-256" in reasons
    assert "disqualification marker" in reasons


def test_mismatched_clean_reproduction_fails_closed(tmp_path: Path) -> None:
    model, revision = make_candidate(
        tmp_path, "mismatch", clean_ndcg=0.90, robust_floor=0.80, intrusion=0.01
    )
    safe = model.replace("/", "__")
    path = tmp_path / "robust" / safe / revision / "summary.json"
    payload = json.loads(path.read_text())
    payload["conditions"]["prompt_on/noise_0.00"]["ndcg_at_10"] = 0.89
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="No complete verified local candidate"):
        select(tmp_path)


def test_atomic_output(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "selection.json"
    atomic_write_json(output, {"ok": True})
    assert json.loads(output.read_text()) == {"ok": True}
    assert not list(output.parent.glob("*.tmp"))
