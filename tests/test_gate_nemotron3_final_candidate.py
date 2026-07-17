from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from scripts.gate_nemotron3_final_candidate import build


def test_final_gate_requires_clean_guards_and_strict_sionic_win() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model = root / "model"
        model.mkdir()
        weights = "a" * 64
        (model / "merge_report.json").write_text(
            json.dumps({"model": {"weights_sha256": weights}})
        )
        revision = f"model-{weights[:12]}"
        decision = root / "decision.json"
        decision.write_text(
            json.dumps(
                {
                    "decision": "adopt_nemotron3_raw_and_run_short_public_lora",
                    "target": 0.793,
                    "thresholds": {
                        "clean_absolute_ndcg": 0.01,
                        "domain_absolute_ndcg": 0.015,
                    },
                    "scores": {
                        "legal_ndcg_at_10": {"qwen3": 0.8, "comsat": 0.81},
                        "multidomain_macro_ndcg_at_10": {"qwen3": 0.82, "comsat": 0.81},
                        "multidomain_domain_ndcg_at_10": {
                            "qwen3": {"finance": 0.8, "knowledge": 0.84},
                            "comsat": {"finance": 0.81, "knowledge": 0.83},
                        },
                    },
                }
            )
        )
        common = {"model": str(model), "requested_revision": revision}
        legal = root / "legal.json"
        legal.write_text(
            json.dumps(
                {
                    **common,
                    "protocol_id": "legal-source-document-heldout-i-v2-text-strict",
                    "metrics": {"ndcg_at_10": 0.805},
                }
            )
        )
        multi = root / "multi.json"
        multi.write_text(
            json.dumps(
                {
                    **common,
                    "protocol_id": "multidomain-selection-heldout-v1",
                    "metrics": {"macro_domain_ndcg_at_10": 0.815},
                    "domain_metrics": {
                        "finance": {"ndcg_at_10": 0.8},
                        "knowledge": {"ndcg_at_10": 0.83},
                    },
                }
            )
        )
        sionic = root / "sionic.json"
        sionic.write_text(
            json.dumps(
                {
                    **common,
                    "protocol_id": "sionic9-fixed-prompt-v1",
                    "completed_tasks": 9,
                    "scores": {str(index): 0.8 for index in range(9)},
                    "average": 0.8,
                }
            )
        )
        args = SimpleNamespace(
            base_decision=decision,
            legal_summary=legal,
            multidomain_summary=multi,
            sionic_summary=sionic,
            model_dir=model,
        )
        assert build(args)["status"] == "pass"
        payload = json.loads(sionic.read_text())
        payload["average"] = 0.793
        sionic.write_text(json.dumps(payload))
        assert build(args)["status"] == "fail"
