from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from scripts.approve_nemotron3_public_release import PROTOCOLS, build


def test_approval_binds_exact_rights_safe_winner_and_evaluations() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model = root / "model"
        model.mkdir()
        weights = "a" * 64
        (model / "merge_report.json").write_text(
            json.dumps({"model": {"weights_sha256": weights}})
        )
        training = root / "training.json"
        training.write_text(
            json.dumps(
                {
                    "training_track": "rights-safe-release",
                    "release_eligible": True,
                    "release_blockers": [],
                    "visibility": "public",
                }
            )
        )
        gate = root / "gate.json"
        gate.write_text(
            json.dumps({"status": "pass", "model": {"weights_sha256": weights}})
        )
        summaries = {}
        for label, protocol in PROTOCOLS.items():
            path = root / f"{label}.json"
            path.write_text(json.dumps({"protocol_id": protocol}))
            summaries[label] = path
        args = SimpleNamespace(
            model_dir=model,
            repo_id="org/public-model",
            training_manifest=training,
            final_gate=gate,
            sionic_summary=summaries["sionic9"],
            official_summary=summaries["official_korean_v1"],
            comprehensive_summary=summaries["comprehensive_text_v1"],
            clean_summary=summaries["clean"],
            robustness_summary=summaries["robustness"],
            approved_by="fixture-owner",
        )
        result = build(args)
        assert result["decision"] == "approved"
        assert result["target"]["visibility"] == "public"
        assert result["model"]["weights_sha256"] == weights
        assert set(result["evaluations"]) == set(PROTOCOLS)
