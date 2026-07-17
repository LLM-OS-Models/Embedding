from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from scripts.model_lineage import lineage_from_local_model, resolve_base_lineage


QWEN = "Qwen/Qwen3-Embedding-8B"
QWEN_REV = "1" * 40
COMSAT = "sionic-ai/comsat-embed-ko-8b-preview"
COMSAT_REV = "a" * 40


def write_evidence(root: Path, value: dict) -> None:
    root.mkdir(parents=True)
    (root / "merge_report.json").write_text(
        json.dumps({"status": "pass", **value}), encoding="utf-8"
    )


def test_local_continual_model_inherits_pinned_hub_base(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    write_evidence(parent, {"base_model": COMSAT, "base_revision": COMSAT_REV})
    write_evidence(child, {"base_model": str(parent), "base_revision": ""})
    assert lineage_from_local_model(child) == [
        {"model": COMSAT, "revision": COMSAT_REV}
    ]


def test_mixed_lineage_is_deduplicated_without_losing_comsat(tmp_path: Path) -> None:
    model = tmp_path / "mixed"
    write_evidence(
        model,
        {
            "upstream_base_models": [
                {"model": QWEN, "revision": QWEN_REV},
                {"model": COMSAT, "revision": COMSAT_REV},
                {"model": QWEN, "revision": QWEN_REV},
            ]
        },
    )
    assert lineage_from_local_model(model) == [
        {"model": QWEN, "revision": QWEN_REV},
        {"model": COMSAT, "revision": COMSAT_REV},
    ]


def test_unpinned_or_ambiguous_lineage_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pinned commit"):
        resolve_base_lineage(QWEN, "main")
    ambiguous = tmp_path / "ambiguous"
    write_evidence(ambiguous, {"base_model": QWEN, "base_revision": QWEN_REV})
    (ambiguous / "full_tuning_report.json").write_text(
        json.dumps({"status": "pass"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly one"):
        lineage_from_local_model(ambiguous)


def test_legacy_soup_lineage_is_bound_to_source_evidence_hash(tmp_path: Path) -> None:
    source = tmp_path / "source"
    soup = tmp_path / "soup"
    write_evidence(source, {"base_model": QWEN, "base_revision": QWEN_REV})
    evidence_path = source / "merge_report.json"
    source_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    soup.mkdir()
    (soup / "soup_report.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "sources": [
                    {
                        "model": str(source),
                        "evidence_file": evidence_path.name,
                        "evidence_sha256": source_sha,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert lineage_from_local_model(soup) == [
        {"model": QWEN, "revision": QWEN_REV}
    ]
    payload = json.loads((soup / "soup_report.json").read_text())
    payload["sources"][0]["evidence_sha256"] = "0" * 64
    (soup / "soup_report.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence hash drifted"):
        lineage_from_local_model(soup)
