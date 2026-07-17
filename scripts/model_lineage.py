#!/usr/bin/env python3
"""Resolve exact Hub ancestry for derived embedding artifacts.

Runtime training may use an absolute local model path as its immediate base.
That path is neither portable nor suitable model-card metadata. Every derived
artifact therefore carries a recursively inherited list of pinned Hub bases in
``upstream_base_models``. The helper fails closed when a local parent has
ambiguous or unpinned evidence so provenance cannot silently collapse to the
default Qwen checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


EVIDENCE_NAMES = (
    "merge_report.json",
    "full_tuning_report.json",
    "soup_report.json",
)
UPSTREAM_FIELD = "upstream_base_models"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _normalize_rows(raw: Any, *, context: str) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{context} has no non-empty {UPSTREAM_FIELD}")
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"{context} lineage row {index} is not an object")
        model = row.get("model")
        revision = row.get("revision")
        if (
            not isinstance(model, str)
            or not model
            or model.startswith("/")
            or model.count("/") != 1
        ):
            raise ValueError(f"{context} lineage row {index} has no Hub model ID")
        if not isinstance(revision, str) or not COMMIT_RE.fullmatch(revision):
            raise ValueError(f"{context} lineage row {index} has no pinned commit")
        key = (model, revision)
        if key not in seen:
            normalized.append({"model": model, "revision": revision})
            seen.add(key)
    return normalized


def unique_evidence_path(model_dir: Path) -> Path:
    present = [
        model_dir / name for name in EVIDENCE_NAMES if (model_dir / name).is_file()
    ]
    if len(present) != 1:
        raise ValueError(
            f"Local base must contain exactly one model evidence report: {model_dir}"
        )
    return present[0]


def _read_evidence(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "pass":
        raise ValueError(f"Local base evidence is not a passing object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lineage_from_evidence(
    evidence: dict[str, Any],
    *,
    evidence_dir: Path | None = None,
    context: str = "model evidence",
    _seen_dirs: set[Path] | None = None,
) -> list[dict[str, str]]:
    """Return pinned Hub bases, recursively upgrading legacy local evidence."""

    if UPSTREAM_FIELD in evidence:
        return _normalize_rows(evidence[UPSTREAM_FIELD], context=context)

    base_model = evidence.get("base_model")
    base_revision = evidence.get("base_revision")
    if isinstance(base_model, str) and base_model:
        candidate = Path(base_model).expanduser()
        if candidate.is_absolute():
            return lineage_from_local_model(candidate, _seen_dirs=_seen_dirs)
        return _normalize_rows(
            [{"model": base_model, "revision": base_revision}], context=context
        )

    # Old soup reports can be upgraded only while their exact local sources
    # still exist. New reports store their inherited lineages directly.
    sources = evidence.get("sources")
    if isinstance(sources, list) and sources:
        inherited: list[dict[str, str]] = []
        for index, source in enumerate(sources):
            if not isinstance(source, dict) or not isinstance(source.get("model"), str):
                raise ValueError(f"{context} soup source {index} is malformed")
            source_dir = Path(source["model"]).expanduser()
            if not source_dir.is_absolute() and evidence_dir is not None:
                source_dir = evidence_dir / source_dir
            source_dir = source_dir.resolve()
            source_evidence = unique_evidence_path(source_dir)
            declared_file = source.get("evidence_file")
            declared_sha = source.get("evidence_sha256")
            if declared_file is not None and declared_file != source_evidence.name:
                raise ValueError(f"{context} soup source {index} evidence file drifted")
            if declared_sha is not None and declared_sha != _sha256_file(source_evidence):
                raise ValueError(f"{context} soup source {index} evidence hash drifted")
            inherited.extend(lineage_from_local_model(source_dir, _seen_dirs=_seen_dirs))
        return _normalize_rows(inherited, context=f"{context} inherited soup lineage")

    raise ValueError(f"{context} has no resolvable pinned Hub ancestry")


def lineage_from_local_model(
    model_dir: Path, *, _seen_dirs: set[Path] | None = None
) -> list[dict[str, str]]:
    resolved = model_dir.expanduser().resolve()
    seen = set() if _seen_dirs is None else set(_seen_dirs)
    if resolved in seen:
        raise ValueError(f"Cycle in local model lineage: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"Local base model is unavailable: {resolved}")
    seen.add(resolved)
    evidence_path = unique_evidence_path(resolved)
    return lineage_from_evidence(
        _read_evidence(evidence_path),
        evidence_dir=resolved,
        context=str(evidence_path),
        _seen_dirs=seen,
    )


def resolve_base_lineage(base_model: str, base_revision: str) -> list[dict[str, str]]:
    candidate = Path(base_model).expanduser()
    if candidate.is_absolute():
        return lineage_from_local_model(candidate)
    return _normalize_rows(
        [{"model": base_model, "revision": base_revision}], context="requested base"
    )


def merge_lineages(*groups: list[dict[str, str]]) -> list[dict[str, str]]:
    return _normalize_rows(
        [row for group in groups for row in group], context="merged model lineage"
    )
