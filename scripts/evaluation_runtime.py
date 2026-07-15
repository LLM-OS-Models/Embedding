"""Shared, fail-closed runtime contracts for embedding evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def effective_attention(requested: str, evaluation_dtype: str) -> str:
    """Return an attention backend supported by the evaluation dtype."""

    if evaluation_dtype == "float32" and requested.startswith("flash_attention"):
        return "sdpa"
    return requested


def runtime_contract(
    *,
    protocol_id: str,
    protocol_path: Path,
    model: str,
    revision: str | None,
    batch_size: int,
    max_length: int,
    requested_attention: str,
    attention: str,
    evaluation_dtype: str,
    loader_contract: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    protocol_bytes = protocol_path.resolve().read_bytes()
    value: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "model": model,
        "revision": revision,
        "batch_size": batch_size,
        "max_length": max_length,
        "requested_attention": requested_attention,
        "attention": attention,
        "evaluation_dtype": evaluation_dtype,
        "loader_contract": loader_contract,
    }
    if extra:
        value["extra"] = extra
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    value["profile_id"] = f"eval-{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"
    return value


def _completed_results(run_dir: Path) -> list[Path]:
    result_root = run_dir / "mteb_cache" / "results"
    return sorted(result_root.rglob("*.json")) if result_root.is_dir() else []


def enforce_runtime_contract(run_dir: Path, contract: dict[str, Any]) -> Path:
    """Persist a runtime contract and reject mixed-profile completed results.

    A failed attempt may change batch size before any task result is complete. Once
    even one MTEB task JSON exists, callers must use a fresh output directory for a
    different profile so summaries can never combine heterogeneous inference runs.
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "runtime_contract.json"
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current == contract:
            return path
        completed = _completed_results(run_dir)
        if completed:
            raise RuntimeError(
                "Evaluation runtime changed after completed task results were cached; "
                f"use a fresh --output-dir (existing profile={current.get('profile_id')}, "
                f"requested profile={contract.get('profile_id')}, completed={len(completed)})"
            )

    descriptor = json.dumps(contract, ensure_ascii=False, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(dir=run_dir, prefix=".runtime-contract-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(descriptor)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return path
