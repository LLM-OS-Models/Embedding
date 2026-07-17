#!/usr/bin/env python3
"""Run the pinned, text-only comprehensive Korean embedding diagnostics offline.

This runner deliberately excludes unregistered K-HATERS and every visual-document
asset.  It validates the MTEB package, checkout, complete task metadata, selected
subsets, dataset revisions, and repository-local snapshots before model loading.
Network access is disabled in-process and no Hugging Face credential is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

try:
    from evaluation_runtime import (
        effective_attention,
        enforce_runtime_contract,
        runtime_contract,
    )
except ModuleNotFoundError:
    from scripts.evaluation_runtime import (
        effective_attention,
        enforce_runtime_contract,
        runtime_contract,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "comprehensive_text_v1_protocol.json"
ASSET_MANIFEST = ROOT / "configs" / "comprehensive_eval_assets.json"
HF_HOME = ROOT / ".cache" / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"
HF_DATASETS_CACHE = HF_HOME / "datasets"

FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REPO_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

EXPECTED_TASKS = (
    "XPQARetrieval",
    "FloresBitextMining",
    "KorSarcasmClassification.v2",
    "KorHateClassification.v2",
    "KorFin",
    "KorHateSpeechMLClassification",
    "KorNLI",
)
EXPECTED_ASSET_KEYS = (
    "xpqa",
    "flores_bitext",
    "kor_sarcasm",
    "kor_hate",
    "korfin_asc",
    "kor_hate_speech_ml",
    "kor_nli",
)
EXPECTED_EXCLUSIONS = {
    "k_haters",
    "sds_kopub_vdr_t2it",
    "kovidore_v2_cybersecurity",
    "kovidore_v2_economic",
    "kovidore_v2_energy",
    "kovidore_v2_hr",
}
EXPECTED_LOADER_CONTRACT = {
    "allowed_modes": [
        "sentence-transformer",
        "registered-task-instruction",
        "qwen3-task-instruction",
    ],
    "padding_side": "left",
    "normalize_embeddings": True,
    "similarity": "model_declared_cosine_compatible",
    "resume": "mteb_result_cache_only_missing_with_optional_exact_embedding_cache",
}
EXPECTED_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


class ProtocolError(RuntimeError):
    """The committed evaluation protocol or installed registry has drifted."""


class OfflineAssetError(RuntimeError):
    """A required immutable local dataset/model snapshot is unavailable."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline comprehensive Korean text-only MTEB diagnostics"
    )
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation/comprehensive_text_v1"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    loader_group = parser.add_mutually_exclusive_group()
    loader_group.add_argument("--registered-loader", action="store_true")
    loader_group.add_argument("--qwen3-instruction-loader", action="store_true")
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        help="Exact float32 embedding chunks used to resume interrupted encoding",
    )
    parser.add_argument(
        "--task",
        action="append",
        help="Task name or asset key; repeat for resumable partial execution",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Validate the offline assets and task registry without loading a model",
    )
    return parser.parse_args(argv)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Could not read {label} ({type(exc).__name__})") from None
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ordered_sequence_sha256(values: Sequence[str]) -> str:
    encoded = json.dumps(
        list(values), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(path, "comprehensive text protocol")
    if protocol.get("schema_version") != "1.0.0":
        raise ProtocolError("Unsupported comprehensive text protocol schema")
    if protocol.get("protocol_id") != "comprehensive-korean-text-v1-mteb-2.18.0":
        raise ProtocolError("Protocol ID does not identify the canonical text-only suite")
    if protocol.get("scope") != "offline_text_only_diagnostic_evaluation":
        raise ProtocolError("Protocol scope must remain offline and text-only")
    if protocol.get("mteb_version") != "2.18.0":
        raise ProtocolError("Protocol must pin MTEB 2.18.0")
    if protocol.get("seed") != 42:
        raise ProtocolError("Protocol must pin MTEB task seed 42")
    if not FULL_SHA_RE.fullmatch(str(protocol.get("mteb_git_revision", ""))):
        raise ProtocolError("Protocol must pin a full MTEB git revision")
    if protocol.get("mteb_module_relative_path") != "third_party/mteb/mteb/__init__.py":
        raise ProtocolError("MTEB module must resolve from the pinned local checkout")
    if protocol.get("loader_contract") != EXPECTED_LOADER_CONTRACT:
        raise ProtocolError("Comprehensive evaluator loader contract changed")

    offline = protocol.get("offline_contract")
    if not isinstance(offline, dict):
        raise ProtocolError("Offline contract is missing")
    if offline != {
        "local_files_only": True,
        "repository_local_hf_home": ".cache/huggingface",
        "environment": EXPECTED_OFFLINE_ENVIRONMENT,
    }:
        raise ProtocolError("Offline/local-only contract changed")

    manifest = protocol.get("asset_manifest")
    if not isinstance(manifest, dict) or manifest != {
        "path": "configs/comprehensive_eval_assets.json",
        "sha256": manifest.get("sha256") if isinstance(manifest, dict) else None,
        "cache_relative_path": ".cache/huggingface/hub",
    }:
        raise ProtocolError("Asset manifest path or cache contract changed")
    if not SHA256_RE.fullmatch(str(manifest.get("sha256", ""))):
        raise ProtocolError("Asset manifest must be pinned by SHA256")

    tasks = protocol.get("tasks")
    if not isinstance(tasks, list) or tuple(spec.get("name") for spec in tasks) != EXPECTED_TASKS:
        raise ProtocolError("Protocol must contain the exact seven supported text tasks")
    if tuple(spec.get("asset_key") for spec in tasks) != EXPECTED_ASSET_KEYS:
        raise ProtocolError("Protocol task-to-asset mapping changed")

    for spec in tasks:
        name = str(spec.get("name"))
        required_strings = (
            "asset_key",
            "task_class",
            "type",
            "split",
            "category",
            "main_score",
            "license",
            "instruction_fallback",
            "metadata_sha256",
            "contamination_grade",
            "claim_policy",
        )
        for field in required_strings:
            if not isinstance(spec.get(field), str) or not spec[field]:
                raise ProtocolError(f"Task {name} has invalid field {field}")
        if spec.get("modalities") != ["text"]:
            raise ProtocolError(f"Task {name} is not explicitly text-only")
        if spec.get("task_prompt") is not None:
            raise ProtocolError(f"Task {name} task prompt contract changed")
        if spec.get("available_splits") != [spec.get("split")]:
            raise ProtocolError(f"Task {name} split contract changed")
        dataset = spec.get("dataset")
        if not isinstance(dataset, dict) or not REPO_ID_RE.fullmatch(
            str(dataset.get("path", ""))
        ):
            raise ProtocolError(f"Task {name} has an invalid dataset repository")
        if not FULL_SHA_RE.fullmatch(str(dataset.get("revision", ""))):
            raise ProtocolError(f"Task {name} dataset revision is not immutable")
        if not SHA256_RE.fullmatch(spec["metadata_sha256"]):
            raise ProtocolError(f"Task {name} metadata digest is invalid")

        registry = spec.get("registry_hf_subsets")
        selection = spec.get("hf_subset_selection")
        for label, value in (("registry", registry), ("selection", selection)):
            if not isinstance(value, dict):
                raise ProtocolError(f"Task {name} {label} subset contract is missing")
            if not isinstance(value.get("count"), int) or value["count"] < 1:
                raise ProtocolError(f"Task {name} {label} subset count is invalid")
            if not SHA256_RE.fullmatch(str(value.get("ordered_sha256", ""))):
                raise ProtocolError(f"Task {name} {label} subset digest is invalid")
        mode = selection.get("mode")
        if mode == "exact":
            values = selection.get("values")
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value for value in values
            ):
                raise ProtocolError(f"Task {name} exact subset values are invalid")
            if len(values) != selection["count"]:
                raise ProtocolError(f"Task {name} exact subset count is inconsistent")
        elif mode == "exact_component":
            if selection.get("separator") != "-" or not isinstance(
                selection.get("component"), str
            ):
                raise ProtocolError(f"Task {name} component selector is invalid")
        else:
            raise ProtocolError(f"Task {name} has unsupported subset selection mode")

    exclusions = protocol.get("explicit_exclusions")
    if not isinstance(exclusions, list) or {
        item.get("asset_key") for item in exclusions if isinstance(item, dict)
    } != EXPECTED_EXCLUSIONS:
        raise ProtocolError("K-HATERS and visual-document exclusions must stay explicit")
    if any(
        not isinstance(item.get("reason"), str) or not item["reason"]
        for item in exclusions
    ):
        raise ProtocolError("Every excluded asset must include a reason")
    return protocol


def validate_asset_manifest(
    protocol: Mapping[str, Any], path: Path = ASSET_MANIFEST
) -> dict[str, Any]:
    expected = protocol["asset_manifest"]
    if file_sha256(path) != expected["sha256"]:
        raise ProtocolError("Comprehensive asset manifest SHA256 mismatch")
    manifest = _load_json(path, "comprehensive asset manifest")
    if manifest.get("schema_version") != "1.0.0":
        raise ProtocolError("Unsupported comprehensive asset manifest schema")
    if manifest.get("cache_relative_path") != expected["cache_relative_path"]:
        raise ProtocolError("Asset manifest no longer uses the repository-local cache")
    records = manifest.get("assets")
    if not isinstance(records, list):
        raise ProtocolError("Comprehensive asset manifest has no asset records")
    by_key = {
        record.get("key"): record for record in records if isinstance(record, dict)
    }
    accounted = set(EXPECTED_ASSET_KEYS) | EXPECTED_EXCLUSIONS
    if set(by_key) != accounted:
        raise ProtocolError("Text runner does not account for every comprehensive asset")

    task_assets: list[dict[str, Any]] = []
    for spec in protocol["tasks"]:
        record = by_key[spec["asset_key"]]
        if record.get("repo_id") != spec["dataset"]["path"]:
            raise ProtocolError(f"Asset repository drifted for {spec['asset_key']}")
        if record.get("revision") != spec["dataset"]["revision"]:
            raise ProtocolError(f"Asset revision drifted for {spec['asset_key']}")
        if record.get("download_tier") != "small":
            raise ProtocolError(f"Text asset is not a complete small snapshot: {spec['asset_key']}")
        if record.get("contamination_grade") != spec["contamination_grade"]:
            raise ProtocolError(f"Contamination grade drifted for {spec['asset_key']}")
        task_assets.append(
            {
                "asset_key": spec["asset_key"],
                "repo_id": record["repo_id"],
                "revision": record["revision"],
                "license": record["license"],
                "usage_policy": record["usage_policy"],
                "contamination_grade": record["contamination_grade"],
            }
        )

    for excluded in protocol["explicit_exclusions"]:
        if excluded["asset_key"] not in by_key:
            raise ProtocolError(f"Excluded asset is absent: {excluded['asset_key']}")
    return {
        "manifest_id": manifest.get("manifest_id"),
        "manifest_sha256": expected["sha256"],
        "assets": task_assets,
    }


def enforce_offline_environment(env: dict[str, str] | os._Environ[str] = os.environ) -> dict[str, str]:
    """Force all Hugging Face loaders offline and keep credentials out of the run."""

    for key, value in EXPECTED_OFFLINE_ENVIRONMENT.items():
        env[key] = value
    env["HF_HOME"] = str(HF_HOME)
    env["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
    env["HF_DATASETS_CACHE"] = str(HF_DATASETS_CACHE)
    for secret_name in (
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "GITHUB",
        "GITHUB_TOKEN",
    ):
        env.pop(secret_name, None)
    return {
        **EXPECTED_OFFLINE_ENVIRONMENT,
        "HF_HOME": ".cache/huggingface",
        "HF_HUB_CACHE": ".cache/huggingface/hub",
        "HF_DATASETS_CACHE": ".cache/huggingface/datasets",
        "credentials_available_to_runner": "none",
    }


SnapshotResolver = Callable[..., str]


def _snapshot_download() -> SnapshotResolver:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise OfflineAssetError("huggingface_hub is unavailable") from None
    return snapshot_download


def verify_local_asset_snapshots(
    protocol: Mapping[str, Any],
    *,
    cache_dir: Path = HF_HUB_CACHE,
    resolver: SnapshotResolver | None = None,
) -> list[dict[str, Any]]:
    if resolver is None:
        resolver = _snapshot_download()
    verified: list[dict[str, Any]] = []
    for spec in protocol["tasks"]:
        repo_id = spec["dataset"]["path"]
        revision = spec["dataset"]["revision"]
        try:
            snapshot = Path(
                resolver(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=revision,
                    cache_dir=cache_dir,
                    local_files_only=True,
                    token=False,
                )
            )
        except Exception as exc:
            raise OfflineAssetError(
                f"Pinned local snapshot unavailable for {spec['asset_key']} "
                f"({type(exc).__name__})"
            ) from None
        try:
            exists = snapshot.is_dir()
            resolved_revision = snapshot.resolve().name
        except OSError as exc:
            raise OfflineAssetError(
                f"Could not inspect local snapshot for {spec['asset_key']} "
                f"({type(exc).__name__})"
            ) from None
        if not exists or resolved_revision != revision:
            raise OfflineAssetError(
                f"Local snapshot revision mismatch for {spec['asset_key']}"
            )
        verified.append(
            {
                "asset_key": spec["asset_key"],
                "repo_id": repo_id,
                "revision": revision,
                "local_files_only": True,
            }
        )
    return verified


def validate_mteb_runtime(mteb: Any, protocol: Mapping[str, Any]) -> str:
    if getattr(mteb, "__version__", None) != protocol["mteb_version"]:
        raise ProtocolError(
            f"MTEB version mismatch: expected {protocol['mteb_version']}, "
            f"found {getattr(mteb, '__version__', None)}"
        )
    expected_module = (ROOT / protocol["mteb_module_relative_path"]).resolve()
    module_path = Path(str(getattr(mteb, "__file__", ""))).resolve()
    if module_path != expected_module:
        raise ProtocolError("Imported MTEB package is not the pinned local checkout")
    checkout = ROOT / "third_party" / "mteb"
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ProtocolError(f"Cannot inspect pinned MTEB checkout ({type(exc).__name__})") from None
    if revision != protocol["mteb_git_revision"]:
        raise ProtocolError(
            f"MTEB git mismatch: expected {protocol['mteb_git_revision']}, found {revision}"
        )
    return revision


def _metadata_json(metadata: Any) -> dict[str, Any]:
    if not hasattr(metadata, "model_dump"):
        raise ProtocolError("MTEB task metadata does not support canonical serialization")
    value = metadata.model_dump(mode="json")
    if not isinstance(value, dict):
        raise ProtocolError("MTEB task metadata did not serialize to an object")
    return value


def _task_class_name(task: Any) -> str:
    cls = type(task)
    return f"{cls.__module__}.{cls.__qualname__}"


def _normalize_dataset(dataset: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: dataset[key]
        for key in ("path", "name", "revision")
        if key in dataset
    }


def selected_hf_subsets(spec: Mapping[str, Any], registry: Sequence[str]) -> list[str]:
    selection = spec["hf_subset_selection"]
    if selection["mode"] == "exact":
        selected = list(selection["values"])
        if any(subset not in registry for subset in selected):
            raise ProtocolError(f"Selected subset disappeared for {spec['name']}")
    else:
        separator = selection["separator"]
        component = selection["component"]
        selected = [
            subset for subset in registry if component in subset.split(separator)
        ]
    if len(selected) != selection["count"]:
        raise ProtocolError(f"Selected subset count drifted for {spec['name']}")
    if ordered_sequence_sha256(selected) != selection["ordered_sha256"]:
        raise ProtocolError(f"Selected subset order/content drifted for {spec['name']}")
    return selected


def _resolve_selectors(
    specs: Sequence[Mapping[str, Any]], selectors: Sequence[str] | None
) -> set[str]:
    if not selectors:
        return {str(spec["name"]) for spec in specs}
    mapping = {
        selector: str(spec["name"])
        for spec in specs
        for selector in (str(spec["name"]), str(spec["asset_key"]))
    }
    missing = set(selectors) - set(mapping)
    if missing:
        raise ProtocolError(f"Unknown task selectors: {sorted(missing)}")
    return {mapping[selector] for selector in selectors}


def resolve_and_validate_tasks(
    mteb: Any,
    protocol: Mapping[str, Any],
    selectors: Sequence[str] | None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    selected_names = _resolve_selectors(protocol["tasks"], selectors)
    selected_tasks: list[Any] = []
    resolved: list[dict[str, Any]] = []
    for spec in protocol["tasks"]:
        full_task = mteb.get_task(spec["name"])
        metadata = full_task.metadata
        registry_subsets = list(full_task.hf_subsets)
        registry_contract = spec["registry_hf_subsets"]
        if len(registry_subsets) != registry_contract["count"]:
            raise ProtocolError(f"Registry subset count drifted for {spec['name']}")
        if ordered_sequence_sha256(registry_subsets) != registry_contract["ordered_sha256"]:
            raise ProtocolError(f"Registry subset order/content drifted for {spec['name']}")

        metadata_digest = canonical_sha256(_metadata_json(metadata))
        actual = {
            "name": metadata.name,
            "task_class": _task_class_name(full_task),
            "type": metadata.type,
            "available_splits": list(metadata.eval_splits),
            "dataset": _normalize_dataset(dict(metadata.dataset)),
            "modalities": list(metadata.modalities),
            "category": metadata.category,
            "main_score": metadata.main_score,
            "license": metadata.license,
            "task_prompt": metadata.prompt,
            "instruction_fallback": full_task.abstask_prompt,
            "metadata_sha256": metadata_digest,
            "evaluation_seed": full_task.seed,
        }
        expected = {
            key: spec[key]
            for key in (
                "name",
                "task_class",
                "type",
                "available_splits",
                "dataset",
                "modalities",
                "category",
                "main_score",
                "license",
                "task_prompt",
                "instruction_fallback",
                "metadata_sha256",
            )
        }
        expected["evaluation_seed"] = protocol["seed"]
        if actual != expected:
            raise ProtocolError(
                f"Installed MTEB metadata drifted for {spec['name']}: "
                f"expected={expected!r}, resolved={actual!r}"
            )
        subsets = selected_hf_subsets(spec, registry_subsets)
        subset_language_map = metadata.hf_subsets_to_langscripts
        try:
            selected_subset_languages = {
                subset: list(subset_language_map[subset]) for subset in subsets
            }
        except (KeyError, TypeError) as exc:
            raise ProtocolError(
                f"MTEB subset-language mapping drifted for {spec['name']} "
                f"({type(exc).__name__})"
            ) from None
        is_selected = spec["name"] in selected_names
        resolved.append(
            {
                "asset_key": spec["asset_key"],
                **actual,
                "registry_hf_subset_count": len(registry_subsets),
                "registry_hf_subsets_ordered_sha256": ordered_sequence_sha256(
                    registry_subsets
                ),
                "selected_split": spec["split"],
                "selected_hf_subsets": subsets,
                "selected_hf_subsets_ordered_sha256": ordered_sequence_sha256(subsets),
                "selected_hf_subset_languages": selected_subset_languages,
                "selected": is_selected,
                "contamination_grade": spec["contamination_grade"],
                "claim_policy": spec["claim_policy"],
            }
        )
        if not is_selected:
            continue
        task = mteb.get_task(
            spec["name"], eval_splits=[spec["split"]], hf_subsets=subsets
        )
        if list(task.eval_splits) != [spec["split"]] or list(task.hf_subsets) != subsets:
            raise ProtocolError(f"MTEB selector contract drifted for {spec['name']}")
        if _task_class_name(task) != spec["task_class"]:
            raise ProtocolError(f"Selected task class drifted for {spec['name']}")
        if canonical_sha256(_metadata_json(task.metadata)) != spec["metadata_sha256"]:
            raise ProtocolError(f"Selected task metadata drifted for {spec['name']}")
        selected_tasks.append(task)
    return selected_tasks, resolved


def build_resolved_protocol(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    asset_evidence: Mapping[str, Any],
    snapshot_evidence: Sequence[Mapping[str, Any]],
    resolved_tasks: Sequence[Mapping[str, Any]],
    checkout_revision: str,
    offline_environment: Mapping[str, str],
) -> dict[str, Any]:
    task_contract = {
        "protocol_id": protocol["protocol_id"],
        "mteb_version": protocol["mteb_version"],
        "mteb_git_revision": checkout_revision,
        "asset_manifest_sha256": asset_evidence["manifest_sha256"],
        "loader_contract": protocol["loader_contract"],
        "offline_contract": protocol["offline_contract"],
        "tasks": [
            {key: value for key, value in row.items() if key != "selected"}
            for row in resolved_tasks
        ],
        "explicit_exclusions": protocol["explicit_exclusions"],
    }
    try:
        relative_protocol = str(protocol_path.resolve().relative_to(ROOT))
    except ValueError:
        relative_protocol = protocol_path.name
    return {
        "schema_version": "1.0.0",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative_protocol,
        "protocol_sha256": file_sha256(protocol_path),
        "resolved_task_contract_sha256": canonical_sha256(task_contract),
        "validated_environment": {
            "mteb_version": protocol["mteb_version"],
            "mteb_git_revision": checkout_revision,
            "mteb_module_relative_path": protocol["mteb_module_relative_path"],
            "offline": dict(offline_environment),
        },
        "asset_manifest": dict(asset_evidence),
        "verified_local_snapshots": list(snapshot_evidence),
        "resolved_tasks": list(resolved_tasks),
        "explicit_exclusions": protocol["explicit_exclusions"],
        "claim_policy": protocol["aggregate"]["claim_policy"],
    }


def _result_json_files(run_dir: Path) -> list[Path]:
    root = run_dir / "mteb_cache" / "results"
    return sorted(root.rglob("*.json")) if root.is_dir() else []


def validate_existing_result_contract(
    run_dir: Path, resolved: Mapping[str, Any]
) -> None:
    task_files = [path for path in _result_json_files(run_dir) if path.name != "model_meta.json"]
    if not task_files:
        return
    runtime_path = run_dir / "runtime_contract.json"
    resolved_path = run_dir / "protocol_resolved.json"
    if not runtime_path.is_file() or not resolved_path.is_file():
        raise ProtocolError(
            "Existing comprehensive results lack runtime/protocol evidence; "
            "use a fresh --output-dir"
        )
    existing = _load_json(resolved_path, "existing resolved comprehensive protocol")
    if existing.get("resolved_task_contract_sha256") != resolved[
        "resolved_task_contract_sha256"
    ]:
        raise ProtocolError(
            "Existing comprehensive results use a different task contract; "
            "use a fresh --output-dir"
        )


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> str:
    content = (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    atomic_write(path, content)
    atomic_write(
        path.with_name(path.name + ".sha256"),
        f"{digest}  {path.name}\n".encode("ascii"),
    )
    return digest


def _local_model_revision(model_path: Path, requested: str | None) -> tuple[str, dict[str, Any]]:
    evidence_path = None
    evidence: dict[str, Any] | None = None
    for filename in ("merge_report.json", "full_tuning_report.json", "soup_report.json"):
        candidate = model_path / filename
        if candidate.is_file():
            evidence_path = candidate
            evidence = _load_json(candidate, "local model evidence")
            break
    if evidence is None or evidence_path is None:
        raise OfflineAssetError(
            "Local model requires merge_report.json, full_tuning_report.json, or "
            "soup_report.json weight evidence"
        )
    weights_sha = evidence.get("model", {}).get("weights_sha256")
    if not isinstance(weights_sha, str) or not SHA256_RE.fullmatch(weights_sha):
        raise OfflineAssetError("Local model weight SHA256 evidence is invalid")
    revision = f"model-{weights_sha[:12]}"
    if requested is not None and requested != revision:
        raise OfflineAssetError(
            f"Local model revision mismatch: expected {revision}, requested {requested}"
        )
    return revision, {
        "source": "local_reviewed_model_package",
        "weights_sha256": weights_sha,
        "evidence_file": evidence_path.name,
        "evidence_file_sha256": file_sha256(evidence_path),
    }


def resolve_offline_model(
    model: str,
    revision: str | None,
    *,
    resolver: SnapshotResolver | None = None,
) -> tuple[str, dict[str, Any]]:
    local = Path(model).expanduser()
    if local.is_dir():
        return _local_model_revision(local, revision)
    if not REPO_ID_RE.fullmatch(model):
        raise OfflineAssetError("Model must be a local reviewed package or Hugging Face repo ID")
    if revision is None or not FULL_SHA_RE.fullmatch(revision):
        raise OfflineAssetError("Hub model --revision must be a full immutable commit SHA")
    if resolver is None:
        resolver = _snapshot_download()
    try:
        snapshot = Path(
            resolver(
                repo_id=model,
                repo_type="model",
                revision=revision,
                cache_dir=HF_HUB_CACHE,
                local_files_only=True,
                token=False,
            )
        )
    except Exception as exc:
        raise OfflineAssetError(
            f"Pinned local model snapshot unavailable ({type(exc).__name__})"
        ) from None
    if not snapshot.is_dir() or snapshot.resolve().name != revision:
        raise OfflineAssetError("Local model snapshot revision mismatch")
    return revision, {
        "source": "repository_local_huggingface_snapshot",
        "repo_id": model,
        "revision": revision,
        "local_files_only": True,
    }


def local_merge_dtype(model: str) -> str:
    report = Path(model).expanduser() / "merge_report.json"
    if report.is_file():
        evidence = _load_json(report, "local merge report")
        if evidence.get("merge", {}).get("dtype") == "float32":
            return "float32"
    return "bfloat16"


def _safe_model_name(model: str) -> str:
    local = Path(model).expanduser()
    raw = local.name if local.is_dir() else model.replace("/", "__")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    if not safe:
        raise ValueError("Model name does not produce a safe output directory")
    return safe


def public_model_reference(model: str) -> str:
    """Return provenance that never exposes an absolute shared-machine path."""

    local = Path(model).expanduser()
    return f"local:{local.name}" if local.is_dir() else model


def _load_raw_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Unreadable cached MTEB result ({type(exc).__name__})") from None
    if not isinstance(value, dict):
        raise ProtocolError("Cached MTEB result must be a JSON object")
    return value


def _finite_score(value: Any, *, task: str, subset: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"Non-numeric {field} for {task}/{subset}")
    score = float(value)
    if not math.isfinite(score):
        raise ProtocolError(f"Non-finite {field} for {task}/{subset}")
    return score


def normalize_cached_results(
    run_dir: Path,
    protocol: Mapping[str, Any],
    resolved_tasks: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = {spec["name"]: spec for spec in protocol["tasks"]}
    resolved_by_name = {row["name"]: row for row in resolved_tasks}
    if set(resolved_by_name) != set(specs):
        raise ProtocolError("Resolved task evidence does not cover the complete protocol")
    raw_by_task: dict[str, tuple[Path, dict[str, Any]]] = {}
    inventory: list[dict[str, Any]] = []
    for path in _result_json_files(run_dir):
        relative = str(path.relative_to(run_dir))
        inventory.append(
            {
                "path": relative,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
        if path.name == "model_meta.json":
            continue
        payload = _load_raw_json(path)
        task_name = payload.get("task_name")
        if task_name not in specs:
            raise ProtocolError(f"Unexpected task result in comprehensive cache: {task_name!r}")
        if task_name in raw_by_task:
            raise ProtocolError(f"Duplicate cached result for {task_name}")
        raw_by_task[task_name] = (path, payload)

    normalized: list[dict[str, Any]] = []
    for spec in protocol["tasks"]:
        if spec["name"] not in raw_by_task:
            continue
        _, payload = raw_by_task[spec["name"]]
        if payload.get("dataset_revision") != spec["dataset"]["revision"]:
            raise ProtocolError(f"Cached dataset revision drifted for {spec['name']}")
        if payload.get("mteb_version") != protocol["mteb_version"]:
            raise ProtocolError(f"Cached MTEB version drifted for {spec['name']}")
        scores = payload.get("scores")
        if not isinstance(scores, dict) or set(scores) != {spec["split"]}:
            raise ProtocolError(f"Cached split contract drifted for {spec['name']}")
        raw_rows = scores[spec["split"]]
        if not isinstance(raw_rows, list):
            raise ProtocolError(f"Cached scores are malformed for {spec['name']}")
        rows_by_subset: dict[str, Mapping[str, Any]] = {}
        for row in raw_rows:
            if not isinstance(row, dict) or not isinstance(row.get("hf_subset"), str):
                raise ProtocolError(f"Cached subset row is malformed for {spec['name']}")
            subset = row["hf_subset"]
            if subset in rows_by_subset:
                raise ProtocolError(f"Duplicate cached subset for {spec['name']}/{subset}")
            rows_by_subset[subset] = row
        expected_subsets = selected_hf_subsets(spec, list(rows_by_subset))
        # ``selected_hf_subsets`` preserves the supplied registry order for a
        # component selector.  Independently require exact protocol order here.
        protocol_subsets = (
            list(spec["hf_subset_selection"]["values"])
            if spec["hf_subset_selection"]["mode"] == "exact"
            else list(rows_by_subset)
        )
        if set(rows_by_subset) != set(expected_subsets) or list(rows_by_subset) != protocol_subsets:
            raise ProtocolError(f"Cached subset contract drifted for {spec['name']}")

        subset_rows: list[dict[str, Any]] = []
        for subset in protocol_subsets:
            raw = rows_by_subset[subset]
            main_score = _finite_score(
                raw.get("main_score"), task=spec["name"], subset=subset, field="main_score"
            )
            metric_score = _finite_score(
                raw.get(spec["main_score"]),
                task=spec["name"],
                subset=subset,
                field=spec["main_score"],
            )
            if not math.isclose(main_score, metric_score, rel_tol=0.0, abs_tol=1e-12):
                raise ProtocolError(f"Main metric mismatch for {spec['name']}/{subset}")
            languages = raw.get("languages")
            if not isinstance(languages, list) or not all(
                isinstance(language, str) for language in languages
            ):
                raise ProtocolError(f"Cached languages are malformed for {spec['name']}/{subset}")
            expected_languages = resolved_by_name[spec["name"]][
                "selected_hf_subset_languages"
            ].get(subset)
            if languages != expected_languages:
                raise ProtocolError(
                    f"Cached subset-language mapping drifted for {spec['name']}/{subset}"
                )
            subset_rows.append(
                {
                    "hf_subset": subset,
                    "languages": expected_languages,
                    "score": main_score,
                }
            )
        task_score = fmean(row["score"] for row in subset_rows)
        normalized.append(
            {
                "asset_key": spec["asset_key"],
                "task_name": spec["name"],
                "task_type": spec["type"],
                "split": spec["split"],
                "dataset_revision": spec["dataset"]["revision"],
                "main_metric": spec["main_score"],
                "task_score": task_score,
                "leaderboard_points": 100.0 * task_score,
                "subset_count": len(subset_rows),
                "subsets": subset_rows,
                "contamination_grade": spec["contamination_grade"],
                "claim_policy": spec["claim_policy"],
            }
        )
    return normalized, inventory


def build_summary(
    *,
    protocol: Mapping[str, Any],
    resolved: Mapping[str, Any],
    contract: Mapping[str, Any],
    model: str,
    revision: str,
    model_evidence: Mapping[str, Any],
    normalized_tasks: Sequence[Mapping[str, Any]],
    raw_inventory: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    type_scores: dict[str, list[float]] = defaultdict(list)
    for row in normalized_tasks:
        type_scores[str(row["task_type"])].append(float(row["task_score"]))
    means_by_type = {
        task_type: fmean(scores) for task_type, scores in sorted(type_scores.items())
    }
    completed = len(normalized_tasks)
    total = len(protocol["tasks"])
    subset_count = sum(int(row["subset_count"]) for row in normalized_tasks)
    normalized_digest = canonical_sha256(list(normalized_tasks))
    inventory_digest = canonical_sha256(list(raw_inventory))
    return {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": resolved["protocol_sha256"],
        "resolved_task_contract_sha256": resolved[
            "resolved_task_contract_sha256"
        ],
        "runtime_profile_id": contract["profile_id"],
        "scope": protocol["scope"],
        "complete": completed == total,
        "completed_tasks": completed,
        "total_tasks": total,
        "completed_subsets": subset_count,
        "expected_subsets": sum(
            spec["hf_subset_selection"]["count"] for spec in protocol["tasks"]
        ),
        "model": {
            "name_or_path": model,
            "revision": revision,
            "evidence": dict(model_evidence),
        },
        "aggregate": {
            "mean_task": (
                fmean(float(row["task_score"]) for row in normalized_tasks)
                if normalized_tasks
                else None
            ),
            "mean_task_type": fmean(means_by_type.values()) if means_by_type else None,
            "means_by_type": means_by_type,
            "method": protocol["aggregate"],
        },
        "normalized_results_sha256": normalized_digest,
        "tasks": list(normalized_tasks),
        "raw_result_inventory_sha256": inventory_digest,
        "raw_result_files": list(raw_inventory),
        "provenance": {
            "asset_manifest": resolved["asset_manifest"],
            "verified_local_snapshots": resolved["verified_local_snapshots"],
            "mteb_version": protocol["mteb_version"],
            "mteb_git_revision": protocol["mteb_git_revision"],
            "mteb_module_relative_path": protocol["mteb_module_relative_path"],
            "offline": resolved["validated_environment"]["offline"],
            "runtime_contract_sha256": canonical_sha256(contract),
            "environment": dict(environment),
        },
        "claim_status": {
            "diagnostic_only": True,
            "clean_claim_allowed": False,
            "visual_document_ready": False,
            "k_haters_ready": False,
            "excluded_assets": protocol["explicit_exclusions"],
        },
    }


def gpu_names() -> list[str]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1 or args.max_length < 1:
        raise SystemExit("--batch-size and --max-length must be positive")
    if not args.list_only and not args.model:
        raise SystemExit("--model is required unless --list-only is used")

    offline_environment = enforce_offline_environment()
    protocol = load_protocol(args.protocol)
    asset_evidence = validate_asset_manifest(protocol)
    try:
        snapshot_evidence = verify_local_asset_snapshots(protocol)
    except OfflineAssetError as exc:
        raise SystemExit(str(exc)) from None

    import mteb

    checkout_revision = validate_mteb_runtime(mteb, protocol)
    tasks, resolved_tasks = resolve_and_validate_tasks(mteb, protocol, args.task)
    resolved = build_resolved_protocol(
        protocol,
        args.protocol,
        asset_evidence,
        snapshot_evidence,
        resolved_tasks,
        checkout_revision,
        offline_environment,
    )
    if args.list_only:
        print(json.dumps(resolved, ensure_ascii=False, indent=2, allow_nan=False))
        return

    assert args.model is not None
    try:
        revision, model_evidence = resolve_offline_model(args.model, args.revision)
    except OfflineAssetError as exc:
        raise SystemExit(str(exc)) from None

    if args.registered_loader:
        loader_mode = "registered-task-instruction"
    elif args.qwen3_instruction_loader:
        loader_mode = "qwen3-task-instruction"
    else:
        loader_mode = "sentence-transformer"
    if loader_mode not in protocol["loader_contract"]["allowed_modes"]:
        raise ProtocolError("Requested loader mode is not pinned by the protocol")

    evaluation_dtype = local_merge_dtype(args.model)
    attention = effective_attention(args.attn_implementation, evaluation_dtype)
    safe_name = _safe_model_name(args.model)
    model_reference = public_model_reference(args.model)
    run_dir = args.output_dir / safe_name / revision
    contract = runtime_contract(
        protocol_id=protocol["protocol_id"],
        protocol_path=args.protocol,
        model=model_reference,
        revision=revision,
        batch_size=args.batch_size,
        max_length=args.max_length,
        requested_attention=args.attn_implementation,
        attention=attention,
        evaluation_dtype=evaluation_dtype,
        loader_contract=loader_mode,
        extra={
            "resolved_task_contract_sha256": resolved[
                "resolved_task_contract_sha256"
            ],
            "local_files_only": True,
            "normalize_embeddings": True,
        },
    )
    validate_existing_result_contract(run_dir, resolved)
    enforce_runtime_contract(run_dir, contract)
    atomic_write_json(run_dir / "protocol_resolved.json", resolved)

    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    torch_dtype = torch.float32 if evaluation_dtype == "float32" else torch.bfloat16
    model_kwargs = {
        "attn_implementation": attention,
        "torch_dtype": torch_dtype,
    }
    common_loader_kwargs = {
        "device": args.device,
        "model_kwargs": model_kwargs,
        "tokenizer_kwargs": {"padding_side": "left"},
        "local_files_only": True,
        "cache_folder": str(HF_HUB_CACHE),
        "token": False,
    }
    if args.registered_loader:
        model = mteb.get_model(
            args.model, revision=revision, **common_loader_kwargs
        )
    elif args.qwen3_instruction_loader:
        from mteb.models.model_implementations.qwen3_models import q3e_instruct_loader

        model = q3e_instruct_loader(
            args.model, revision=revision, **common_loader_kwargs
        )
    else:
        model = SentenceTransformer(
            args.model,
            revision=revision,
            trust_remote_code=args.trust_remote_code,
            **common_loader_kwargs,
        )

    inner_model = model.model if hasattr(model, "model") else model
    if not hasattr(inner_model, "max_seq_length"):
        raise TypeError("Resolved embedding model has no max_seq_length contract")
    inner_model.max_seq_length = args.max_length
    cache_namespace = (
        f"{model_reference}@{revision}|profile={contract['profile_id']}|"
        f"max={args.max_length}|batch={args.batch_size}|attn={attention}|"
        f"dtype={evaluation_dtype}|loader={loader_mode}"
    )
    if args.embedding_cache_dir is not None:
        try:
            from resumable_sentence_transformer import install_exact_encode_cache
        except ModuleNotFoundError:
            from scripts.resumable_sentence_transformer import install_exact_encode_cache

        install_exact_encode_cache(
            inner_model,
            embedding_cache_dir=args.embedding_cache_dir,
            embedding_cache_namespace=cache_namespace,
            counter_target=model,
        )

    results = mteb.evaluate(
        model,
        tasks=tasks,
        cache=mteb.ResultCache(cache_path=run_dir / "mteb_cache"),
        overwrite_strategy="always" if args.overwrite else "only-missing",
        prediction_folder=(run_dir / "predictions") if args.save_predictions else None,
        encode_kwargs={
            "batch_size": args.batch_size,
            "normalize_embeddings": True,
            "show_progress_bar": True,
        },
    )
    if results.model_revision != revision:
        raise ProtocolError(
            f"Resolved model revision drifted: expected {revision}, found {results.model_revision}"
        )

    normalized_tasks, raw_inventory = normalize_cached_results(
        run_dir, protocol, resolved["resolved_tasks"]
    )
    summary = build_summary(
        protocol=protocol,
        resolved=resolved,
        contract=contract,
        model=model_reference,
        revision=revision,
        model_evidence=model_evidence,
        normalized_tasks=normalized_tasks,
        raw_inventory=raw_inventory,
        environment={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "mteb": mteb.__version__,
            "device": args.device,
            "gpu": gpu_names(),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "requested_attention": args.attn_implementation,
            "attention": attention,
            "evaluation_dtype": evaluation_dtype,
            "loader_mode": loader_mode,
            "embedding_cache_enabled": args.embedding_cache_dir is not None,
            "embedding_cache_hits": getattr(model, "embedding_cache_hits", 0),
            "embedding_cache_misses": getattr(model, "embedding_cache_misses", 0),
        },
    )
    summary_sha = atomic_write_json(run_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "summary": summary,
                "summary_file_sha256": summary_sha,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
