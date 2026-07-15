#!/usr/bin/env python3
"""Restore pinned comprehensive-evaluation datasets to a repository-local cache.

The Hugging Face token is taken from ``HF_TOKEN`` (or
``HUGGINGFACE_HUB_TOKEN``) in process environment, then from the repository's
ignored ``.env`` file.  The token is passed directly to ``snapshot_download``;
this script never prints it, places it on the command line, or persists it.

Modes are deliberately conservative:

* ``--small`` downloads the complete snapshots of the small regression sets.
* ``--metadata`` downloads only cards, queries, qrels, and metadata for the
  large SDS KoPub and KoViDoRe assets.
* ``--all`` downloads every complete snapshot, including the large corpora.
* ``--local-only`` adds Hugging Face's hard no-network cache-only constraint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "comprehensive_eval_assets.json"
CACHE_DIR = ROOT / ".cache" / "huggingface" / "hub"
DOTENV_PATH = ROOT / ".env"

FULL_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
REPO_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
METADATA_ALLOW_PATTERNS = (
    "README.md",
    "queries/**",
    "qrels/**",
    "metadata/**",
)
VALID_DOWNLOAD_TIERS = {"small", "metadata_first"}
VALID_CONTAMINATION_GRADES = {"low", "medium", "high"}


class ManifestError(RuntimeError):
    """The committed asset manifest does not satisfy the restore contract."""


class RestoreError(RuntimeError):
    """A restore failed without exposing credentials or local paths."""


@dataclass(frozen=True)
class EvalAsset:
    key: str
    repo_id: str
    revision: str
    license: str
    purpose: str
    usage_policy: str
    contamination_grade: str
    contamination_notes: str
    download_tier: str
    metadata_allow_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannedAsset:
    asset: EvalAsset
    metadata_only: bool


SnapshotDownload = Callable[..., str]


def _required_string(record: Mapping[str, object], field: str, key: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"Asset {key!r} has an invalid {field!r} field")
    return value


def _parse_asset(record: object, index: int) -> EvalAsset:
    if not isinstance(record, dict):
        raise ManifestError(f"Asset at index {index} is not an object")
    raw_key = record.get("key")
    key = raw_key if isinstance(raw_key, str) and raw_key else f"index-{index}"
    repo_id = _required_string(record, "repo_id", key)
    revision = _required_string(record, "revision", key)
    license_id = _required_string(record, "license", key)
    purpose = _required_string(record, "purpose", key)
    usage_policy = _required_string(record, "usage_policy", key)
    contamination_grade = _required_string(record, "contamination_grade", key)
    contamination_notes = _required_string(record, "contamination_notes", key)
    download_tier = _required_string(record, "download_tier", key)

    if not REPO_ID_RE.fullmatch(repo_id):
        raise ManifestError(f"Asset {key!r} has an invalid Hugging Face repo ID")
    if not FULL_REVISION_RE.fullmatch(revision):
        raise ManifestError(f"Asset {key!r} is not pinned to a full commit SHA")
    if download_tier not in VALID_DOWNLOAD_TIERS:
        raise ManifestError(f"Asset {key!r} has an invalid download tier")
    if contamination_grade not in VALID_CONTAMINATION_GRADES:
        raise ManifestError(f"Asset {key!r} has an invalid contamination grade")

    raw_patterns = record.get("metadata_allow_patterns", [])
    if not isinstance(raw_patterns, list) or not all(
        isinstance(pattern, str) and pattern for pattern in raw_patterns
    ):
        raise ManifestError(f"Asset {key!r} has invalid metadata allow patterns")
    patterns = tuple(raw_patterns)
    if download_tier == "metadata_first" and patterns != METADATA_ALLOW_PATTERNS:
        raise ManifestError(
            f"Asset {key!r} must use the reviewed metadata-only allow patterns"
        )
    if download_tier == "small" and patterns:
        raise ManifestError(f"Small asset {key!r} must not define allow patterns")

    return EvalAsset(
        key=key,
        repo_id=repo_id,
        revision=revision,
        license=license_id,
        purpose=purpose,
        usage_policy=usage_policy,
        contamination_grade=contamination_grade,
        contamination_notes=contamination_notes,
        download_tier=download_tier,
        metadata_allow_patterns=patterns,
    )


def load_assets(path: Path = MANIFEST_PATH) -> tuple[EvalAsset, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(
            f"Could not load the evaluation asset manifest ({type(exc).__name__})"
        ) from None
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0.0":
        raise ManifestError("Unsupported evaluation asset manifest schema")
    if payload.get("cache_relative_path") != str(CACHE_DIR.relative_to(ROOT)):
        raise ManifestError("Manifest cache contract is not repository-local")
    records = payload.get("assets")
    if not isinstance(records, list) or not records:
        raise ManifestError("Evaluation asset manifest has no assets")
    assets = tuple(_parse_asset(record, index) for index, record in enumerate(records))
    keys = [asset.key for asset in assets]
    repo_ids = [asset.repo_id for asset in assets]
    if len(keys) != len(set(keys)):
        raise ManifestError("Evaluation asset keys are not unique")
    if len(repo_ids) != len(set(repo_ids)):
        raise ManifestError("Evaluation asset repo IDs are not unique")
    return assets


def _parse_dotenv_value(raw_value: str) -> str | None:
    try:
        parsed = shlex.split(raw_value, comments=False, posix=True)
    except ValueError:
        return None
    return parsed[0] if parsed else None


def read_hf_token(env_file: Path = DOTENV_PATH) -> str | None:
    """Read a token without logging it or copying it into process environment."""

    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(name)
        if token:
            return token
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError):
        return None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        raw_key, raw_value = line.split("=", 1)
        key = raw_key.removeprefix("export ").strip()
        if key not in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"}:
            continue
        token = _parse_dotenv_value(raw_value)
        if token:
            return token
    return None


def build_plan(
    assets: Iterable[EvalAsset],
    *,
    small: bool,
    metadata: bool,
    all_assets: bool,
) -> tuple[PlannedAsset, ...]:
    assets = tuple(assets)
    if all_assets and (small or metadata):
        raise ManifestError("--all cannot be combined with --small or --metadata")
    if all_assets:
        return tuple(PlannedAsset(asset, metadata_only=False) for asset in assets)
    if not small and not metadata:
        small = True

    plan: list[PlannedAsset] = []
    for asset in assets:
        if small and asset.download_tier == "small":
            plan.append(PlannedAsset(asset, metadata_only=False))
        elif metadata and asset.download_tier == "metadata_first":
            plan.append(PlannedAsset(asset, metadata_only=True))
    return tuple(plan)


def load_snapshot_download() -> SnapshotDownload:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RestoreError(
            "huggingface_hub is unavailable; install requirements/hf-tools.txt"
        ) from None
    return snapshot_download


def restore_asset(
    planned: PlannedAsset,
    *,
    token: str | None,
    local_only: bool,
    max_workers: int,
    cache_dir: Path = CACHE_DIR,
    downloader: SnapshotDownload | None = None,
) -> None:
    if max_workers < 1:
        raise RestoreError("max_workers must be positive")
    if downloader is None:
        downloader = load_snapshot_download()
    asset = planned.asset
    kwargs: dict[str, object] = {
        "repo_id": asset.repo_id,
        "repo_type": "dataset",
        "revision": asset.revision,
        "cache_dir": cache_dir,
        "token": token,
        "max_workers": max_workers,
        "local_files_only": local_only,
    }
    if planned.metadata_only:
        kwargs["allow_patterns"] = list(asset.metadata_allow_patterns)
    try:
        downloader(**kwargs)
    except Exception as exc:
        raise RestoreError(
            f"Restore failed for {asset.key} ({type(exc).__name__})"
        ) from None
    scope = "metadata" if planned.metadata_only else "snapshot"
    print(f"RESTORED {asset.key} {scope}: {asset.repo_id}@{asset.revision[:12]}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore pinned comprehensive evaluation assets"
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="restore complete snapshots of the eight small evaluation assets (default)",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="restore only reviewed metadata/query/qrels files for SDS and KoViDoRe",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="restore every complete snapshot, including all large image corpora",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="use cached files only and never read a token or contact the Hub",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        assets = load_assets()
        plan = build_plan(
            assets,
            small=args.small,
            metadata=args.metadata,
            all_assets=args.all,
        )
    except ManifestError as exc:
        raise SystemExit(str(exc)) from None
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")

    token = None if args.local_only else read_hf_token()
    if not args.local_only and not token:
        raise SystemExit(
            "HF token unavailable; configure HF_TOKEN in environment or the ignored .env file"
        )
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"Could not prepare repository-local cache ({type(exc).__name__})") from None

    try:
        downloader = load_snapshot_download()
        for planned in plan:
            restore_asset(
                planned,
                token=token,
                local_only=args.local_only,
                max_workers=args.max_workers,
                downloader=downloader,
            )
    except RestoreError as exc:
        raise SystemExit(str(exc)) from None
    print(f"COMPLETE assets={len(plan)} local_only={str(args.local_only).lower()}")


if __name__ == "__main__":
    main()
