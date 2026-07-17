#!/usr/bin/env python3
"""Restore pinned Hugging Face assets into the queue's local path contract.

The published dataset repositories use ``data/`` and ``metadata/`` layouts,
while the campaign predates publication and expects selected files at the
local dataset root.  This tool downloads exact Hub revisions, verifies the
training bytes, and creates relative symlinks for the legacy queue names.

Public assets are downloaded anonymously by default with ``token=False`` so a
credential persisted by another user on a shared machine is never reused.  If
``--use-token`` is explicitly requested, the token is read into process memory
from ``HF_TOKEN`` or the repository's ignored ``.env`` file.  It is never
printed or persisted by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / ".cache" / "huggingface" / "hub"


@dataclass(frozen=True)
class FileContract:
    source: str
    alias: str | None
    sha256: str
    rows: int | None = None


@dataclass(frozen=True)
class DatasetAsset:
    key: str
    repo_id: str
    revision: str
    destination: str
    files: tuple[FileContract, ...]
    requires_token: bool = False


@dataclass(frozen=True)
class ModelAsset:
    key: str
    repo_id: str
    revision: str
    group: str


DATASETS = (
    DatasetAsset(
        "bcai-finance-triplet",
        "BCCard/BCAI-Finance-Kor-Embedding-Triplet",
        "f63d59969dba9916bd34c86c82112331890b11da",
        "outputs/assets/bcai-finance-kor-embedding-triplet",
        (
            FileContract(
                "data/train-00000-of-00001.parquet",
                None,
                "a84665bc2214ca0cd9d72d6e3355228a0ac3ba76b5275041a2cffdc00af70384",
            ),
            FileContract(
                "data/validation-00000-of-00001.parquet",
                None,
                "eb0d31a4d8ae6fb6ca81d8813f7a2a4389e31cc1f0233f295ffa9202058e8c63",
            ),
            FileContract(
                "data/test-00000-of-00001.parquet",
                None,
                "dc92e3f876dc24bd7712b61964c1099efec90cecbfedec8b61844c8bd8358c4e",
            ),
        ),
    ),
    DatasetAsset(
        "bcai-finance-pair",
        "BCCard/BCAI-Finance-Kor-Embedding-Pair",
        "e022cb013f2907e0716ebe40d13f30ed93ffa9b0",
        "outputs/assets/bcai-finance-kor-embedding-pair",
        (
            FileContract(
                "data/train-00000-of-00001.parquet",
                None,
                "0390e3ef89033cca3161914816e495ad239e5d8033f1a7bc14a922adb081888d",
            ),
        ),
    ),
    DatasetAsset(
        "kotsqa-v2",
        "etri-lirs/KoTSQA-v.2.0",
        "ff9349df469a765b4561959e36ef1b3f377765cd",
        "outputs/assets/kotsqa-v2",
        (
            FileContract(
                "train.parquet",
                None,
                "e38c3509fd9ff844488dc0687f6b919c3dfb4db13d989e063b60aa48826758ee",
            ),
            FileContract(
                "test.parquet",
                None,
                "5e8719d1d48b45dc9d325dd448a201b6d739c1e9921274faae66c2883656e949",
            ),
        ),
    ),
    DatasetAsset(
        "pilot10k",
        "LLM-OS-Models/korean-embedding-ko-triplet-hn-pilot-10k",
        "0865276985dd2eae5efec33a4fa181ee3086bd5f",
        "data/processed/ko_triplet_pilot_10k",
        (
            FileContract(
                "data/train.jsonl",
                "train.hn-qwen3-r095-n4.jsonl",
                "3df507549ea801d9e1c4aba54d9bf95a88b6690b6b27a0f1e1a05b3c0c525adc",
                10_000,
            ),
            FileContract(
                "data/validation.jsonl",
                "validation.hn-qwen3-r095-n4.jsonl",
                "f121f7eb3011ee2bfd796cb7622efd4b6f8f8ad80d09525cf083eeb18c7a9ede",
                512,
            ),
            FileContract(
                "metadata/train_mining_audit.jsonl",
                "train.hn-qwen3-r095-n4.jsonl.audit.jsonl",
                "fe1b25159067a6c33615c0bb0c950c897daa518fb34367534f01471f54fbefae",
                10_000,
            ),
            FileContract(
                "metadata/validation_mining_audit.jsonl",
                "validation.hn-qwen3-r095-n4.jsonl.audit.jsonl",
                "366e4b46abae9871eed371070cc48db3c1882e5e13c02781bcb6008691c22c08",
                512,
            ),
        ),
    ),
    DatasetAsset(
        "performance200k",
        "LLM-OS-Models/korean-embedding-performance-v1-ablation-200k",
        "f605128d3233e7cc488dc741b8f2af9ecf68b6fa",
        "outputs/data/performance-v1/ablation-200k",
        (
            FileContract("data/train.jsonl", "train.jsonl", "087c543e97975115b826455318bdae37bce371e63c396e2242ad7ef5fbd4a3c2", 200_000),
            FileContract("metadata/provenance.jsonl", "provenance.jsonl", "3114c455cf4a4604401a1ea0c723ff1fa5918478f97d0c70da72a9cff0bf9cd5", 200_000),
            FileContract("data/train.homogeneous-b16-length-bucketed.jsonl", "train.homogeneous-b16.jsonl", "8e2731ab25299ff558af675f067b253a6ce4375a850aa925acfe3b3117505e3c", 199_904),
            FileContract("metadata/provenance.homogeneous-b16-length-bucketed.jsonl", "provenance.homogeneous-b16.jsonl", "89f90133a95e5bbad2ddb392a1494c2a6480e94888100434c24504c8ac2cc0ea", 199_904),
        ),
    ),
    DatasetAsset(
        "performance1m",
        "LLM-OS-Models/korean-embedding-performance-v1-performance-1m",
        "5a2a3ab7f0928c6570929cc231eaefdd3fa203e1",
        "outputs/data/performance-v1/performance-1m",
        (
            FileContract("data/train.jsonl", "train.jsonl", "056abaf6b72d7521f9f60483e7ee1267011e3fe4803ee0429e19db4df323d2fa", 1_000_000),
            FileContract("metadata/provenance.jsonl", "provenance.jsonl", "81dac008cc9523cb64983aaa8f623b7cb81c5075f6cc045ac6fd74e04e6bd4f4", 1_000_000),
            FileContract("data/train.homogeneous-b16-length-bucketed.jsonl", "train.homogeneous-b16.jsonl", "7f2641a0a0256e89b2abb3462aa2c8b415b3e605fd4ca413178d2ad4abfc9009", 999_936),
            FileContract("metadata/provenance.homogeneous-b16-length-bucketed.jsonl", "provenance.homogeneous-b16.jsonl", "b036f6ef28d1a09d16aab4cedbb896d44e4bccd98a0dee94959aa0b4bae56646", 999_936),
        ),
    ),
    DatasetAsset(
        "squad60k",
        "LLM-OS-Models/korean-embedding-performance-v1-sionic-squad-train-60k",
        "8fbc6d6d5c93c3493456079d930921ac90ec6801",
        "outputs/data/performance-v1/sionic-squad-train-60k",
        (
            FileContract("data/train.jsonl", "train.jsonl", "5def1584d2e9b62cbedb3428cc49b1e7eeed674c48ec7e514f40ec54b6a63e07", 60_000),
            FileContract("metadata/provenance.jsonl", "provenance.jsonl", "e26d81fc3ca5a957c36353c522d280606de0195986c2ea784b8101df45646ea5", 60_000),
        ),
    ),
    DatasetAsset(
        "health100k",
        "LLM-OS-Models/korean-embedding-performance-v1-sionic-health-100k",
        "5fc4bb817f6970a710be53376f35e0225201d2e2",
        "outputs/data/performance-v1/sionic-health-multilingual-100k",
        (
            FileContract("data/train.jsonl", "train.jsonl", "6f9715bb130e1d58bac74f13d4b6d1996840bf45b1569ab281a92f632ac15302", 100_000),
            FileContract("metadata/provenance.jsonl", "provenance.jsonl", "cc9e41b7d4c7442ea7f78a4071ed9d94bb439e9374297ab54216b062d67054db", 100_000),
        ),
    ),
    DatasetAsset(
        "autorag100k",
        "LLM-OS-Models/korean-embedding-performance-v1-sionic-autorag-100k",
        "9140e9e02bb3f40ac1c22a6e595d58208770f696",
        "outputs/data/performance-v1/sionic-autorag-domain-100k",
        (
            FileContract("data/train.jsonl", "train.jsonl", "9b636831e1f4c5eb5d453c0b5f18eb642115035ba13d75a4d70ffd9fb905b835", 100_000),
            FileContract("metadata/provenance.jsonl", "provenance.jsonl", "05006632636b7c619152dca259db1dd71b32fb9d3263bb30e024c702e34d0f01", 100_000),
        ),
    ),
    DatasetAsset(
        "retrieval4146",
        "LLM-OS-Models/korean-embedding-performance-v1-sionic-retrieval-train-family-4146",
        "c9513a66ad64e5eab586969f6fdde7f9c8abd922",
        "outputs/data/performance-v1/sionic-retrieval-train-family-4146",
        (
            FileContract("data/train.jsonl", "train.jsonl", "6837367935ea56912375fe6a476360eb7dd0efcc0100459901e92a44029b7c60", 4_146),
            FileContract("metadata/provenance.jsonl", "provenance.jsonl", "9d97802b378b6c2d3bd15824db2ab3a680315f9ee6fb95492b16778c093d015e", 4_146),
        ),
    ),
    DatasetAsset(
        "legal250k",
        "LLM-OS-Models/korean-legal-retrieval-source-native-250k",
        "ec2f09a220dc5aa326c5d63b8e49adbf3a5524bc",
        "outputs/data/legal-performance-v1",
        (
            FileContract("data/train.jsonl", "train.bootstrap.jsonl", "1d81364bed3b4dab83a6979ef0874dd39bddb108830d35a43be7fd417d134c90", 250_000),
            FileContract("metadata/provenance.jsonl", "provenance.jsonl", "a1b3cda735df2e112832ebfbd8e07f3ec7d889ba875f17ff2f51cb9133a9de3e", 250_000),
        ),
    ),
    DatasetAsset(
        "cleanlegal10k",
        "LLM-OS-Models/korean-legal-source-heldout-retrieval-v1",
        "ee1300f04ea03d66bb51e23bbbda34376fece3f0",
        "outputs/evaluation/legal-source-heldout-i-v1-shards12-15",
        (
            FileContract("queries.jsonl", None, "9360d05b22656c5bb88ac1ce5cb59fc70b656340c5c353b6803fea607a2bee57", 10_000),
            FileContract("corpus.jsonl", None, "39824ac40bfecbc157cd41a0f5a956f55071faf60d796d6df99188146167a25a", 10_000),
            FileContract("qrels.jsonl", None, "a38310bf22a90b9d9dc8c25960cf7060a9afb51069166448542150c8c44012ca", 10_000),
            FileContract("provenance.jsonl", None, "6bb921fe9aff5f428d8ed4a0795311572710241ef9b1597626aa5268594d748e", 10_000),
        ),
    ),
    DatasetAsset(
        "cleanlegal10k-v2-text-strict",
        "LLM-OS-Models2/korean-legal-source-heldout-retrieval-v2-text-strict",
        "ce9d3bb57ca4dc5144753f6d0f8b4a2256851e97",
        "outputs/evaluation/legal-source-heldout-i-v2-text-strict",
        (
            FileContract("queries.jsonl", None, "412bdf86ae0c87515948c7925477b210455ffe71584fe1916d976e164265fc49", 10_000),
            FileContract("corpus.jsonl", None, "04b90c45f22ae3a09cae08eb84409a792725be28515700cf76b50eceb6fba94c", 10_000),
            FileContract("qrels.jsonl", None, "1b33d648c4669f01a52d33712e227fcaa01dd117e5de1048b67e85dfb823782e", 10_000),
            FileContract("provenance.jsonl", None, "1232dcdd7cc78f0bb2032d8a832e08f22c0dea2599d10b53bed8d8947ebe25bb", 10_000),
            FileContract("manifest.json", None, "5455459ee9474430e0ba9f61be84d7a0a577f8f1a1f73f8981aefb6ef61a216e"),
        ),
        requires_token=True,
    ),
    DatasetAsset(
        "legal-validation-v2-text-strict-512",
        "LLM-OS-Models2/korean-embedding-legal-validation-v2-text-strict-512",
        "8fdd1cad0007a9bfadf328d1702dcf6973c3c03d",
        "outputs/data/validation/legal-source-heldout-i-v2-text-strict-512",
        (
            FileContract("validation.jsonl", None, "e95ccdf34d6de00292f84f22bfb28ae95eb0bcd9ed8cbb2120216f89701b2703", 512),
            FileContract("provenance.jsonl", None, "a30cafb5491de73c991d15914556297a0df08e155d4b517658a8cbfadff5c517", 512),
            FileContract("manifest.json", None, "1a108fe8b5c7c29a842773c11012acb04d12cda939321c8f782bec966eed6aa4"),
        ),
        requires_token=True,
    ),
    DatasetAsset(
        "blocklist",
        "LLM-OS-Models/korean-embedding-benchmark-blocklist-v1",
        "5e876f26606830cd4d663cd62806d1f4c36387c9",
        "outputs/decontamination/benchmark_blocklist",
        (),
    ),
)


MODELS = (
    ModelAsset("qwen-base", "Qwen/Qwen3-Embedding-8B", "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af", "core"),
    ModelAsset("qwen-official", "Qwen/Qwen3-Embedding-8B", "4e423935c619ae4df87b646a3ce949610c66241c", "core"),
    ModelAsset("comsat", "sionic-ai/comsat-embed-ko-8b-preview", "a5cc22b651c1b2e51cdd8bf671774ae93584f0ab", "core"),
    ModelAsset("f2", "codefuse-ai/F2LLM-v2-8B", "e5725783762d69b4f8ba7e09a8872ce19a7a5ec3", "comparison"),
    ModelAsset("pwc", "SamilPwC-AXNode-GenAI/PwC-Embedding_expr", "6c5196980c685db45b58f67bd3be2f79d794351e", "comparison"),
    ModelAsset("harrier", "microsoft/harrier-oss-v1-27b", "0c0fc62f6d8af9e8604cb818c412301b103a0093", "comparison"),
    ModelAsset("kalm", "tencent/KaLM-Embedding-Gemma3-12B-2511", "98c19ba34197906fbc93f6f1ef79402ca3a33956", "comparison"),
    ModelAsset("nemotron", "nvidia/llama-embed-nemotron-8b", "aa3b43a495a9b280d1bdb716da37c54bb495d630", "comparison"),
    ModelAsset("nemotron3", "nvidia/Nemotron-3-Embed-8B-BF16", "2b29550c4ab0646bb6bb47032dda54ea11f6dfe2", "comparison"),
    ModelAsset("qwen-reranker-teacher", "Qwen/Qwen3-Reranker-8B", "77d193c791ed757ca307ee72715aa132723da912", "teacher"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            count += block.count(b"\n")
    return count


def read_dotenv_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    path = ROOT / ".env"
    if not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.removeprefix("export ").strip() != "HF_TOKEN":
            continue
        parsed = shlex.split(value, comments=False, posix=True)
        return parsed[0] if parsed else ""
    return None


def ensure_alias(source: Path, alias: Path, expected_sha: str) -> None:
    if alias == source:
        return
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.exists() or alias.is_symlink():
        if alias.resolve() == source.resolve() or (alias.is_file() and sha256(alias) == expected_sha):
            return
        raise RuntimeError(f"Refusing to replace mismatched local path: {alias}")
    relative = os.path.relpath(source, alias.parent)
    alias.symlink_to(relative)


def metadata_aliases(asset: DatasetAsset, destination: Path) -> Iterable[tuple[Path, Path]]:
    common = {
        "metadata/manifest.json": "manifest.json",
        "metadata/homogeneous-b16-length-bucketed.manifest.json": "homogeneous-b16.manifest.json",
        "metadata/source_manifest.json": "manifest.json",
        "metadata/train_hn_manifest.json": "train.hn-qwen3-r095-n4.jsonl.manifest.json",
        "metadata/validation_hn_manifest.json": "validation.hn-qwen3-r095-n4.jsonl.manifest.json",
    }
    for source_name, alias_name in common.items():
        source = destination / source_name
        if source.is_file():
            yield source, destination / alias_name


def restore_dataset(asset: DatasetAsset, token: str | bool, cache_dir: Path, max_workers: int, local_only: bool) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install requirements/hf-tools.txt in a tools environment") from exc
    destination = ROOT / asset.destination
    print(f"DATASET {asset.key}: {asset.repo_id}@{asset.revision[:12]}")
    if asset.requires_token and token is False and not local_only:
        raise RuntimeError(f"Private dataset {asset.key} requires explicit --use-token")
    if local_only:
        if not destination.is_dir():
            raise FileNotFoundError(destination)
    else:
        snapshot_download(
            repo_id=asset.repo_id,
            repo_type="dataset",
            revision=asset.revision,
            local_dir=destination,
            cache_dir=cache_dir,
            token=token,
            max_workers=max_workers,
        )
    for contract in asset.files:
        source = destination / contract.source
        if not source.is_file():
            raise FileNotFoundError(source)
        actual_sha = sha256(source)
        if actual_sha != contract.sha256:
            raise RuntimeError(f"SHA-256 mismatch for {source}: {actual_sha}")
        if contract.rows is not None:
            actual_rows = line_count(source)
            if actual_rows != contract.rows:
                raise RuntimeError(f"Row-count mismatch for {source}: {actual_rows}")
        if contract.alias:
            ensure_alias(source, destination / contract.alias, contract.sha256)
    for source, alias in metadata_aliases(asset, destination):
        ensure_alias(source, alias, sha256(source))
    print(f"VERIFIED {asset.key}")


def restore_model(asset: ModelAsset, token: str | bool, cache_dir: Path, max_workers: int, local_only: bool) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install requirements/hf-tools.txt in a tools environment") from exc
    print(f"MODEL {asset.key}: {asset.repo_id}@{asset.revision[:12]}")
    path = snapshot_download(
        repo_id=asset.repo_id,
        revision=asset.revision,
        cache_dir=cache_dir,
        token=token,
        max_workers=max_workers,
        local_files_only=local_only,
    )
    print(f"CACHED {asset.key}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", action="store_true", help="restore all required public datasets")
    parser.add_argument("--core-models", action="store_true", help="cache Qwen and Comsat revisions")
    parser.add_argument("--comparison-models", action="store_true", help="cache the five additional comparison models")
    parser.add_argument("--teacher-models", action="store_true", help="cache pinned teacher and reranker models")
    parser.add_argument("--all", action="store_true", help="restore datasets and every model")
    parser.add_argument("--asset", action="append", default=[], help="restore only a named dataset/model key")
    parser.add_argument("--local-only", action="store_true", help="verify/link without network access")
    parser.add_argument(
        "--use-token",
        action="store_true",
        help="opt in to reading HF_TOKEN or the ignored .env file; public downloads are anonymous by default",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = set(args.asset)
    if not any((args.datasets, args.core_models, args.comparison_models, args.teacher_models, args.all, requested)):
        args.datasets = True
    known = {asset.key for asset in DATASETS} | {asset.key for asset in MODELS}
    unknown = requested - known
    if unknown:
        raise SystemExit(f"Unknown asset key(s): {', '.join(sorted(unknown))}")
    # Passing False explicitly prevents huggingface_hub from falling back to a
    # token persisted by another user on this shared machine.  Public assets do
    # not need authentication.  Token use is an explicit opt-in and the value
    # is never accepted as a command-line argument or printed.
    token: str | bool = False
    if args.use_token:
        loaded_token = read_dotenv_token()
        if not loaded_token:
            raise SystemExit("--use-token was requested but HF_TOKEN is unavailable")
        token = loaded_token
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    requested_datasets = [
        asset for asset in DATASETS if args.all or args.datasets or asset.key in requested
    ]
    private_explicit_without_token = [
        asset.key
        for asset in requested_datasets
        if asset.key in requested and asset.requires_token and not args.use_token and not args.local_only
    ]
    if private_explicit_without_token:
        raise SystemExit(
            "Private dataset asset requires --use-token: "
            + ", ".join(private_explicit_without_token)
        )
    selected_datasets = [
        asset
        for asset in requested_datasets
        if not asset.requires_token or args.use_token or args.local_only
    ]
    skipped_private = [
        asset.key for asset in requested_datasets if asset not in selected_datasets
    ]
    if skipped_private:
        print(
            "SKIP private datasets without --use-token: "
            + ", ".join(skipped_private)
        )
    selected_models = [
        asset
        for asset in MODELS
        if args.all
        or asset.key in requested
        or (args.core_models and asset.group == "core")
        or (args.comparison_models and asset.group == "comparison")
        or (args.teacher_models and asset.group == "teacher")
    ]
    for asset in selected_datasets:
        restore_dataset(asset, token, args.cache_dir, args.max_workers, args.local_only)
    for asset in selected_models:
        restore_model(asset, token, args.cache_dir, args.max_workers, args.local_only)


if __name__ == "__main__":
    main()
