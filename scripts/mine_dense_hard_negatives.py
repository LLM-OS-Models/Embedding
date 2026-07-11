#!/usr/bin/env python3
"""Mine deterministic, positive-aware dense hard negatives.

The input and output use this repository's strict ms-swift embedding JSONL
schema.  Queries and positive documents are embedded independently, then every
query is compared with every unique positive document.  The exhaustive search
is blockwise, so the score matrix is never materialized in CPU or GPU memory.

This program deliberately does not read an evaluation corpus.  The operator
must explicitly assert that the supplied JSONL contains training data only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
PINNED_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class InputRow:
    query: str
    positive: str
    source_negatives: tuple[str, ...]
    query_normalized: str
    positive_normalized: str


@dataclass(frozen=True)
class Document:
    normalized: str
    text: str
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exhaustively mine dense negatives from an ms-swift training JSONL"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        help="Defaults to manifest.json next to --input when that file exists",
    )
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep-work-dir", action="store_true")

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--allow-unpinned-model", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-device", default="auto")
    parser.add_argument("--model-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--document-prefix", default="")

    parser.add_argument("--candidate-pool-size", type=int, default=24)
    parser.add_argument("--num-negatives", type=int, default=4)
    parser.add_argument("--positive-relative-ratio", type=float, default=0.95)
    parser.add_argument("--query-block-size", type=int, default=64)
    parser.add_argument("--corpus-block-size", type=int, default=2048)
    parser.add_argument(
        "--include-source-negatives",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Score existing row negatives and let eligible ones compete with mined candidates",
    )
    parser.add_argument(
        "--exclude-query-document-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude a corpus document whose normalized text exactly equals the query",
    )
    parser.add_argument(
        "--insufficient-policy",
        choices=("drop", "error"),
        default="drop",
        help="Action when fewer than --num-negatives survive filtering",
    )
    parser.add_argument(
        "--duplicate-row-policy",
        choices=("drop", "error"),
        default="error",
        help="Action if mining makes two strict output rows identical",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--assert-no-benchmark-data",
        action="store_true",
        help="Required for a real run; asserts --input is training-only and contains no benchmark rows",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize input without importing or executing the embedding model",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_message(value: Any, field: str, path: Path, line_number: int) -> str:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{path}:{line_number}: {field} must contain exactly one message")
    message = value[0]
    if not isinstance(message, dict) or set(message) != {"role", "content"}:
        raise ValueError(f"{path}:{line_number}: {field} has an invalid message object")
    if message["role"] != "user":
        raise ValueError(f"{path}:{line_number}: {field} must contain a user message")
    content = message["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{path}:{line_number}: {field} content must be non-empty")
    return content


def nested_messages(value: Any, field: str, path: Path, line_number: int) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}:{line_number}: {field} must be a non-empty list")
    return [one_message(group, f"{field}[{index}]", path, line_number) for index, group in enumerate(value)]


def read_rows(path: Path) -> list[InputRow]:
    rows: list[InputRow] = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            raw = json.loads(line)
            expected = {"messages", "positive_messages", "negative_messages"}
            if not isinstance(raw, dict) or set(raw) != expected:
                keys = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
                raise ValueError(f"{path}:{line_number}: unexpected fields/type: {keys}")
            query = one_message(raw["messages"], "messages", path, line_number)
            positives = nested_messages(
                raw["positive_messages"], "positive_messages", path, line_number
            )
            negatives = nested_messages(
                raw["negative_messages"], "negative_messages", path, line_number
            )
            if len(positives) != 1:
                raise ValueError(f"{path}:{line_number}: exactly one positive is required")
            query_norm = normalize_text(query)
            positive_norm = normalize_text(positives[0])
            if not query_norm or not positive_norm:
                raise ValueError(f"{path}:{line_number}: normalization produced empty text")
            negative_norms = [normalize_text(item) for item in negatives]
            if any(not item for item in negative_norms):
                raise ValueError(f"{path}:{line_number}: normalization produced an empty negative")
            if positive_norm in negative_norms:
                raise ValueError(f"{path}:{line_number}: positive duplicated as a negative after normalization")
            identity = text_hash("\0".join((query, positives[0], *negatives)))
            if identity in identities:
                raise ValueError(f"{path}:{line_number}: duplicate input row")
            identities.add(identity)
            rows.append(
                InputRow(
                    query=query,
                    positive=positives[0],
                    source_negatives=tuple(negatives),
                    query_normalized=query_norm,
                    positive_normalized=positive_norm,
                )
            )
    if len(rows) < 2:
        raise ValueError(f"{path}: at least two rows are required")
    return rows


def canonical_documents(texts: Iterable[str]) -> tuple[list[Document], dict[str, int]]:
    representatives: dict[str, str] = {}
    for text in texts:
        normalized = normalize_text(text)
        previous = representatives.get(normalized)
        if previous is None or text < previous:
            representatives[normalized] = text
    documents = [
        Document(normalized=normalized, text=text, sha256=text_hash(normalized))
        for normalized, text in representatives.items()
    ]
    documents.sort(key=lambda item: (item.sha256, item.normalized, item.text))
    return documents, {item.normalized: index for index, item in enumerate(documents)}


def automatic_manifest_path(args: argparse.Namespace) -> Path | None:
    if args.input_manifest:
        return args.input_manifest
    candidate = args.input.parent / "manifest.json"
    return candidate if candidate.exists() else None


def load_input_manifest(args: argparse.Namespace) -> tuple[dict[str, Any] | None, Path | None, bool]:
    path = automatic_manifest_path(args)
    if path is None:
        return None, None, False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    verified = False
    declared = manifest.get("files", {}).get(args.input.name)
    if declared and declared.get("sha256"):
        actual = file_hash(args.input)
        if actual != declared["sha256"]:
            raise ValueError(
                f"Input hash does not match {path}: expected {declared['sha256']}, got {actual}"
            )
        verified = True
    return manifest, path, verified


def validate_arguments(args: argparse.Namespace, row_count: int, corpus_count: int) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.resolve() == args.input.resolve():
        raise ValueError("--output must not overwrite --input")
    if args.num_negatives < 1:
        raise ValueError("--num-negatives must be positive")
    if args.candidate_pool_size < args.num_negatives:
        raise ValueError("--candidate-pool-size must be at least --num-negatives")
    if not 0.0 < args.positive_relative_ratio <= 1.0:
        raise ValueError("--positive-relative-ratio must be in (0, 1]")
    for field in (
        "max_seq_length",
        "encode_batch_size",
        "query_block_size",
        "corpus_block_size",
    ):
        if getattr(args, field) < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if row_count < 2 or corpus_count < 2:
        raise ValueError("At least two rows and two unique positive documents are required")
    if not args.allow_unpinned_model and not PINNED_REVISION_RE.fullmatch(args.revision or ""):
        raise ValueError("--revision must be a 40-character commit SHA unless --allow-unpinned-model is set")


def resolve_device(requested: str, torch: Any) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(args: argparse.Namespace) -> tuple[Any, Any, str, str]:
    import torch
    from sentence_transformers import SentenceTransformer

    device = resolve_device(args.device, torch)
    dtype_name = args.model_dtype
    if device == "cpu" and dtype_name != "float32":
        dtype_name = "float32"
    torch_dtype = getattr(torch, dtype_name)
    model = SentenceTransformer(
        args.model,
        revision=args.revision,
        device=device,
        trust_remote_code=args.trust_remote_code,
        model_kwargs={
            "attn_implementation": args.attn_implementation,
            "torch_dtype": torch_dtype,
        },
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = args.max_seq_length
    return model, torch, device, dtype_name


def encode_to_memmap(
    model: Any,
    texts: Sequence[str],
    prefix: str,
    path: Path,
    batch_size: int,
) -> tuple[Any, int]:
    import numpy as np

    if not texts:
        raise ValueError("Cannot encode an empty text collection")
    storage = None
    dimension = 0
    report_interval = max(batch_size, math.ceil(len(texts) / 20))
    next_report = 0
    for start in range(0, len(texts), batch_size):
        rendered = [prefix + text for text in texts[start : start + batch_size]]
        encoded = model.encode(
            rendered,
            prompt="",  # Disable any model-card default prompt; prefixes are explicit above.
            batch_size=len(rendered),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        array = np.asarray(encoded, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(rendered):
            raise RuntimeError(f"Unexpected embedding shape {array.shape}")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if not np.isfinite(array).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
            raise RuntimeError("Encoder produced non-finite or zero embeddings")
        array /= norms
        if storage is None:
            dimension = int(array.shape[1])
            storage = np.memmap(
                path, mode="w+", dtype=np.float32, shape=(len(texts), dimension)
            )
        elif array.shape[1] != dimension:
            raise RuntimeError("Embedding dimension changed between batches")
        storage[start : start + len(rendered)] = array
        completed = start + len(rendered)
        if start == 0 or completed >= next_report or completed == len(texts):
            print(f"[encode:{path.stem}] {completed}/{len(texts)}", file=sys.stderr)
            next_report = completed + report_interval
    assert storage is not None
    storage.flush()
    return storage, dimension


def stable_top_pool(
    query_embeddings: Any,
    document_embeddings: Any,
    positive_indices: Sequence[int],
    query_match_indices: Sequence[int],
    ratio: float,
    pool_size: int,
    query_block_size: int,
    corpus_block_size: int,
    torch: Any,
    score_device: str,
):
    """Yield blockwise exact top pools and score/exclusion metadata.

    Documents are globally ordered by their normalized-text hash.  Stable sorts
    therefore resolve exactly equal cosine scores by ascending document hash.
    """

    import numpy as np

    row_count = query_embeddings.shape[0]
    corpus_count = document_embeddings.shape[0]
    for query_start in range(0, row_count, query_block_size):
        query_end = min(query_start + query_block_size, row_count)
        block_count = query_end - query_start
        queries_np = np.asarray(query_embeddings[query_start:query_end], dtype=np.float32)
        queries = torch.from_numpy(queries_np).to(score_device)
        own_indices_np = np.asarray(positive_indices[query_start:query_end], dtype=np.int64)
        own_documents_np = np.asarray(document_embeddings[own_indices_np], dtype=np.float32)
        own_documents = torch.from_numpy(own_documents_np).to(score_device)
        positive_scores = torch.sum(queries * own_documents, dim=1)
        thresholds = positive_scores * ratio

        best_scores = torch.full(
            (block_count, pool_size), -torch.inf, dtype=torch.float32, device=score_device
        )
        best_indices = torch.full(
            (block_count, pool_size), -1, dtype=torch.long, device=score_device
        )
        exclusion_counts = {
            "own_positive": 0,
            "query_document_exact_match": 0,
            "above_positive_relative_threshold": 0,
            "non_finite": 0,
        }

        for document_start in range(0, corpus_count, corpus_block_size):
            document_end = min(document_start + corpus_block_size, corpus_count)
            documents_np = np.asarray(
                document_embeddings[document_start:document_end], dtype=np.float32
            )
            documents = torch.from_numpy(documents_np).to(score_device)
            scores = queries @ documents.T
            finite = torch.isfinite(scores)
            structural = torch.zeros_like(scores, dtype=torch.bool)

            row_ids = torch.arange(block_count, device=score_device)
            own_local = torch.as_tensor(own_indices_np - document_start, device=score_device)
            own_in_block = (own_local >= 0) & (own_local < document_end - document_start)
            if own_in_block.any():
                structural[row_ids[own_in_block], own_local[own_in_block]] = True
                exclusion_counts["own_positive"] += int(own_in_block.sum().item())

            query_match_np = np.asarray(
                query_match_indices[query_start:query_end], dtype=np.int64
            )
            query_local = torch.as_tensor(query_match_np - document_start, device=score_device)
            query_in_block = (query_local >= 0) & (query_local < document_end - document_start)
            distinct_query_match = query_in_block & (query_local != own_local)
            if distinct_query_match.any():
                structural[row_ids[distinct_query_match], query_local[distinct_query_match]] = True
                exclusion_counts["query_document_exact_match"] += int(
                    distinct_query_match.sum().item()
                )

            above = finite & ~structural & (scores >= thresholds[:, None])
            exclusion_counts["above_positive_relative_threshold"] += int(above.sum().item())
            exclusion_counts["non_finite"] += int((~finite).sum().item())
            eligible = finite & ~structural & ~above
            scores = scores.masked_fill(~eligible, -torch.inf)

            # Stable sort + globally hash-sorted corpus makes exact-score ties deterministic.
            block_order = torch.argsort(scores, dim=1, descending=True, stable=True)
            take = min(pool_size, document_end - document_start)
            block_order = block_order[:, :take]
            block_scores = torch.gather(scores, 1, block_order)
            block_indices = block_order + document_start
            if take < pool_size:
                pad = pool_size - take
                block_scores = torch.cat(
                    [
                        block_scores,
                        torch.full(
                            (block_count, pad),
                            -torch.inf,
                            dtype=torch.float32,
                            device=score_device,
                        ),
                    ],
                    dim=1,
                )
                block_indices = torch.cat(
                    [
                        block_indices,
                        torch.full(
                            (block_count, pad), -1, dtype=torch.long, device=score_device
                        ),
                    ],
                    dim=1,
                )

            merged_scores = torch.cat([best_scores, block_scores], dim=1)
            merged_indices = torch.cat([best_indices, block_indices], dim=1)
            merge_order = torch.argsort(merged_scores, dim=1, descending=True, stable=True)[
                :, :pool_size
            ]
            best_scores = torch.gather(merged_scores, 1, merge_order)
            best_indices = torch.gather(merged_indices, 1, merge_order)

        yield {
            "query_start": query_start,
            "query_end": query_end,
            "query_tensor": queries,
            "positive_scores": positive_scores.detach().cpu().numpy(),
            "thresholds": thresholds.detach().cpu().numpy(),
            "indices": best_indices.detach().cpu().numpy(),
            "scores": best_scores.detach().cpu().numpy(),
            "exclusions": exclusion_counts,
        }


def score_source_candidates(
    rows: Sequence[InputRow],
    query_start: int,
    query_end: int,
    query_tensor: Any,
    source_norms_by_row: Sequence[Sequence[str]],
    corpus_lookup: dict[str, int],
    corpus_embeddings: Any,
    source_only_lookup: dict[str, int],
    source_only_embeddings: Any | None,
    torch: Any,
    score_device: str,
) -> list[list[tuple[str, float]]]:
    """Return (normalized text, cosine) source candidates for each row in a block."""

    import numpy as np

    pair_rows: list[int] = []
    pair_norms: list[str] = []
    pair_vectors: list[Any] = []
    for local_index, global_index in enumerate(range(query_start, query_end)):
        for normalized in source_norms_by_row[global_index]:
            if normalized in corpus_lookup:
                vector = corpus_embeddings[corpus_lookup[normalized]]
            else:
                if source_only_embeddings is None:
                    raise RuntimeError("Missing source-only embeddings")
                vector = source_only_embeddings[source_only_lookup[normalized]]
            pair_rows.append(local_index)
            pair_norms.append(normalized)
            pair_vectors.append(vector)
    output: list[list[tuple[str, float]]] = [[] for _ in range(query_end - query_start)]
    if not pair_rows:
        return output
    vector_array = np.asarray(pair_vectors, dtype=np.float32)
    vectors = torch.from_numpy(vector_array).to(score_device)
    row_tensor = torch.as_tensor(pair_rows, dtype=torch.long, device=score_device)
    scores = torch.sum(query_tensor[row_tensor] * vectors, dim=1).detach().cpu().numpy()
    for local_index, normalized, score in zip(pair_rows, pair_norms, scores, strict=True):
        output[local_index].append((normalized, float(score)))
    return output


def strict_output_row(row: InputRow, negatives: Sequence[str]) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": row.query}],
        "positive_messages": [[{"role": "user", "content": row.positive}]],
        "negative_messages": [
            [{"role": "user", "content": negative}] for negative in negatives
        ],
    }


def output_identity(row: InputRow, negatives: Sequence[str]) -> str:
    return text_hash("\0".join((row.query, row.positive, *negatives)))


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    import numpy as np

    if not values:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def resolved_model_revision(model: Any) -> str | None:
    candidates: list[Any] = []
    try:
        candidates.append(model[0].auto_model.config)
    except (AttributeError, IndexError, KeyError, TypeError):
        pass
    candidates.extend([getattr(model, "tokenizer", None), getattr(model, "_model_config", None)])
    for candidate in candidates:
        if candidate is None:
            continue
        for key in ("_commit_hash", "commit_hash"):
            value = getattr(candidate, key, None)
            if value:
                return str(value)
        init_kwargs = getattr(candidate, "init_kwargs", {})
        if isinstance(init_kwargs, dict) and init_kwargs.get("_commit_hash"):
            return str(init_kwargs["_commit_hash"])
    return None


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    corpus, corpus_lookup = canonical_documents(row.positive for row in rows)
    validate_arguments(args, len(rows), len(corpus))
    input_manifest, input_manifest_path, input_manifest_hash_verified = load_input_manifest(args)

    source_norms_by_row: list[list[str]] = []
    source_representatives: dict[str, str] = {}
    if args.include_source_negatives:
        for row in rows:
            seen: set[str] = set()
            normalized_items: list[str] = []
            for text in row.source_negatives:
                normalized = normalize_text(text)
                if normalized == row.positive_normalized:
                    continue
                if args.exclude_query_document_match and normalized == row.query_normalized:
                    continue
                if normalized in seen:
                    continue
                seen.add(normalized)
                normalized_items.append(normalized)
                previous = source_representatives.get(normalized)
                if previous is None or text < previous:
                    source_representatives[normalized] = text
            source_norms_by_row.append(normalized_items)
    else:
        source_norms_by_row = [[] for _ in rows]

    source_only_texts = [
        source_representatives[normalized]
        for normalized in source_representatives
        if normalized not in corpus_lookup
    ]
    source_only_documents, source_only_lookup = canonical_documents(source_only_texts)

    plan = {
        "input": str(args.input),
        "input_sha256": file_hash(args.input),
        "rows": len(rows),
        "unique_positive_corpus_documents": len(corpus),
        "collapsed_exact_normalized_positive_duplicates": len(rows) - len(corpus),
        "source_negatives_enabled": args.include_source_negatives,
        "unique_source_only_documents": len(source_only_documents),
        "model": args.model,
        "revision": args.revision,
        "candidate_pool_size": args.candidate_pool_size,
        "num_negatives": args.num_negatives,
        "positive_relative_ratio": args.positive_relative_ratio,
        "query_block_size": args.query_block_size,
        "corpus_block_size": args.corpus_block_size,
        "exhaustive_dot_products": len(rows) * len(corpus),
        "input_manifest": str(input_manifest_path) if input_manifest_path else None,
        "input_manifest_hash_verified": input_manifest_hash_verified,
        "release_eligible_inherited": bool(
            input_manifest.get("release_eligible", False) if input_manifest else False
        ),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if not args.assert_no_benchmark_data:
        raise ValueError(
            "A real run requires --assert-no-benchmark-data. Do not mine from dev/test or benchmark data."
        )

    # Set deterministic CUDA behavior before importing torch or initializing CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    started = time.monotonic()
    model, torch, encoder_device, effective_model_dtype = load_encoder(args)
    score_device = resolve_device(args.score_device, torch)
    cuda_active = encoder_device.startswith("cuda") or score_device.startswith("cuda")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    if torch.cuda.is_available() and cuda_active:
        torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)

    output = args.output
    manifest_output = args.manifest_output or Path(str(output) + ".manifest.json")
    audit_output = args.audit_output or Path(str(output) + ".audit.jsonl")
    for path in (output, manifest_output, audit_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    work_parent = args.work_dir or output.parent
    work_parent.mkdir(parents=True, exist_ok=True)
    work_path = Path(tempfile.mkdtemp(prefix=".dense-hard-negative-", dir=work_parent))
    output_tmp = output.with_name(output.name + ".tmp")
    audit_tmp = audit_output.with_name(audit_output.name + ".tmp")

    positive_scores_all: list[float] = []
    threshold_scores_all: list[float] = []
    selected_scores_all: list[float] = []
    dense_pool_sizes: list[float] = []
    eligible_source_sizes: list[float] = []
    output_identities: set[str] = set()
    output_rows = 0
    dropped_insufficient = 0
    dropped_duplicate = 0
    source_selected = 0
    exclusions_total = {
        "own_positive": 0,
        "query_document_exact_match": 0,
        "above_positive_relative_threshold": 0,
        "non_finite": 0,
    }

    try:
        query_embeddings, query_dimension = encode_to_memmap(
            model,
            [row.query for row in rows],
            args.query_prefix,
            work_path / "queries.f32",
            args.encode_batch_size,
        )
        corpus_embeddings, corpus_dimension = encode_to_memmap(
            model,
            [document.text for document in corpus],
            args.document_prefix,
            work_path / "corpus.f32",
            args.encode_batch_size,
        )
        if query_dimension != corpus_dimension:
            raise RuntimeError(
                f"Query/document dimensions differ: {query_dimension} != {corpus_dimension}"
            )
        source_only_embeddings = None
        if source_only_documents:
            source_only_embeddings, source_dimension = encode_to_memmap(
                model,
                [document.text for document in source_only_documents],
                args.document_prefix,
                work_path / "source_only.f32",
                args.encode_batch_size,
            )
            if source_dimension != query_dimension:
                raise RuntimeError("Source-negative embedding dimension differs")

        positive_indices = [corpus_lookup[row.positive_normalized] for row in rows]
        if args.exclude_query_document_match:
            query_match_indices = [corpus_lookup.get(row.query_normalized, -1) for row in rows]
        else:
            query_match_indices = [-1 for _ in rows]

        with output_tmp.open("w", encoding="utf-8") as output_handle, audit_tmp.open(
            "w", encoding="utf-8"
        ) as audit_handle:
            mining_report_interval = max(args.query_block_size, math.ceil(len(rows) / 20))
            next_mining_report = 0
            for mined in stable_top_pool(
                query_embeddings=query_embeddings,
                document_embeddings=corpus_embeddings,
                positive_indices=positive_indices,
                query_match_indices=query_match_indices,
                ratio=args.positive_relative_ratio,
                pool_size=args.candidate_pool_size,
                query_block_size=args.query_block_size,
                corpus_block_size=args.corpus_block_size,
                torch=torch,
                score_device=score_device,
            ):
                query_start = mined["query_start"]
                query_end = mined["query_end"]
                if query_start == 0 or query_end >= next_mining_report or query_end == len(rows):
                    print(f"[mine] {query_end}/{len(rows)} queries", file=sys.stderr)
                    next_mining_report = query_end + mining_report_interval
                source_scores = score_source_candidates(
                    rows=rows,
                    query_start=query_start,
                    query_end=query_end,
                    query_tensor=mined["query_tensor"],
                    source_norms_by_row=source_norms_by_row,
                    corpus_lookup=corpus_lookup,
                    corpus_embeddings=corpus_embeddings,
                    source_only_lookup=source_only_lookup,
                    source_only_embeddings=source_only_embeddings,
                    torch=torch,
                    score_device=score_device,
                )
                for key, value in mined["exclusions"].items():
                    exclusions_total[key] += value

                for local_index, global_index in enumerate(range(query_start, query_end)):
                    row = rows[global_index]
                    positive_score = float(mined["positive_scores"][local_index])
                    threshold = float(mined["thresholds"][local_index])
                    positive_scores_all.append(positive_score)
                    threshold_scores_all.append(threshold)

                    candidates: dict[str, dict[str, Any]] = {}
                    dense_pool: list[dict[str, Any]] = []
                    for document_index, score in zip(
                        mined["indices"][local_index], mined["scores"][local_index], strict=True
                    ):
                        if int(document_index) < 0 or not math.isfinite(float(score)):
                            continue
                        document = corpus[int(document_index)]
                        item = {
                            "document_sha256": document.sha256,
                            "score": float(score),
                            "origin": "dense",
                            "text": document.text,
                        }
                        dense_pool.append({key: value for key, value in item.items() if key != "text"})
                        candidates[document.normalized] = item

                    eligible_sources: list[dict[str, Any]] = []
                    for normalized, score in source_scores[local_index]:
                        if not math.isfinite(score) or not score < threshold:
                            continue
                        if normalized == row.positive_normalized:
                            continue
                        if args.exclude_query_document_match and normalized == row.query_normalized:
                            continue
                        if normalized in corpus_lookup:
                            document = corpus[corpus_lookup[normalized]]
                        else:
                            document = source_only_documents[source_only_lookup[normalized]]
                        eligible_sources.append(
                            {"document_sha256": document.sha256, "score": score, "origin": "source"}
                        )
                        previous = candidates.get(normalized)
                        if previous is not None:
                            previous["origin"] = "dense+source"
                        else:
                            candidates[normalized] = {
                                "document_sha256": document.sha256,
                                "score": score,
                                "origin": "source",
                                "text": document.text,
                            }

                    ordered = sorted(
                        candidates.values(),
                        key=lambda item: (-item["score"], item["document_sha256"], item["origin"]),
                    )
                    dense_pool_sizes.append(float(len(dense_pool)))
                    eligible_source_sizes.append(float(len(eligible_sources)))
                    drop_reason = None
                    if len(ordered) < args.num_negatives:
                        drop_reason = "insufficient_eligible_negatives"
                        if args.insufficient_policy == "error":
                            raise RuntimeError(
                                f"Row {global_index} has {len(ordered)} eligible negatives; "
                                f"requested {args.num_negatives}"
                            )
                        dropped_insufficient += 1

                    selected = ordered[: args.num_negatives] if drop_reason is None else []
                    if selected:
                        identity = output_identity(row, [item["text"] for item in selected])
                        if identity in output_identities:
                            drop_reason = "duplicate_output_row"
                            selected = []
                            if args.duplicate_row_policy == "error":
                                raise RuntimeError(
                                    f"Mining produced a duplicate strict output row at input row {global_index}"
                                )
                            dropped_duplicate += 1
                        else:
                            output_identities.add(identity)

                    output_row_index: int | None = None
                    if selected:
                        output_row_index = output_rows
                        output_rows += 1
                        selected_scores_all.extend(float(item["score"]) for item in selected)
                        source_selected += sum("source" in item["origin"] for item in selected)
                        strict = strict_output_row(row, [item["text"] for item in selected])
                        output_handle.write(
                            json.dumps(strict, ensure_ascii=False, separators=(",", ":")) + "\n"
                        )

                    audit = {
                        "input_row_index": global_index,
                        "output_row_index": output_row_index,
                        "query_sha256": text_hash(row.query_normalized),
                        "positive_sha256": text_hash(row.positive_normalized),
                        "positive_score": positive_score,
                        "positive_relative_threshold": threshold,
                        "dense_pool": dense_pool,
                        "eligible_source_candidates": eligible_sources,
                        "selected": [
                            {key: value for key, value in item.items() if key != "text"}
                            for item in selected
                        ],
                        "drop_reason": drop_reason,
                    }
                    audit_handle.write(
                        json.dumps(audit, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )

        if output_rows < 2:
            raise RuntimeError(f"Only {output_rows} output rows survived; at least two are required")

        # Use the same strict validator as training entry points before publishing atomically.
        scripts_directory = str(Path(__file__).resolve().parent)
        if scripts_directory not in sys.path:
            sys.path.insert(0, scripts_directory)
        from validate_embedding_jsonl import validate

        validate(output_tmp)
        output_tmp.replace(output)
        audit_tmp.replace(audit_output)
        validation = validate(output)

        import numpy as np
        import sentence_transformers
        import transformers

        cuda_memory = None
        if torch.cuda.is_available() and cuda_active:
            cuda_memory = {
                "device": torch.cuda.get_device_name(torch.cuda.current_device()),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            }

        release_eligible = bool(
            input_manifest.get("release_eligible", False) if input_manifest else False
        )
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "training-only-dense-hard-negative-mining",
            "release_eligible": release_eligible,
            "release_eligibility": {
                "inherited_from_input_manifest": bool(input_manifest),
                "input_value": release_eligible,
                "default_when_manifest_missing": False,
                "input_release_blocker": (
                    input_manifest.get("release_blocker") if input_manifest else "input manifest missing"
                ),
            },
            "benchmark_data": {
                "included": False,
                "operator_assertion_required": True,
                "operator_asserted_training_only": args.assert_no_benchmark_data,
                "policy": "No benchmark/dev/test query, corpus, qrel, or row may enter this miner.",
            },
            "input": {
                "path": str(args.input),
                "sha256": plan["input_sha256"],
                "rows": len(rows),
                "manifest_path": str(input_manifest_path) if input_manifest_path else None,
                "manifest_sha256": file_hash(input_manifest_path) if input_manifest_path else None,
                "manifest_file_hash_verified": input_manifest_hash_verified,
            },
            "model": {
                "id": args.model,
                "requested_revision": args.revision,
                "resolved_revision": resolved_model_revision(model) or args.revision,
                "encoder_device": encoder_device,
                "model_dtype": effective_model_dtype,
                "attention_implementation": args.attn_implementation,
                "max_seq_length": args.max_seq_length,
                "embedding_dimension": query_dimension,
                "normalize_embeddings": True,
            },
            "prompt": {
                "query_source": "messages[0].content as stored",
                "query_prefix": args.query_prefix,
                "document_source": "positive_messages[0][0].content as stored",
                "document_prefix": args.document_prefix,
                "sentence_transformers_implicit_prompt": "disabled with prompt=''",
            },
            "mining": {
                "algorithm": "exhaustive blockwise normalized dot product",
                "score": "cosine (L2-normalized dot product)",
                "score_device": score_device,
                "score_dtype": "float32",
                "tf32": False,
                "seed": args.seed,
                "candidate_pool_size": args.candidate_pool_size,
                "num_explicit_negatives": args.num_negatives,
                "positive_relative_filter": {
                    "strict_inequality": "candidate_score < ratio * positive_score",
                    "ratio": args.positive_relative_ratio,
                },
                "source_negatives_enabled": args.include_source_negatives,
                "exclude_normalized_query_document_match": args.exclude_query_document_match,
                "exact_duplicate_definition": "NFKC + collapsed whitespace + exact string equality",
                "tie_break": "score descending, then normalized-document SHA-256 ascending",
                "query_block_size": args.query_block_size,
                "corpus_block_size": args.corpus_block_size,
                "exhaustive_dot_products": plan["exhaustive_dot_products"],
            },
            "corpus": {
                "input_positive_rows": len(rows),
                "unique_positive_documents": len(corpus),
                "collapsed_normalized_duplicates": len(rows) - len(corpus),
                "unique_source_only_documents": len(source_only_documents),
            },
            "results": {
                "output_rows": output_rows,
                "dropped_insufficient": dropped_insufficient,
                "dropped_duplicate": dropped_duplicate,
                "source_candidates_selected": source_selected,
                "exclusions": exclusions_total,
                "positive_scores": distribution(positive_scores_all),
                "positive_relative_thresholds": distribution(threshold_scores_all),
                "selected_negative_scores": distribution(selected_scores_all),
                "dense_pool_sizes": distribution(dense_pool_sizes),
                "eligible_source_candidates_per_row": distribution(eligible_source_sizes),
                "elapsed_seconds": time.monotonic() - started,
            },
            "files": {
                output.name: {
                    "path": str(output),
                    "sha256": file_hash(output),
                    "rows": output_rows,
                    "strict_validator": validation,
                },
                audit_output.name: {
                    "path": str(audit_output),
                    "sha256": file_hash(audit_output),
                    "rows": len(rows),
                    "contains_document_text": False,
                },
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "sentence_transformers": sentence_transformers.__version__,
                "numpy": np.__version__,
                "cuda_memory": cuda_memory,
                "git": git_state(Path(__file__).resolve().parents[1]),
            },
        }
        manifest_output.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        for temporary in (output_tmp, audit_tmp):
            if temporary.exists():
                temporary.unlink()
        if not args.keep_work_dir:
            shutil.rmtree(work_path, ignore_errors=True)
        else:
            print(f"Kept work directory: {work_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
