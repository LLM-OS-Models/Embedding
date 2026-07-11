#!/usr/bin/env python3
"""Mine scalable, positive-aware hard negatives with a pinned FAISS IVF index."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from mine_dense_hard_negatives import (
        DEFAULT_MODEL,
        DEFAULT_REVISION,
        canonical_documents,
        encode_to_memmap,
        file_hash,
        load_encoder,
        normalize_text,
        read_rows,
        strict_output_row,
        text_hash,
    )
except ModuleNotFoundError:
    from scripts.mine_dense_hard_negatives import (
        DEFAULT_MODEL,
        DEFAULT_REVISION,
        canonical_documents,
        encode_to_memmap,
        file_hash,
        load_encoder,
        normalize_text,
        read_rows,
        strict_output_row,
        text_hash,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--candidate-pool-size", type=int, default=24)
    parser.add_argument("--search-k", type=int, default=256)
    parser.add_argument("--num-negatives", type=int, default=7)
    parser.add_argument("--positive-relative-ratio", type=float, default=0.95)
    parser.add_argument("--nlist", type=int, default=512)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--training-points", type=int, default=50000)
    parser.add_argument("--add-block-size", type=int, default=16384)
    parser.add_argument("--query-block-size", type=int, default=2048)
    parser.add_argument("--faiss-threads", type=int, default=min(32, max(1, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--assert-no-benchmark-data", action="store_true")
    parser.add_argument(
        "--allow-target-adapted",
        action="store_true",
        help="Allow target-like domain/corpus data while forcing target-adapted disclosure",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, rows: int, corpus: int) -> None:
    positive = (
        "max_seq_length",
        "encode_batch_size",
        "candidate_pool_size",
        "search_k",
        "num_negatives",
        "nlist",
        "nprobe",
        "training_points",
        "add_block_size",
        "query_block_size",
        "faiss_threads",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.search_k < args.candidate_pool_size:
        raise ValueError("--search-k must be >= --candidate-pool-size")
    if args.candidate_pool_size < args.num_negatives:
        raise ValueError("--candidate-pool-size must be >= --num-negatives")
    if not 0 < args.positive_relative_ratio <= 1:
        raise ValueError("--positive-relative-ratio must be in (0, 1]")
    if rows < 2 or corpus < 2:
        raise ValueError("At least two rows and documents are required")


def cache_namespace(args: argparse.Namespace, input_sha: str, role: str, rows: int) -> dict:
    return {
        "schema": 1,
        "input_sha256": input_sha,
        "role": role,
        "rows": rows,
        "model": args.model,
        "revision": args.revision,
        "max_seq_length": args.max_seq_length,
        "model_dtype": args.model_dtype,
        "attention": args.attn_implementation,
        "prefix": "",
        "normalized": True,
        "dtype": "float32",
    }


def load_cached_memmap(path: Path, metadata_path: Path, expected: dict) -> tuple[Any, int] | None:
    if not path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("namespace") != expected:
        return None
    dimension = metadata.get("dimension")
    if not isinstance(dimension, int) or dimension < 1:
        return None
    expected_bytes = expected["rows"] * dimension * np.dtype(np.float32).itemsize
    if path.stat().st_size != expected_bytes:
        return None
    return np.memmap(path, mode="r", dtype=np.float32, shape=(expected["rows"], dimension)), dimension


def encode_or_resume(
    model: Any,
    texts: Sequence[str],
    path: Path,
    namespace: dict,
    batch_size: int,
) -> tuple[Any, int, bool]:
    metadata_path = path.with_suffix(path.suffix + ".json")
    cached = load_cached_memmap(path, metadata_path, namespace)
    if cached is not None:
        print(f"[resume:{path.stem}] rows={len(texts)}", file=sys.stderr)
        return cached[0], cached[1], True
    path.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)
    array, dimension = encode_to_memmap(model, texts, "", path, batch_size)
    metadata_path.write_text(
        json.dumps({"namespace": namespace, "dimension": dimension}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return array, dimension, False


def build_or_resume_index(
    corpus_embeddings: Any,
    dimension: int,
    args: argparse.Namespace,
    work_dir: Path,
    namespace: dict,
) -> tuple[Any, dict, bool]:
    import faiss

    faiss.omp_set_num_threads(args.faiss_threads)
    corpus_count = int(corpus_embeddings.shape[0])
    nlist = min(args.nlist, max(1, corpus_count // 39))
    nprobe = min(args.nprobe, nlist)
    index_path = work_dir / "corpus.ivfflat.faiss"
    metadata_path = work_dir / "corpus.ivfflat.json"
    expected = {
        "schema": 1,
        "embedding_namespace": namespace,
        "dimension": dimension,
        "corpus_count": corpus_count,
        "nlist": nlist,
        "nprobe": nprobe,
        "metric": "inner_product",
        "faiss": faiss.__version__,
        "seed": args.seed,
    }
    if index_path.is_file() and metadata_path.is_file():
        try:
            if json.loads(metadata_path.read_text()) == expected:
                index = faiss.read_index(str(index_path))
                index.nprobe = nprobe
                if index.ntotal == corpus_count:
                    return index, expected, True
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    index.cp.seed = args.seed
    train_count = min(corpus_count, max(args.training_points, nlist * 40))
    train_indices = np.linspace(0, corpus_count - 1, train_count, dtype=np.int64)
    training = np.ascontiguousarray(corpus_embeddings[train_indices], dtype=np.float32)
    print(f"[faiss:train] points={train_count} nlist={nlist}", file=sys.stderr)
    index.train(training)
    for start in range(0, corpus_count, args.add_block_size):
        end = min(corpus_count, start + args.add_block_size)
        index.add(np.ascontiguousarray(corpus_embeddings[start:end], dtype=np.float32))
        if start == 0 or end == corpus_count or end % (args.add_block_size * 10) == 0:
            print(f"[faiss:add] {end}/{corpus_count}", file=sys.stderr)
    index.nprobe = nprobe
    faiss.write_index(index, str(index_path))
    metadata_path.write_text(json.dumps(expected, sort_keys=True) + "\n")
    return index, expected, False


def select_candidates(
    query: np.ndarray,
    positive: np.ndarray,
    indices: np.ndarray,
    corpus_embeddings: Any,
    own_index: int,
    query_match_index: int,
    ratio: float,
    pool_size: int,
) -> tuple[float, float, list[tuple[int, float]], dict[str, int]]:
    positive_score = float(np.dot(query, positive))
    threshold = positive_score * ratio
    exclusions = {"own_positive": 0, "query_document_exact_match": 0, "above_threshold": 0}
    candidates = []
    seen = set()
    for raw_index in indices:
        index = int(raw_index)
        if index < 0 or index in seen:
            continue
        seen.add(index)
        if index == own_index:
            exclusions["own_positive"] += 1
            continue
        if index == query_match_index:
            exclusions["query_document_exact_match"] += 1
            continue
        score = float(np.dot(query, np.asarray(corpus_embeddings[index], dtype=np.float32)))
        if not math.isfinite(score) or score >= threshold:
            exclusions["above_threshold"] += 1
            continue
        candidates.append((index, score))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return positive_score, threshold, candidates[:pool_size], exclusions


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    corpus, corpus_lookup = canonical_documents(row.positive for row in rows)
    validate_args(args, len(rows), len(corpus))
    input_sha = file_hash(args.input)
    plan = {
        "rows": len(rows),
        "corpus": len(corpus),
        "input_sha256": input_sha,
        "backend": "faiss_ivfflat_candidate_generation_then_exact_candidate_dot",
        "search_k": args.search_k,
        "candidate_pool_size": args.candidate_pool_size,
        "num_negatives": args.num_negatives,
        "positive_relative_ratio": args.positive_relative_ratio,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if args.assert_no_benchmark_data == args.allow_target_adapted:
        raise ValueError(
            "A real run requires exactly one of --assert-no-benchmark-data or "
            "--allow-target-adapted"
        )

    output = args.output.resolve()
    audit_output = (args.audit_output or Path(str(output) + ".audit.jsonl")).resolve()
    manifest_output = (args.manifest_output or Path(str(output) + ".manifest.json")).resolve()
    for path in (output, audit_output, manifest_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = (args.work_dir or output.parent / f".{output.name}.faiss-work").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    model, _, encoder_device, effective_dtype = load_encoder(args)
    query_namespace = cache_namespace(args, input_sha, "queries", len(rows))
    corpus_namespace = cache_namespace(args, input_sha, "positive_corpus", len(corpus))
    query_embeddings, query_dimension, query_resumed = encode_or_resume(
        model, [row.query for row in rows], work_dir / "queries.f32", query_namespace, args.encode_batch_size
    )
    corpus_embeddings, corpus_dimension, corpus_resumed = encode_or_resume(
        model,
        [document.text for document in corpus],
        work_dir / "corpus.f32",
        corpus_namespace,
        args.encode_batch_size,
    )
    del model
    if query_dimension != corpus_dimension:
        raise RuntimeError("Query/corpus dimensions differ")
    index, index_config, index_resumed = build_or_resume_index(
        corpus_embeddings, corpus_dimension, args, work_dir, corpus_namespace
    )

    positive_indices = np.asarray(
        [corpus_lookup[row.positive_normalized] for row in rows], dtype=np.int64
    )
    query_match_indices = np.asarray(
        [corpus_lookup.get(row.query_normalized, -1) for row in rows], dtype=np.int64
    )
    output_tmp = output.with_name(output.name + ".tmp")
    audit_tmp = audit_output.with_name(audit_output.name + ".tmp")
    output_rows = dropped = 0
    score_values = []
    exclusion_totals = {"own_positive": 0, "query_document_exact_match": 0, "above_threshold": 0}
    with output_tmp.open("w", encoding="utf-8") as output_handle, audit_tmp.open(
        "w", encoding="utf-8"
    ) as audit_handle:
        for start in range(0, len(rows), args.query_block_size):
            end = min(len(rows), start + args.query_block_size)
            query_block = np.ascontiguousarray(query_embeddings[start:end], dtype=np.float32)
            _, approximate_indices = index.search(query_block, args.search_k)
            for local, global_index in enumerate(range(start, end)):
                own_index = int(positive_indices[global_index])
                positive_score, threshold, pool, exclusions = select_candidates(
                    query_block[local],
                    np.asarray(corpus_embeddings[own_index], dtype=np.float32),
                    approximate_indices[local],
                    corpus_embeddings,
                    own_index,
                    int(query_match_indices[global_index]),
                    args.positive_relative_ratio,
                    args.candidate_pool_size,
                )
                for key, value in exclusions.items():
                    exclusion_totals[key] += value
                selected = pool[: args.num_negatives]
                output_index = None
                if len(selected) == args.num_negatives:
                    output_index = output_rows
                    output_rows += 1
                    negatives = [corpus[index].text for index, _ in selected]
                    score_values.extend(score for _, score in selected)
                    output_handle.write(
                        json.dumps(strict_output_row(rows[global_index], negatives), ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                else:
                    dropped += 1
                audit_handle.write(
                    json.dumps(
                        {
                            "input_row_index": global_index,
                            "output_row_index": output_index,
                            "query_sha256": text_hash(normalize_text(rows[global_index].query)),
                            "positive_sha256": text_hash(rows[global_index].positive_normalized),
                            "positive_score": positive_score,
                            "threshold": threshold,
                            "selected": [
                                {"document_sha256": corpus[index].sha256, "score": score}
                                for index, score in selected
                            ],
                            "ann_search_k": args.search_k,
                            "drop_reason": None if output_index is not None else "insufficient_ann_candidates",
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            print(f"[mine:faiss] {end}/{len(rows)}", file=sys.stderr)
    os.replace(output_tmp, output)
    os.replace(audit_tmp, audit_output)
    manifest = {
        **plan,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "output_rows": output_rows,
        "dropped_rows": dropped,
        "embedding_dimension": query_dimension,
        "embedding_cache_resumed": {"queries": query_resumed, "corpus": corpus_resumed},
        "faiss_index_resumed": index_resumed,
        "faiss_index": index_config,
        "exclusions": exclusion_totals,
        "selected_score": {
            "count": len(score_values),
            "min": min(score_values) if score_values else None,
            "mean": float(np.mean(score_values)) if score_values else None,
            "max": max(score_values) if score_values else None,
        },
        "model": args.model,
        "revision": args.revision,
        "encoder_device": encoder_device,
        "effective_model_dtype": effective_dtype,
        "faiss_version": __import__("faiss").__version__,
        "numpy_version": np.__version__,
        "elapsed_seconds": time.monotonic() - started,
        "files": {
            output.name: {"rows": output_rows, "sha256": file_hash(output)},
            audit_output.name: {"rows": len(rows), "sha256": file_hash(audit_output)},
        },
        "claim_scope": "ANN candidate generation; selected candidate dot scores are exact float32, but ANN recall is approximate and teacher/reranker validation remains required",
        "benchmark_adaptation": (
            "target-adapted; never claim clean zero-shot"
            if args.allow_target_adapted
            else "operator asserted training-only input with no benchmark data"
        ),
    }
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
