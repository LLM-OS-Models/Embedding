#!/usr/bin/env python3
"""Evaluate an embedding model on the verified 10K legal source holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

try:
    from evaluate_sionic9 import canonical_local_revision, local_merge_dtype
except ModuleNotFoundError:
    from scripts.evaluate_sionic9 import canonical_local_revision, local_merge_dtype


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "outputs/evaluation/legal-source-heldout-i-v1-shards12-15"
QUERY_PROMPT = (
    "Instruct: Given a Korean web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/evaluation/legal-source-heldout"
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=ROOT / "outputs/embedding-cache/legal-source-heldout",
    )
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--query-block-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def validate_dataset(dataset: Path) -> tuple[list[dict], list[dict], dict[str, str], dict]:
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assertions = manifest.get("assertions", {})
    required_zero = (
        "selected_query_hash_overlap_with_benchmark",
        "selected_positive_hash_overlap_with_benchmark",
        "selected_source_candidate_id_overlap_with_training",
        "selected_source_document_sha256_overlap_with_training",
    )
    if manifest.get("status") != "complete" or any(assertions.get(key) != 0 for key in required_zero):
        raise ValueError("Dataset manifest did not pass leakage assertions")
    queries = read_jsonl(dataset / "queries.jsonl")
    corpus = read_jsonl(dataset / "corpus.jsonl")
    qrels = read_jsonl(dataset / "qrels.jsonl")
    if len(queries) != 10000 or len(corpus) != 10000 or len(qrels) != 10000:
        raise ValueError("Legal holdout must contain exactly 10K queries/corpus/qrels")
    query_ids = {str(row["_id"]) for row in queries}
    corpus_ids = {str(row["_id"]) for row in corpus}
    positives: dict[str, str] = {}
    for row in qrels:
        query_id = str(row["query-id"])
        corpus_id = str(row["corpus-id"])
        if row.get("score") != 1 or query_id in positives:
            raise ValueError("Expected exactly one score-1 qrel per query")
        positives[query_id] = corpus_id
    if set(positives) != query_ids or not set(positives.values()) <= corpus_ids:
        raise ValueError("Query/corpus/qrel identifiers are inconsistent")
    return queries, corpus, positives, manifest


def gpu_name() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_length < 1 or args.query_block_size < 1:
        raise ValueError("Batch, max length and query block size must be positive")
    dataset = args.dataset_dir.resolve()
    queries, corpus, positives, manifest = validate_dataset(dataset)
    revision = canonical_local_revision(args.model, args.revision)
    if not revision:
        raise ValueError("--revision is required for a remote model")

    import sentence_transformers
    import torch
    import transformers

    try:
        from resumable_sentence_transformer import ResumableSentenceTransformer
    except ModuleNotFoundError:
        from scripts.resumable_sentence_transformer import ResumableSentenceTransformer

    evaluation_dtype = local_merge_dtype(args.model)
    torch_dtype = torch.float32 if evaluation_dtype == "float32" else torch.bfloat16
    effective_batch = min(args.batch_size, 96) if evaluation_dtype == "float32" else args.batch_size
    cache_namespace = (
        f"{args.model}@{revision}|legal-source-heldout-i-v1|manifest={sha256(dataset / 'manifest.json')}|"
        f"max={args.max_length}|attn={args.attn_implementation}|dtype={evaluation_dtype}|"
        f"prompt={hashlib.sha256(QUERY_PROMPT.encode()).hexdigest()}"
    )
    model = ResumableSentenceTransformer(
        args.model,
        revision=revision,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        model_kwargs={
            "attn_implementation": args.attn_implementation,
            "torch_dtype": torch_dtype,
        },
        tokenizer_kwargs={"padding_side": "left"},
        embedding_cache_dir=args.embedding_cache_dir,
        embedding_cache_namespace=cache_namespace,
    )
    model.max_seq_length = args.max_length

    query_rows = sorted(((str(row["_id"]), str(row["text"])) for row in queries))
    # `text` is the exact source-native positive and already contains its
    # heading/title where applicable. Do not prepend the metadata title twice.
    corpus_rows = sorted((str(row["_id"]), str(row["text"]).strip()) for row in corpus)
    query_vectors = np.asarray(
        model.encode(
            [QUERY_PROMPT + text for _, text in query_rows],
            batch_size=effective_batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    corpus_vectors = np.asarray(
        model.encode(
            [text for _, text in corpus_rows],
            batch_size=effective_batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    if query_vectors.shape != (10000, 4096) or corpus_vectors.shape != (10000, 4096):
        raise RuntimeError("Unexpected embedding matrix shape")

    corpus_ids = [item[0] for item in corpus_rows]
    corpus_index = {value: index for index, value in enumerate(corpus_ids)}
    corpus_tensor = torch.from_numpy(corpus_vectors).to(args.device)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    ranks: list[int] = []
    try:
        for start in range(0, len(query_rows), args.query_block_size):
            end = min(len(query_rows), start + args.query_block_size)
            query_tensor = torch.from_numpy(query_vectors[start:end]).to(args.device)
            scores = query_tensor @ corpus_tensor.T
            positive_indices = torch.tensor(
                [corpus_index[positives[query_rows[index][0]]] for index in range(start, end)],
                device=scores.device,
                dtype=torch.long,
            )
            positive_scores = scores.gather(1, positive_indices[:, None])
            greater = (scores > positive_scores).sum(dim=1)
            # corpus_rows is ID-sorted; this gives a deterministic ID-ascending tie break.
            corpus_positions = torch.arange(scores.shape[1], device=scores.device)[None, :]
            tied_before = ((scores == positive_scores) & (corpus_positions < positive_indices[:, None])).sum(dim=1)
            ranks.extend((greater + tied_before + 1).cpu().tolist())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    ndcg10 = [1.0 / math.log2(rank + 1) if rank <= 10 else 0.0 for rank in ranks]
    recall10 = [1.0 if rank <= 10 else 0.0 for rank in ranks]
    reciprocal10 = [1.0 / rank if rank <= 10 else 0.0 for rank in ranks]
    safe = args.model.replace("/", "__")
    run_dir = args.output_dir.resolve() / safe / revision
    run_dir.mkdir(parents=True, exist_ok=True)
    per_query = run_dir / "ranks.jsonl"
    with per_query.open("w", encoding="utf-8") as handle:
        for (query_id, _), rank in zip(query_rows, ranks, strict=True):
            handle.write(json.dumps({"query_id": query_id, "positive_rank": rank}) + "\n")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": "legal-source-document-heldout-i-v1",
        "model": args.model,
        "requested_revision": revision,
        "dataset": {
            "path": str(dataset),
            "manifest_sha256": sha256(dataset / "manifest.json"),
            "independence_grade": manifest["independence"]["grade"],
            "not_grade": manifest["independence"]["not_grade"],
            "rows": 10000,
        },
        "metrics": {
            "ndcg_at_10": fmean(ndcg10),
            "recall_at_10": fmean(recall10),
            "mrr_at_10": fmean(reciprocal10),
            "recall_at_100": fmean(1.0 if rank <= 100 else 0.0 for rank in ranks),
            "mean_positive_rank": fmean(ranks),
            "median_positive_rank": float(np.median(ranks)),
        },
        "query_prompt": QUERY_PROMPT,
        "score_contract": "exact float32 normalized dot; TF32 disabled; rank ties by corpus ID ascending",
        "limitations": "one source-native positive qrel per query; relevance judgments are not exhaustive",
        "files": {"ranks.jsonl": {"rows": 10000, "sha256": sha256(per_query)}},
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "gpu": gpu_name(),
            "batch_size": effective_batch,
            "requested_batch_size": args.batch_size,
            "max_length": args.max_length,
            "attention": args.attn_implementation,
            "torch_dtype": evaluation_dtype,
            "embedding_cache_hits": getattr(model, "embedding_cache_hits", 0),
            "embedding_cache_misses": getattr(model, "embedding_cache_misses", 0),
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
