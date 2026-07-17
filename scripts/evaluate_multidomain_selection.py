#!/usr/bin/env python3
"""Evaluate the fixed finance/knowledge internal retrieval selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

try:
    from scripts.evaluate_sionic9 import canonical_local_revision, local_merge_dtype
    from scripts.evaluation_runtime import effective_attention
except ImportError:  # pragma: no cover - direct script execution fallback
    from evaluate_sionic9 import canonical_local_revision, local_merge_dtype
    from evaluation_runtime import effective_attention


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "outputs/evaluation/multidomain-selection-heldout-v1"
QUERY_PROMPT = (
    "Instruct: Given a Korean web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)
PROTOCOL_ID = "multidomain-selection-heldout-v1"
DATASET_MANIFEST_SHA256 = (
    "86fea553c6652388b1f67160c0e2e6b7626acf8929f86c1a2708156bd89b3c46"
)
SCORE_CONTRACT = (
    "exact float32 normalized dot; TF32 disabled; per-domain corpus; "
    "rank ties by corpus ID ascending"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/evaluation/multidomain-selection",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=ROOT / "outputs/embedding-cache/multidomain-selection",
    )
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--max-length", type=int, default=8192)
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
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def validate_dataset(
    root: Path,
    *,
    expected_manifest_sha256: str | None = DATASET_MANIFEST_SHA256,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, set[str]],
    dict[str, Any],
]:
    manifest_path = root / "manifest.json"
    if expected_manifest_sha256 is not None and sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Multidomain selector manifest identity drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Multidomain selector manifest is not complete")
    assertions = manifest.get("assertions", {})
    if (
        assertions.get("all_selected_query_exact_training_text_overlap") != 0
        or assertions.get("knowledge_query_and_corpus_exact_training_text_overlap") != 0
        or assertions.get("all_selected_query_and_corpus_benchmark_blocklist_overlap") != 0
        or assertions.get("public_benchmark_score_used_for_selection") is not False
    ):
        raise ValueError("Multidomain selector leakage assertions failed")
    for name, descriptor in manifest.get("files", {}).items():
        path = root / name
        if not path.is_file() or sha256(path) != descriptor.get("sha256"):
            raise ValueError(f"Multidomain selector file hash drifted: {name}")
        with path.open("rb") as handle:
            if sum(1 for _ in handle) != descriptor.get("rows"):
                raise ValueError(f"Multidomain selector row count drifted: {name}")
    queries = read_jsonl(root / "queries.jsonl")
    corpus = read_jsonl(root / "corpus.jsonl")
    qrel_rows = read_jsonl(root / "qrels.jsonl")
    query_map = {str(row["_id"]): row for row in queries}
    corpus_map = {str(row["_id"]): row for row in corpus}
    if len(query_map) != len(queries) or len(corpus_map) != len(corpus):
        raise ValueError("Duplicate multidomain query/corpus ID")
    expected_domains = {
        domain: int(descriptor["queries"])
        for domain, descriptor in manifest.get("domains", {}).items()
    }
    observed_domains = {
        domain: sum(row.get("domain") == domain for row in queries)
        for domain in expected_domains
    }
    if observed_domains != expected_domains or set(observed_domains) != {"finance", "knowledge"}:
        raise ValueError("Multidomain query domain counts drifted")
    qrels: dict[str, set[str]] = {query_id: set() for query_id in query_map}
    for row in qrel_rows:
        query_id = str(row.get("query-id"))
        corpus_id = str(row.get("corpus-id"))
        if row.get("score") != 1 or query_id not in query_map or corpus_id not in corpus_map:
            raise ValueError("Malformed multidomain qrel")
        if query_map[query_id].get("domain") != corpus_map[corpus_id].get("domain"):
            raise ValueError("Cross-domain qrel is forbidden")
        qrels[query_id].add(corpus_id)
    if any(not values for values in qrels.values()):
        raise ValueError("Every multidomain query needs a positive qrel")
    return queries, corpus, qrels, manifest


def metrics_from_ranks(ranks: list[list[int]]) -> dict[str, float]:
    if not ranks or any(not row for row in ranks):
        raise ValueError("Rank lists must be non-empty")
    ndcg10: list[float] = []
    recalls10: list[float] = []
    reciprocal10: list[float] = []
    recalls100: list[float] = []
    for values in ranks:
        ordered = sorted(values)
        relevant = len(ordered)
        dcg = sum(1.0 / math.log2(rank + 1) for rank in ordered if rank <= 10)
        ideal = sum(
            1.0 / math.log2(rank + 1) for rank in range(1, min(relevant, 10) + 1)
        )
        ndcg10.append(dcg / ideal)
        recalls10.append(sum(rank <= 10 for rank in ordered) / relevant)
        reciprocal10.append(1.0 / ordered[0] if ordered[0] <= 10 else 0.0)
        recalls100.append(sum(rank <= 100 for rank in ordered) / relevant)
    return {
        "ndcg_at_10": fmean(ndcg10),
        "recall_at_10": fmean(recalls10),
        "mrr_at_10": fmean(reciprocal10),
        "recall_at_100": fmean(recalls100),
        "mean_best_positive_rank": fmean(min(row) for row in ranks),
    }


def gpu_name() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("Batch size and max length must be positive")
    dataset = args.dataset_dir.expanduser().resolve()
    queries, corpus, qrels, manifest = validate_dataset(dataset)
    revision = canonical_local_revision(args.model, args.revision)
    if not revision:
        raise ValueError("--revision is required for a remote model")

    import sentence_transformers
    import torch
    import transformers

    try:
        from scripts.resumable_sentence_transformer import ResumableSentenceTransformer
    except ImportError:  # pragma: no cover - direct script execution fallback
        from resumable_sentence_transformer import ResumableSentenceTransformer

    evaluation_dtype = local_merge_dtype(args.model)
    torch_dtype = torch.float32 if evaluation_dtype == "float32" else torch.bfloat16
    effective_batch = min(args.batch_size, 96) if evaluation_dtype == "float32" else args.batch_size
    attention = effective_attention(args.attn_implementation, evaluation_dtype)
    manifest_sha = sha256(dataset / "manifest.json")
    cache_namespace = (
        f"{args.model}@{revision}|{PROTOCOL_ID}|manifest={manifest_sha}|"
        f"max={args.max_length}|batch={effective_batch}|attn={attention}|dtype={evaluation_dtype}|"
        f"prompt={hashlib.sha256(QUERY_PROMPT.encode()).hexdigest()}"
    )
    model = ResumableSentenceTransformer(
        args.model,
        revision=revision,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        model_kwargs={"attn_implementation": attention, "torch_dtype": torch_dtype},
        tokenizer_kwargs={"padding_side": "left"},
        embedding_cache_dir=args.embedding_cache_dir,
        embedding_cache_namespace=cache_namespace,
    )
    model.max_seq_length = args.max_length
    query_rows = sorted(
        (str(row["_id"]), str(row["text"]), str(row["domain"])) for row in queries
    )
    corpus_rows = sorted(
        (str(row["_id"]), str(row["text"]), str(row["domain"])) for row in corpus
    )
    query_vectors = np.asarray(
        model.encode(
            [QUERY_PROMPT + text for _, text, _ in query_rows],
            batch_size=effective_batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    corpus_vectors = np.asarray(
        model.encode(
            [text for _, text, _ in corpus_rows],
            batch_size=effective_batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    if query_vectors.shape != (len(query_rows), 4096) or corpus_vectors.shape != (
        len(corpus_rows),
        4096,
    ):
        raise RuntimeError("Unexpected multidomain embedding shape")

    ranks_by_query: dict[str, list[int]] = {}
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for domain in ("finance", "knowledge"):
            query_indices = [i for i, row in enumerate(query_rows) if row[2] == domain]
            corpus_indices = [i for i, row in enumerate(corpus_rows) if row[2] == domain]
            domain_corpus_ids = [corpus_rows[i][0] for i in corpus_indices]
            corpus_tensor = torch.from_numpy(corpus_vectors[corpus_indices]).to(args.device)
            for start in range(0, len(query_indices), 256):
                selected = query_indices[start : start + 256]
                query_tensor = torch.from_numpy(query_vectors[selected]).to(args.device)
                scores = (query_tensor @ corpus_tensor.T).cpu().numpy()
                for row_index, score_row in zip(selected, scores, strict=True):
                    query_id = query_rows[row_index][0]
                    order = np.lexsort((np.asarray(domain_corpus_ids), -score_row))
                    positions = {
                        domain_corpus_ids[corpus_position]: rank
                        for rank, corpus_position in enumerate(order, 1)
                    }
                    ranks_by_query[query_id] = sorted(
                        positions[positive_id] for positive_id in qrels[query_id]
                    )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    domain_metrics = {
        domain: metrics_from_ranks(
            [
                ranks_by_query[query_id]
                for query_id, _, query_domain in query_rows
                if query_domain == domain
            ]
        )
        for domain in ("finance", "knowledge")
    }
    macro_ndcg = fmean(row["ndcg_at_10"] for row in domain_metrics.values())
    safe = args.model.replace("/", "__")
    run_dir = args.output_dir.expanduser().resolve() / safe / revision
    run_dir.mkdir(parents=True, exist_ok=True)
    ranks_path = run_dir / "ranks.jsonl"
    with ranks_path.open("w", encoding="utf-8") as handle:
        for query_id, _, domain in query_rows:
            handle.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "domain": domain,
                        "relevant_ranks": ranks_by_query[query_id],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "model": args.model,
        "requested_revision": revision,
        "dataset": {
            "path": str(dataset),
            "manifest_sha256": manifest_sha,
            "selection_only": True,
            "public_benchmark": False,
            "domains": manifest["domains"],
        },
        "metrics": {
            "macro_domain_ndcg_at_10": macro_ndcg,
            "mean_query_ndcg_at_10": fmean(
                metrics_from_ranks([values])["ndcg_at_10"]
                for values in ranks_by_query.values()
            ),
        },
        "domain_metrics": domain_metrics,
        "query_prompt": QUERY_PROMPT,
        "score_contract": SCORE_CONTRACT,
        "files": {
            "ranks.jsonl": {"rows": len(query_rows), "sha256": sha256(ranks_path)}
        },
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "gpu": gpu_name(),
            "batch_size": effective_batch,
            "requested_batch_size": args.batch_size,
            "max_length": args.max_length,
            "requested_attention": args.attn_implementation,
            "attention": attention,
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
