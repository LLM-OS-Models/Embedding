#!/usr/bin/env python3
"""Evaluate prompt/noise robustness on the verified 10K legal holdout."""

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
    from evaluate_legal_source_holdout import (
        DEFAULT_DATASET,
        QUERY_PROMPT,
        sha256,
        validate_dataset,
    )
    from evaluate_sionic9 import canonical_local_revision, local_merge_dtype
except ModuleNotFoundError:
    from scripts.evaluate_legal_source_holdout import (
        DEFAULT_DATASET,
        QUERY_PROMPT,
        sha256,
        validate_dataset,
    )
    from scripts.evaluate_sionic9 import canonical_local_revision, local_merge_dtype

try:
    from evaluation_runtime import effective_attention
except ModuleNotFoundError:
    from scripts.evaluation_runtime import effective_attention


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "legal-conversational-noise-i-v2-text-strict"
NOISE_RATIOS = (0.0, 0.01, 0.05)
NOISE_TEMPLATES = (
    "[시스템 메타데이터]\n세션 ID: {token}\n상태: 처리 완료\nassistant: 알겠습니다. 추가 질문이 있으면 말씀해 주세요.",
    "system: 대화 기록을 불러왔습니다.\nuser: 네\nassistant: 확인했습니다. 무엇을 도와드릴까요?\n기록 번호: {token}",
    "[대화 요약]\n사용자가 인사했습니다. 도우미가 응답했습니다. 구체적인 사실 정보는 없습니다.\n참조: {token}",
    "assistant/analysis: 요청을 확인하는 중입니다.\nassistant/final: 요청이 확인되었습니다.\ntrace: {token}",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/evaluation/conversational-noise-robustness",
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


def gpu_name() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def build_noise_documents(clean_corpus_size: int) -> list[tuple[str, str]]:
    count = math.ceil(clean_corpus_size * max(NOISE_RATIOS))
    rows: list[tuple[str, str]] = []
    for index in range(count):
        token = hashlib.sha256(f"{PROTOCOL_ID}:{index}".encode()).hexdigest()[:16]
        text = NOISE_TEMPLATES[index % len(NOISE_TEMPLATES)].format(token=token)
        rows.append((f"noise-dialogue-v1-{index:05d}", text))
    return rows


def summarize_condition(
    positive_ranks: list[int], noise_ranks: list[int] | None
) -> dict[str, Any]:
    ndcg10 = [
        1.0 / math.log2(rank + 1) if rank <= 10 else 0.0 for rank in positive_ranks
    ]
    result: dict[str, Any] = {
        "ndcg_at_10": fmean(ndcg10),
        "recall_at_10": fmean(1.0 if rank <= 10 else 0.0 for rank in positive_ranks),
        "mean_positive_rank": fmean(positive_ranks),
        "median_positive_rank": float(np.median(positive_ranks)),
    }
    if noise_ranks is not None:
        result["highest_noise_mean_rank"] = fmean(noise_ranks)
        result["highest_noise_median_rank"] = float(np.median(noise_ranks))
        for cutoff in (1, 5, 10):
            result[f"noise_intrusion_at_{cutoff}"] = fmean(
                1.0 if rank <= cutoff else 0.0 for rank in noise_ranks
            )
    return result


def rank_with_appended_noise(
    clean_scores: Any, noise_scores: Any, positive_indices: Any
) -> tuple[Any, Any | None]:
    """Rank clean positives and the best noise under the fixed corpus ordering."""

    import torch

    positive_scores = clean_scores.gather(1, positive_indices[:, None])
    clean_positions = torch.arange(clean_scores.shape[1], device=clean_scores.device)[
        None, :
    ]
    clean_greater = (clean_scores > positive_scores).sum(dim=1)
    clean_tied_before = (
        (clean_scores == positive_scores)
        & (clean_positions < positive_indices[:, None])
    ).sum(dim=1)
    if noise_scores.shape[1] == 0:
        return clean_greater + clean_tied_before + 1, None

    # A clean positive precedes every appended noise row, so exact noise ties
    # do not outrank the positive.
    noise_before_positive = (noise_scores > positive_scores).sum(dim=1)
    positive_ranks = clean_greater + clean_tied_before + noise_before_positive + 1
    best_noise_scores = noise_scores.max(dim=1).values[:, None]
    # All clean rows precede noise rows. torch.max chooses the first tied noise,
    # leaving only clean equal-score rows ahead of the best noise row.
    best_noise_ranks = 1 + (clean_scores >= best_noise_scores).sum(dim=1)
    return positive_ranks, best_noise_ranks


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
    effective_batch = (
        min(args.batch_size, 96) if evaluation_dtype == "float32" else args.batch_size
    )
    attention = effective_attention(args.attn_implementation, evaluation_dtype)
    manifest_hash = sha256(dataset / "manifest.json")
    # Match the clean evaluator namespace exactly so prompted queries and clean
    # corpus embeddings are cache hits. Raw queries and noise remain distinct
    # because the exact input strings are part of each cache key.
    cache_namespace = (
        f"{args.model}@{revision}|legal-source-heldout-i-v2-text-strict|manifest={manifest_hash}|"
        f"max={args.max_length}|batch={effective_batch}|attn={attention}|dtype={evaluation_dtype}|"
        f"prompt={hashlib.sha256(QUERY_PROMPT.encode()).hexdigest()}"
    )
    model = ResumableSentenceTransformer(
        args.model,
        revision=revision,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        model_kwargs={
            "attn_implementation": attention,
            "torch_dtype": torch_dtype,
        },
        tokenizer_kwargs={"padding_side": "left"},
        embedding_cache_dir=args.embedding_cache_dir,
        embedding_cache_namespace=cache_namespace,
    )
    model.max_seq_length = args.max_length

    query_rows = sorted((str(row["_id"]), str(row["text"])) for row in queries)
    corpus_rows = sorted((str(row["_id"]), str(row["text"]).strip()) for row in corpus)
    noise_rows = build_noise_documents(len(corpus_rows))
    encode_kwargs = {
        "batch_size": effective_batch,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": True,
    }
    prompted_vectors = np.asarray(
        model.encode([QUERY_PROMPT + text for _, text in query_rows], **encode_kwargs),
        dtype=np.float32,
    )
    raw_vectors = np.asarray(
        model.encode([text for _, text in query_rows], **encode_kwargs),
        dtype=np.float32,
    )
    corpus_vectors = np.asarray(
        model.encode([text for _, text in corpus_rows], **encode_kwargs),
        dtype=np.float32,
    )
    noise_vectors = np.asarray(
        model.encode([text for _, text in noise_rows], **encode_kwargs),
        dtype=np.float32,
    )
    matrices = (prompted_vectors, raw_vectors, corpus_vectors, noise_vectors)
    if any(array.ndim != 2 for array in matrices):
        raise RuntimeError("Embedding matrices must all be two-dimensional")
    dimensions = {array.shape[1] for array in matrices}
    if len(dimensions) != 1:
        raise RuntimeError("Embedding matrices have incompatible shapes")

    corpus_ids = [item[0] for item in corpus_rows]
    corpus_index = {value: index for index, value in enumerate(corpus_ids)}
    corpus_tensor = torch.from_numpy(corpus_vectors).to(args.device)
    noise_tensor = torch.from_numpy(noise_vectors).to(args.device)
    per_query: dict[str, dict[str, dict[str, int | None]]] = {
        query_id: {} for query_id, _ in query_rows
    }
    condition_metrics: dict[str, dict[str, Any]] = {}
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for prompt_name, query_vectors in (
            ("prompt_on", prompted_vectors),
            ("prompt_off", raw_vectors),
        ):
            condition_ranks = {
                ratio: {"positive": [], "noise": []} for ratio in NOISE_RATIOS
            }
            for start in range(0, len(query_rows), args.query_block_size):
                end = min(len(query_rows), start + args.query_block_size)
                query_tensor = torch.from_numpy(query_vectors[start:end]).to(
                    args.device
                )
                clean_scores = query_tensor @ corpus_tensor.T
                all_noise_scores = query_tensor @ noise_tensor.T
                positive_indices = torch.tensor(
                    [
                        corpus_index[positives[query_rows[index][0]]]
                        for index in range(start, end)
                    ],
                    device=clean_scores.device,
                    dtype=torch.long,
                )
                for ratio in NOISE_RATIOS:
                    noise_count = math.ceil(len(corpus_rows) * ratio)
                    positive_ranks, best_noise_ranks = rank_with_appended_noise(
                        clean_scores,
                        all_noise_scores[:, :noise_count],
                        positive_indices,
                    )
                    if best_noise_ranks is not None:
                        condition_ranks[ratio]["noise"].extend(
                            best_noise_ranks.cpu().tolist()
                        )
                    condition_ranks[ratio]["positive"].extend(
                        positive_ranks.cpu().tolist()
                    )

            clean_ndcg = summarize_condition(condition_ranks[0.0]["positive"], None)[
                "ndcg_at_10"
            ]
            for ratio in NOISE_RATIOS:
                ratio_key = f"{ratio:.2f}"
                noise_ranks = condition_ranks[ratio]["noise"] or None
                metrics = summarize_condition(
                    condition_ranks[ratio]["positive"], noise_ranks
                )
                metrics["ndcg_retention_vs_same_prompt_clean"] = (
                    metrics["ndcg_at_10"] / clean_ndcg if clean_ndcg else None
                )
                condition_metrics[f"{prompt_name}/noise_{ratio_key}"] = metrics
                for index, (query_id, _) in enumerate(query_rows):
                    per_query[query_id].setdefault(prompt_name, {})[ratio_key] = {
                        "positive_rank": condition_ranks[ratio]["positive"][index],
                        "highest_noise_rank": (
                            noise_ranks[index] if noise_ranks is not None else None
                        ),
                    }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    safe = args.model.replace("/", "__")
    run_dir = args.output_dir.resolve() / safe / revision
    run_dir.mkdir(parents=True, exist_ok=True)
    ranks_path = run_dir / "ranks.jsonl"
    with ranks_path.open("w", encoding="utf-8") as handle:
        for query_id, _ in query_rows:
            handle.write(
                json.dumps(
                    {"query_id": query_id, "conditions": per_query[query_id]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "model": args.model,
        "requested_revision": revision,
        "dataset": {
            "path": str(dataset),
            "manifest_sha256": manifest_hash,
            "independence_grade": manifest["independence"]["grade"],
            "not_grade": manifest["independence"]["not_grade"],
            "clean_queries": len(query_rows),
            "clean_corpus": len(corpus_rows),
            "max_noise_documents": len(noise_rows),
        },
        "conditions": condition_metrics,
        "query_prompt": QUERY_PROMPT,
        "noise": {
            "ratios": list(NOISE_RATIOS),
            "templates_sha256": hashlib.sha256(
                json.dumps(
                    NOISE_TEMPLATES, ensure_ascii=False, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "ordering": "clean corpus ID ascending, then synthetic noise ID ascending",
        },
        "score_contract": "exact float32 normalized dot; TF32 disabled; ties by combined corpus ordering",
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
            "embedding_dimensions": dimensions.pop(),
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
