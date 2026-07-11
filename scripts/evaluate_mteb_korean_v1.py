#!/usr/bin/env python3
"""Evaluate a dense encoder on the exact official MTEB Korean v1 benchmark.

The benchmark's task/split/subset/revision metadata is checked against a
committed protocol before model weights are loaded. For an unregistered
SentenceTransformer such as Comsat, this matches the current ``mteb run``
fallback: the model's own query/document prompts are used on asymmetric tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/mteb_korean_v1_protocol.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="sionic-ai/comsat-embed-ko-8b-preview"
    )
    parser.add_argument("--revision")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/evaluation/mteb_korean_v1")
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        help="Persist exact float32 encode chunks so interrupted retrieval can resume",
    )
    parser.add_argument(
        "--task",
        action="append",
        help="Run only this official task; repeat for a resumable partial run",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Validate and print the resolved protocol without loading weights",
    )
    return parser.parse_args()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol["benchmark"] != "MTEB(kor, v1)":
        raise ValueError("Protocol must target MTEB(kor, v1)")
    if len(protocol["tasks"]) != 6:
        raise ValueError("Official MTEB Korean v1 must contain exactly six tasks")
    return protocol


def _normalize_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    return {key: dataset[key] for key in ("path", "name", "revision") if key in dataset}


def resolve_and_validate_tasks(
    mteb: Any, protocol: dict[str, Any], selectors: list[str] | None
) -> tuple[list[Any], list[dict[str, Any]]]:
    benchmark = mteb.get_benchmark(protocol["benchmark"])
    benchmark_tasks = {task.metadata.name: task for task in benchmark.tasks}
    expected_names = [spec["name"] for spec in protocol["tasks"]]
    if set(benchmark_tasks) != set(expected_names):
        raise RuntimeError(
            "Installed MTEB benchmark membership changed: "
            f"expected={expected_names}, resolved={list(benchmark_tasks)}"
        )

    if selectors:
        unknown = set(selectors) - set(expected_names)
        if unknown:
            raise ValueError(f"Unknown Korean v1 tasks: {sorted(unknown)}")
        selected = set(selectors)
    else:
        selected = set(expected_names)

    tasks: list[Any] = []
    resolved: list[dict[str, Any]] = []
    for spec in protocol["tasks"]:
        task = benchmark_tasks[spec["name"]]
        actual = {
            "name": task.metadata.name,
            "type": task.metadata.type,
            "split": list(task.metadata.eval_splits),
            "hf_subsets": list(task.hf_subsets),
            "dataset": _normalize_dataset(dict(task.metadata.dataset)),
            "main_score": task.metadata.main_score,
            "task_prompt": task.metadata.prompt,
            "instruction_fallback": task.abstask_prompt,
        }
        expected = {
            "name": spec["name"],
            "type": spec["type"],
            "split": [spec["split"]],
            "hf_subsets": spec["hf_subsets"],
            "dataset": _normalize_dataset(spec["dataset"]),
            "main_score": spec["main_score"],
            "task_prompt": spec["task_prompt"],
            "instruction_fallback": spec["instruction_fallback"],
        }
        if actual != expected:
            raise RuntimeError(
                f"Installed MTEB metadata drifted for {spec['name']}: "
                f"expected={expected!r}, resolved={actual!r}"
            )
        resolved.append({**actual, "selected": spec["name"] in selected})
        if spec["name"] in selected:
            tasks.append(task)
    return tasks, resolved


def gpu_names() -> list[str]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def validate_mteb_checkout(protocol: dict[str, Any]) -> None:
    checkout = ROOT / "third_party/mteb"
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Cannot inspect pinned MTEB checkout at {checkout}") from error
    if revision != protocol["mteb_git_revision"]:
        raise RuntimeError(
            f"MTEB git mismatch: expected {protocol['mteb_git_revision']}, "
            f"found {revision}"
        )


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    validate_mteb_checkout(protocol)

    import mteb

    if mteb.__version__ != protocol["mteb_version"]:
        raise RuntimeError(
            f"MTEB version mismatch: expected {protocol['mteb_version']}, "
            f"found {mteb.__version__}"
        )
    tasks, resolved_tasks = resolve_and_validate_tasks(mteb, protocol, args.task)
    resolved_protocol = {
        **protocol,
        "protocol_path": str(args.protocol.resolve()),
        "resolved_tasks": resolved_tasks,
    }
    if args.list_only:
        print(json.dumps(resolved_protocol, ensure_ascii=False, indent=2))
        return

    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    revision = args.revision
    max_length = args.max_length
    if args.model == protocol["comsat"]["model"]:
        revision = revision or protocol["comsat"]["revision"]
        max_length = max_length or protocol["comsat"]["max_tokens"]
    if not revision:
        raise ValueError("--revision is required for models not pinned by this protocol")
    if not max_length:
        raise ValueError("--max-length is required for models not pinned by this protocol")

    model_class = SentenceTransformer
    model_extra: dict[str, Any] = {}
    if args.embedding_cache_dir is not None:
        try:
            from resumable_sentence_transformer import ResumableSentenceTransformer
        except ModuleNotFoundError:
            from scripts.resumable_sentence_transformer import ResumableSentenceTransformer
        model_class = ResumableSentenceTransformer
        model_extra = {
            "embedding_cache_dir": args.embedding_cache_dir,
            "embedding_cache_namespace": (
                f"{args.model}@{revision}|protocol={protocol['protocol_id']}|"
                f"max={max_length}|attn={args.attn_implementation}|"
                f"prompts={json.dumps(protocol['comsat']['effective_prompts'], sort_keys=True)}"
            ),
        }

    model = model_class(
        args.model,
        revision=revision,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        model_kwargs={
            "attn_implementation": args.attn_implementation,
            "torch_dtype": torch.bfloat16,
        },
        tokenizer_kwargs={"padding_side": "left"},
        **model_extra,
    )
    model.max_seq_length = max_length

    if args.model == protocol["comsat"]["model"]:
        expected_prompts = {
            "query": protocol["comsat"]["effective_prompts"]["query"],
            "document": protocol["comsat"]["effective_prompts"]["document"],
        }
        if model.prompts != expected_prompts or model.default_prompt_name is not None:
            raise RuntimeError(
                "Pinned Comsat prompt configuration changed: "
                f"prompts={model.prompts!r}, default={model.default_prompt_name!r}"
            )

    safe_model_name = args.model.replace("/", "__")
    run_dir = args.output_dir / safe_model_name / revision
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "protocol_resolved.json").write_text(
        json.dumps(resolved_protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    results = mteb.evaluate(
        model,
        tasks=tasks,
        cache=mteb.ResultCache(cache_path=run_dir / "mteb_cache"),
        overwrite_strategy="always" if args.overwrite else "only-missing",
        prediction_folder=run_dir / "predictions" if args.save_predictions else None,
        encode_kwargs={
            "batch_size": args.batch_size,
            "normalize_embeddings": True,
            "show_progress_bar": True,
        },
    )

    result_by_name = {result.task_name: result for result in results.task_results}
    score_rows: dict[str, dict[str, Any]] = {}
    type_scores: dict[str, list[float]] = defaultdict(list)
    for spec in protocol["tasks"]:
        if spec["name"] not in result_by_name:
            continue
        score = float(
            result_by_name[spec["name"]].get_score(splits=[spec["split"]])
        )
        score_rows[spec["name"]] = {
            "task_type": spec["type"],
            "metric": spec["main_score"],
            "score": score,
            "leaderboard_points": 100.0 * score,
        }
        type_scores[spec["type"]].append(score)

    means_by_type = {
        task_type: fmean(scores) for task_type, scores in type_scores.items()
    }
    complete = len(score_rows) == 6
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "benchmark": protocol["benchmark"],
        "model": args.model,
        "requested_revision": revision,
        "resolved_revision": results.model_revision,
        "completed_tasks": len(score_rows),
        "total_tasks": 6,
        "complete": complete,
        "mean_task": fmean(row["score"] for row in score_rows.values()),
        "mean_task_leaderboard_points": 100.0
        * fmean(row["score"] for row in score_rows.values()),
        "mean_task_type": fmean(means_by_type.values()),
        "mean_task_type_leaderboard_points": 100.0 * fmean(means_by_type.values()),
        "means_by_type": means_by_type,
        "scores": score_rows,
        "rank_borda": None,
        "rank_note": "Requires comparison against complete official leaderboard rows.",
        "environment": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "mteb": mteb.__version__,
            "gpu": gpu_names(),
            "batch_size": args.batch_size,
            "max_length": max_length,
            "attention": args.attn_implementation,
            "embedding_cache_dir": (
                str(args.embedding_cache_dir.resolve())
                if args.embedding_cache_dir is not None
                else None
            ),
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
