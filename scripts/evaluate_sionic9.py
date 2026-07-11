#!/usr/bin/env python3
"""Run the pinned nine-task Korean retrieval comparison.

The default mode applies one fixed Qwen/Sionic-style query prompt to every
model. Official MTEB evaluation with registered task-specific prompts is a
separate protocol and must not be mixed into this table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/sionic9_protocol.json",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation/sionic9"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=8192)
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
        help="Run only this task label or MTEB task name; repeat for a partial smoke evaluation",
    )
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if len(protocol["tasks"]) != 9:
        raise ValueError("The Sionic comparison protocol must contain exactly nine tasks")
    return protocol


def build_tasks(mteb: Any, specs: list[dict[str, Any]]) -> list[Any]:
    tasks = []
    for spec in specs:
        task = mteb.get_task(
            spec["name"],
            eval_splits=[spec["split"]],
            hf_subsets=spec.get("hf_subsets"),
        )
        if "dataset_revision" in spec:
            task.metadata.dataset["revision"] = spec["dataset_revision"]
        if len(task.hf_subsets) != 1:
            raise ValueError(
                f"{spec['name']} resolved to {task.hf_subsets}; exactly one Korean subset is required"
            )
        tasks.append(task)
    return tasks


def resolved_protocol(
    protocol: dict[str, Any], specs: list[dict[str, Any]], tasks: list[Any]
) -> dict[str, Any]:
    resolved = dict(protocol)
    resolved["resolved_tasks"] = []
    for spec, task in zip(specs, tasks, strict=True):
        resolved["resolved_tasks"].append(
            {
                **spec,
                "resolved_hf_subsets": list(task.hf_subsets),
                "dataset": task.metadata.dataset,
                "main_score": task.metadata.main_score,
                "license": task.metadata.license,
            }
        )
    return resolved


def gpu_name() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    selected_specs = protocol["tasks"]
    if args.task:
        requested = set(args.task)
        selected_specs = [
            spec
            for spec in selected_specs
            if spec["label"] in requested or spec["name"] in requested
        ]
        matched = {spec["label"] for spec in selected_specs} | {
            spec["name"] for spec in selected_specs
        }
        missing = requested - matched
        if missing:
            raise ValueError(f"Unknown task selectors: {sorted(missing)}")

    import mteb

    tasks = build_tasks(mteb, selected_specs)
    resolved = resolved_protocol(protocol, selected_specs, tasks)
    if args.list_only:
        print(json.dumps(resolved, ensure_ascii=False, indent=2, default=str))
        return
    if not args.model:
        raise ValueError("--model is required unless --list-only is used")

    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

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
                f"{args.model}@{args.revision or 'unresolved'}|"
                f"protocol={protocol['protocol_id']}|max={args.max_length}|"
                f"attn={args.attn_implementation}|"
                f"prompts={json.dumps({'query': protocol['query_prompt'], 'document': protocol['document_prompt']}, sort_keys=True)}"
            ),
        }

    model = model_class(
        args.model,
        revision=args.revision,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        model_kwargs={
            "attn_implementation": args.attn_implementation,
            "torch_dtype": torch.bfloat16,
        },
        tokenizer_kwargs={"padding_side": "left"},
        **model_extra,
    )
    model.max_seq_length = args.max_length
    model.prompts = {
        "query": protocol["query_prompt"],
        "document": protocol["document_prompt"],
    }

    safe_model_name = args.model.replace("/", "__")
    run_dir = args.output_dir / safe_model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "protocol_resolved.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    cache = mteb.ResultCache(cache_path=run_dir / "mteb_cache")
    prediction_folder = run_dir / "predictions" if args.save_predictions else None
    results = mteb.evaluate(
        model,
        tasks=tasks,
        cache=cache,
        overwrite_strategy="always" if args.overwrite else "only-missing",
        prediction_folder=prediction_folder,
        encode_kwargs={
            "batch_size": args.batch_size,
            "normalize_embeddings": True,
            "show_progress_bar": True,
        },
    )

    result_by_name = {result.task_name: result for result in results.task_results}
    scores: dict[str, float] = {}
    for spec in selected_specs:
        task_result = result_by_name[spec["name"]]
        scores[spec["label"]] = float(task_result.get_score(splits=[spec["split"]]))

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": results.model_revision,
        "average": fmean(scores.values()) if len(scores) == 9 else None,
        "partial_average": fmean(scores.values()),
        "completed_tasks": len(scores),
        "total_protocol_tasks": 9,
        "scores": scores,
        "environment": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "mteb": mteb.__version__,
            "gpu": gpu_name(),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
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
