#!/usr/bin/env python3
"""Evaluate pinned Korean MTEB baselines with the offline vLLM pooler.

This is intentionally separate from ``evaluate_mteb_korean_v1.py`` so that
SentenceTransformers and vLLM caches cannot be confused.  ``--list-only``
validates the benchmark and the small Hugging Face configuration files without
importing vLLM or initializing CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

try:
    from evaluate_mteb_korean_v1 import (
        DEFAULT_PROTOCOL,
        gpu_names,
        load_protocol,
        resolve_and_validate_tasks,
        validate_mteb_checkout,
    )
except ModuleNotFoundError:  # Supports ``python -m scripts...`` from the repo root.
    from scripts.evaluate_mteb_korean_v1 import (
        DEFAULT_PROTOCOL,
        gpu_names,
        load_protocol,
        resolve_and_validate_tasks,
        validate_mteb_checkout,
    )


QWEN_MODEL = "Qwen/Qwen3-Embedding-8B"
QWEN_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
QWEN_ARCHITECTURES = {"Qwen3ForCausalLM"}
COMSAT_ARCHITECTURES = {"Qwen3Model"}
DEFAULT_OUTPUT_DIR = Path("outputs/evaluation/mteb_korean_v1_vllm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="sionic-ai/comsat-embed-ko-8b-preview",
        choices=["sionic-ai/comsat-embed-ko-8b-preview", QWEN_MODEL],
    )
    parser.add_argument(
        "--revision",
        help="Optional assertion; must equal the committed revision for the selected model",
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="MTEB DataLoader batch size; vLLM performs its own continuous batching",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Pinned truncation limit used by the SentenceTransformers baseline",
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16"],
        help="Kept fixed for numerical and model-card parity",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--task",
        action="append",
        help="Run only this official Korean-v1 task; repeat for multiple tasks",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Validate protocol/model configs without importing vLLM or initializing CUDA",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hub_json(repo_id: str, revision: str, filename: str) -> tuple[dict[str, Any], str]:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id=repo_id, revision=revision, filename=filename))
    return json.loads(path.read_text(encoding="utf-8")), sha256_file(path)


def resolve_model(protocol: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    query_prompt = protocol["comsat"]["effective_prompts"]["query"]
    document_prompt = protocol["comsat"]["effective_prompts"]["document"]
    specs = {
        protocol["comsat"]["model"]: {
            "revision": protocol["comsat"]["revision"],
            "max_tokens": protocol["comsat"]["max_tokens"],
            "architectures": sorted(COMSAT_ARCHITECTURES),
        },
        QWEN_MODEL: {
            "revision": QWEN_REVISION,
            "max_tokens": 8192,
            "architectures": sorted(QWEN_ARCHITECTURES),
        },
    }
    spec = {"model": args.model, **specs[args.model]}
    if args.revision is not None and args.revision != spec["revision"]:
        raise ValueError(
            f"Revision mismatch for {args.model}: expected {spec['revision']}, "
            f"received {args.revision}"
        )
    if args.max_model_len != spec["max_tokens"]:
        raise ValueError(
            f"max_model_len is protocol-pinned to {spec['max_tokens']} for {args.model}; "
            f"received {args.max_model_len}"
        )
    if args.tensor_parallel_size != 1:
        raise ValueError("This entry point is pinned to one H100 (tensor_parallel_size=1)")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("--gpu-memory-utilization must be between 0 and 1")
    if args.max_num_batched_tokens < args.max_model_len:
        raise ValueError("--max-num-batched-tokens must be at least --max-model-len")
    if args.batch_size <= 0 or args.max_num_seqs <= 0:
        raise ValueError("batch sizes must be positive")
    spec["prompts"] = {"query": query_prompt, "document": document_prompt}
    return spec


def validate_model_card(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact pooler/prompt contract without loading model weights."""

    repo_id = spec["model"]
    revision = spec["revision"]
    config, config_hash = load_hub_json(repo_id, revision, "config.json")
    modules, modules_hash = load_hub_json(repo_id, revision, "modules.json")
    st_config, st_config_hash = load_hub_json(
        repo_id, revision, "config_sentence_transformers.json"
    )

    architectures = set(config.get("architectures", []))
    if architectures != set(spec["architectures"]):
        raise RuntimeError(
            f"Architecture drift for {repo_id}@{revision}: "
            f"expected={sorted(spec['architectures'])}, found={sorted(architectures)}"
        )

    pooling_modules = [item for item in modules if item["type"].split(".")[-1] == "Pooling"]
    normalize_modules = [
        item for item in modules if item["type"].split(".")[-1] == "Normalize"
    ]
    if len(pooling_modules) != 1 or len(normalize_modules) != 1:
        raise RuntimeError(
            f"Expected exactly one Pooling and one Normalize module, found {modules!r}"
        )
    pooling_path = pooling_modules[0]["path"]
    pooling_config, pooling_hash = load_hub_json(
        repo_id, revision, f"{pooling_path}/config.json"
    )
    if "pooling_mode" in pooling_config:
        last_token_only = pooling_config["pooling_mode"] == "lasttoken"
    else:
        disabled_modes = [
            "pooling_mode_cls_token",
            "pooling_mode_mean_tokens",
            "pooling_mode_max_tokens",
            "pooling_mode_mean_sqrt_len_tokens",
            "pooling_mode_weightedmean_tokens",
        ]
        last_token_only = pooling_config.get("pooling_mode_lasttoken") is True and not any(
            pooling_config.get(key, False) for key in disabled_modes
        )
    if not last_token_only:
        raise RuntimeError(f"Expected last-token-only pooling, found {pooling_config!r}")
    if pooling_config.get("include_prompt") is not True:
        raise RuntimeError("Pooling config must include the query prompt in the representation")

    actual_prompts = st_config.get("prompts")
    if actual_prompts != spec["prompts"]:
        raise RuntimeError(
            f"Prompt drift for {repo_id}@{revision}: "
            f"expected={spec['prompts']!r}, found={actual_prompts!r}"
        )
    if st_config.get("default_prompt_name") is not None:
        raise RuntimeError("default_prompt_name must remain null for symmetric Korean tasks")
    if st_config.get("similarity_fn_name") != "cosine":
        raise RuntimeError("The pinned models must declare cosine similarity")

    return {
        "architectures": sorted(architectures),
        "module_types": [item["type"] for item in modules],
        "pooling": "last_token",
        "normalization": "l2_via_sentence_transformers_normalize_module",
        "pooler_override": None,
        "prompts": actual_prompts,
        "default_prompt_name": st_config.get("default_prompt_name"),
        "similarity_fn_name": st_config.get("similarity_fn_name"),
        "config_sha256": config_hash,
        "modules_sha256": modules_hash,
        "pooling_config_sha256": pooling_hash,
        "sentence_transformers_config_sha256": st_config_hash,
    }


def engine_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "runner": "pooling",
        "convert": "embed",
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enforce_eager": args.enforce_eager,
        "trust_remote_code": args.trust_remote_code,
        "pooler_config": None,
    }


def summarize_results(
    *,
    protocol: dict[str, Any],
    results: Any,
    resolved_revision: str,
    model_spec: dict[str, Any],
    model_validation: dict[str, Any],
    args: argparse.Namespace,
    elapsed_seconds: float,
    versions: dict[str, str],
) -> dict[str, Any]:
    result_by_name = {result.task_name: result for result in results.task_results}
    score_rows: dict[str, dict[str, Any]] = {}
    type_scores: dict[str, list[float]] = defaultdict(list)
    for task_spec in protocol["tasks"]:
        if task_spec["name"] not in result_by_name:
            continue
        score = float(
            result_by_name[task_spec["name"]].get_score(splits=[task_spec["split"]])
        )
        score_rows[task_spec["name"]] = {
            "task_type": task_spec["type"],
            "metric": task_spec["main_score"],
            "score": score,
            "leaderboard_points": 100.0 * score,
        }
        type_scores[task_spec["type"]].append(score)

    means_by_type = {
        task_type: fmean(scores) for task_type, scores in type_scores.items()
    }
    task_mean = fmean(row["score"] for row in score_rows.values())
    type_mean = fmean(means_by_type.values())
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "benchmark": protocol["benchmark"],
        "backend": "vllm_offline_pooling",
        "model": model_spec["model"],
        "requested_revision": model_spec["revision"],
        "resolved_revision": resolved_revision,
        "completed_tasks": len(score_rows),
        "total_tasks": len(protocol["tasks"]),
        "complete": len(score_rows) == len(protocol["tasks"]),
        "mean_task": task_mean,
        "mean_task_leaderboard_points": 100.0 * task_mean,
        "mean_task_type": type_mean,
        "mean_task_type_leaderboard_points": 100.0 * type_mean,
        "means_by_type": means_by_type,
        "scores": score_rows,
        "rank_borda": None,
        "rank_note": "Requires comparison against complete official leaderboard rows.",
        "model_contract": model_validation,
        "engine": engine_config(args),
        "environment": {
            "python": os.sys.version,
            **versions,
            "gpu": gpu_names(),
            "mteb_dataloader_batch_size": args.batch_size,
            "elapsed_seconds": elapsed_seconds,
        },
    }


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
    model_spec = resolve_model(protocol, args)
    model_validation = validate_model_card(model_spec)
    resolved_protocol = {
        **protocol,
        "protocol_path": str(args.protocol.resolve()),
        "backend": "vllm_offline_pooling",
        "model": model_spec,
        "model_contract": model_validation,
        "engine": engine_config(args),
        "resolved_tasks": resolved_tasks,
    }
    if args.list_only:
        print(json.dumps(resolved_protocol, ensure_ascii=False, indent=2))
        return

    # Deliberately delayed: --list-only must not import vLLM or touch CUDA.
    import torch
    import transformers
    import vllm
    from huggingface_hub import __version__ as huggingface_hub_version
    from mteb.models.vllm_wrapper import VllmEncoderWrapper

    model = VllmEncoderWrapper(
        model=model_spec["model"],
        revision=model_spec["revision"],
        prompt_dict=model_spec["prompts"],
        use_instructions=False,
        apply_instruction_to_documents=False,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=args.enable_prefix_caching,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    # MTEB 2.18's static-prompt branch selects a prompt name through
    # ``model_prompts`` and then looks up its value in ``prompts_dict``.
    # Set both explicitly so query/document behavior cannot silently drift.
    model.model_prompts = dict(model_spec["prompts"])

    safe_model_name = model_spec["model"].replace("/", "__")
    run_dir = args.output_dir / safe_model_name / model_spec["revision"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "protocol_resolved.json").write_text(
        json.dumps(resolved_protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    started = time.perf_counter()
    results = mteb.evaluate(
        model,
        tasks=tasks,
        cache=mteb.ResultCache(cache_path=run_dir / "mteb_cache"),
        overwrite_strategy="always" if args.overwrite else "only-missing",
        prediction_folder=run_dir / "predictions" if args.save_predictions else None,
        encode_kwargs={"batch_size": args.batch_size},
    )
    elapsed_seconds = time.perf_counter() - started
    summary = summarize_results(
        protocol=protocol,
        results=results,
        resolved_revision=results.model_revision,
        model_spec=model_spec,
        model_validation=model_validation,
        args=args,
        elapsed_seconds=elapsed_seconds,
        versions={
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
            "mteb": mteb.__version__,
            "huggingface_hub": huggingface_hub_version,
        },
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
