#!/usr/bin/env python3
"""Run the pinned nine-task Korean retrieval comparison.

The default mode applies one fixed Qwen/Sionic-style query prompt to every
model. Official MTEB evaluation with registered task-specific prompts is a
separate protocol and must not be mixed into this table.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

try:
    from evaluation_runtime import (
        effective_attention,
        enforce_runtime_contract,
        runtime_contract,
    )
except ModuleNotFoundError:
    from scripts.evaluation_runtime import (
        effective_attention,
        enforce_runtime_contract,
        runtime_contract,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/sionic9_protocol.json"
EXPECTED_LOADER_CONTRACT = {
    "id": "fixed-query-prompt-sentence-transformer-v1",
    "model_api": "sentence_transformers.SentenceTransformer",
    "prompt_mode": "fixed_query_document",
    "default_prompt_name": None,
    "padding_side": "left",
    "normalize_embeddings": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
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
    if protocol.get("protocol_id") != "sionic9-fixed-prompt-v1":
        raise ValueError("Protocol must target the fixed Sionic-9 comparison")
    if len(protocol["tasks"]) != 9:
        raise ValueError("The Sionic comparison protocol must contain exactly nine tasks")
    if protocol.get("mteb_version") != "2.18.0":
        raise ValueError("The Sionic comparison protocol must pin MTEB 2.18.0")
    git_revision = protocol.get("mteb_git_revision")
    if not isinstance(git_revision, str) or len(git_revision) != 40:
        raise ValueError("The Sionic comparison protocol must pin a full MTEB git SHA")
    if protocol.get("loader_contract") != EXPECTED_LOADER_CONTRACT:
        raise ValueError(
            "The Sionic fixed-prompt loader contract changed; update the evaluator "
            "and protocol together instead of reusing canonical results"
        )
    labels = [spec.get("label") for spec in protocol["tasks"]]
    names = [spec.get("name") for spec in protocol["tasks"]]
    if len(set(labels)) != 9 or len(set(names)) != 9:
        raise ValueError("Sionic task labels and MTEB task names must be unique")
    return protocol


def _normalize_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    return {key: dataset[key] for key in ("path", "name", "revision") if key in dataset}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve_selectors(
    specs: list[dict[str, Any]], selectors: list[str] | None
) -> set[str]:
    if not selectors:
        return {spec["name"] for spec in specs}
    by_selector = {
        selector: spec["name"]
        for spec in specs
        for selector in (spec["label"], spec["name"])
    }
    missing = set(selectors) - set(by_selector)
    if missing:
        raise ValueError(f"Unknown task selectors: {sorted(missing)}")
    return {by_selector[selector] for selector in selectors}


def resolve_and_validate_tasks(
    mteb: Any,
    protocol: dict[str, Any],
    selectors: list[str] | None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Resolve all nine tasks and reject any installed-MTEB metadata drift."""

    specs = protocol["tasks"]
    selected_names = _resolve_selectors(specs, selectors)
    tasks: list[Any] = []
    resolved: list[dict[str, Any]] = []
    for spec in specs:
        task = mteb.get_task(
            spec["name"],
            eval_splits=[spec["split"]],
            hf_subsets=spec["hf_subsets"],
        )
        metadata = task.metadata
        registry_dataset = _normalize_dataset(dict(metadata.dataset))
        expected_registry_dataset = _normalize_dataset(dict(spec["dataset"]))
        if "registry_dataset_revision" in spec:
            expected_registry_dataset["revision"] = spec["registry_dataset_revision"]
        actual = {
            "label": spec["label"],
            "name": metadata.name,
            "type": metadata.type,
            "selected_splits": list(task.eval_splits),
            "available_splits": list(metadata.eval_splits),
            "hf_subsets": list(task.hf_subsets),
            "registry_dataset": registry_dataset,
            "modalities": list(metadata.modalities),
            "category": metadata.category,
            "main_score": metadata.main_score,
            "license": metadata.license,
            "task_prompt": metadata.prompt,
            "instruction_fallback": task.abstask_prompt,
        }
        expected = {
            "label": spec["label"],
            "name": spec["name"],
            "type": spec["type"],
            "selected_splits": [spec["split"]],
            "available_splits": spec["available_splits"],
            "hf_subsets": spec["hf_subsets"],
            "registry_dataset": expected_registry_dataset,
            "modalities": spec["modalities"],
            "category": spec["category"],
            "main_score": spec["main_score"],
            "license": spec["license"],
            "task_prompt": spec["task_prompt"],
            "instruction_fallback": spec["instruction_fallback"],
        }
        if actual != expected:
            raise RuntimeError(
                f"Installed MTEB metadata drifted for {spec['name']}: "
                f"expected={expected!r}, resolved={actual!r}"
            )

        # Never mutate MTEB's shared class-level TaskMetadata.  PublicHealthQA
        # advertises ``main`` in the registry, while this protocol deliberately
        # loads a reviewed immutable dataset revision.
        task.metadata = copy.deepcopy(metadata)
        task.metadata.dataset = dict(task.metadata.dataset)
        task.metadata.dataset["revision"] = spec["dataset"]["revision"]
        row = {
            **actual,
            "dataset": _normalize_dataset(dict(task.metadata.dataset)),
            "selected": spec["name"] in selected_names,
        }
        resolved.append(row)
        if row["selected"]:
            tasks.append(task)
    return tasks, resolved


def validate_mteb_checkout(protocol: dict[str, Any]) -> str:
    checkout = ROOT / "third_party/mteb"
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"Cannot inspect pinned MTEB checkout at {checkout}"
        ) from error
    if revision != protocol["mteb_git_revision"]:
        raise RuntimeError(
            f"MTEB git mismatch: expected {protocol['mteb_git_revision']}, "
            f"found {revision}"
        )
    return revision


def validate_mteb_package(mteb: Any, protocol: dict[str, Any]) -> None:
    if mteb.__version__ != protocol["mteb_version"]:
        raise RuntimeError(
            f"MTEB version mismatch: expected {protocol['mteb_version']}, "
            f"found {mteb.__version__}"
        )


def build_resolved_protocol(
    protocol: dict[str, Any],
    protocol_path: Path,
    resolved_tasks: list[dict[str, Any]],
    checkout_revision: str,
) -> dict[str, Any]:
    task_contract = {
        "protocol_id": protocol["protocol_id"],
        "mteb_version": protocol["mteb_version"],
        "mteb_git_revision": checkout_revision,
        "loader_contract": protocol["loader_contract"],
        "query_prompt": protocol["query_prompt"],
        "document_prompt": protocol["document_prompt"],
        "similarity": protocol["similarity"],
        "metric": protocol["metric"],
        "aggregate": protocol["aggregate"],
        "resolved_tasks": [
            {key: value for key, value in row.items() if key != "selected"}
            for row in resolved_tasks
        ],
    }
    return {
        **protocol,
        "protocol_path": str(protocol_path.resolve()),
        "validated_environment": {
            "mteb_version": protocol["mteb_version"],
            "mteb_git_revision": checkout_revision,
        },
        "resolved_task_contract_sha256": _canonical_sha256(task_contract),
        "resolved_tasks": resolved_tasks,
    }


def _cached_result_jsons(run_dir: Path) -> list[Path]:
    result_root = run_dir / "mteb_cache" / "results"
    return sorted(result_root.rglob("*.json")) if result_root.is_dir() else []


def validate_existing_result_contract(
    run_dir: Path, resolved: dict[str, Any]
) -> None:
    """Refuse legacy or drifted cached task results before MTEB can reuse them."""

    cached = _cached_result_jsons(run_dir)
    if not cached:
        return
    runtime_path = run_dir / "runtime_contract.json"
    if not runtime_path.is_file():
        raise RuntimeError(
            "Existing completed Sionic result cache has no runtime_contract.json; "
            "legacy canonical results cannot be reused safely. Use a fresh --output-dir."
        )
    resolved_path = run_dir / "protocol_resolved.json"
    if not resolved_path.is_file():
        raise RuntimeError(
            "Existing completed Sionic result cache has no protocol_resolved.json; "
            "use a fresh --output-dir."
        )
    try:
        existing = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Existing Sionic protocol evidence is unreadable; use a fresh --output-dir."
        ) from error
    expected_sha = resolved["resolved_task_contract_sha256"]
    if existing.get("resolved_task_contract_sha256") != expected_sha:
        raise RuntimeError(
            "Existing completed Sionic results were produced under a legacy or "
            "different task/loader contract; use a fresh --output-dir "
            f"(expected task contract {expected_sha})."
        )


def gpu_name() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def local_merge_dtype(model: str) -> str:
    report = Path(model).expanduser() / "merge_report.json"
    if report.is_file():
        evidence = json.loads(report.read_text(encoding="utf-8"))
        if evidence.get("merge", {}).get("dtype") == "float32":
            return "float32"
    return "bfloat16"


def canonical_local_revision(model: str, requested: str | None) -> str | None:
    root = Path(model).expanduser()
    for name in ("merge_report.json", "full_tuning_report.json", "soup_report.json"):
        report = root / name
        if not report.is_file():
            continue
        evidence = json.loads(report.read_text(encoding="utf-8"))
        model_sha = evidence.get("model", {}).get("weights_sha256")
        if not isinstance(model_sha, str) or len(model_sha) != 64:
            raise ValueError(f"Invalid local model weight evidence: {report}")
        canonical = f"model-{model_sha[:12]}"
        if requested != canonical:
            print(
                f"Canonicalizing local revision {requested!r} -> {canonical!r}",
                file=sys.stderr,
            )
        return canonical
    return requested


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    checkout_revision = validate_mteb_checkout(protocol)

    import mteb

    validate_mteb_package(mteb, protocol)
    tasks, resolved_tasks = resolve_and_validate_tasks(mteb, protocol, args.task)
    resolved = build_resolved_protocol(
        protocol,
        args.protocol,
        resolved_tasks,
        checkout_revision,
    )
    if args.list_only:
        print(json.dumps(resolved, ensure_ascii=False, indent=2, default=str))
        return
    if not args.model:
        raise ValueError("--model is required unless --list-only is used")
    args.revision = canonical_local_revision(args.model, args.revision)

    evaluation_dtype = local_merge_dtype(args.model)
    effective_batch_size = (
        min(args.batch_size, 96)
        if evaluation_dtype == "float32"
        else args.batch_size
    )
    attention = effective_attention(args.attn_implementation, evaluation_dtype)

    safe_model_name = args.model.replace("/", "__")
    run_dir = args.output_dir / safe_model_name
    task_contract_sha = resolved["resolved_task_contract_sha256"]
    contract = runtime_contract(
        protocol_id=protocol["protocol_id"],
        protocol_path=args.protocol,
        model=args.model,
        revision=args.revision,
        batch_size=effective_batch_size,
        max_length=args.max_length,
        requested_attention=args.attn_implementation,
        attention=attention,
        evaluation_dtype=evaluation_dtype,
        loader_contract=protocol["loader_contract"]["id"],
        extra={
            "query_prompt": protocol["query_prompt"],
            "document_prompt": protocol["document_prompt"],
            "loader_contract": protocol["loader_contract"],
            "resolved_task_contract_sha256": task_contract_sha,
            "trust_remote_code": args.trust_remote_code,
        },
    )
    validate_existing_result_contract(run_dir, resolved)
    enforce_runtime_contract(run_dir, contract)

    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    torch_dtype = torch.float32 if evaluation_dtype == "float32" else torch.bfloat16

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
                f"profile={contract['profile_id']}|max={args.max_length}|"
                f"batch={effective_batch_size}|attn={attention}|dtype={evaluation_dtype}|"
                f"loader={protocol['loader_contract']['id']}|tasks={task_contract_sha}|"
                f"prompts={json.dumps({'query': protocol['query_prompt'], 'document': protocol['document_prompt']}, sort_keys=True)}"
            ),
        }

    model = model_class(
        args.model,
        revision=args.revision,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        model_kwargs={
            "attn_implementation": attention,
            "torch_dtype": torch_dtype,
        },
        tokenizer_kwargs={
            "padding_side": protocol["loader_contract"]["padding_side"]
        },
        **model_extra,
    )
    model.max_seq_length = args.max_length
    model.prompts = {
        "query": protocol["query_prompt"],
        "document": protocol["document_prompt"],
    }
    model.default_prompt_name = protocol["loader_contract"]["default_prompt_name"]
    expected_prompts = {
        "query": protocol["query_prompt"],
        "document": protocol["document_prompt"],
    }
    tokenizer = getattr(model, "tokenizer", None)
    if (
        model.prompts != expected_prompts
        or model.default_prompt_name is not None
        or tokenizer is None
        or tokenizer.padding_side != protocol["loader_contract"]["padding_side"]
    ):
        raise RuntimeError(
            "The fixed Sionic query/document prompt loader contract was not applied"
        )

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
            "batch_size": effective_batch_size,
            "normalize_embeddings": protocol["loader_contract"][
                "normalize_embeddings"
            ],
            "show_progress_bar": True,
        },
    )

    result_by_name = {result.task_name: result for result in results.task_results}
    scores: dict[str, float] = {}
    selected_specs = [
        spec
        for spec, row in zip(protocol["tasks"], resolved_tasks, strict=True)
        if row["selected"]
    ]
    for spec in selected_specs:
        task_result = result_by_name[spec["name"]]
        scores[spec["label"]] = float(task_result.get_score(splits=[spec["split"]]))

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "runtime_profile_id": contract["profile_id"],
        "resolved_task_contract_sha256": task_contract_sha,
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
            "batch_size": effective_batch_size,
            "requested_batch_size": args.batch_size,
            "max_length": args.max_length,
            "requested_attention": args.attn_implementation,
            "attention": attention,
            "torch_dtype": evaluation_dtype,
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
