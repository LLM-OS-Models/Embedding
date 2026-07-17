#!/usr/bin/env python3
"""Merge a selected Nemotron-3 LoRA while preserving its embedding contract."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.merge_embedding_adapter import (
        model_weights_sha256,
        parity_metrics,
        sha256,
        validate_adapter,
        validate_adapter_base_reference,
        write_json,
    )
    from scripts.model_lineage import resolve_base_lineage
    from scripts.train_nemotron3_public_lora import QUERY_PROMPT
except ImportError:  # pragma: no cover
    from merge_embedding_adapter import (
        model_weights_sha256,
        parity_metrics,
        sha256,
        validate_adapter,
        validate_adapter_base_reference,
        write_json,
    )
    from model_lineage import resolve_base_lineage
    from train_nemotron3_public_lora import QUERY_PROMPT


BASE_MODEL = "nvidia/Nemotron-3-Embed-8B-BF16"
BASE_REVISION = "2b29550c4ab0646bb6bb47032dda54ea11f6dfe2"
BASE_LICENSE = "OpenMDW-1.1"
EXPECTED_MODULE_TYPES = (
    "sentence_transformers.models.Transformer",
    "sentence_transformers.models.Pooling",
    "sentence_transformers.models.Normalize",
)
PROBE_ROWS = (
    QUERY_PROMPT + "대한민국의 수도는 어디인가?",
    QUERY_PROMPT + "계약을 해제할 수 있는 요건은 무엇인가?",
    "대한민국의 수도는 서울특별시이다.",
    "계약 해제는 법률 또는 약정에서 정한 요건에 따라 이루어진다.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-row-cosine", type=float, default=0.9999)
    parser.add_argument("--max-pairwise-score-difference", type=float, default=0.002)
    parser.add_argument("--contract-only", action="store_true")
    return parser.parse_args()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def validate_embedding_contract(model_dir: Path) -> dict[str, Any]:
    config = read_object(model_dir / "config.json")
    modules = json.loads((model_dir / "modules.json").read_text(encoding="utf-8"))
    pooling = read_object(model_dir / "1_Pooling/config.json")
    st_config = read_object(model_dir / "config_sentence_transformers.json")
    if config.get("model_type") != "ministral3" or config.get("architectures") != [
        "Ministral3Model"
    ]:
        raise ValueError("Nemotron bidirectional backbone contract drifted")
    if config.get("hidden_size") != 4096:
        raise ValueError("Nemotron hidden size drifted")
    if not isinstance(modules, list) or tuple(row.get("type") for row in modules) != EXPECTED_MODULE_TYPES:
        raise ValueError("SentenceTransformers module graph drifted")
    expected_pooling = {
        "pooling_mode_cls_token": False,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False,
        "pooling_mode_lasttoken": False,
        "include_prompt": True,
        "word_embedding_dimension": 4096,
    }
    if any(pooling.get(key) != value for key, value in expected_pooling.items()):
        raise ValueError("Nemotron masked-mean pooling contract drifted")
    prompts = st_config.get("prompts")
    if prompts not in (
        {"query": "query: ", "document": "passage: "},
        {"query": QUERY_PROMPT, "document": ""},
    ):
        raise ValueError("Nemotron prompt contract drifted")
    if st_config.get("default_prompt_name") is not None:
        raise ValueError("Nemotron must not apply an implicit default prompt")
    if st_config.get("similarity_fn_name") != "cosine":
        raise ValueError("Nemotron similarity contract drifted")
    return {
        "architecture": "Ministral3Model",
        "hidden_size": 4096,
        "pooling": "masked_mean",
        "normalize": True,
        "prompts": prompts,
        "default_prompt_name": None,
        "similarity_fn_name": "cosine",
    }


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if args.base_revision != BASE_REVISION:
        raise ValueError("Nemotron base revision drifted")
    if args.max_length < 32:
        raise ValueError("--max-length is too small")
    base = args.base_model.resolve()
    adapter = args.adapter.resolve()
    selection_path = args.selection.resolve()
    manifest_path = args.training_manifest.resolve()
    for path in (base, adapter, selection_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    base_contract = validate_embedding_contract(base)
    adapter_info = validate_adapter(adapter)
    validate_adapter_base_reference(adapter_info["config"], str(base), BASE_REVISION)
    selection = read_object(selection_path)
    manifest = read_object(manifest_path)
    selected = selection.get("selected", {})
    if selection.get("status") != "pass" or Path(str(selected.get("checkpoint", ""))).resolve() != adapter:
        raise ValueError("Adapter is not the selected checkpoint")
    if selected.get("adapter_weights_sha256") != adapter_info["weights_sha256"]:
        raise ValueError("Selected adapter weights SHA drifted")
    if selected.get("adapter_config_sha256") != adapter_info["config_sha256"]:
        raise ValueError("Selected adapter config SHA drifted")
    if manifest.get("release_eligible") is not True or manifest.get("release_blockers"):
        raise ValueError("Training manifest is not release eligible")
    if manifest.get("visibility") != "public":
        raise ValueError("Training manifest is not public")
    selection_manifest = selection.get("training_manifest", {})
    if selection_manifest.get("sha256") != sha256(manifest_path):
        raise ValueError("Selection belongs to a different training manifest")
    return {
        "base": base,
        "adapter": adapter,
        "selection": selection,
        "selection_sha256": sha256(selection_path),
        "manifest": manifest,
        "manifest_sha256": sha256(manifest_path),
        "adapter_info": adapter_info,
        "base_contract": base_contract,
    }


def encode_probe(model: Any) -> Any:
    return model.encode(
        list(PROBE_ROWS),
        prompt="",
        batch_size=len(PROBE_ROWS),
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=False,
    ).float().cpu()


def merge(args: argparse.Namespace, validated: dict[str, Any], staging: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = SentenceTransformer(
        str(validated["base"]),
        device=args.device,
        local_files_only=True,
        model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = args.max_length
    transformer = model[0]
    peft_model = PeftModel.from_pretrained(
        transformer.auto_model,
        str(validated["adapter"]),
        is_trainable=False,
        low_cpu_mem_usage=True,
    )
    transformer.model = peft_model
    before = encode_probe(model)
    transformer.model = peft_model.merge_and_unload(safe_merge=True, progressbar=True)
    after = encode_probe(model)
    parity = parity_metrics(before, after)
    if parity.minimum_row_cosine < args.min_row_cosine:
        raise RuntimeError("Nemotron adapter/merge row cosine parity failed")
    if parity.maximum_pairwise_score_difference > args.max_pairwise_score_difference:
        raise RuntimeError("Nemotron adapter/merge pairwise score parity failed")
    model.prompts = {"query": QUERY_PROMPT, "document": ""}
    model.default_prompt_name = None
    model.save_pretrained(str(staging), create_model_card=False, safe_serialization=True)
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        source = validated["base"] / name
        if source.is_file():
            shutil.copy2(source, staging / name)
    saved_contract = validate_embedding_contract(staging)
    weights_sha = model_weights_sha256(staging)
    selected = validated["selection"]["selected"]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "artifact_type": "nemotron3-peft-safe-merge",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "base_license": BASE_LICENSE,
        "upstream_base_models": resolve_base_lineage(BASE_MODEL, BASE_REVISION),
        "adapter": {
            "checkpoint": f"checkpoint-{selected['step']}",
            "step": selected["step"],
            "weights_sha256": validated["adapter_info"]["weights_sha256"],
            "config_sha256": validated["adapter_info"]["config_sha256"],
        },
        "selection": {
            "report_sha256": validated["selection_sha256"],
            "signal": validated["selection"]["selection_signal"],
            "public_benchmark_used": False,
        },
        "training_manifest": {
            "sha256": validated["manifest_sha256"],
            "visibility": "public",
            "release_eligible": True,
        },
        "merge": {"safe_merge": True, "dtype": "bfloat16"},
        "parity": asdict(parity),
        "sentence_transformers_contract": saved_contract,
        "model": {"weights_sha256": weights_sha},
    }
    write_json(staging / "merge_report.json", report)
    return report


def main() -> None:
    args = parse_args()
    validated = validate_inputs(args)
    if args.contract_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "base_contract": validated["base_contract"],
                    "adapter_weights_sha256": validated["adapter_info"]["weights_sha256"],
                    "selection_sha256": validated["selection_sha256"],
                    "training_manifest_sha256": validated["manifest_sha256"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Output directory already exists; refuse to overwrite")
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        report = merge(args, validated, staging)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
