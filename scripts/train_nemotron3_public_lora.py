#!/usr/bin/env python3
"""Resumable SentenceTransformers/PEFT LoRA training for Nemotron-3 Embed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REVISION_RE = re.compile(r"[0-9a-f]{40}")
QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer "
    "the query\nQuery:"
)
TRAINING_PROMPTS = {"anchor": QUERY_PROMPT}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path)
    parser.add_argument("--eval-manifest", type=Path)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mini-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--contract-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def validate_contract(args: argparse.Namespace) -> dict[str, Any]:
    if not REVISION_RE.fullmatch(args.revision):
        raise ValueError("--revision must be an immutable 40-hex commit")
    for path in (args.model, args.train, args.training_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.eval is not None and not args.eval.is_file():
        raise FileNotFoundError(args.eval)
    if args.eval_manifest is not None and not args.eval_manifest.is_file():
        raise FileNotFoundError(args.eval_manifest)
    if (args.eval is None) != (args.eval_manifest is None):
        raise ValueError("--eval and --eval-manifest must be supplied together")
    if args.max_steps > 1 and (args.eval is None or args.eval_manifest is None):
        raise ValueError(
            "Multi-step public training requires --eval and --eval-manifest for checkpoint validation"
        )
    eval_contract = None
    if args.eval is not None:
        if args.eval_manifest is None and args.max_steps > 1:
            raise ValueError("Multi-step public training requires --eval-manifest")
        if args.eval_manifest is not None:
            eval_manifest = read_object(args.eval_manifest)
            declared_eval = eval_manifest.get("files", {}).get(args.eval.name, {})
            if declared_eval.get("sha256") != sha256(args.eval):
                raise ValueError("Evaluation JSONL SHA does not match its manifest")
            assertions = eval_manifest.get("assertions", {})
            required_zero = (
                "selected_query_training_text_overlap",
                "selected_positive_training_text_overlap",
                "selected_negative_training_text_overlap",
                "selected_source_document_training_provenance_overlap",
            )
            if assertions.get("source_holdout_contract_verified") is not True or any(
                assertions.get(field) != 0 for field in required_zero
            ):
                raise ValueError("Evaluation manifest independence assertions failed")
            eval_contract = {
                "path": str(args.eval.resolve()),
                "sha256": sha256(args.eval),
                "manifest_path": str(args.eval_manifest.resolve()),
                "manifest_sha256": sha256(args.eval_manifest),
                "independence_verified": True,
            }
    config = read_object(args.model / "config.json")
    if config.get("model_type") != "ministral3" or config.get("architectures") != [
        "Ministral3Model"
    ]:
        raise ValueError("Model is not the pinned bidirectional Nemotron-3 backbone")
    modules = json.loads((args.model / "modules.json").read_text(encoding="utf-8"))
    if not isinstance(modules, list):
        raise ValueError("SentenceTransformers modules.json is not a list")
    pooling = read_object(args.model / "1_Pooling/config.json")
    if pooling.get("pooling_mode_mean_tokens") is not True:
        raise ValueError("Nemotron-3 mean-pooling contract drifted")
    if [item.get("type") for item in modules] != [
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Normalize",
    ]:
        raise ValueError("SentenceTransformers module graph drifted")
    manifest = read_object(args.training_manifest)
    if manifest.get("release_eligible") is not True or manifest.get("release_blockers"):
        raise ValueError("Public model training requires a rights-safe training manifest")
    if manifest.get("visibility") != "public":
        raise ValueError("Training manifest visibility must be public")
    declared = manifest.get("outputs", {}).get("train", {})
    if declared.get("sha256") != sha256(args.train):
        raise ValueError("Training JSONL SHA does not match the public manifest")
    if not isinstance(declared.get("rows"), int) or declared["rows"] < 2:
        raise ValueError("Training manifest has no valid row count")
    if args.max_steps < 1 or args.batch_size < 2 or args.mini_batch_size < 1:
        raise ValueError("Invalid training schedule")
    if args.batch_size % args.mini_batch_size:
        raise ValueError("--batch-size must be divisible by --mini-batch-size")
    if args.max_length < 32 or args.save_steps < 1:
        raise ValueError("Invalid length/checkpoint schedule")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": "nvidia/Nemotron-3-Embed-8B-BF16",
        "base_path": str(args.model.resolve()),
        "base_revision": args.revision,
        "base_license": "OpenMDW-1.1",
        "architecture": "Ministral3Model",
        "pooling": "masked-mean",
        "normalization": "l2",
        "training_prompts": {
            "anchor": QUERY_PROMPT,
            "positive": "",
            "negatives": "",
        },
        "training_data": {
            "path": str(args.train.resolve()),
            "sha256": sha256(args.train),
            "rows": declared["rows"],
            "manifest_path": str(args.training_manifest.resolve()),
            "manifest_sha256": sha256(args.training_manifest),
            "release_eligible": True,
            "visibility": "public",
        },
        **({"evaluation_data": eval_contract} if eval_contract is not None else {}),
        "input_contract": {
            "query": "stored Instruct/Query prefix stripped, then exact training_prompts.anchor prepended by the collator",
            "document": "source-native positive/negative text; no prefix",
            "max_length": args.max_length,
        },
        "adapter": {
            "type": "LORA",
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "optimization": {
            "loss": "CachedMultipleNegativesRankingLoss",
            "scale": 50.0,
            "hardness_mode": "all_negatives",
            "batch_size": args.batch_size,
            "mini_batch_size": args.mini_batch_size,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "bf16": True,
            "gradient_checkpointing": True,
        },
    }


def extract_text(group: Any, field: str, line_number: int) -> str:
    if not isinstance(group, list) or len(group) != 1:
        raise ValueError(f"line {line_number}: {field} must contain one message")
    message = group[0]
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError(f"line {line_number}: malformed {field}")
    return message["content"]


def strip_stored_query_instruction(text: str, line_number: int) -> str:
    stripped = text.strip()
    if not stripped.startswith("Instruct:") or "\nQuery:" not in stripped:
        raise ValueError(
            f"line {line_number}: query must contain one explicit stored Instruct/Query prefix"
        )
    query = stripped.rpartition("Query:")[2].strip()
    if not query:
        raise ValueError(f"line {line_number}: stored query body is empty")
    return query


def convert_example(example: dict[str, Any], index: int) -> dict[str, str]:
    positives = example.get("positive_messages")
    negatives = example.get("negative_messages")
    if not isinstance(positives, list) or len(positives) != 1:
        raise ValueError(f"line {index + 1}: exactly one positive is required")
    if not isinstance(negatives, list) or not negatives:
        raise ValueError(f"line {index + 1}: at least one negative is required")
    converted = {
        "anchor": strip_stored_query_instruction(
            extract_text(example.get("messages"), "messages", index + 1), index + 1
        ),
        "positive": extract_text(positives[0], "positive_messages[0]", index + 1),
    }
    for negative_index, group in enumerate(negatives):
        converted[f"negative_{negative_index + 1}"] = extract_text(
            group, f"negative_messages[{negative_index}]", index + 1
        )
    return converted


def latest_complete_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if not match:
            continue
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
        if all((path / name).is_file() for name in required):
            checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(0, None))[1]


def write_run_contract(path: Path, contract: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    comparable = dict(contract)
    comparable.pop("created_at_utc", None)
    if path.exists():
        existing = read_object(path)
        existing.pop("created_at_utc", None)
        if existing != comparable:
            raise ValueError("Existing output directory has a different run contract")
        return
    path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train(args: argparse.Namespace, contract: dict[str, Any]) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    model = SentenceTransformer(
        str(args.model),
        device="cuda",
        local_files_only=True,
        model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
        processor_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = args.max_length
    model.prompts = {}
    model.default_prompt_name = None
    model.add_adapter(
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
    )
    def load_strict_dataset(path: Path, description: str):
        dataset = load_dataset("json", data_files=str(path), split="train")
        original_columns = dataset.column_names
        return dataset.map(
            convert_example,
            with_indices=True,
            remove_columns=original_columns,
            desc=description,
        )

    dataset = load_strict_dataset(args.train, "Converting strict training rows")
    eval_dataset = (
        load_strict_dataset(args.eval, "Converting strict evaluation rows")
        if args.eval is not None
        else None
    )
    loss = CachedMultipleNegativesRankingLoss(
        model,
        scale=50.0,
        mini_batch_size=args.mini_batch_size,
        hardness_mode="all_negatives",
    )
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=5,
        logging_first_step=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=2,
        dataloader_persistent_workers=True,
        # Match the fixed Sionic/Qwen comparison contract exactly: only the
        # query/anchor receives an instruction. Documents and hard negatives
        # remain source-native text without a prefix.
        prompts=TRAINING_PROMPTS,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )
    resume = args.resume_from_checkpoint or latest_complete_checkpoint(args.output_dir)
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_model(str(args.output_dir / "final-adapter"))


def main() -> None:
    args = parse_args()
    contract = validate_contract(args)
    write_run_contract(args.output_dir / "run_contract.json", contract)
    if args.contract_only:
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return
    train(args, contract)


if __name__ == "__main__":
    main()
