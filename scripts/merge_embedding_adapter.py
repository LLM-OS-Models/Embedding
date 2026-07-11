#!/usr/bin/env python3
"""Merge a Qwen3-Embedding PEFT adapter and preserve its embedding contract.

This script deliberately treats weight merging and SentenceTransformers metadata
as two separate operations.  ``transformers``/PEFT only save the transformer;
without the metadata written here, a consumer can silently use the wrong pooler,
omit normalization, or lose the query prompt.

The heavy ML dependencies are imported only inside :func:`merge_adapter`, so the
contract helpers and their tests run without downloading or loading model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_BASE_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_BASE_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer "
    "the query\nQuery:"
)
DOCUMENT_PROMPT = ""
EXPECTED_MODULES = [
    {
        "idx": 0,
        "name": "0",
        "path": "",
        "type": "sentence_transformers.models.Transformer",
    },
    {
        "idx": 1,
        "name": "1",
        "path": "1_Pooling",
        "type": "sentence_transformers.models.Pooling",
    },
    {
        "idx": 2,
        "name": "2",
        "path": "2_Normalize",
        "type": "sentence_transformers.models.Normalize",
    },
]
EXPECTED_ST_CONFIG = {
    "prompts": {"query": QUERY_PROMPT, "document": DOCUMENT_PROMPT},
    "default_prompt_name": None,
    "similarity_fn_name": "cosine",
}
PROBE_ROWS = (
    ("query", "대한민국의 수도는 어디인가?"),
    ("query", "계약을 해제할 수 있는 요건은 무엇인가?"),
    ("query", "What is contrastive learning?"),
    ("document", "대한민국의 수도는 서울특별시이다."),
    ("document", "계약 해제는 법률 또는 약정에서 정한 요건에 따라 이루어진다."),
    (
        "document",
        "Contrastive learning trains representations by bringing positive pairs "
        "closer and separating negatives.",
    ),
)


@dataclass(frozen=True)
class ParityMetrics:
    rows: int
    dimensions: int
    minimum_row_cosine: float
    mean_row_cosine: float
    maximum_absolute_difference: float
    maximum_pairwise_score_difference: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge an ms-swift/PEFT Qwen3-Embedding LoRA into its base model, "
            "restore the SentenceTransformers contract, and verify embedding parity."
        )
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base-revision", default=DEFAULT_BASE_REVISION)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "auto"),
        help="Where to load and merge weights; CPU is the safe default.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--probe-max-length", type=int, default=256)
    parser.add_argument("--min-row-cosine", type=float, default=0.999)
    parser.add_argument("--max-absolute-difference", type=float, default=0.05)
    parser.add_argument("--max-pairwise-score-difference", type=float, default=0.01)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--keep-failed-staging",
        action="store_true",
        help="Keep the temporary output after a failed merge for debugging.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_weights_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    shards = sorted(root.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No model safetensors under {root}")
    for shard in shards:
        digest.update(shard.name.encode() + b"\0")
        with shard.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def adapter_weight_path(adapter_dir: Path) -> Path:
    candidates = (
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_model.bin",
    )
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise ValueError(
            "Adapter must contain exactly one of adapter_model.safetensors or "
            f"adapter_model.bin: {adapter_dir}"
        )
    return present[0]


def validate_adapter(adapter_dir: Path) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing adapter_config.json: {adapter_dir}")
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError("adapter_config.json must contain a JSON object")
    if config.get("peft_type") != "LORA":
        raise ValueError(
            f"Only LoRA adapters can be merged, found {config.get('peft_type')!r}"
        )
    if not isinstance(config.get("r"), int) or config["r"] <= 0:
        raise ValueError("Adapter rank must be a positive integer")
    if (
        not isinstance(config.get("target_modules"), list)
        or not config["target_modules"]
    ):
        raise ValueError("Adapter target_modules must be a non-empty list")
    weight_path = adapter_weight_path(adapter_dir)
    return {
        "config": config,
        "config_sha256": sha256(config_path),
        "weights_filename": weight_path.name,
        "weights_sha256": sha256(weight_path),
        "weights_bytes": weight_path.stat().st_size,
    }


def validate_adapter_base_reference(
    adapter_config: dict[str, Any], base_model: str, base_revision: str
) -> str:
    reference = adapter_config.get("base_model_name_or_path")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Adapter does not declare base_model_name_or_path")
    normalized_reference = reference.rstrip("/")
    normalized_base = base_model.rstrip("/")
    compatible = normalized_reference == normalized_base
    try:
        reference_path = Path(normalized_reference).expanduser().resolve()
        base_path = Path(normalized_base).expanduser().resolve()
        compatible = compatible or reference_path == base_path
        # ms-swift resolves a pinned Hub model to .../snapshots/<commit> before
        # PEFT writes adapter_config.json. Accept that exact immutable snapshot.
        compatible = compatible or (
            reference_path.name == base_revision
            and reference_path.parent.name == "snapshots"
        )
    except (OSError, RuntimeError):
        pass
    if not compatible:
        raise ValueError(
            "Adapter/base mismatch: adapter declares "
            f"{reference!r}, command requested {base_model!r}@{base_revision}"
        )
    return reference


def sentence_transformers_contract(hidden_size: int) -> dict[str, Any]:
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    return {
        "modules": EXPECTED_MODULES,
        "sentence_transformers": EXPECTED_ST_CONFIG,
        "pooling": {
            "word_embedding_dimension": hidden_size,
            "pooling_mode_cls_token": False,
            "pooling_mode_mean_tokens": False,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
            "pooling_mode_weightedmean_tokens": False,
            "pooling_mode_lasttoken": True,
            "include_prompt": True,
        },
    }


def write_sentence_transformers_contract(model_dir: Path, hidden_size: int) -> None:
    contract = sentence_transformers_contract(hidden_size)
    write_json(model_dir / "modules.json", contract["modules"])
    write_json(
        model_dir / "config_sentence_transformers.json",
        contract["sentence_transformers"],
    )
    write_json(model_dir / "1_Pooling" / "config.json", contract["pooling"])
    # Normalize has no parameters/config, but a real local directory makes the
    # module graph explicit and survives non-Hub packaging formats.
    (model_dir / "2_Normalize").mkdir(parents=True, exist_ok=True)


def validate_sentence_transformers_contract(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "tokenizer_config.json"
    required = (
        config_path,
        tokenizer_path,
        model_dir / "modules.json",
        model_dir / "config_sentence_transformers.json",
        model_dir / "1_Pooling" / "config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete merged model contract; missing={missing}")

    model_config = read_json(config_path)
    hidden_size = model_config.get("hidden_size")
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError(f"Invalid transformer hidden_size: {hidden_size!r}")
    expected = sentence_transformers_contract(hidden_size)
    actual_modules = read_json(model_dir / "modules.json")
    actual_st = read_json(model_dir / "config_sentence_transformers.json")
    actual_pool = read_json(model_dir / "1_Pooling" / "config.json")
    if actual_modules != expected["modules"]:
        raise ValueError(f"SentenceTransformers module graph drift: {actual_modules!r}")
    if actual_st != expected["sentence_transformers"]:
        raise ValueError(f"SentenceTransformers prompt/similarity drift: {actual_st!r}")
    if actual_pool != expected["pooling"]:
        raise ValueError(f"SentenceTransformers pooling drift: {actual_pool!r}")

    tokenizer_config = read_json(tokenizer_path)
    if tokenizer_config.get("padding_side") != "left":
        raise ValueError(
            "tokenizer_config.json must persist padding_side='left'; "
            f"found {tokenizer_config.get('padding_side')!r}"
        )
    if tokenizer_config.get("eos_token") != "<|im_end|>":
        raise ValueError("Qwen3-Embedding eos_token contract changed")
    if tokenizer_config.get("pad_token") != "<|endoftext|>":
        raise ValueError("Qwen3-Embedding pad_token contract changed")

    architectures = model_config.get("architectures")
    if architectures != ["Qwen3ForCausalLM"]:
        raise ValueError(
            "Merged checkpoint must retain Qwen3ForCausalLM for PEFT/vLLM parity; "
            f"found {architectures!r}"
        )
    return {
        "status": "pass",
        "hidden_size": hidden_size,
        "architectures": architectures,
        "pooling": "last_token",
        "normalize": True,
        "prompts": expected["sentence_transformers"]["prompts"],
        "default_prompt_name": None,
        "similarity_fn_name": "cosine",
        "padding_side": "left",
    }


def format_probe_rows(rows: Sequence[tuple[str, str]] = PROBE_ROWS) -> list[str]:
    formatted: list[str] = []
    for kind, text in rows:
        if kind == "query":
            formatted.append(QUERY_PROMPT + text)
        elif kind == "document":
            formatted.append(DOCUMENT_PROMPT + text)
        else:
            raise ValueError(f"Unknown probe row kind: {kind!r}")
    return formatted


def parity_metrics(before: Any, after: Any) -> ParityMetrics:
    """Compute normalized-row parity; accepts array-like objects.

    This helper intentionally uses only Python arithmetic so contract tests do
    not depend on NumPy or PyTorch. The real merge passes CPU ``Tensor.tolist``
    results into it.
    """

    left = before.tolist() if hasattr(before, "tolist") else before
    right = after.tolist() if hasattr(after, "tolist") else after
    if not left or len(left) != len(right):
        raise ValueError("Parity matrices must have the same non-zero row count")
    dimensions = len(left[0])
    if dimensions == 0 or any(len(row) != dimensions for row in (*left, *right)):
        raise ValueError("Parity matrices must be rectangular and shape-compatible")

    row_cosines: list[float] = []
    max_abs = 0.0
    for x, y in zip(left, right, strict=True):
        dot = sum(float(a) * float(b) for a, b in zip(x, y, strict=True))
        x_norm = math.sqrt(sum(float(a) ** 2 for a in x))
        y_norm = math.sqrt(sum(float(b) ** 2 for b in y))
        if x_norm == 0.0 or y_norm == 0.0:
            raise ValueError("Zero-norm embedding in parity probe")
        row_cosines.append(dot / (x_norm * y_norm))
        max_abs = max(
            max_abs,
            max(abs(float(a) - float(b)) for a, b in zip(x, y, strict=True)),
        )

    max_score_delta = 0.0
    for i in range(len(left)):
        for j in range(len(left)):
            before_score = sum(
                float(a) * float(b) for a, b in zip(left[i], left[j], strict=True)
            )
            after_score = sum(
                float(a) * float(b) for a, b in zip(right[i], right[j], strict=True)
            )
            max_score_delta = max(max_score_delta, abs(before_score - after_score))
    return ParityMetrics(
        rows=len(left),
        dimensions=dimensions,
        minimum_row_cosine=min(row_cosines),
        mean_row_cosine=sum(row_cosines) / len(row_cosines),
        maximum_absolute_difference=max_abs,
        maximum_pairwise_score_difference=max_score_delta,
    )


def _model_device(model: Any) -> Any:
    try:
        return model.get_input_embeddings().weight.device
    except AttributeError:
        return next(model.parameters()).device


def _backbone(model: Any) -> Any:
    root = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(root, getattr(root, "base_model_prefix", "model"), None)
    if backbone is None or backbone is root:
        backbone = getattr(root, "model", None)
    if backbone is None:
        raise TypeError(f"Cannot locate transformer backbone in {type(root).__name__}")
    return backbone


def encode_probe(model: Any, tokenizer: Any, max_length: int) -> Any:
    import torch
    import torch.nn.functional as functional

    tokenizer.padding_side = "left"
    inputs = tokenizer(
        format_probe_rows(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = _model_device(model)
    inputs = {name: value.to(device) for name, value in inputs.items()}
    backbone = _backbone(model)
    model.eval()
    with torch.inference_mode():
        outputs = backbone(**inputs, return_dict=True, use_cache=False)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"]
        if bool((mask[:, -1].sum() == mask.shape[0]).item()):
            pooled = hidden[:, -1]
        else:
            sequence_lengths = mask.sum(dim=1) - 1
            pooled = hidden[
                torch.arange(hidden.shape[0], device=hidden.device), sequence_lengths
            ]
        return functional.normalize(pooled.float(), p=2, dim=1).cpu()


def _load_kwargs(args: argparse.Namespace, dtype: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "revision": args.base_revision or None,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.device == "auto":
        kwargs["device_map"] = "auto"
    elif args.device == "cuda":
        kwargs["device_map"] = {"": "cuda:0"}
    else:
        kwargs["device_map"] = {"": "cpu"}
    return kwargs


def merge_adapter(args: argparse.Namespace, staging_dir: Path) -> dict[str, Any]:
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.probe_max_length <= 0:
        raise ValueError("--probe-max-length must be positive")
    if not -1.0 <= args.min_row_cosine <= 1.0:
        raise ValueError("--min-row-cosine must be in [-1, 1]")
    if args.max_absolute_difference < 0.0:
        raise ValueError("--max-absolute-difference must be non-negative")
    if args.max_pairwise_score_difference < 0.0:
        raise ValueError("--max-pairwise-score-difference must be non-negative")

    adapter_dir = args.adapter.expanduser().resolve()
    adapter = validate_adapter(adapter_dir)
    validate_adapter_base_reference(
        adapter["config"], args.base_model, args.base_revision
    )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.base_revision or None,
        padding_side="left",
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    def load_merge_probe(load_dtype: Any) -> tuple[Any, ParityMetrics]:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, **_load_kwargs(args, load_dtype)
        )
        if base.config.architectures != ["Qwen3ForCausalLM"]:
            raise RuntimeError(
                f"Unexpected base architecture: {base.config.architectures!r}"
            )
        peft_model = PeftModel.from_pretrained(
            base,
            str(adapter_dir),
            is_trainable=False,
            low_cpu_mem_usage=True,
        )
        before = encode_probe(peft_model, tokenizer, args.probe_max_length)
        merged_model = peft_model.merge_and_unload(safe_merge=True, progressbar=True)
        after = encode_probe(merged_model, tokenizer, args.probe_max_length)
        return merged_model, parity_metrics(before, after)

    def parity_passes(value: ParityMetrics) -> bool:
        return (
            value.minimum_row_cosine >= args.min_row_cosine
            and value.maximum_absolute_difference <= args.max_absolute_difference
            and value.maximum_pairwise_score_difference
            <= args.max_pairwise_score_difference
        )

    effective_dtype = args.dtype
    merged, metrics = load_merge_probe(dtype)
    if not parity_passes(metrics) and args.dtype != "float32":
        # PEFT keeps trained adapter matrices in FP32. Folding a large/high-rank
        # update directly into a BF16 base can lose enough information to move
        # retrieval scores. Retry in FP32 instead of weakening the parity gate.
        del merged
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        effective_dtype = "float32"
        print(
            "BF16 adapter folding missed the parity gate; retrying an exact "
            "FP32 merge.",
            file=sys.stderr,
        )
        merged, metrics = load_merge_probe(torch.float32)
    if metrics.minimum_row_cosine < args.min_row_cosine:
        raise RuntimeError(
            "LoRA/merged embedding parity failed: minimum row cosine "
            f"{metrics.minimum_row_cosine:.9f} < {args.min_row_cosine:.9f}"
        )
    if metrics.maximum_absolute_difference > args.max_absolute_difference:
        raise RuntimeError(
            "LoRA/merged embedding parity failed: maximum absolute difference "
            f"{metrics.maximum_absolute_difference:.9f} > "
            f"{args.max_absolute_difference:.9f}"
        )
    if metrics.maximum_pairwise_score_difference > args.max_pairwise_score_difference:
        raise RuntimeError(
            "LoRA/merged embedding parity failed: maximum pairwise score difference "
            f"{metrics.maximum_pairwise_score_difference:.9f} > "
            f"{args.max_pairwise_score_difference:.9f}"
        )

    # Qwen's embedding checkpoint intentionally has no lm_head weights even
    # though its config names Qwen3ForCausalLM. Transformers initializes a
    # random head on CausalLM load; it is unrelated to last-token embeddings,
    # non-deterministic, and very large. Preserve the upstream encoder-only
    # artifact contract instead of publishing that random tensor.
    omitted_heads: list[str] = []
    if getattr(merged, "lm_head", None) is not None:
        merged.lm_head = None
        omitted_heads.append("lm_head")

    # Saving only happens after numerical parity has passed.  The caller keeps
    # this in a temporary sibling and atomically renames it after all checks.
    merged.save_pretrained(
        staging_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(staging_dir)
    write_sentence_transformers_contract(staging_dir, int(merged.config.hidden_size))
    contract = validate_sentence_transformers_contract(staging_dir)

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "adapter": {key: value for key, value in adapter.items() if key != "config"},
        "model": {"weights_sha256": model_weights_sha256(staging_dir)},
        "adapter_config": {
            key: adapter["config"].get(key)
            for key in (
                "peft_type",
                "r",
                "lora_alpha",
                "lora_dropout",
                "target_modules",
                "use_dora",
                "use_rslora",
            )
        },
        "merge": {
            "safe_merge": True,
            "requested_dtype": args.dtype,
            "dtype": effective_dtype,
            "device": args.device,
            "max_shard_size": args.max_shard_size,
            "omitted_random_untrained_heads": omitted_heads,
        },
        "probe": {
            "rows": [
                {"kind": kind, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
                for kind, text in PROBE_ROWS
            ],
            "formatted_query_prompt": QUERY_PROMPT,
            "max_length": args.probe_max_length,
            "metrics": asdict(metrics),
            "thresholds": {
                "minimum_row_cosine": args.min_row_cosine,
                "maximum_absolute_difference": args.max_absolute_difference,
                "maximum_pairwise_score_difference": args.max_pairwise_score_difference,
            },
        },
        "sentence_transformers_contract": contract,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output path already exists; refusing to overwrite: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.merging-{uuid.uuid4().hex}"
    if staging_dir.exists():
        raise FileExistsError(f"Unexpected staging collision: {staging_dir}")
    staging_dir.mkdir()

    try:
        report = merge_adapter(args, staging_dir)
        write_json(staging_dir / "merge_report.json", report)
        # Verify that writing the report did not disturb model metadata.
        validate_sentence_transformers_contract(staging_dir)
        os.replace(staging_dir, output_dir)
    except BaseException:
        if not args.keep_failed_staging:
            shutil.rmtree(staging_dir, ignore_errors=True)
        else:
            print(f"Failed staging retained at {staging_dir}", file=sys.stderr)
        raise
    print(
        json.dumps(
            {**report, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
