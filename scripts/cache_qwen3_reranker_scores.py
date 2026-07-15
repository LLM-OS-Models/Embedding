#!/usr/bin/env python3
"""Build a resumable, provenance-complete Qwen3 reranker score cache.

The production backend is deliberately fixed to the locally cached
``Qwen/Qwen3-Reranker-8B`` commit.  It follows the prompt and yes/no next-token
scoring example in the official model card.  No token is read, no network
fallback is allowed, and ``trust_remote_code`` is always false.

The input JSONL contract is one query and its positive/candidate documents per
line.  The output never repeats query or document text: it stores stable hashes,
raw no/yes logits, and the normalized yes probability needed by the synthetic
query compiler.  Completed shards, state, the canonical combined JSONL, and the
manifest are all replaced atomically.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sqlite3
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, Sequence


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_HF_CACHE = ROOT / ".cache" / "huggingface" / "hub"
MODEL_ID = "Qwen/Qwen3-Reranker-8B"
MODEL_REVISION = "77d193c791ed757ca307ee72715aa132723da912"
DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
OFFICIAL_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n'
)
OFFICIAL_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROMPT_CONTRACT = "qwen3-reranker-official-model-card-yes-no-v1"
PRODUCTION_SCORE_FIELD = "reranker_score"
MOCK_SCORE_FIELD = "mock_reranker_score"
STATE_SCHEMA_VERSION = 1
OUTPUT_SCHEMA_VERSION = 1
MOCK_ALGORITHM = "sha256-logits-v1-non-production"
STATE_NAME = "state.json"
MANIFEST_NAME = "manifest.json"
SCORES_NAME = "scores.jsonl"
SHARD_DIRECTORY = "shards"
LOCK_NAME = ".score-cache.lock"
SHARD_RE = re.compile(r"part-([0-9]{6})\.jsonl\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SENSITIVE_VALUE_RES = (
    re.compile(r"\bhf_[A-Za-z0-9]{10,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{10,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{10,}", re.I),
)
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


class ScoreCacheError(RuntimeError):
    """A path-, text-, and secret-free failure suitable for CLI output."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class InputEvidence:
    sha256: str
    rows: int
    bytes: int

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class DocumentInput:
    candidate_id: str
    text: str
    retriever_score: float | None


@dataclass(frozen=True)
class InputRow:
    generated_id: str
    query: str
    positive: DocumentInput
    candidates: tuple[DocumentInput, ...]

    @property
    def documents(self) -> tuple[DocumentInput, ...]:
        return (self.positive, *self.candidates)


@dataclass(frozen=True)
class RawLogits:
    no: float
    yes: float


@dataclass(frozen=True)
class CacheOptions:
    input_path: Path
    output_dir: Path
    instruction: str = DEFAULT_INSTRUCTION
    shard_size: int = 64
    model_batch_size: int = 8
    max_documents_per_row: int = 201
    max_text_characters: int = 1_000_000
    max_length: int = 8192
    device: str = "cuda"
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"


class ScoreBackend(Protocol):
    score_field: str

    def scorer_provenance(self) -> dict[str, Any]: ...

    def runtime_provenance(self) -> dict[str, Any]: ...

    def score(
        self, instruction: str, query: str, documents: Sequence[str]
    ) -> list[RawLogits]: ...


def emit(event: str, **fields: Any) -> None:
    """Emit only explicitly selected safe fields, never local paths or text."""

    print(
        json.dumps(
            {"event": event, **fields},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )


def canonical_json_bytes(value: Any, *, newline: bool = True) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ScoreCacheError(
            "read_failed", f"artifact read failed ({type(error).__name__})"
        ) from None
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON object key")
        output[key] = value
    return output


def strict_json_loads(value: str) -> Any:
    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON number")

    return json.loads(
        value,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=reject_constant,
    )


def _expect_exact_keys(
    value: dict[str, Any], *, required: set[str], optional: set[str], label: str
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ScoreCacheError("invalid_schema", f"{label} is missing required fields")
    if unknown:
        raise ScoreCacheError("invalid_schema", f"{label} has unknown fields")


def _contains_sensitive_value(value: str) -> bool:
    return any(pattern.search(value) for pattern in SENSITIVE_VALUE_RES)


def _validate_public_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ScoreCacheError(
            "invalid_schema", f"{label} must be a non-empty trimmed string"
        )
    if len(value) > 512 or any(ord(character) < 32 for character in value):
        raise ScoreCacheError(
            "invalid_schema", f"{label} is not a safe public identifier"
        )
    if value.startswith(("/", "~/")) or re.match(r"[A-Za-z]:[\\/]", value):
        raise ScoreCacheError("unsafe_identifier", f"{label} must not be a local path")
    if _contains_sensitive_value(value):
        raise ScoreCacheError("unsafe_identifier", f"{label} resembles a credential")
    return value


def _validate_text(value: Any, label: str, max_characters: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScoreCacheError("invalid_schema", f"{label} must be non-empty text")
    if "\x00" in value or len(value) > max_characters:
        raise ScoreCacheError(
            "invalid_schema", f"{label} violates the text-size contract"
        )
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoreCacheError("invalid_schema", f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ScoreCacheError("nonfinite", f"{label} must be finite")
    return result


def parse_document(
    value: Any, *, label: str, max_text_characters: int
) -> DocumentInput:
    if not isinstance(value, dict):
        raise ScoreCacheError("invalid_schema", f"{label} must be an object")
    _expect_exact_keys(
        value,
        required={"candidate_id", "text"},
        optional={"retriever_score"},
        label=label,
    )
    retriever_score = value.get("retriever_score")
    return DocumentInput(
        candidate_id=_validate_public_id(
            value["candidate_id"], f"{label}.candidate_id"
        ),
        text=_validate_text(value["text"], f"{label}.text", max_text_characters),
        retriever_score=(
            None
            if retriever_score is None
            else _finite_number(retriever_score, f"{label}.retriever_score")
        ),
    )


def parse_input_row(
    value: Any,
    *,
    max_documents_per_row: int,
    max_text_characters: int,
) -> InputRow:
    if not isinstance(value, dict):
        raise ScoreCacheError("invalid_schema", "input row must be an object")
    _expect_exact_keys(
        value,
        required={"generated_id", "query", "positive", "candidates"},
        optional=set(),
        label="input row",
    )
    candidates_raw = value["candidates"]
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ScoreCacheError("invalid_schema", "candidates must be a non-empty list")
    if len(candidates_raw) + 1 > max_documents_per_row:
        raise ScoreCacheError(
            "invalid_schema", "input row exceeds the document-count contract"
        )
    positive = parse_document(
        value["positive"], label="positive", max_text_characters=max_text_characters
    )
    candidates = tuple(
        parse_document(
            candidate,
            label="candidates[]",
            max_text_characters=max_text_characters,
        )
        for candidate in candidates_raw
    )
    candidate_ids = [positive.candidate_id, *(item.candidate_id for item in candidates)]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ScoreCacheError("duplicate_candidate", "a row repeats a candidate_id")
    return InputRow(
        generated_id=_validate_public_id(value["generated_id"], "generated_id"),
        query=_validate_text(value["query"], "query", max_text_characters),
        positive=positive,
        candidates=candidates,
    )


def decode_input_line(raw: bytes, line_number: int, options: CacheOptions) -> InputRow:
    if not raw.strip():
        raise ScoreCacheError("invalid_jsonl", f"input line {line_number} is blank")
    try:
        text = raw.decode("utf-8")
        value = strict_json_loads(text)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ScoreCacheError(
            "invalid_jsonl",
            f"input line {line_number} is invalid ({type(error).__name__})",
        ) from None
    return parse_input_row(
        value,
        max_documents_per_row=options.max_documents_per_row,
        max_text_characters=options.max_text_characters,
    )


def iter_input_rows(path: Path, options: CacheOptions) -> Iterator[InputRow]:
    try:
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                yield decode_input_line(raw, line_number, options)
    except ScoreCacheError:
        raise
    except OSError as error:
        raise ScoreCacheError(
            "input_read_failed", f"input read failed ({type(error).__name__})"
        ) from None


def preflight_input(path: Path, options: CacheOptions) -> InputEvidence:
    """Validate every row and detect duplicate generated IDs without retaining text."""

    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ScoreCacheError(
                "invalid_input", "input must resolve to a regular file"
            )
    except FileNotFoundError:
        raise ScoreCacheError("missing_input", "input JSONL is unavailable") from None
    except ScoreCacheError:
        raise
    except OSError as error:
        raise ScoreCacheError(
            "input_read_failed", f"input stat failed ({type(error).__name__})"
        ) from None

    digest = hashlib.sha256()
    rows = 0
    byte_count = 0
    with tempfile.TemporaryDirectory(prefix="qwen-reranker-preflight-") as temporary:
        connection = sqlite3.connect(Path(temporary) / "ids.sqlite3")
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE generated_ids (id TEXT PRIMARY KEY)")
            try:
                with path.open("rb") as handle:
                    for line_number, raw in enumerate(handle, 1):
                        digest.update(raw)
                        byte_count += len(raw)
                        row = decode_input_line(raw, line_number, options)
                        try:
                            connection.execute(
                                "INSERT INTO generated_ids VALUES (?)",
                                (row.generated_id,),
                            )
                        except sqlite3.IntegrityError:
                            raise ScoreCacheError(
                                "duplicate_generated_id",
                                f"input repeats generated_id at line {line_number}",
                            ) from None
                        rows += 1
                connection.commit()
            except ScoreCacheError:
                raise
            except OSError as error:
                raise ScoreCacheError(
                    "input_read_failed",
                    f"input read failed ({type(error).__name__})",
                ) from None
        finally:
            connection.close()
    if rows < 1:
        raise ScoreCacheError("empty_input", "input JSONL has no rows")
    return InputEvidence(sha256=digest.hexdigest(), rows=rows, bytes=byte_count)


def validate_options(options: CacheOptions) -> None:
    if not 1 <= options.shard_size <= 10_000:
        raise ScoreCacheError("invalid_config", "shard_size must be in [1,10000]")
    if not 1 <= options.model_batch_size <= 1024:
        raise ScoreCacheError("invalid_config", "model_batch_size must be in [1,1024]")
    if not 2 <= options.max_documents_per_row <= 10_000:
        raise ScoreCacheError(
            "invalid_config", "max_documents_per_row must be in [2,10000]"
        )
    if not 1 <= options.max_text_characters <= 10_000_000:
        raise ScoreCacheError(
            "invalid_config", "max_text_characters must be in [1,10000000]"
        )
    if not 128 <= options.max_length <= 32_768:
        raise ScoreCacheError("invalid_config", "max_length must be in [128,32768]")
    if options.device not in {"cuda", "cpu"}:
        raise ScoreCacheError("invalid_config", "device must be cuda or cpu")
    if options.dtype not in {"bfloat16", "float16", "float32"}:
        raise ScoreCacheError("invalid_config", "unsupported dtype")
    if options.attention_implementation not in {"sdpa", "eager", "flash_attention_2"}:
        raise ScoreCacheError("invalid_config", "unsupported attention implementation")
    if not isinstance(options.instruction, str) or not options.instruction.strip():
        raise ScoreCacheError("invalid_config", "instruction must be non-empty")
    if len(options.instruction) > 4096 or "\x00" in options.instruction:
        raise ScoreCacheError(
            "invalid_config", "instruction violates the size contract"
        )
    if _contains_sensitive_value(options.instruction):
        raise ScoreCacheError(
            "unsafe_instruction", "instruction resembles a credential"
        )


def format_instruction(instruction: str, query: str, document: str) -> str:
    """The exact body formatter published in the official Qwen model card."""

    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def normalized_yes_probability(no_logit: float, yes_logit: float) -> float:
    """Stable float64 equivalent of softmax([no, yes])[yes]."""

    no_value = _finite_number(no_logit, "raw_no_logit")
    yes_value = _finite_number(yes_logit, "raw_yes_logit")
    difference = no_value - yes_value
    if difference >= 0:
        exponent = math.exp(-difference)
        result = exponent / (1.0 + exponent)
    else:
        exponent = math.exp(difference)
        result = 1.0 / (1.0 + exponent)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ScoreCacheError("nonfinite", "normalized yes probability is invalid")
    return result


def _metadata_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def enforce_offline_no_credentials(
    env: dict[str, str] | os._Environ[str] = os.environ,
) -> dict[str, Any]:
    """Keep the production scorer offline and remove inherited credentials."""

    for key, value in OFFLINE_ENVIRONMENT.items():
        env[key] = value
    removed = []
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "GITHUB", "GITHUB_TOKEN"):
        if key in env:
            removed.append(key)
            env.pop(key, None)
    return {
        "offline": True,
        "telemetry_disabled": True,
        "credential_names_removed": sorted(removed),
    }


def resolve_pinned_snapshot(
    snapshot_download: Callable[..., str] | None = None,
    *,
    cache_dir: Path = REPOSITORY_HF_CACHE,
    verify_weight_content: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Resolve only the exact local revision and validate its built-in Qwen layout."""

    if snapshot_download is None:
        try:
            from huggingface_hub import snapshot_download as hf_snapshot_download
        except ImportError:
            raise ScoreCacheError(
                "missing_dependency", "huggingface_hub is required"
            ) from None
        snapshot_download = hf_snapshot_download
    try:
        cache_root = cache_dir.resolve(strict=True)
        raw_path = snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            local_files_only=True,
            token=False,
        )
        unresolved_snapshot = Path(raw_path)
        snapshot_info = unresolved_snapshot.lstat()
        if unresolved_snapshot.is_symlink() or not stat.S_ISDIR(snapshot_info.st_mode):
            raise ValueError("unsafe snapshot directory")
        if snapshot_info.st_uid != os.geteuid():
            raise ValueError("unexpected snapshot owner")
        snapshot = unresolved_snapshot.resolve(strict=True)
    except Exception as error:
        raise ScoreCacheError(
            "missing_model",
            f"pinned local model snapshot is unavailable ({type(error).__name__})",
        ) from None
    if (
        snapshot.name != MODEL_REVISION
        or not snapshot.is_dir()
        or not snapshot.is_relative_to(cache_root)
    ):
        raise ScoreCacheError(
            "wrong_revision", "local model snapshot revision is not exact"
        )

    metadata_names = (
        "config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    )
    metadata: dict[str, str] = {}
    weight_blobs: dict[str, str] = {}
    weight_bytes: dict[str, int] = {}

    def checked_snapshot_file(name: str) -> Path:
        candidate = snapshot / name
        info = candidate.lstat()
        if info.st_uid != os.geteuid():
            raise ValueError("unexpected snapshot file owner")
        resolved = candidate.resolve(strict=True)
        resolved_info = resolved.lstat()
        if (
            not stat.S_ISREG(resolved_info.st_mode)
            or resolved_info.st_uid != os.geteuid()
            or not resolved.is_relative_to(cache_root)
        ):
            raise ValueError("unsafe snapshot file")
        return resolved

    try:
        for name in metadata_names:
            resolved = checked_snapshot_file(name)
            metadata[name] = sha256_file(resolved)
        config = strict_json_loads(
            (snapshot / "config.json").read_text(encoding="utf-8")
        )
        index = strict_json_loads(
            (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        if not isinstance(config, dict) or config.get("model_type") != "qwen3":
            raise ValueError("unexpected model type")
        architectures = config.get("architectures")
        if architectures != ["Qwen3ForCausalLM"]:
            raise ValueError("unexpected architecture")
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("missing weight map")
        shard_names = sorted(set(weight_map.values()))
        if any(
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
            for name in shard_names
        ):
            raise ValueError("unsafe weight shard name")
        for name in shard_names:
            resolved = checked_snapshot_file(name)
            # Hugging Face LFS blob names are their content SHA-256.  Binding the
            # revision to these local blob IDs avoids re-reading roughly 16 GB on
            # every resume while still exposing exact weight provenance.
            if not SHA256_RE.fullmatch(resolved.name):
                raise ValueError("weight blob has no SHA-256 identity")
            if verify_weight_content and sha256_file(resolved) != resolved.name:
                raise ValueError("weight blob content does not match its SHA-256 identity")
            weight_blobs[name] = resolved.name
            weight_bytes[name] = resolved.stat().st_size
    except ScoreCacheError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ScoreCacheError(
            "invalid_model",
            f"pinned local snapshot validation failed ({type(error).__name__})",
        ) from None
    evidence = {
        "metadata_sha256": metadata,
        "weight_shard_sha256": weight_blobs,
        "weight_shard_bytes": weight_bytes,
        "weight_shard_count": len(shard_names),
        "weight_content_verified": verify_weight_content,
    }
    evidence["snapshot_fingerprint_sha256"] = sha256_bytes(
        canonical_json_bytes({"revision": MODEL_REVISION, **evidence}, newline=False)
    )
    return snapshot, evidence


class Qwen3RerankerBackend:
    """Pinned local Qwen scorer implementing the official yes/no contract."""

    score_field = PRODUCTION_SCORE_FIELD

    @staticmethod
    def tokenizer_load_kwargs() -> dict[str, Any]:
        return {
            "local_files_only": True,
            "trust_remote_code": False,
            "token": False,
            "padding_side": "left",
        }

    @staticmethod
    def model_load_kwargs(
        *, torch_dtype: Any, attention_implementation: str
    ) -> dict[str, Any]:
        return {
            "local_files_only": True,
            "trust_remote_code": False,
            "token": False,
            "torch_dtype": torch_dtype,
            "attn_implementation": attention_implementation,
            "low_cpu_mem_usage": True,
        }

    def __init__(self, options: CacheOptions):
        self.options = options
        enforce_offline_no_credentials()
        snapshot, snapshot_evidence = resolve_pinned_snapshot(
            verify_weight_content=True
        )
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ScoreCacheError(
                "missing_dependency",
                f"model runtime is unavailable ({type(error).__name__})",
            ) from None
        try:
            transformers.logging.set_verbosity_error()
            transformers.logging.disable_progress_bar()
            if options.device == "cuda" and not torch.cuda.is_available():
                raise ScoreCacheError("missing_cuda", "CUDA is unavailable")
            dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[options.dtype]
            tokenizer = AutoTokenizer.from_pretrained(
                snapshot, **self.tokenizer_load_kwargs()
            )
            model = AutoModelForCausalLM.from_pretrained(
                snapshot,
                **self.model_load_kwargs(
                    torch_dtype=dtype,
                    attention_implementation=options.attention_implementation,
                ),
            )
            model = model.to(options.device).eval()
            actual_attention = getattr(model.config, "_attn_implementation", None)
            if actual_attention != options.attention_implementation:
                raise ScoreCacheError(
                    "runtime_mismatch",
                    "loaded attention implementation differs from the requested contract",
                )
            try:
                first_parameter = next(model.parameters())
            except StopIteration:
                raise ScoreCacheError(
                    "invalid_model", "loaded model has no parameters"
                ) from None
            actual_dtype = str(first_parameter.dtype).removeprefix("torch.")
            actual_device = first_parameter.device.type
            if actual_dtype != options.dtype or actual_device != options.device:
                raise ScoreCacheError(
                    "runtime_mismatch",
                    "loaded model dtype or device differs from contract",
                )
            torch.use_deterministic_algorithms(True)
            if options.device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = False
            token_no_id = tokenizer.convert_tokens_to_ids("no")
            token_yes_id = tokenizer.convert_tokens_to_ids("yes")
            if (
                not isinstance(token_no_id, int)
                or not isinstance(token_yes_id, int)
                or token_no_id == token_yes_id
                or tokenizer.encode("no", add_special_tokens=False) != [token_no_id]
                or tokenizer.encode("yes", add_special_tokens=False) != [token_yes_id]
            ):
                raise ScoreCacheError(
                    "invalid_tokens",
                    "official no/yes tokens are not single distinct tokens",
                )
            prefix_tokens = tokenizer.encode(OFFICIAL_PREFIX, add_special_tokens=False)
            suffix_tokens = tokenizer.encode(OFFICIAL_SUFFIX, add_special_tokens=False)
            if options.max_length <= len(prefix_tokens) + len(suffix_tokens):
                raise ScoreCacheError(
                    "invalid_config", "max_length leaves no body tokens"
                )
        except ScoreCacheError:
            raise
        except Exception as error:
            raise ScoreCacheError(
                "model_load_failed", f"local model load failed ({type(error).__name__})"
            ) from None
        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model
        self.token_no_id = token_no_id
        self.token_yes_id = token_yes_id
        self.prefix_tokens = prefix_tokens
        self.suffix_tokens = suffix_tokens
        self.snapshot_evidence = snapshot_evidence
        self._runtime = {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "huggingface_hub": _metadata_version("huggingface-hub"),
            "device_type": options.device,
            "device_name": (
                torch.cuda.get_device_name() if options.device == "cuda" else "cpu"
            ),
            "cuda_runtime": (
                str(torch.version.cuda)
                if options.device == "cuda"
                else "not_applicable"
            ),
            "cudnn": (
                str(torch.backends.cudnn.version())
                if options.device == "cuda"
                else "not_applicable"
            ),
            "compute_capability": (
                ".".join(str(value) for value in torch.cuda.get_device_capability())
                if options.device == "cuda"
                else "not_applicable"
            ),
            "dtype": options.dtype,
            "attention_implementation": actual_attention,
            "model_class": model.__class__.__name__,
            "tokenizer_class": tokenizer.__class__.__name__,
            "deterministic_algorithms": True,
            "tf32": False,
            "snapshot": snapshot_evidence,
        }

    def scorer_provenance(self) -> dict[str, Any]:
        return build_scorer_provenance(
            backend="pinned-local-qwen3-reranker",
            instruction=self.options.instruction,
            max_length=self.options.max_length,
            token_no_id=self.token_no_id,
            token_yes_id=self.token_yes_id,
            dtype=self.options.dtype,
            attention_implementation=self.options.attention_implementation,
        )

    def runtime_provenance(self) -> dict[str, Any]:
        return self._runtime

    def score(
        self, instruction: str, query: str, documents: Sequence[str]
    ) -> list[RawLogits]:
        if not documents:
            return []
        body_budget = (
            self.options.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        output: list[RawLogits] = []
        try:
            for start in range(0, len(documents), self.options.model_batch_size):
                batch_documents = documents[
                    start : start + self.options.model_batch_size
                ]
                formatted = [
                    format_instruction(instruction, query, document)
                    for document in batch_documents
                ]
                inputs = self.tokenizer(
                    formatted,
                    padding=False,
                    truncation="longest_first",
                    return_attention_mask=False,
                    max_length=body_budget,
                )
                for index, token_ids in enumerate(inputs["input_ids"]):
                    inputs["input_ids"][index] = (
                        self.prefix_tokens + token_ids + self.suffix_tokens
                    )
                inputs = self.tokenizer.pad(
                    inputs,
                    padding=True,
                    return_tensors="pt",
                    max_length=self.options.max_length,
                )
                inputs = {
                    key: value.to(self.model.device) for key, value in inputs.items()
                }
                with self.torch.inference_mode():
                    final_logits = self.model(**inputs).logits[:, -1, :]
                    no_logits = (
                        final_logits[:, self.token_no_id].float().cpu().tolist()
                    )
                    yes_logits = (
                        final_logits[:, self.token_yes_id].float().cpu().tolist()
                    )
                output.extend(
                    RawLogits(
                        no=_finite_number(no_value, "raw_no_logit"),
                        yes=_finite_number(yes_value, "raw_yes_logit"),
                    )
                    for no_value, yes_value in zip(
                        no_logits, yes_logits, strict=True
                    )
                )
        except Exception as error:
            raise ScoreCacheError(
                "inference_failed",
                f"reranker inference failed ({type(error).__name__})",
            ) from None
        if len(output) != len(documents):
            raise ScoreCacheError(
                "inference_failed", "reranker output count differs from input"
            )
        return output


class DeterministicMockBackend:
    """CPU-only test backend whose artifacts are explicitly non-admissible."""

    score_field = MOCK_SCORE_FIELD

    def __init__(self, options: CacheOptions):
        self.options = options

    def scorer_provenance(self) -> dict[str, Any]:
        return build_scorer_provenance(
            backend="mock-non-production",
            instruction=self.options.instruction,
            max_length=self.options.max_length,
            token_no_id=1000,
            token_yes_id=1001,
            dtype="float64-mock",
            attention_implementation="none-mock",
        )

    def runtime_provenance(self) -> dict[str, Any]:
        return {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "algorithm": MOCK_ALGORITHM,
            "device_type": "cpu-mock",
        }

    def score(
        self, instruction: str, query: str, documents: Sequence[str]
    ) -> list[RawLogits]:
        output: list[RawLogits] = []
        for document in documents:
            digest = hashlib.sha256(
                canonical_json_bytes(
                    [MOCK_ALGORITHM, instruction, query, document], newline=False
                )
            ).digest()
            no_integer = int.from_bytes(digest[:8], "big")
            yes_integer = int.from_bytes(digest[8:16], "big")
            scale = float(2**64 - 1)
            output.append(
                RawLogits(
                    no=(no_integer / scale) * 8.0 - 4.0,
                    yes=(yes_integer / scale) * 8.0 - 4.0,
                )
            )
        return output


def build_scorer_provenance(
    *,
    backend: str,
    instruction: str,
    max_length: int,
    token_no_id: int,
    token_yes_id: int,
    dtype: str,
    attention_implementation: str,
) -> dict[str, Any]:
    prompt_material = {
        "prefix": OFFICIAL_PREFIX,
        "body_format": "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}",
        "suffix": OFFICIAL_SUFFIX,
    }
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "backend": backend,
        "local_files_only": True,
        "trust_remote_code": False,
        "prompt_contract": PROMPT_CONTRACT,
        "prompt": instruction,
        "instruction": instruction,
        "instruction_sha256": sha256_bytes(instruction.encode("utf-8")),
        "prompt_template_sha256": sha256_bytes(
            canonical_json_bytes(prompt_material, newline=False)
        ),
        "max_length": max_length,
        "truncation": "longest_first_before_fixed_prefix_suffix",
        "padding_side": "left",
        "token_no": "no",
        "token_no_id": token_no_id,
        "token_yes": "yes",
        "token_yes_id": token_yes_id,
        "raw_logit_semantics": "last-position next-token logits before normalization",
        "normalization": "stable_float64_softmax_over_[no,yes]",
        "score_semantics": "normalized yes-token probability in [0,1]",
        "dtype": dtype,
        "attention_implementation": attention_implementation,
    }


def score_input_row(
    row: InputRow, backend: ScoreBackend, options: CacheOptions
) -> dict[str, Any]:
    documents = row.documents
    raw_scores: list[RawLogits] = []
    for start in range(0, len(documents), options.model_batch_size):
        batch = documents[start : start + options.model_batch_size]
        scores = backend.score(
            options.instruction, row.query, [document.text for document in batch]
        )
        if len(scores) != len(batch):
            raise ScoreCacheError(
                "backend_contract", "scorer returned an unexpected result count"
            )
        raw_scores.extend(scores)
    scored_documents: list[dict[str, Any]] = []
    for index, (document, logits) in enumerate(zip(documents, raw_scores, strict=True)):
        no_logit = _finite_number(logits.no, "raw_no_logit")
        yes_logit = _finite_number(logits.yes, "raw_yes_logit")
        output_document: dict[str, Any] = {
            "candidate_id": document.candidate_id,
            "role": "positive" if index == 0 else "candidate",
            "text_sha256": sha256_bytes(document.text.encode("utf-8")),
            "raw_no_logit": no_logit,
            "raw_yes_logit": yes_logit,
            backend.score_field: normalized_yes_probability(no_logit, yes_logit),
        }
        if document.retriever_score is not None:
            output_document["retriever_score"] = document.retriever_score
        scored_documents.append(output_document)
    return {
        "generated_id": row.generated_id,
        "query_sha256": sha256_bytes(row.query.encode("utf-8")),
        "score_field": backend.score_field,
        "scorer": backend.scorer_provenance(),
        "documents": scored_documents,
    }


def _safe_output_directory(path: Path) -> None:
    try:
        if path.exists() or path.is_symlink():
            info = path.lstat()
            if path.is_symlink() or not stat.S_ISDIR(info.st_mode):
                raise ScoreCacheError(
                    "unsafe_output", "output root must be a regular directory"
                )
            if info.st_uid != os.geteuid():
                raise ScoreCacheError(
                    "unsafe_output", "output root has an unexpected owner"
                )
        else:
            path.mkdir(parents=True, mode=0o700)
        os.chmod(path, 0o700)
        shard_dir = path / SHARD_DIRECTORY
        if shard_dir.exists() or shard_dir.is_symlink():
            shard_info = shard_dir.lstat()
            if shard_dir.is_symlink() or not stat.S_ISDIR(shard_info.st_mode):
                raise ScoreCacheError(
                    "unsafe_output", "shard root must be a regular directory"
                )
            if shard_info.st_uid != os.geteuid():
                raise ScoreCacheError(
                    "unsafe_output", "shard root has an unexpected owner"
                )
        else:
            shard_dir.mkdir(mode=0o700)
        os.chmod(shard_dir, 0o700)
    except ScoreCacheError:
        raise
    except OSError as error:
        raise ScoreCacheError(
            "output_setup_failed", f"output setup failed ({type(error).__name__})"
        ) from None


@contextlib.contextmanager
def exclusive_lock(output_dir: Path) -> Iterator[None]:
    lock_path = output_dir / LOCK_NAME
    try:
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            raise ScoreCacheError(
                "already_running", "another scorer owns the output lock"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
    except ScoreCacheError:
        raise
    except OSError as error:
        raise ScoreCacheError(
            "lock_failed", f"output lock failed ({type(error).__name__})"
        ) from None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: Path, payload: bytes) -> None:
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    except OSError as error:
        raise ScoreCacheError(
            "atomic_write_failed", f"atomic write failed ({type(error).__name__})"
        ) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        info = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
        ):
            raise ScoreCacheError("unsafe_artifact", f"{label} must be a regular file")
        value = strict_json_loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ScoreCacheError("missing_artifact", f"{label} is unavailable") from None
    except ScoreCacheError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ScoreCacheError(
            "invalid_artifact", f"{label} is invalid ({type(error).__name__})"
        ) from None
    if not isinstance(value, dict):
        raise ScoreCacheError("invalid_artifact", f"{label} must be an object")
    return value


def _contract(options: CacheOptions, backend: ScoreBackend) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "instruction_sha256": sha256_bytes(options.instruction.encode("utf-8")),
        "prompt_contract": PROMPT_CONTRACT,
        "score_field": backend.score_field,
        "shard_size": options.shard_size,
        "model_batch_size": options.model_batch_size,
        "max_documents_per_row": options.max_documents_per_row,
        "max_text_characters": options.max_text_characters,
        "max_length": options.max_length,
        "device_type": (
            options.device
            if backend.score_field == PRODUCTION_SCORE_FIELD
            else "cpu-mock"
        ),
        "dtype": (
            options.dtype
            if backend.score_field == PRODUCTION_SCORE_FIELD
            else "float64-mock"
        ),
        "attention_implementation": (
            options.attention_implementation
            if backend.score_field == PRODUCTION_SCORE_FIELD
            else "none-mock"
        ),
    }


def _run_identity(
    evidence: InputEvidence, options: CacheOptions, backend: ScoreBackend
) -> dict[str, Any]:
    scorer = backend.scorer_provenance()
    runtime = backend.runtime_provenance()
    contract = _contract(options, backend)
    material = {
        "input": evidence.as_dict(),
        "contract": contract,
        "scorer": scorer,
        "runtime": runtime,
    }
    return {
        **material,
        "run_fingerprint_sha256": sha256_bytes(
            canonical_json_bytes(material, newline=False)
        ),
    }


def new_state(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "status": "running",
        **identity,
        "completed_shards": [],
        "next_row": 0,
    }


def validate_state(
    state: dict[str, Any], identity: dict[str, Any], evidence: InputEvidence
) -> None:
    required = {
        "schema_version",
        "status",
        "input",
        "contract",
        "scorer",
        "runtime",
        "run_fingerprint_sha256",
        "completed_shards",
        "next_row",
    }
    optional = {"output"}
    _expect_exact_keys(state, required=required, optional=optional, label="state")
    if state["schema_version"] != STATE_SCHEMA_VERSION:
        raise ScoreCacheError("state_mismatch", "state schema version differs")
    if state["status"] not in {"running", "complete"}:
        raise ScoreCacheError("state_mismatch", "state status is invalid")
    for key in (
        "input",
        "contract",
        "scorer",
        "runtime",
        "run_fingerprint_sha256",
    ):
        if state[key] != identity[key]:
            raise ScoreCacheError(
                "state_mismatch", f"state {key} differs from this run"
            )
    if state["input"] != evidence.as_dict():
        raise ScoreCacheError("input_drift", "input SHA or row count changed")
    shards = state["completed_shards"]
    if not isinstance(shards, list):
        raise ScoreCacheError("invalid_state", "completed_shards must be a list")
    expected_start = 0
    for expected_index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ScoreCacheError("invalid_state", "shard state must be an object")
        _expect_exact_keys(
            shard,
            required={
                "index",
                "name",
                "start_row",
                "end_row_exclusive",
                "rows",
                "bytes",
                "sha256",
            },
            optional=set(),
            label="shard state",
        )
        expected_name = f"part-{expected_index:06d}.jsonl"
        if (
            shard["index"] != expected_index
            or shard["name"] != expected_name
            or shard["start_row"] != expected_start
            or not isinstance(shard["rows"], int)
            or shard["rows"] < 1
            or shard["rows"] > state["contract"]["shard_size"]
            or shard["end_row_exclusive"] != expected_start + shard["rows"]
            or not isinstance(shard["bytes"], int)
            or shard["bytes"] < 1
            or not isinstance(shard["sha256"], str)
            or not SHA256_RE.fullmatch(shard["sha256"])
        ):
            raise ScoreCacheError(
                "invalid_state", "shard state is not contiguous and exact"
            )
        expected_start = shard["end_row_exclusive"]
    if state["next_row"] != expected_start or not 0 <= expected_start <= evidence.rows:
        raise ScoreCacheError("invalid_state", "state next_row is inconsistent")
    if state["status"] == "complete" and "output" not in state:
        raise ScoreCacheError("invalid_state", "complete state has no output evidence")
    if state["status"] == "running" and "output" in state:
        raise ScoreCacheError(
            "invalid_state", "running state unexpectedly has output evidence"
        )


def _load_or_initialize_state(
    output_dir: Path, identity: dict[str, Any], evidence: InputEvidence
) -> dict[str, Any]:
    state_path = output_dir / STATE_NAME
    if state_path.exists() or state_path.is_symlink():
        state = read_json_object(state_path, "state")
        validate_state(state, identity, evidence)
        return state
    shard_dir = output_dir / SHARD_DIRECTORY
    existing_shards = list(shard_dir.iterdir())
    if existing_shards or any(
        (output_dir / name).exists() or (output_dir / name).is_symlink()
        for name in (SCORES_NAME, MANIFEST_NAME)
    ):
        raise ScoreCacheError(
            "orphan_artifact", "output root has artifacts but no state"
        )
    state = new_state(identity)
    atomic_write(state_path, canonical_json_bytes(state))
    return state


def _decode_output_line(raw: bytes, label: str) -> dict[str, Any]:
    if not raw.strip():
        raise ScoreCacheError("invalid_shard", f"{label} contains a blank line")
    try:
        value = strict_json_loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ScoreCacheError(
            "invalid_shard", f"{label} contains invalid JSON ({type(error).__name__})"
        ) from None
    if not isinstance(value, dict):
        raise ScoreCacheError("invalid_shard", f"{label} row must be an object")
    return value


def validate_output_row(
    output: dict[str, Any],
    expected: InputRow,
    *,
    scorer: dict[str, Any],
    score_field: str,
) -> None:
    _expect_exact_keys(
        output,
        required={"generated_id", "query_sha256", "score_field", "scorer", "documents"},
        optional=set(),
        label="score row",
    )
    if (
        output["generated_id"] != expected.generated_id
        or output["query_sha256"] != sha256_bytes(expected.query.encode("utf-8"))
        or output["score_field"] != score_field
        or output["scorer"] != scorer
    ):
        raise ScoreCacheError(
            "output_mismatch", "score row identity differs from input or scorer"
        )
    documents = output["documents"]
    if not isinstance(documents, list) or len(documents) != len(expected.documents):
        raise ScoreCacheError("output_mismatch", "score row document count differs")
    for index, (actual, source) in enumerate(
        zip(documents, expected.documents, strict=True)
    ):
        if not isinstance(actual, dict):
            raise ScoreCacheError("invalid_shard", "scored document must be an object")
        optional = {"retriever_score"} if source.retriever_score is not None else set()
        _expect_exact_keys(
            actual,
            required={
                "candidate_id",
                "role",
                "text_sha256",
                "raw_no_logit",
                "raw_yes_logit",
                score_field,
            },
            optional=optional,
            label="scored document",
        )
        if (
            actual["candidate_id"] != source.candidate_id
            or actual["role"] != ("positive" if index == 0 else "candidate")
            or actual["text_sha256"] != sha256_bytes(source.text.encode("utf-8"))
        ):
            raise ScoreCacheError("output_mismatch", "scored document identity differs")
        if source.retriever_score is not None:
            if (
                _finite_number(actual.get("retriever_score"), "retriever_score")
                != source.retriever_score
            ):
                raise ScoreCacheError("output_mismatch", "retriever score differs")
        no_logit = _finite_number(actual["raw_no_logit"], "raw_no_logit")
        yes_logit = _finite_number(actual["raw_yes_logit"], "raw_yes_logit")
        probability = _finite_number(actual[score_field], score_field)
        expected_probability = normalized_yes_probability(no_logit, yes_logit)
        if not 0.0 <= probability <= 1.0 or not math.isclose(
            probability, expected_probability, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ScoreCacheError(
                "normalization_mismatch", "yes probability does not match logits"
            )


def _validate_shard(
    path: Path,
    *,
    expected_rows: Iterator[InputRow],
    scorer: dict[str, Any],
    score_field: str,
    expected_metadata: dict[str, Any] | None,
    index: int,
    start_row: int,
    max_rows: int,
) -> dict[str, Any]:
    try:
        info = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
        ):
            raise ScoreCacheError("unsafe_artifact", "shard must be a regular file")
        digest = hashlib.sha256()
        byte_count = 0
        rows = 0
        with path.open("rb") as handle:
            for raw in handle:
                digest.update(raw)
                byte_count += len(raw)
                rows += 1
                if rows > max_rows:
                    raise ScoreCacheError(
                        "invalid_shard", "shard exceeds configured row count"
                    )
                try:
                    source = next(expected_rows)
                except StopIteration:
                    raise ScoreCacheError(
                        "invalid_shard", "shard exceeds input row count"
                    ) from None
                validate_output_row(
                    _decode_output_line(raw, "shard"),
                    source,
                    scorer=scorer,
                    score_field=score_field,
                )
    except ScoreCacheError:
        raise
    except OSError as error:
        raise ScoreCacheError(
            "shard_read_failed", f"shard read failed ({type(error).__name__})"
        ) from None
    if rows < 1:
        raise ScoreCacheError("invalid_shard", "shard is empty")
    metadata = {
        "index": index,
        "name": f"part-{index:06d}.jsonl",
        "start_row": start_row,
        "end_row_exclusive": start_row + rows,
        "rows": rows,
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
    }
    if expected_metadata is not None and metadata != expected_metadata:
        raise ScoreCacheError("shard_drift", "shard SHA, size, or row count changed")
    return metadata


def _remove_incomplete_temporary_shards(shard_dir: Path) -> None:
    for candidate in shard_dir.iterdir():
        if not candidate.name.startswith(".part-") or not candidate.name.endswith(
            ".tmp"
        ):
            continue
        info = candidate.lstat()
        if (
            candidate.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
        ):
            raise ScoreCacheError("unsafe_artifact", "temporary shard is unsafe")
        candidate.unlink()


def validate_and_recover_shards(
    output_dir: Path,
    state: dict[str, Any],
    options: CacheOptions,
) -> Iterator[InputRow]:
    """Validate tracked shards and adopt an atomic shard from a crash window."""

    shard_dir = output_dir / SHARD_DIRECTORY
    _remove_incomplete_temporary_shards(shard_dir)
    input_rows = iter_input_rows(options.input_path, options)
    tracked_names: set[str] = set()
    for metadata in state["completed_shards"]:
        tracked_names.add(metadata["name"])
        _validate_shard(
            shard_dir / metadata["name"],
            expected_rows=input_rows,
            scorer=state["scorer"],
            score_field=state["contract"]["score_field"],
            expected_metadata=metadata,
            index=metadata["index"],
            start_row=metadata["start_row"],
            max_rows=state["contract"]["shard_size"],
        )

    while True:
        index = len(state["completed_shards"])
        name = f"part-{index:06d}.jsonl"
        candidate = shard_dir / name
        if not candidate.exists() and not candidate.is_symlink():
            break
        metadata = _validate_shard(
            candidate,
            expected_rows=input_rows,
            scorer=state["scorer"],
            score_field=state["contract"]["score_field"],
            expected_metadata=None,
            index=index,
            start_row=state["next_row"],
            max_rows=state["contract"]["shard_size"],
        )
        state["completed_shards"].append(metadata)
        state["next_row"] = metadata["end_row_exclusive"]
        atomic_write(output_dir / STATE_NAME, canonical_json_bytes(state))
        tracked_names.add(name)
        emit("orphan_shard_recovered", shard=index, rows=metadata["rows"])

    unexpected = []
    for candidate in shard_dir.iterdir():
        match = SHARD_RE.fullmatch(candidate.name)
        if match is None or candidate.name not in tracked_names:
            unexpected.append(candidate.name)
    if unexpected:
        raise ScoreCacheError(
            "unexpected_artifact", "shard root has unexpected artifacts"
        )
    return input_rows


def _write_new_shard(
    *,
    output_dir: Path,
    state: dict[str, Any],
    input_rows: Iterator[InputRow],
    backend: ScoreBackend,
    options: CacheOptions,
    remaining_rows: int,
) -> dict[str, Any]:
    index = len(state["completed_shards"])
    name = f"part-{index:06d}.jsonl"
    final_path = output_dir / SHARD_DIRECTORY / name
    if final_path.exists() or final_path.is_symlink():
        raise ScoreCacheError("orphan_artifact", "next shard already exists")
    descriptor = -1
    temporary: Path | None = None
    digest = hashlib.sha256()
    byte_count = 0
    rows = 0
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{name}.", suffix=".tmp", dir=final_path.parent
        )
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            target_rows = min(options.shard_size, remaining_rows)
            for _ in range(target_rows):
                try:
                    source = next(input_rows)
                except StopIteration:
                    raise ScoreCacheError(
                        "input_drift", "input ended during scoring"
                    ) from None
                payload = canonical_json_bytes(
                    score_input_row(source, backend, options)
                )
                handle.write(payload)
                digest.update(payload)
                byte_count += len(payload)
                rows += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final_path)
        temporary = None
        _fsync_directory(final_path.parent)
    except ScoreCacheError:
        raise
    except OSError as error:
        raise ScoreCacheError(
            "shard_write_failed", f"shard write failed ({type(error).__name__})"
        ) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()
    return {
        "index": index,
        "name": name,
        "start_row": state["next_row"],
        "end_row_exclusive": state["next_row"] + rows,
        "rows": rows,
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def _assemble_scores(output_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    final_path = output_dir / SCORES_NAME
    descriptor = -1
    temporary: Path | None = None
    digest = hashlib.sha256()
    byte_count = 0
    rows = 0
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{SCORES_NAME}.", suffix=".tmp", dir=output_dir
        )
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            for shard in state["completed_shards"]:
                source_path = output_dir / SHARD_DIRECTORY / shard["name"]
                with source_path.open("rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                        byte_count += len(block)
                rows += shard["rows"]
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, final_path)
        temporary = None
        _fsync_directory(output_dir)
    except OSError as error:
        raise ScoreCacheError(
            "assembly_failed", f"score assembly failed ({type(error).__name__})"
        ) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()
    return {
        "name": SCORES_NAME,
        "rows": rows,
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def is_training_admissible(state: dict[str, Any]) -> bool:
    scorer = state.get("scorer")
    contract = state.get("contract")
    return bool(
        isinstance(scorer, dict)
        and isinstance(contract, dict)
        and contract.get("score_field") == PRODUCTION_SCORE_FIELD
        and scorer.get("backend") == "pinned-local-qwen3-reranker"
        and scorer.get("model") == MODEL_ID
        and scorer.get("revision") == MODEL_REVISION
        and scorer.get("local_files_only") is True
        and scorer.get("trust_remote_code") is False
    )


def build_manifest(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "artifact_type": "qwen3_reranker_teacher_score_cache",
        "status": "complete",
        "admissible_for_training": is_training_admissible(state),
        "input": state["input"],
        "output": state["output"]["scores"],
        "shards": state["completed_shards"],
        "scorer": state["scorer"],
        "runtime": state["runtime"],
        "execution_contract": state["contract"],
        "run_fingerprint_sha256": state["run_fingerprint_sha256"],
        "determinism_scope": (
            "canonical row order/JSON, atomic shard recovery, exact input and runtime contract; "
            "raw logits may differ under a different hardware or numerical runtime"
        ),
        "privacy_contract": (
            "no query/document text, local path, environment value, or authentication token"
        ),
    }


def _complete_run(output_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    scores = _assemble_scores(output_dir, state)
    if scores["rows"] != state["input"]["rows"]:
        raise ScoreCacheError(
            "assembly_failed", "combined score row count differs from input"
        )
    state["output"] = {"scores": scores}
    manifest = build_manifest(state)
    manifest_payload = canonical_json_bytes(manifest)
    atomic_write(output_dir / MANIFEST_NAME, manifest_payload)
    state["output"]["manifest"] = {
        "name": MANIFEST_NAME,
        "bytes": len(manifest_payload),
        "sha256": sha256_bytes(manifest_payload),
    }
    state["status"] = "complete"
    atomic_write(output_dir / STATE_NAME, canonical_json_bytes(state))
    return state


def verify_complete_artifacts(
    options: CacheOptions, *, expected_identity: dict[str, Any] | None = None
) -> dict[str, Any]:
    state = read_json_object(options.output_dir / STATE_NAME, "state")
    effective_options = options
    if expected_identity is None:
        try:
            contract = state["contract"]
            stored_scorer = state["scorer"]
            effective_options = dataclasses.replace(
                options,
                instruction=stored_scorer["instruction"],
                shard_size=contract["shard_size"],
                model_batch_size=contract["model_batch_size"],
                max_documents_per_row=contract["max_documents_per_row"],
                max_text_characters=contract["max_text_characters"],
                max_length=contract["max_length"],
            )
            validate_options(effective_options)
        except (KeyError, TypeError, ScoreCacheError) as error:
            raise ScoreCacheError(
                "invalid_state",
                f"stored verification contract is invalid ({type(error).__name__})",
            ) from None
    evidence = preflight_input(effective_options.input_path, effective_options)
    if expected_identity is None:
        identity = {
            key: state.get(key)
            for key in (
                "input",
                "contract",
                "scorer",
                "runtime",
                "run_fingerprint_sha256",
            )
        }
    else:
        identity = expected_identity
    validate_state(state, identity, evidence)
    if state["status"] != "complete":
        raise ScoreCacheError("incomplete", "score cache is not complete")
    input_rows = iter_input_rows(effective_options.input_path, effective_options)
    for metadata in state["completed_shards"]:
        _validate_shard(
            effective_options.output_dir / SHARD_DIRECTORY / metadata["name"],
            expected_rows=input_rows,
            scorer=state["scorer"],
            score_field=state["contract"]["score_field"],
            expected_metadata=metadata,
            index=metadata["index"],
            start_row=metadata["start_row"],
            max_rows=state["contract"]["shard_size"],
        )
    try:
        next(input_rows)
    except StopIteration:
        pass
    else:
        raise ScoreCacheError(
            "output_mismatch", "score shards do not cover every input row"
        )
    scores_path = effective_options.output_dir / SCORES_NAME
    try:
        scores_info = scores_path.lstat()
        if (
            scores_path.is_symlink()
            or not stat.S_ISREG(scores_info.st_mode)
            or scores_info.st_uid != os.geteuid()
        ):
            raise ScoreCacheError(
                "unsafe_artifact", "combined scores must be a regular file"
            )
    except FileNotFoundError:
        raise ScoreCacheError(
            "missing_artifact", "combined scores are unavailable"
        ) from None
    scores_evidence = state["output"]["scores"]
    if (
        scores_info.st_size != scores_evidence["bytes"]
        or sha256_file(scores_path) != scores_evidence["sha256"]
        or scores_evidence["rows"] != evidence.rows
    ):
        raise ScoreCacheError(
            "output_drift", "combined score SHA, size, or row count changed"
        )
    combined_digest = hashlib.sha256()
    for shard in state["completed_shards"]:
        with (effective_options.output_dir / SHARD_DIRECTORY / shard["name"]).open(
            "rb"
        ) as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                combined_digest.update(block)
    if combined_digest.hexdigest() != scores_evidence["sha256"]:
        raise ScoreCacheError(
            "output_drift", "combined scores differ from ordered shards"
        )
    manifest = read_json_object(
        effective_options.output_dir / MANIFEST_NAME, "manifest"
    )
    if manifest != build_manifest(state):
        raise ScoreCacheError("manifest_drift", "manifest differs from complete state")
    manifest_payload = canonical_json_bytes(manifest)
    manifest_evidence = state["output"]["manifest"]
    if (
        len(manifest_payload) != manifest_evidence["bytes"]
        or sha256_bytes(manifest_payload) != manifest_evidence["sha256"]
    ):
        raise ScoreCacheError("manifest_drift", "manifest evidence differs")
    return state


def run_score_cache(options: CacheOptions, backend: ScoreBackend) -> dict[str, Any]:
    validate_options(options)
    evidence = preflight_input(options.input_path, options)
    identity = _run_identity(evidence, options, backend)
    _safe_output_directory(options.output_dir)
    with exclusive_lock(options.output_dir):
        state = _load_or_initialize_state(options.output_dir, identity, evidence)
        if state["status"] == "complete":
            verified = verify_complete_artifacts(options, expected_identity=identity)
            emit(
                "score_cache_already_complete",
                input_rows=evidence.rows,
                output_sha256=verified["output"]["scores"]["sha256"],
            )
            return verified
        input_rows = validate_and_recover_shards(options.output_dir, state, options)
        while state["next_row"] < evidence.rows:
            metadata = _write_new_shard(
                output_dir=options.output_dir,
                state=state,
                input_rows=input_rows,
                backend=backend,
                options=options,
                remaining_rows=evidence.rows - state["next_row"],
            )
            state["completed_shards"].append(metadata)
            state["next_row"] = metadata["end_row_exclusive"]
            atomic_write(options.output_dir / STATE_NAME, canonical_json_bytes(state))
            emit(
                "shard_complete",
                shard=metadata["index"],
                rows=metadata["rows"],
                completed_rows=state["next_row"],
                input_rows=evidence.rows,
                sha256=metadata["sha256"],
            )
        # Detect a file mutation during a long inference run before publication.
        final_input_evidence = preflight_input(options.input_path, options)
        if final_input_evidence != evidence:
            raise ScoreCacheError("input_drift", "input changed while scoring")
        state = _complete_run(options.output_dir, state)
        state = verify_complete_artifacts(options, expected_identity=identity)
        emit(
            "score_cache_complete",
            input_rows=evidence.rows,
            output_rows=state["output"]["scores"]["rows"],
            output_sha256=state["output"]["scores"]["sha256"],
            manifest_sha256=state["output"]["manifest"]["sha256"],
            admissible_for_training=(is_training_admissible(state)),
        )
        return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an exact, local-only Qwen3 reranker teacher score cache"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--model-batch-size", type=int, default=8)
    parser.add_argument("--max-documents-per-row", type=int, default=201)
    parser.add_argument("--max-text-characters", type=int, default=1_000_000)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--backend",
        choices=("qwen", "mock"),
        default="qwen",
        help="mock is CPU-only test output and is never training-admissible",
    )
    parser.add_argument(
        "--allow-mock-output",
        action="store_true",
        help="required acknowledgement before writing non-production mock scores",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate input and print a path-free plan; load no tokenizer/model and write nothing",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify input/state/shards/output/manifest without loading the model",
    )
    return parser


def options_from_args(args: argparse.Namespace) -> CacheOptions:
    return CacheOptions(
        input_path=args.input,
        output_dir=args.output_dir,
        instruction=args.instruction,
        shard_size=args.shard_size,
        model_batch_size=args.model_batch_size,
        max_documents_per_row=args.max_documents_per_row,
        max_text_characters=args.max_text_characters,
        max_length=args.max_length,
        device=args.device,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
    )


def cli_main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = options_from_args(args)
    try:
        enforce_offline_no_credentials()
        validate_options(options)
        if args.dry_run and args.verify_only:
            raise ScoreCacheError(
                "invalid_config", "dry-run and verify-only are mutually exclusive"
            )
        if args.backend == "mock" and not args.allow_mock_output and not args.dry_run:
            raise ScoreCacheError(
                "mock_not_acknowledged", "mock output requires --allow-mock-output"
            )
        if args.verify_only:
            state = verify_complete_artifacts(options)
            emit(
                "score_cache_verified",
                input_rows=state["input"]["rows"],
                output_sha256=state["output"]["scores"]["sha256"],
                admissible_for_training=(is_training_admissible(state)),
            )
            return 0
        if args.dry_run:
            evidence = preflight_input(options.input_path, options)
            emit(
                "dry_run_valid",
                model=MODEL_ID,
                revision=MODEL_REVISION,
                backend=args.backend,
                local_files_only=True,
                trust_remote_code=False,
                input_rows=evidence.rows,
                input_sha256=evidence.sha256,
                instruction_sha256=sha256_bytes(options.instruction.encode("utf-8")),
                output_written=False,
                model_loaded=False,
            )
            return 0
        backend: ScoreBackend
        if args.backend == "mock":
            backend = DeterministicMockBackend(options)
        else:
            backend = Qwen3RerankerBackend(options)
        run_score_cache(options, backend)
        return 0
    except ScoreCacheError as error:
        emit("error", code=error.code, message=str(error))
        return 2
    except KeyboardInterrupt:
        emit(
            "interrupted",
            message="scoring interrupted; completed atomic shards remain resumable",
        )
        return 130
    except (
        Exception
    ) as error:  # Fail closed without leaking paths or environment values.
        emit(
            "error",
            code="unexpected",
            message=f"scoring failed ({type(error).__name__})",
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(cli_main())
