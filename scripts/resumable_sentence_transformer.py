"""Exact on-disk embedding cache for long MTEB SentenceTransformer runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


def _canonical_cache_options(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """Return output-affecting JSON options, or None for an unsupported call."""

    if kwargs.get("convert_to_tensor") or kwargs.get("convert_to_numpy") is False:
        return None
    if kwargs.get("output_value", "sentence_embedding") != "sentence_embedding":
        return None
    if kwargs.get("pool") is not None or kwargs.get("chunk_size") is not None:
        return None
    allowed = {
        "batch_size",
        "device",
        "prompt_name",
        "prompt",
        "normalize_embeddings",
        "task",
        "truncate_dim",
        "precision",
    }
    non_output = {
        "chunk_size",
        "convert_to_numpy",
        "convert_to_tensor",
        "output_value",
        "pool",
        "show_progress_bar",
    }
    if set(kwargs) - allowed - non_output:
        return None
    options = {key: kwargs.get(key) for key in sorted(allowed) if key in kwargs}
    try:
        json.dumps(options, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return None
    return options


def embedding_cache_key(
    *, namespace: str, sentences: Sequence[str], options: dict[str, Any]
) -> str:
    digest = hashlib.sha256()
    header = json.dumps(
        {"schema": 1, "namespace": namespace, "options": options},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(len(sentences).to_bytes(8, "big"))
    for sentence in sentences:
        if not isinstance(sentence, str):
            raise TypeError("The resumable cache supports string inputs only")
        encoded = sentence.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class ExactNpyCache:
    """Store float32 arrays atomically without changing numerical results."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def paths(self, key: str) -> tuple[Path, Path]:
        folder = self.root / key[:2]
        return folder / f"{key}.npy", folder / f"{key}.json"

    def load(self, key: str, expected_rows: int) -> np.ndarray | None:
        array_path, metadata_path = self.paths(key)
        if not array_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata != {
                "cache_key": key,
                "dtype": "float32",
                "rows": expected_rows,
                "schema_version": 1,
            }:
                return None
            array = np.load(array_path, allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if array.dtype != np.float32 or array.ndim != 2 or array.shape[0] != expected_rows:
            return None
        return array

    def store(self, key: str, array: np.ndarray) -> None:
        value = np.asarray(array)
        if value.dtype != np.float32 or value.ndim != 2:
            raise ValueError(
                f"Only exact float32 2D embeddings are cached, got {value.dtype} {value.shape}"
            )
        array_path, metadata_path = self.paths(key)
        array_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "cache_key": key,
            "dtype": "float32",
            "rows": int(value.shape[0]),
            "schema_version": 1,
        }
        array_fd, array_name = tempfile.mkstemp(
            dir=array_path.parent, prefix=f".{key}.", suffix=".npy"
        )
        metadata_fd, metadata_name = tempfile.mkstemp(
            dir=metadata_path.parent, prefix=f".{key}.", suffix=".json"
        )
        try:
            with os.fdopen(array_fd, "wb") as handle:
                np.save(handle, value, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            with os.fdopen(metadata_fd, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(array_name, array_path)
            os.replace(metadata_name, metadata_path)
        finally:
            for temporary in (array_name, metadata_name):
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass


class ResumableSentenceTransformer(SentenceTransformer):
    """SentenceTransformer whose normal NumPy encode calls survive restarts."""

    def __init__(
        self,
        *args: Any,
        embedding_cache_dir: Path,
        embedding_cache_namespace: str,
        **kwargs: Any,
    ) -> None:
        self._exact_embedding_cache = ExactNpyCache(embedding_cache_dir)
        self._embedding_cache_namespace = embedding_cache_namespace
        self.embedding_cache_hits = 0
        self.embedding_cache_misses = 0
        super().__init__(*args, **kwargs)

    def encode(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        if args:
            return super().encode(inputs, *args, **kwargs)
        if isinstance(inputs, str):
            return super().encode(inputs, *args, **kwargs)
        if not isinstance(inputs, (list, tuple)) or not inputs:
            return super().encode(inputs, *args, **kwargs)
        options = _canonical_cache_options(kwargs)
        if options is None or not all(isinstance(item, str) for item in inputs):
            return super().encode(inputs, *args, **kwargs)
        key = embedding_cache_key(
            namespace=self._embedding_cache_namespace,
            sentences=inputs,
            options=options,
        )
        cached = self._exact_embedding_cache.load(key, len(inputs))
        if cached is not None:
            self.embedding_cache_hits += 1
            return cached
        result = super().encode(inputs, *args, **kwargs)
        array = np.asarray(result)
        if array.dtype == np.float32 and array.ndim == 2:
            self._exact_embedding_cache.store(key, array)
            self.embedding_cache_misses += 1
        return result
