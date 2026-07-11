#!/usr/bin/env python3
"""CPU-only invariants for the exact MTEB chunk cache."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from resumable_sentence_transformer import ExactNpyCache, embedding_cache_key


def main() -> None:
    sentences = ["가", "나\n다", "emoji: 🛰️"]
    options = {"normalize_embeddings": True, "prompt_name": "document"}
    key = embedding_cache_key(namespace="model@revision", sentences=sentences, options=options)
    assert key == embedding_cache_key(
        namespace="model@revision", sentences=list(sentences), options=dict(options)
    )
    assert key != embedding_cache_key(
        namespace="model@revision", sentences=list(reversed(sentences)), options=options
    )
    with tempfile.TemporaryDirectory() as directory:
        cache = ExactNpyCache(Path(directory))
        assert cache.load(key, 3) is None
        expected = np.arange(12, dtype=np.float32).reshape(3, 4)
        cache.store(key, expected)
        actual = cache.load(key, 3)
        assert actual is not None
        assert actual.dtype == np.float32
        assert np.array_equal(actual, expected)
        assert cache.load(key, 2) is None
        array_path, _ = cache.paths(key)
        array_path.write_bytes(b"corrupt")
        assert cache.load(key, 3) is None
    print({"cache_key": key, "status": "pass"})


if __name__ == "__main__":
    main()
