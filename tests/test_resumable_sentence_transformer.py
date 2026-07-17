from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.resumable_sentence_transformer import _encode_with_oom_backoff


def test_cuda_oom_halves_only_internal_microbatch() -> None:
    observed: list[int] = []
    counters = SimpleNamespace(
        embedding_oom_retries=0, minimum_effective_batch_size=None
    )

    def encode(_inputs, *, batch_size: int):
        observed.append(batch_size)
        if batch_size > 16:
            raise torch.OutOfMemoryError("fixture")
        return "ok"

    assert (
        _encode_with_oom_backoff(
            encode, ["a"], (), {"batch_size": 64}, counters
        )
        == "ok"
    )
    assert observed == [64, 32, 16]
    assert counters.embedding_oom_retries == 2
    assert counters.minimum_effective_batch_size == 16


def test_cuda_oom_at_batch_one_is_not_hidden() -> None:
    counters = SimpleNamespace(
        embedding_oom_retries=0, minimum_effective_batch_size=None
    )

    def encode(_inputs, *, batch_size: int):
        raise torch.OutOfMemoryError("fixture")

    try:
        _encode_with_oom_backoff(encode, ["a"], (), {"batch_size": 1}, counters)
    except torch.OutOfMemoryError:
        pass
    else:  # pragma: no cover
        raise AssertionError("batch-one OOM must propagate")
