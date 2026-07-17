from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.listwise_distillation import (
    ListwiseDistillationLoss,
    ListwiseKDConfig,
    normalize_teacher_scores,
    split_embedding_groups,
)
from scripts.validate_embedding_jsonl import validate


LABELS = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
TEACHER = [[0.9, 0.6, 0.1], [0.85, 0.5, 0.2]]


def embeddings(*, aligned: bool = True) -> torch.Tensor:
    query1 = [1.0, 0.0, 0.0]
    query2 = [0.0, 1.0, 0.0]
    if aligned:
        group1 = [query1, [1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [-1.0, 0.0, 0.0]]
        group2 = [query2, [0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [0.0, -1.0, 0.0]]
    else:
        group1 = [query1, [0.1, 1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        group2 = [query2, [1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    return torch.tensor(group1 + group2)


def test_group_alignment_and_teacher_contract() -> None:
    groups = split_embedding_groups(embeddings(), LABELS)
    assert [group.shape for group in groups] == [(4, 3), (4, 3)]
    scores = normalize_teacher_scores(
        TEACHER, [3, 3], device=torch.device("cpu"), dtype=torch.float32
    )
    assert scores is not None and scores[0].tolist() == pytest.approx(TEACHER[0])
    with pytest.raises(ValueError, match="positive"):
        normalize_teacher_scores(
            [[0.4, 0.8, 0.1], TEACHER[1]],
            [3, 3],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_listwise_kl_prefers_teacher_aligned_student() -> None:
    config = ListwiseKDConfig(hard_weight=0.0, kd_weight=1.0, queue_size=0)
    loss = ListwiseDistillationLoss(config)
    aligned = loss(
        {"last_hidden_state": embeddings(aligned=True)},
        LABELS,
        TEACHER,
        training=False,
    )
    misaligned = loss(
        {"last_hidden_state": embeddings(aligned=False)},
        LABELS,
        TEACHER,
        training=False,
    )
    assert torch.isfinite(aligned)
    assert aligned < misaligned


def test_margin_mse_and_stop_gradient_queue_are_finite() -> None:
    config = ListwiseKDConfig(mode="margin_mse", queue_size=5)
    loss = ListwiseDistillationLoss(config)
    first = loss(
        {"last_hidden_state": embeddings()}, LABELS, TEACHER, training=True
    )
    assert torch.isfinite(first) and loss.queue_rows == 5
    second = loss(
        {"last_hidden_state": embeddings()}, LABELS, TEACHER, training=False
    )
    assert torch.isfinite(second) and loss.queue_rows == 5


def test_missing_teacher_scores_falls_back_to_hard_infonce() -> None:
    loss = ListwiseDistillationLoss(ListwiseKDConfig())
    result = loss(
        {"last_hidden_state": embeddings()}, LABELS, None, training=False
    )
    assert torch.isfinite(result) and result > 0


def write_rows(path: Path, *, scores: list[float] | None) -> None:
    rows = []
    for index in range(2):
        row = {
            "messages": [{"role": "user", "content": f"query {index}"}],
            "positive_messages": [[{"role": "user", "content": f"positive {index}"}]],
            "negative_messages": [
                [{"role": "user", "content": f"negative {index} a"}],
                [{"role": "user", "content": f"negative {index} b"}],
            ],
        }
        if scores is not None:
            row["teacher_scores"] = scores
        rows.append(row)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_strict_jsonl_validator_supports_and_can_require_teacher_scores(
    tmp_path: Path,
) -> None:
    scored = tmp_path / "scored.jsonl"
    plain = tmp_path / "plain.jsonl"
    write_rows(scored, scores=[0.9, 0.5, 0.1])
    write_rows(plain, scores=None)
    assert validate(scored, require_teacher_scores=True)["teacher_score_rows"] == 2
    assert validate(plain)["teacher_score_rows"] == 0
    with pytest.raises(ValueError, match="every row"):
        validate(plain, require_teacher_scores=True)


def test_validator_rejects_misaligned_or_nonfinite_teacher_scores(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.jsonl"
    write_rows(path, scores=[0.9, 0.4])
    with pytest.raises(ValueError, match="align"):
        validate(path)
    write_rows(path, scores=[0.4, 0.8, 0.1])
    with pytest.raises(ValueError, match="positive"):
        validate(path)

