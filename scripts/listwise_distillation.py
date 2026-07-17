#!/usr/bin/env python3
"""First-party listwise reranker distillation loss for embedding training.

The ms-swift integration lives in
``experiments/030_teacher_distillation/listwise_kd_plugin.py``.  This module is
kept framework-light so the score alignment, KL/MarginMSE objectives, and
stop-gradient document queue can be tested independently.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F


def _finite_float(name: str, value: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} violates its finite range")
    return result


def _nonnegative_int(name: str, value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True)
class ListwiseKDConfig:
    hard_weight: float = 0.3
    kd_weight: float = 0.7
    infonce_temperature: float = 0.02
    student_temperature: float = 0.02
    teacher_temperature: float = 1.0
    mode: str = "kl"
    queue_size: int = 0
    queue_false_negative_margin: float = 0.02
    probability_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        for name in (
            "hard_weight",
            "kd_weight",
            "infonce_temperature",
            "student_temperature",
            "teacher_temperature",
            "queue_false_negative_margin",
            "probability_epsilon",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.hard_weight + self.kd_weight <= 0:
            raise ValueError("at least one loss weight must be positive")
        if min(
            self.infonce_temperature,
            self.student_temperature,
            self.teacher_temperature,
        ) <= 0:
            raise ValueError("all temperatures must be positive")
        if self.mode not in {"kl", "margin_mse"}:
            raise ValueError("mode must be kl or margin_mse")
        if self.queue_size < 0:
            raise ValueError("queue_size must be nonnegative")
        if not 0 < self.probability_epsilon < 0.5:
            raise ValueError("probability_epsilon must be in (0, 0.5)")

    @classmethod
    def from_environment(cls) -> "ListwiseKDConfig":
        return cls(
            hard_weight=_finite_float(
                "EMBEDDING_KD_HARD_WEIGHT",
                os.environ.get("EMBEDDING_KD_HARD_WEIGHT", "0.3"),
                minimum=0,
            ),
            kd_weight=_finite_float(
                "EMBEDDING_KD_WEIGHT",
                os.environ.get("EMBEDDING_KD_WEIGHT", "0.7"),
                minimum=0,
            ),
            infonce_temperature=_finite_float(
                "INFONCE_TEMPERATURE",
                os.environ.get("INFONCE_TEMPERATURE", "0.02"),
                minimum=0,
            ),
            student_temperature=_finite_float(
                "EMBEDDING_KD_STUDENT_TEMPERATURE",
                os.environ.get("EMBEDDING_KD_STUDENT_TEMPERATURE", "0.02"),
                minimum=0,
            ),
            teacher_temperature=_finite_float(
                "EMBEDDING_KD_TEACHER_TEMPERATURE",
                os.environ.get("EMBEDDING_KD_TEACHER_TEMPERATURE", "1.0"),
                minimum=0,
            ),
            mode=os.environ.get("EMBEDDING_KD_MODE", "kl"),
            queue_size=_nonnegative_int(
                "EMBEDDING_KD_QUEUE_SIZE",
                os.environ.get("EMBEDDING_KD_QUEUE_SIZE", "0"),
            ),
            queue_false_negative_margin=_finite_float(
                "EMBEDDING_KD_QUEUE_FALSE_NEGATIVE_MARGIN",
                os.environ.get("EMBEDDING_KD_QUEUE_FALSE_NEGATIVE_MARGIN", "0.02"),
                minimum=0,
            ),
            probability_epsilon=_finite_float(
                "EMBEDDING_KD_PROBABILITY_EPSILON",
                os.environ.get("EMBEDDING_KD_PROBABILITY_EPSILON", "1e-5"),
                minimum=0,
            ),
        )


def split_embedding_groups(
    sentences: torch.Tensor, labels: torch.Tensor
) -> list[torch.Tensor]:
    """Split flattened anchor/positive/negative embeddings using Swift labels."""

    if sentences.ndim != 2 or labels.ndim != 1:
        raise ValueError("sentences must be [N,D] and labels must be [documents]")
    positive_positions = torch.nonzero(labels == 1, as_tuple=False).flatten().tolist()
    if not positive_positions or positive_positions[0] != 0:
        raise ValueError("every embedding batch must begin with a positive label")
    boundaries = positive_positions + [labels.numel()]
    groups: list[torch.Tensor] = []
    for group_index in range(len(positive_positions)):
        document_start = boundaries[group_index]
        document_end = boundaries[group_index + 1]
        sentence_start = document_start + group_index
        sentence_end = document_end + group_index + 1
        group = sentences[sentence_start:sentence_end]
        if group.shape[0] < 3:
            raise ValueError("each row requires anchor, positive, and a negative")
        groups.append(group)
    if sum(group.shape[0] for group in groups) != sentences.shape[0]:
        raise ValueError("embedding/label group alignment failed")
    return groups


def normalize_teacher_scores(
    teacher_scores: Any,
    expected_lengths: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor] | None:
    if teacher_scores is None:
        return None
    if isinstance(teacher_scores, torch.Tensor):
        rows: Any = teacher_scores.detach().cpu().tolist()
    else:
        rows = teacher_scores
    if not isinstance(rows, (list, tuple)) or len(rows) != len(expected_lengths):
        raise ValueError("teacher score rows do not align with the embedding batch")
    output: list[torch.Tensor] = []
    for row, expected in zip(rows, expected_lengths):
        if isinstance(row, torch.Tensor):
            row = row.detach().cpu().tolist()
        if not isinstance(row, (list, tuple)) or len(row) != expected:
            raise ValueError("teacher score count does not align with row documents")
        values = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("teacher scores must be numeric")
            number = float(value)
            if not math.isfinite(number) or not 0 <= number <= 1:
                raise ValueError("teacher scores must be finite probabilities")
            values.append(number)
        if values[0] <= max(values[1:]):
            raise ValueError("teacher positive must outrank every selected negative")
        output.append(torch.tensor(values, device=device, dtype=dtype))
    return output


class ListwiseDistillationLoss:
    """Hard in-batch InfoNCE plus listwise KL or positive-margin MSE."""

    def __init__(self, config: ListwiseKDConfig):
        self.config = config
        self._document_queue: torch.Tensor | None = None

    @property
    def queue_rows(self) -> int:
        return 0 if self._document_queue is None else self._document_queue.shape[0]

    def _teacher_logits(self, scores: torch.Tensor) -> torch.Tensor:
        epsilon = self.config.probability_epsilon
        scores = scores.clamp(epsilon, 1 - epsilon)
        return torch.logit(scores) / self.config.teacher_temperature

    def _update_queue(self, documents: torch.Tensor) -> None:
        if self.config.queue_size == 0:
            return
        detached = documents.detach()
        if self._document_queue is None or self._document_queue.device != detached.device:
            combined = detached
        else:
            combined = torch.cat((self._document_queue.to(detached.dtype), detached), dim=0)
        self._document_queue = combined[-self.config.queue_size :].contiguous()

    def __call__(
        self,
        outputs: dict[str, torch.Tensor] | Any,
        labels: torch.Tensor,
        teacher_scores: Any = None,
        *,
        training: bool,
        metric_callback: Callable[[str, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        sentences = outputs["last_hidden_state"]
        sentences = F.normalize(sentences.float(), p=2, dim=-1)
        labels = labels.to(sentences.device)
        groups = split_embedding_groups(sentences, labels)
        queries = torch.stack([group[0] for group in groups])
        documents = torch.cat([group[1:] for group in groups])
        positive_indices = []
        cursor = 0
        for group in groups:
            positive_indices.append(cursor)
            cursor += group.shape[0] - 1
        targets = torch.tensor(positive_indices, device=sentences.device)

        hard_logits = queries @ documents.T
        if training and self._document_queue is not None and self._document_queue.numel():
            queue = self._document_queue.to(device=sentences.device, dtype=sentences.dtype)
            queue_logits = queries @ queue.T
            positive_scores = hard_logits[torch.arange(len(groups), device=sentences.device), targets]
            false_negative_threshold = (
                positive_scores.detach().unsqueeze(1)
                - self.config.queue_false_negative_margin
            )
            queue_logits = queue_logits.masked_fill(
                queue_logits >= false_negative_threshold, float("-inf")
            )
            hard_logits = torch.cat((hard_logits, queue_logits), dim=1)
        hard_loss = F.cross_entropy(hard_logits / self.config.infonce_temperature, targets)

        expected_lengths = [group.shape[0] - 1 for group in groups]
        aligned_scores = normalize_teacher_scores(
            teacher_scores,
            expected_lengths,
            device=sentences.device,
            dtype=sentences.dtype,
        )
        if aligned_scores is None or self.config.kd_weight == 0:
            total = hard_loss
            kd_loss = hard_loss.new_zeros(())
        else:
            kd_terms = []
            for group, scores in zip(groups, aligned_scores):
                student_logits = (group[0] @ group[1:].T) / self.config.student_temperature
                teacher_logits = self._teacher_logits(scores)
                if self.config.mode == "kl":
                    teacher_distribution = F.softmax(teacher_logits, dim=-1)
                    kd_terms.append(
                        F.kl_div(
                            F.log_softmax(student_logits, dim=-1),
                            teacher_distribution,
                            reduction="sum",
                        )
                    )
                else:
                    student_margins = student_logits[0] - student_logits[1:]
                    teacher_margins = teacher_logits[0] - teacher_logits[1:]
                    kd_terms.append(F.mse_loss(student_margins, teacher_margins))
            kd_loss = torch.stack(kd_terms).mean()
            weight_sum = self.config.hard_weight + self.config.kd_weight
            total = (
                self.config.hard_weight * hard_loss
                + self.config.kd_weight * kd_loss
            ) / weight_sum

        if metric_callback is not None:
            metric_callback("hard_infonce", hard_loss.detach())
            metric_callback("listwise_kd", kd_loss.detach())
            metric_callback("queue_rows", hard_loss.new_tensor(float(self.queue_rows)))
        if training:
            self._update_queue(documents)
        if not torch.isfinite(total):
            raise FloatingPointError("listwise distillation produced non-finite loss")
        return total
