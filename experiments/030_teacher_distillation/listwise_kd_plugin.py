"""ms-swift external plugin for first-party embedding listwise KD."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.listwise_distillation import (  # noqa: E402
    ListwiseDistillationLoss,
    ListwiseKDConfig,
)
from swift.dataset.preprocessor.core import RowPreprocessor  # noqa: E402
from swift.loss import loss_map  # noqa: E402
from swift.trainers.embedding_trainer import EmbeddingTrainer  # noqa: E402


TEACHER_SCORE_KEY = "teacher_scores"
LOSS_NAME = "listwise_embedding_kd"

if TEACHER_SCORE_KEY not in RowPreprocessor.standard_keys:
    RowPreprocessor.standard_keys.append(TEACHER_SCORE_KEY)


if not getattr(EmbeddingTrainer, "_embedding_kd_score_patch", False):
    _original_compute_loss = EmbeddingTrainer.compute_loss

    def _compute_loss_with_teacher_scores(
        self: EmbeddingTrainer,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        teacher_scores = inputs.pop(TEACHER_SCORE_KEY, None)
        self._embedding_teacher_scores = teacher_scores
        try:
            return _original_compute_loss(
                self,
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        finally:
            self._embedding_teacher_scores = None

    EmbeddingTrainer.compute_loss = _compute_loss_with_teacher_scores
    EmbeddingTrainer._embedding_kd_score_patch = True


class SwiftListwiseEmbeddingKDLoss:
    def __init__(self, args: Any, trainer: EmbeddingTrainer):
        self.trainer = trainer
        self.loss = ListwiseDistillationLoss(ListwiseKDConfig.from_environment())

    def __call__(self, outputs: Any, labels: torch.Tensor, **_: Any) -> torch.Tensor:
        mode = "train" if self.trainer.model.training else "eval"

        def record(name: str, value: torch.Tensor) -> None:
            self.trainer.custom_metrics[mode][name].update(value)

        return self.loss(
            outputs,
            labels,
            getattr(self.trainer, "_embedding_teacher_scores", None),
            training=self.trainer.model.training,
            metric_callback=record,
        )


loss_map[LOSS_NAME] = SwiftListwiseEmbeddingKDLoss

