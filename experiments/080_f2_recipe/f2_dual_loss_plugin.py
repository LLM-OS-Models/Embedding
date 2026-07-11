"""F2LLM-style dual contrastive loss for the pinned ms-swift trainer.

F2LLM-v2 optimizes retrieval batches with two separate cross-entropy terms:
one over the other positives in the batch and one over each query's explicit
hard-negative set.  ms-swift's built-in InfoNCE puts positives and explicit
negatives from every row into one denominator.  Both are sensible objectives,
but they are not mathematically identical, so this plugin makes the paper/code
ablation explicit.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from swift.loss import BaseLoss, loss_map
from swift.loss.embedding import _parse_multi_negative_sentences


def _env_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _gather_equal_batch(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Gather a same-sized positive batch while preserving local gradients."""

    if not dist.is_available() or not dist.is_initialized():
        return tensor, 0
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    sizes_int = [int(item.item()) for item in sizes]
    if len(set(sizes_int)) != 1:
        raise RuntimeError(
            "f2_dual_infonce requires equal per-rank batches; enable dataloader_drop_last"
        )
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.detach())
    gathered[rank] = tensor
    return torch.cat(gathered, dim=0), rank * tensor.shape[0]


class F2DualInfoNCELoss(BaseLoss):
    """Sum in-batch-positive CE and per-row explicit-hard-negative CE."""

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        temperature = _env_float("F2_DUAL_TEMPERATURE", 0.05)
        if temperature == 0:
            raise ValueError("F2_DUAL_TEMPERATURE must be positive")
        inbatch_weight = _env_float("F2_DUAL_INBATCH_WEIGHT", 1.0)
        hard_weight = _env_float("F2_DUAL_HARD_WEIGHT", 1.0)
        hard_negatives_raw = os.environ.get("F2_DUAL_HARD_NEGATIVES")
        hard_negatives = int(hard_negatives_raw) if hard_negatives_raw else None
        if hard_negatives is not None and hard_negatives < 1:
            raise ValueError("F2_DUAL_HARD_NEGATIVES must be positive")

        sentences = outputs["last_hidden_state"]
        groups = _parse_multi_negative_sentences(sentences, labels, hard_negatives)
        if not groups:
            raise ValueError("f2_dual_infonce received an empty batch")
        if any(group.shape[0] < 3 for group in groups):
            raise ValueError("Every row needs a query, positive, and at least one negative")
        group_sizes = {group.shape[0] for group in groups}
        if len(group_sizes) != 1:
            raise ValueError(
                "f2_dual_infonce requires the same explicit-negative count per row; "
                "set F2_DUAL_HARD_NEGATIVES"
            )

        batch = torch.stack(groups, dim=0)
        batch = F.normalize(batch.float(), p=2, dim=-1).to(sentences.dtype)
        queries = batch[:, 0]
        positives = batch[:, 1]
        explicit_docs = batch[:, 1:]

        all_positives, target_offset = _gather_equal_batch(positives)
        inbatch_logits = queries @ all_positives.T / temperature
        inbatch_targets = torch.arange(queries.shape[0], device=queries.device) + target_offset
        inbatch_loss = F.cross_entropy(inbatch_logits.float(), inbatch_targets)

        hard_logits = torch.einsum("bd,bkd->bk", queries, explicit_docs) / temperature
        hard_targets = torch.zeros(queries.shape[0], dtype=torch.long, device=queries.device)
        hard_loss = F.cross_entropy(hard_logits.float(), hard_targets)

        return inbatch_weight * inbatch_loss + hard_weight * hard_loss


loss_map["f2_dual_infonce"] = F2DualInfoNCELoss

