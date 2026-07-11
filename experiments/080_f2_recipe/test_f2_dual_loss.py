#!/usr/bin/env python3
"""CPU invariants for the local F2-style loss plugin."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch


class DummyTrainer:
    pass


def load_plugin():
    path = Path(__file__).with_name("f2_dual_loss_plugin.py")
    spec = importlib.util.spec_from_file_location("f2_dual_loss_plugin", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    os.environ["F2_DUAL_TEMPERATURE"] = "0.05"
    os.environ["F2_DUAL_HARD_NEGATIVES"] = "2"
    module = load_plugin()
    loss_fn = module.F2DualInfoNCELoss(None, DummyTrainer())

    # Two rows laid out as q, positive, negative, negative.
    good = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    # Labels omit each query: 1 marks a positive and 0 marks following negatives.
    labels = torch.tensor([1, 0, 0, 1, 0, 0])
    good_loss = loss_fn({"last_hidden_state": good}, labels)
    assert torch.isfinite(good_loss) and good_loss.item() < 1e-5, good_loss

    bad = good.clone()
    bad[1], bad[2] = good[2].clone(), good[1].clone()
    bad_loss = loss_fn({"last_hidden_state": bad}, labels)
    assert bad_loss > good_loss + 1.0, (good_loss, bad_loss)

    good.requires_grad_(True)
    train_loss = loss_fn({"last_hidden_state": good}, labels)
    train_loss.backward()
    assert good.grad is not None and torch.isfinite(good.grad).all()
    print({"good_loss": good_loss.item(), "bad_loss": bad_loss.item(), "grad_finite": True})


if __name__ == "__main__":
    main()

