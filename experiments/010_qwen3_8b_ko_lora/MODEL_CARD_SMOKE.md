---
base_model: Qwen/Qwen3-Embedding-8B
library_name: transformers
pipeline_tag: feature-extraction
language:
  - ko
license: other
tags:
  - sentence-transformers
  - embedding
  - retrieval
  - korean
  - lora
---

# Qwen3-Embedding-8B Korean smoke adapter

This repository is a **private pipeline-validation artifact**, not a performance release.

## Status

- Base: `Qwen/Qwen3-Embedding-8B` at revision `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`
- Method: BF16 LoRA, InfoNCE, one explicit hard negative, temperature 0.02
- Purpose: verify data formatting, optimization, saving, loading, and evaluation plumbing
- Public benchmark claim: **none**
- Release eligibility: **no**; the smoke source dataset does not declare an explicit license
- Distribution: private adapter artifact only; the base model's Apache-2.0 license does not resolve the missing training-data license

## Measured pipeline checks

- Trainable parameters: 87.294M / 8.276B (1.0548%)
- Training: 20 steps in 43.81 seconds on one H100 80GB
- Peak training memory: 17.07 GiB at an average sequence length of about 39 tokens
- Fresh-process adapter reload: pass; 4096-dimensional embeddings and a positive-minus-negative probe margin of 0.44580

These checks are not benchmark results. The validation loss saturated immediately because the smoke negatives were too easy.

## Planned fair comparison

The public candidate will compare `Qwen/Qwen3-Embedding-8B`,
`sionic-ai/comsat-embed-ko-8b-preview`, and our model under one pinned evaluator.
It will report the same nine retrieval tasks claimed by Sionic, NDCG@10 on the
full corpus, query-only instructions, and per-task scores. It will also report
clean Korean holdouts, MTEB Korean/Multilingual regressions, bootstrap confidence
intervals, latency, memory, context length, and embedding dimension.

No score will be filled in until it is reproduced from raw evaluator output.
