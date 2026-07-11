---
base_model: Qwen/Qwen3-Embedding-8B
library_name: transformers
pipeline_tag: feature-extraction
language:
  - ko
license: apache-2.0
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

## Planned fair comparison

The public candidate will compare `Qwen/Qwen3-Embedding-8B`,
`sionic-ai/comsat-embed-ko-8b-preview`, and our model under one pinned evaluator.
It will report the same nine retrieval tasks claimed by Sionic, NDCG@10 on the
full corpus, query-only instructions, and per-task scores. It will also report
clean Korean holdouts, MTEB Korean/Multilingual regressions, bootstrap confidence
intervals, latency, memory, context length, and embedding dimension.

No score will be filled in until it is reproduced from raw evaluator output.
