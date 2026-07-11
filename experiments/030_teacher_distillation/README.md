# 030 — Teacher distillation

후보 teacher:

- Qwen3-Reranker-8B
- stronger public embedding ensemble
- domain-specific cross-encoder

비교:

- teacher를 filter로만 사용
- hard label InfoNCE
- candidate distribution KL/soft labels
- base Qwen representation replay

## 구현된 data-selection 계약

[`grounded_synthetic_query_factory.py`](../../scripts/grounded_synthetic_query_factory.py)는
Qwen3-Reranker-8B 등의 연속 score 파일을 받아 positive threshold, positive-relative
false-negative filter, text dedup을 적용한다. 기본 `score_rank_quantiles`는 score-sorted
top-24 pool의 양 끝을 포함한 7개 rank quantile을 결정론적으로 선택한다. `top_k`와
`hash_sample_from_top_pool`은 ablation으로 유지한다. 각 row audit에는 선택 index,
최소/최대 teacher score, scorer model/revision을 저장한다.
