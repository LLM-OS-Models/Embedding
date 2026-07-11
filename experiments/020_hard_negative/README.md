# 020 — Hard-negative and false-negative filtering

비교:

- random/in-batch only
- BM25 negatives
- base Qwen dense negatives
- current checkpoint negatives
- reranker-filtered negatives
- positive-relative threshold 0.90/0.95/0.98

주 결과는 NDCG뿐 아니라 false-negative audit sample과 training margin을 포함합니다.
