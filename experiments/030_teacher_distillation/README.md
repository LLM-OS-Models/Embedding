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
