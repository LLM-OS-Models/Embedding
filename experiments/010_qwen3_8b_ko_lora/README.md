# 010 — Qwen3-Embedding-8B Korean LoRA

현재 첫 학습 실험입니다.

가설: Qwen3-Embedding-8B의 geometry를 보존하면서 clean Korean query-positive-negative로 contrastive LoRA를 하면 Comsat의 +1.05 NDCG point를 재현하거나 넘을 수 있습니다.

단계:

1. 100-example overfit
2. 5K smoke
3. 50K pilot
4. 300K~500K Stage A

초기 설정:

- BF16 LoRA, all-linear
- r32와 r64
- InfoNCE tau .02
- positive 1 + hard negative 1
- query 256 / document 512
- public target-9 blocklist 적용

상태: 환경 및 데이터 준비 중.
