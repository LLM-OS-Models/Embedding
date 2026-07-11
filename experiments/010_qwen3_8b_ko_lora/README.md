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

현재 구현:

- `prepare_smoke.sh`: revision이 고정된 288-row 연구용 smoke split 생성 및 검증
- `train_smoke.sh`: H100 1장, Qwen3-Embedding-8B r32 LoRA, 20-step smoke run
- `MODEL_CARD_SMOKE.md`: 성능 주장이 없는 private HF artifact 카드

주의: smoke source인 `nlpai-lab/ko-triplet-v1.0`은 카드에 명시적 라이선스가 없습니다. 따라서 이 런은 optimizer/save/load 검증에만 쓰며, 공개 후보 모델 학습에는 사용하지 않습니다.

상태: 실행 가능.
