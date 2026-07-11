# 010 — Qwen3-Embedding-8B Korean LoRA

첫 학습·저장·재로딩 경로를 검증한 실험입니다. 다음 성능 pilot은 더 어려운 재채굴 negative와 권리 확인 데이터로 분리합니다.

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

실측 결과 (`v1-20260711-210119`, H100 80GB):

- 256 train / 32 validation rows, 평균 길이 약 39 tokens
- 87.294M trainable parameters / 8.276B total (`1.0548%`)
- 20 steps `43.81 s`, peak VRAM `17.07 GiB`
- validation positive cosine `0.74382`, negative cosine `0.22422`, margin `0.39358`
- checkpoint-20을 별도 process에서 재로딩: 4096-d, L2 norm 허용오차 통과, probe margin `0.44580`

loss가 첫 step부터 사실상 0이었으므로 negative가 너무 쉽습니다. 이 값은 embedding pipeline이 작동한다는 증거이지 Comsat 대비 성능 증거가 아닙니다.

주의: smoke source인 `nlpai-lab/ko-triplet-v1.0`은 카드에 명시적 라이선스가 없습니다. 따라서 이 런은 optimizer/save/load 검증에만 쓰며, 공개 후보 모델 학습에는 사용하지 않습니다.

상태: smoke pass. 공개 benchmark 성능 주장은 없음.
