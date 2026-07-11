# 090 — Sionic SQuADKorV1 target adaptation

원본 KorQuAD train 60K를 current-student quantile hard negative 7개로 다시 채굴하고,
general 1M curriculum과 50:50 replay하는 격리 실험이다.

- 가설: 자연어 질문→Wikipedia 문맥 신호가 SQuADKorV1을 올리면서 MIRACL/일반 QA도
  보조할 수 있다.
- 위험: 한 benchmark family 과적합과 Wikipedia 문맥 중복으로 broad/법률 retrieval이
  회귀할 수 있다.
- 차단: 원본 validation, MTEB test query/qrel/corpus는 로드하지 않는다.
- 공개: 결과는 `SQuADKorV1 train-family exposed`, `target-adapted-squad`로 표시한다.
- 선택: Sionic 9 macro, 공식 Korean v1, clean source-heldout를 모두 통과해야 한다.

실행은 [`scripts/run_sionic_squad_adaptation_queue.sh`](../../scripts/run_sionic_squad_adaptation_queue.sh),
데이터와 근거는 [`docs/25_SIONIC_SQUAD_TARGET_ADAPTATION.md`](../../docs/25_SIONIC_SQUAD_TARGET_ADAPTATION.md)에
고정한다. 장기 full campaign에서는 1M general model을 merge·평가한 직후 자동 실행된다.
