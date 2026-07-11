# 000 — Baseline and evaluator audit

목적: Qwen3-Embedding-8B와 Comsat 점수를 동일한 MTEB revision, prompt, max length, split으로 재현합니다.

두 프로토콜을 분리합니다.

1. `MTEB(kor, v1)`: 공식 Korean leaderboard의 classification/reranking/retrieval/STS 6개 task, 모델별 native task prompt, Borda와 Mean(Task) 보고
2. `sionic9-fixed-prompt-v1`: Sionic 카드가 고른 retrieval 9종, 고정 query prompt, full corpus NDCG@10, 9-task 단순 평균

공식 MTEB 점수를 Sionic 표에 끼워 넣지 않습니다. 데이터셋·subset·split·MTEB commit은 [`configs/sionic9_protocol.json`](../../configs/sionic9_protocol.json)에 고정했습니다. Qwen, Comsat, F2LLM, PwC, 우리 모델을 모두 같은 스크립트로 다시 평가합니다.

실행 전 protocol 검증:

```bash
.venv-mteb/bin/python scripts/evaluate_sionic9.py --list-only
```

상태: evaluator 구현, 환경 검증 중.

성공 조건:

- Qwen 카드/Comsat 카드와 허용 오차 내 일치
- 모든 task의 raw JSON과 실행 시간을 보존
- prompt와 truncation을 바꾼 민감도 표 작성
