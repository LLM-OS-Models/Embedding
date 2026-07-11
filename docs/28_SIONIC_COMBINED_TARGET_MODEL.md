# Sionic 9 combined target-domain model

기준일: 2026-07-12 (Asia/Seoul)

## 왜 combined candidate가 필요한가

SQuAD, health, AutoRAG, legal specialist를 각각 만들면 per-task 최고점은 찾기 쉽지만
Sionic 9 평균을 한 모델에서 동시에 올린다는 보장은 없다. 최종 공개 후보는 1M general
winner에서 다시 시작해 네 target domain과 general replay를 한 curriculum에서 학습한다.
개별 specialist를 순차로 덮어쓰지 않아 마지막 domain의 catastrophic forgetting을 줄인다.

공개 카드 기준 Qwen3-Embedding-8B가 Comsat에 뒤지는 폭은 다음과 같다. 이 숫자는
evaluation row를 데이터 생성에 사용한 것이 아니라 두 모델 카드의 고정 aggregate를
비교한 사전 설계 근거다.

| Task | Comsat - Qwen | combined에서의 대응 |
|---|---:|---|
| AutoRAG | +0.0242 | finance/commerce/legal target 10% + legal 일부 |
| MIRACL | +0.0181 | 1M general의 공식 train-family + general replay |
| PublicHealthQA | +0.0150 | medical/health target 10% |
| MLDR | +0.0147 | 1M general의 long retrieval train-family |
| SQuADKorV1 | +0.0105 | KorQuAD train-family target 10% |
| MrTidy | +0.0066 | 1M general의 task-train family |
| Ko-StrategyQA | +0.0031 | 1M general의 공식 train qrels |
| Belebele | +0.0025 | multilingual/cross-lingual general replay |
| LawIRKo | **-0.0007** | Qwen이 이미 우세; legal replay로 회귀 방지·clean 보강 |

따라서 별도 target component는 general 1M만으로 직접 메우기 어려운 AutoRAG,
PublicHealthQA, SQuADKorV1에 두고, MIRACL/MLDR/MrTidy/StrategyQA는 이미 포함된 공식
train-family와 55% general replay로 유지한다. legal 15%는 LawIR 점수만을 위한 비중이
아니라 AutoRAG의 legal slice와 별도 clean 법률 holdout 강건성을 함께 노린다.

## 사전 등록 mixture

| Role | Rows | 비중 | 입력 |
|---|---:|---:|---|
| SQuADKor train-family | 40,000 | 10% | current-student quantile HN7 |
| health/medical | 40,000 | 10% | current-student quantile HN7 |
| AutoRAG finance/commerce/legal | 40,000 | 10% | current-student quantile HN7 |
| MIRACL/MrTidy/MLDR train-family | 4,144 | 1.04% | 2K current-student quantile HN7 |
| Korean legal/public | 60,000 | 15% | current-student quantile HN7 |
| general 1M replay | 215,856 | 53.96% | mined 1M 우선, 아니면 audited homogeneous 1M |
| 합계 | **400,000** | **100%** |  |

실제 mining drop으로 한 component가 부족하면 available complete 16-row batches까지만
사용하며 manifest에 실제 role fraction을 기록한다. 구성은 public test score를 본 뒤
임의로 바꾸지 않고 첫 combined run에 고정한다.

## 데이터·순서 계약

[`scripts/build_multidomain_curriculum.py`](../scripts/build_multidomain_curriculum.py)는
각 입력의 source-homogeneous batch byte offset을 먼저 스캔한다. row를 메모리에 적재하지
않고 complete batch reference만 seed 42로 섞으며 batch 내부 순서는 유지한다.

provenance의 `multidomain_curriculum_batch`에는 다음이 추가된다.

```json
{
  "batch_index": 0,
  "batch_size": 16,
  "role": "health",
  "source_id": "f2_pubmedqa",
  "output_row_index": 0
}
```

감사기는 원 input의 stale `homogeneous_batch`보다 가장 최근
`multidomain_curriculum_batch`를 우선 검증한다. 모든 final row의 training SHA,
provenance alignment, source, query style, negative count와 15-task exact overlap을 다시
계산한다. query/evaluation-text critical overlap이 하나라도 있으면 학습하지 않는다.

## 학습

| 항목 | 값 |
|---|---|
| Base | 1M general winner merged checkpoint |
| Tuner | LoRA r64, alpha 128, all-linear |
| Loss | InfoNCE, in-batch + explicit HN7 |
| LR | `5e-6`, cosine, warmup 5% |
| Effective batch | 64 |
| Length | 512, right truncation |
| Steps | actual rows / 64, 400K이면 6,250 |
| Best selection | held-out InfoNCE minimum; public test로 checkpoint 반복 선택 금지 |

FA2 격리 환경은 실제 backward probe를 통과할 때만 사용하며 실패하면 SDPA로 자동
fallback한다. training 뒤 safe merge parity와 SentenceTransformers last-token/L2
계약을 검증한다.

## 승격·공개

combined model은 다음을 모두 수행한다.

1. Sionic retrieval 9종 전체 NDCG@10
2. 공식 MTEB Korean v1 6-task exact Qwen instruction loader
3. clean legal source-document-heldout
4. model/data/weights SHA와 task-family/corpus exposure가 포함된 모델 카드
5. final 400K curriculum, provenance, quality/overlap audit 공개

개별 specialist, general 1M, combined 후보의 세 보드를 비교해 raw Sionic 9 평균이 가장
좋고 broad/clean 회귀가 허용 범위인 모델만 대표 모델로 승격한다. target-adapted 결과를
clean zero-shot SOTA로 부르지 않는다.

자동 실행은 [`scripts/run_sionic_combined_adaptation_queue.sh`](../scripts/run_sionic_combined_adaptation_queue.sh)이며
법률 mining/adaptation이 끝난 직후 시작한다.
