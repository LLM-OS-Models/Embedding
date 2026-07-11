# Executive summary

## 목표

두 목표를 분리합니다.

1. **숫자 목표:** Comsat 카드의 한국어 retrieval 9종 macro NDCG@10 `0.7930`을 넘는다.
2. **연구 목표:** 공개 test를 학습하지 않고, 새로운 한국어 문서·질의에서도 Qwen3-Embedding-8B와 Comsat보다 낫다는 근거를 만든다.

첫 목표만 원하면 공개 corpus/qrel에 맞춰 쉽게 점수를 올릴 수 있지만, 이는 benchmark chasing입니다. 이 저장소의 기본 track은 두 번째 목표입니다.

## 지금까지 확인한 사실

- Comsat 카드는 `1M+ Korean examples`라고만 밝힙니다. unique 문서 수, 토큰 수, pair/triplet 구조, 출처, loss, hard-negative 방법, dedup/decontamination은 공개하지 않았습니다.
- 모델 계보는 `Qwen3-Embedding-8B -> Comsat`이고, query prompt, last-token pooling, 4096차원, L2 normalization이 사실상 같습니다.
- 따라서 공개 증거에 가장 잘 맞는 해석은 **한국어 retrieval contrastive continued training**입니다. LM next-token CPT라고 볼 근거는 없습니다.
- Qwen 대비 평균 상승은 `+0.0105`이며 9개 중 8개에서 상승, LawIRKo에서 `-0.0007`입니다.
- 이 표는 공식 `MTEB(kor, v1)`이 아니라 Sionic이 고른 공개 retrieval 9종의 자가 보고 결과입니다.

## 현재 최고 모델에서 배울 점

2026-07-11 라이브 `MTEB(Multilingual, v2)` Borda 순위는 다음과 같습니다.

| Borda | 모델 | Mean(Task) | Retrieval | 공개 학습 목록 기준 zero-shot |
|---:|---|---:|---:|---:|
| 1 | Microsoft Harrier OSS v1 27B | 74.27 | 78.27 | 78% |
| 2 | Tencent KaLM-Embedding-Gemma3-12B-2511 | 72.32 | 75.66 | 73% |
| 3 | NVIDIA llama-embed-nemotron-8b | 69.46 | 68.69 | 99% |
| 4 | Qwen3-Embedding-8B | 70.58 | 70.88 | 99% |

순위는 단순 평균순이 아닙니다. 예를 들어 Qwen은 Nemotron보다 평균이 높지만 task별 Borda 합산에서는 아래입니다. 또한 Harrier와 KaLM은 공개 학습 데이터 목록상 benchmark task와 겹치는 비중이 더 큽니다. 점수만 보고 “아키텍처가 무조건 우월하다”고 결론내리면 안 됩니다.

가져올 방법론의 우선순위는 다음입니다.

1. **데이터 품질과 task/domain 균형**
2. **retriever + BM25로 mining하고 reranker로 false negative 제거**
3. **큰 effective batch의 InfoNCE와 explicit hard negatives**
4. **문서 기반 synthetic query, 다중 생성기와 검증기**
5. **서로 다른 data mix/checkpoint의 weight merge**
6. **long-document curriculum과 evidence 위치 분산**
7. 작은 모델이라면 큰 embedder/reranker distillation

## 선택한 첫 학습

- Backbone: `Qwen/Qwen3-Embedding-8B`
- 방식: BF16 LoRA, all-linear, rank 32/64 비교
- Objective: InfoNCE, temperature 0.02 중심; in-batch + 1 explicit hard negative로 시작
- Query: retrieval instruction 포함
- Document: prefix 없음
- 길이: query 256, document 512의 short pilot 이후 2K/4K/8K bucket 추가
- 데이터: 첫 smoke test는 공개 Korean triplet 일부를 엄격히 필터링해 pipeline 검증에만 사용; release 후보는 권리·출처가 명시된 문서에서 새로 만든 pair로 교체
- 목표: 내부 blind dev에서 Qwen 대비 +1.5 NDCG point, public 9종은 마지막에 1회 평가

## 난이도 판단

같은 공개 9개 숫자만 넘기는 것은 **가능성이 높은 편**입니다. 차이가 1.05 NDCG point이고 동일 backbone의 last-mile adaptation이기 때문입니다. 그러나 깨끗한 unseen 평가까지 이기는 것은 데이터 생성·negative 품질·오염 통제가 필요한 실제 연구 과제입니다.
