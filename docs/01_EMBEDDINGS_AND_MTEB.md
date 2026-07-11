# 임베딩 모델과 MTEB

## 임베딩 모델이 하는 일

아주 단순화하면 다음과 같습니다.

```text
문서 d_i  --encoder-->  v_i  (미리 계산)
질의 q    --encoder-->  u    (검색 시 계산)
score_i = cosine(u, v_i)
score가 큰 문서부터 top-k 반환
```

Qwen3-Embedding-8B와 Comsat은 텍스트를 4096차원 벡터로 바꾸고 L2 정규화합니다. 이 경우 내적 `u @ v`가 곧 cosine similarity입니다. 비싼 부분은 8B 모델로 벡터를 만드는 과정이고, 한 번 벡터화한 뒤의 비교는 매우 싸며 FAISS/HNSW 같은 ANN index로 확장할 수 있습니다.

이 구조는 query와 document를 따로 encode하는 **bi-encoder**입니다. query-document를 한 입력으로 함께 읽는 cross-encoder/reranker보다 정밀도는 보통 낮지만, 문서 벡터를 미리 저장할 수 있어 수백만 문서 검색이 가능합니다.

## Comsat 표는 문장 유사도 평균이 아니다

Comsat 표의 모든 항목은 retrieval입니다.

1. query에 instruction을 붙여 벡터화합니다.
2. corpus의 모든 문서를 별도로 벡터화합니다.
3. cosine으로 문서를 정렬합니다.
4. 정답 문서가 top 10의 얼마나 높은 위치에 왔는지 `NDCG@10`으로 평가합니다.

`0.7930`은 평균 cosine도, 79.3% accuracy도 아닙니다. 9개 dataset의 NDCG@10을 각각 구해 동일 가중으로 평균낸 값입니다.

정답 문서가 하나이고 relevance가 binary인 단순한 경우, 정답이 1위면 NDCG는 1, 2위면 약 0.631, 5위면 약 0.387, 10위 밖이면 top-10 기여가 0입니다. 실제 task는 관련 문서가 여러 개거나 graded relevance일 수 있습니다.

## Sionic의 9개 retrieval task

| Task | 주로 보는 능력 |
|---|---|
| MIRACL | 한국어 Wikipedia 질문에서 관련 passage 찾기 |
| MrTyDi | 자연스러운 한국어 질의의 Wikipedia dense retrieval |
| MLDR | 긴 문서에서 질문과 관련된 전체 문서 찾기 |
| AutoRAG | 금융·공공·의료·법률·상거래 PDF/RAG retrieval |
| Ko-StrategyQA | multi-hop 질문의 근거 문서 찾기 |
| PublicHealthQA | CDC/WHO 공중보건 질문과 답변 연결 |
| Belebele | 독해 질문과 원 지문 연결 |
| SQuADKorV1 | 한국어 Wikipedia QA passage retrieval |
| LawIRKo | 법명/조문명 질의와 법 조문 연결 |

작은 task도 큰 task와 똑같이 평균의 1/9를 차지합니다. PublicHealthQA 한국어 subset은 77 query, AutoRAG는 114 query인 반면 SQuADKor는 5,774 query입니다. 따라서 macro 평균은 작은 task의 분산과 오염에 민감합니다.

## MTEB Multilingual은 더 넓다

`MTEB(Multilingual, v2)`는 131개 task, 9개 task type, 250개 이상의 언어를 묶습니다.

- Retrieval: query로 corpus를 검색
- Reranking: 주어진 후보를 similarity로 재정렬
- STS: cosine과 사람의 의미 유사도 점수 간 상관
- Classification / Multilabel: frozen embedding 위의 단순 probe 성능
- Clustering: embedding 공간의 군집 품질
- Pair classification: paraphrase/entailment/중복 쌍 구분
- Bitext mining: 서로 다른 언어의 번역문 nearest-neighbor 정렬
- Instruction reranking: 조건/instruction 변화에 맞춰 순위를 바꾸는 능력

`Mean(Task)`는 task별 main score를 평균하고, `Mean(TaskType)`은 먼저 type별 평균을 낸 뒤 type을 동일 가중합니다. 라이브 `Rank`는 평균순이 아니라 task별 상대 순위를 합치는 Borda count입니다.

따라서 Qwen 카드의 `70.58`과 Comsat의 `0.7930`은 직접 비교할 수 없습니다. Comsat 점수를 79.30으로 바꿔도 범위와 aggregation이 전혀 다릅니다.

## MTEB 점수에 포함되지 않는 것

- 실제 vector DB의 ANN recall/latency
- 문서 index 구축 시간과 저장 비용
- 8B/27B 모델의 GPU 비용
- query score calibration과 threshold 안정성
- 최신 사내 문서에서의 domain shift
- 공개 test contamination
- RAG 최종 답변의 사실성

그러므로 최종 선택은 MTEB와 별도로 사내/비공개 retrieval set, 처리량, 메모리, RAG answer metric을 함께 봐야 합니다.
