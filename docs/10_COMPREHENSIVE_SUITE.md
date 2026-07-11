# Clean Korean 종합 평가 보드

기준일: 2026-07-11. 이 보드는 MTEB Korean 6종이나 Sionic retrieval 9종을 다시 평균내는 제3의 홍보 점수가 아니다. 공개 test에 과적합하지 않고 실제 배포 품질로 checkpoint를 고르기 위한 내부 clean holdout이다.

## 왜 별도 보드가 필요한가

- 공식 Korean MTEB는 폭은 있지만 6개 task뿐이고 일부 상위 모델은 평가 계열을 학습했다.
- Sionic 9종은 모두 retrieval이며 선택된 공개 benchmark에 집중한다.
- 실제 검색에서는 최신 문서, 긴 문맥, OCR 오류, 말투 변화, 법률·보건의 false negative, latency와 저장비용이 함께 중요하다.

따라서 세 보드의 점수를 하나로 합치지 않는다. 승격은 아래 필수 gate와 subscore를 함께 보고 결정한다.

## 평가 축

| 축 | 측정할 것 | 주 metric | 최소 보고 단위 |
|---|---|---|---|
| Clean retrieval | 정부·법률·보건·금융·상거래·일반 검색 | NDCG@10, Recall@10/100, MRR@10 | domain, query style |
| Broad semantic | STS, paraphrase, intent/classification | Spearman, accuracy/F1 | task, label balance |
| Long context | 512/2K/4K/8K, evidence head/middle/tail | NDCG@10, recall | token bucket, evidence position |
| Robustness | OCR, 띄어쓰기, 오탈자, 존댓말/구어체, 짧은 keyword | relative score retention | perturbation type/severity |
| Multilingual regression | English 및 기존 multilingual holdout | task metric delta vs base | language/task |
| Efficiency | encode throughput, p50/p95 latency, VRAM, index size | docs/s, ms/query, GiB | length, batch, dimension |

## 오염 등급

각 row와 평가 세트에는 다음 등급을 기록한다.

- `U`: 모델/학습 데이터와 관계가 확인되지 않은 unknown
- `T`: task 계열 또는 train split 노출
- `I`: instruction/domain만 유사하고 문서·질의 중복은 차단
- `Z`: source URL/hash, exact match, MinHash 및 embedding near-duplicate를 모두 차단한 clean zero-shot

주 결과는 `Z`와 `I`만 사용한다. `T` 결과는 leaderboard-adapted 보조 표로 분리한다.

## 데이터 split과 생성 규칙

1. 원문 권리와 revision/date가 확인된 source에서 문서를 수집한다.
2. 문서 단위로 train/dev/test의 source 또는 시간 구간을 분리한다.
3. test query는 해당 문서에 접근하지 않은 annotator 또는 고정 생성 prompt로 만든 뒤 사람이 relevance를 검수한다.
4. public MTEB/Sionic query·corpus와 exact URL/hash, normalized text, MinHash, dense-neighbor를 비교한다.
5. hard negative mining은 train corpus에서만 수행하며 dev/test corpus를 인덱스에 넣지 않는다.
6. 모든 결과에 dataset manifest SHA, model revision, prompt, max length, dimension, hardware를 남긴다.

## 모델 승격 gate

release candidate는 다음을 모두 만족해야 한다.

- Sionic 9종 전체를 동일 protocol로 직접 재현하고 Comsat 평균 `0.7930`을 넘어설 것
- clean retrieval macro가 Qwen base와 Comsat 모두보다 높고 domain 한 곳의 큰 회귀가 없을 것
- broad semantic과 multilingual holdout에서 사전에 정한 허용 회귀 이내일 것
- 3개 이상 seed 또는 paired bootstrap 95% CI로 개선의 불확실성을 보고할 것
- 데이터 라이선스와 provenance가 public weight 배포를 허용할 것
- latency, peak VRAM, embedding 차원과 index 비용을 모델 카드에 공개할 것

단일 평균은 모델의 실패를 숨길 수 있어 주 순위로 쓰지 않는다. 모든 필수 축을 완주한 모델에 한해 보조 Borda와 Pareto frontier만 제공한다.

## 구현 순서

1. Qwen/Comsat/F2/PwC와 우리 후보의 고정 inference wrapper를 검증한다.
2. 시간 분리된 rights-safe retrieval holdout부터 만들고 Qwen/Comsat 기준선을 고정한다.
3. long/noise 변형은 원본 query와 paired 평가가 되도록 생성한다.
4. throughput harness와 dimension ablation(MRL)을 붙인다.
5. 공개 benchmark는 최종 checkpoint에 한 번 실행하고, selection은 이 clean dev 보드에서만 한다.
