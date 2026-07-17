# 다음 단계 teacher와 추가 데이터

기준일은 2026-07-15이다. 아래 자산은 모두 repository-local Hugging Face cache에 exact
revision으로 다운로드했다. 제3자 원본을 우리 조직에 재업로드하지 않고, 파생 데이터가
생길 때만 원본 revision·변환·감사 SHA를 포함해 별도 공개한다.

> **2026-07-17 상태 정정:** 아래 “다운로드했다”는 2026-07-15 container의 역사적
> 상태다. 재시작 후 local cache는 없으며 revision·SHA 계약을 이용해 NFS에 다시
> 복원해야 한다.

## 고정 자산

| 역할 | Hugging Face asset | Revision | License / 크기 |
|---|---|---|---|
| reranker teacher | `Qwen/Qwen3-Reranker-8B` | `77d193c791ed757ca307ee72715aa132723da912` | Apache-2.0, 5 shards, 약 16GB |
| 금융 triplet | `BCCard/BCAI-Finance-Kor-Embedding-Triplet` | `f63d59969dba9916bd34c86c82112331890b11da` | CC-BY-4.0, train 43,394 / validation 1,000 / test 1,000 |
| 금융 corpus | `BCCard/BCAI-Finance-Kor-Embedding-Pair` | `e022cb013f2907e0716ebe40d13f30ed93ffa9b0` | CC-BY-4.0, 45,589 rows / 36,281 unique corpus chunks |
| temporal QA | `etri-lirs/KoTSQA-v.2.0` | `ff9349df469a765b4561959e36ef1b3f377765cd` | CC-BY-SA-4.0, train 750 / test 6,750 |

`scripts/restore_hf_assets.py`의 `qwen-reranker-teacher`,
`bcai-finance-triplet`, `bcai-finance-pair`, `kotsqa-v2` key로 exact revision과 파일
SHA를 다시 복구·검증할 수 있다.

## 채택 방법

Qwen reranker는 현재 student와 같은 공개 Qwen 계열이며 100개 이상 언어와 32K
context를 지원한다. 첫 사용은 1M 전체 재채점이 아니라 positive를 반드시 포함한
hybrid top-100~200 후보의 연속 yes-token probability cache다. 다음을 같은 clean dev
token budget에서 비교한다.

1. reranker false-negative filter만 적용
2. InfoNCE + score-distribution KL
3. InfoNCE + MarginMSE

모든 score row에는 model revision, instruction, tokenization, yes/no token IDs, raw
logits와 normalization을 남긴다. public test 결과로 KD coefficient를 고르지 않는다.

BCAI 금융 데이터는 `train`만 performance-domain 후보로 사용하고 validation은 제한된
개발 진단, test는 최종 진단으로 봉인한다. 카드 자체 감사가 보고한 다음 한계 때문에
clean selection board로는 사용하지 않는다.

- train∩test 동일 query 60개(6%), train∩validation 50개(5%)
- train hard negative 약 2%가 validation/test parent family에 노출
- triplet의 약 73–77%가 lexical signal로 풀림
- corpus 386 chunks(1.06%)가 서로 다른 ID 아래 byte-identical

따라서 train ingest 전 exact/MinHash dedup, split-aware negative 제거, 현재 200K/1M 및
전체 benchmark blocklist와의 overlap 감사를 수행한다. test score는 duplicate-query
포함/제외를 함께 보고하고 BM25 기준선도 제공한다.

KoTSQA test 6,750개는 unchanged/changed/new와 single-hop/multi-hop/false-premise를
포함하는 시간성 진단이다. 전부 training blocklist에 추가하고, 750개 train도 현재
후보 학습에는 넣지 않는다. 먼저 `question → evidence passage` retrieval 변환의 qrel
완전성과 false-premise 정의를 감사한 뒤 temporal diagnostic으로만 사용한다.

## 실행 순서

1. 중단된 200K r64의 exact data/environment를 복원하고 처음부터 재학습해 clean held-out으로 첫 유효 후보 여부를 판정한다.
2. 개선이 확인되면 Qwen reranker score cache와 filter-only/KD 소규모 A/B를 만든다.
3. BCAI train-only finance shard와 KoTSQA test blocklist의 exact/near-overlap 감사를
   완료한다.
4. clean/worst-domain/다국어 회귀가 좋아질 때만 1M 또는 domain replay에 편입한다.

다운로드 완료 자체는 성능 근거가 아니다. 파생 curriculum은 전수 감사와 학습 manifest,
모델은 clean·public board 결과가 생긴 시점에 각각 Hugging Face에 올린다.
