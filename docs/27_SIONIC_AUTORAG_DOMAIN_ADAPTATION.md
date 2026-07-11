# Sionic AutoRAG finance/commerce/legal domain adaptation

기준일: 2026-07-12 (Asia/Seoul)

## 결론

AutoRAG는 Comsat이 base Qwen보다 가장 크게 앞선 Sionic task 중 하나다
(`0.8522` 대 `0.8276`, 약 `+0.0246`). 평가 corpus/query를 학습하는 대신 F2LLM-v2의
finance, banking, commerce, legal QA를 격리한 100K domain shard를 만들었다. 원본
AutoRAG exact query/corpus overlap은 0이다.

데이터는 영어·중국어 중심이다. 한국어 transfer는 Qwen3의 multilingual alignment에
의존하므로, 1M general winner에 50:50 replay로 짧게 적응시킨 뒤 AutoRAG뿐 아니라
Sionic 9 macro와 공식 Korean/clean 회귀를 모두 본다.

## 고정 100K 구성

revision:
`codefuse-ai/F2LLM-v2@d520b8ad02c86d5e5611441c6196ff65d8888927`.

| Source | Available | Used | Domain |
|---|---:|---:|---|
| FIQA | 7,452 | 7,000 | finance |
| Amazon QA | 59,340 | 31,000 | commerce/product QA |
| Banking77 | 9,993 | 9,000 | banking intents |
| MultiCPR e-commerce | 90,850 | 42,000 | Chinese e-commerce |
| LawZhiDao | 11,899 | 11,000 | Chinese legal QA |
| 합계 | 179,534 | **100,000** |  |

각 row는 positive 1개와 source가 제공한 최대 24개 후보에서 seed 42로 고른 negative
7개를 갖는다. Query body 최소 4자, positive/negative 최소 8자, exact pair de-dup을
적용했다.

## 실제 품질

| 검사 | 결과 |
|---|---:|
| rows / negative count | 100,000 / 모두 7 |
| row SHA mismatch | 0 |
| provenance index mismatch | 0 |
| query body 4자 미만 | 0 |
| short/title style | 53,844 |
| natural question | 36,511 |
| long / comma-keyword | 6,501 / 3,144 |
| query p50/p95 | 24 / 138 chars |
| positive p50/p95 | 83 / 734 chars |
| query / positive duplicate beyond first | 2,928 / 2,468 |

train SHA:
`9b636831e1f4c5eb5d453c0b5f18eb642115035ba13d75a4d70ffd9fb905b835`.

provenance SHA:
`05006632636b7c619152dca259db1dd71b32fb9d3263bb30e024c702e34d0f01`.

## 평가 오염 감사

15-task text-only blocklist에 query full/body 각 100K, positive 100K, negative 700K를
대조했다.

- benchmark query/evaluation-text critical match: **0**
- AutoRAG query/corpus exact match: **0**
- corpus-only match: 고유 1개
- 위치: Amazon QA query-body 1회가 Ko-StrategyQA corpus text hash와 동일

원문은 audit에 저장하지 않는다. 유일한 corpus-only match 때문에 clean zero-shot으로
부르지 않지만 AutoRAG test leakage 근거는 없다.

## 자동 학습 계약

1. 1M general winner로 100K query/positive corpus를 임베딩한다.
2. FAISS IVF512, search-k 256, exact candidate dot 재계산을 사용한다.
3. `s_neg < .95*s_pos`, pool24에서 score-rank quantile 7개를 고른다.
4. complete source-homogeneous batches만 남긴다.
5. target 50% + general 50%, LoRA r64, LR `5e-6`, effective batch 64로 학습한다.
6. Sionic 9, 공식 Korean v1, clean source-heldout를 전부 평가한다.
7. 파생 curriculum과 merged model/card/evaluation raw files를 Hugging Face에 공개한다.

AutoRAG 114 query는 macro 평균에서 1/9를 차지해 분산이 크다. 따라서 그 한 task 점수만
보고 checkpoint를 반복 선택하지 않는다. 50:50이 broad regression을 만들면 25:75로
낮추고, macro gain이 없으면 general winner를 유지한다.

## 공개 artifact

- dataset:
  `LLM-OS-Models/korean-embedding-performance-v1-sionic-autorag-100k@9140e9e02bb3f40ac1c22a6e595d58208770f696`
- quality audit: `reports/sionic-autorag-domain-100k-training-data-audit.json`
- overlap audit: `reports/sionic-autorag-domain-100k-benchmark-overlap-audit.json`
- queue: `scripts/run_sionic_autorag_adaptation_queue.sh`

dataset build만으로 Comsat 우위나 SOTA를 주장하지 않는다. 실제 9-task score를 모델
revision과 함께 얻은 뒤 README 세 보드에 반영한다.
