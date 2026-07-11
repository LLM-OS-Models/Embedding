---
language:
- en
- zh
license: other
task_categories:
- sentence-similarity
- text-retrieval
pretty_name: Korean Embedding Sionic AutoRAG Domain 100K
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# Korean Embedding — Sionic AutoRAG domain 100K

AutoRAG의 금융·상거래·법률 domain retrieval을 보강하기 위한 100,000-row
performance dataset이다. F2LLM-v2 collection의 영어 FIQA/Amazon/Banking77과 중국어
e-commerce/legal QA를 query/positive/negative contrastive schema로 묶었다.

## 사용 조건과 평가 노출

`release_eligible: false`인 performance/non-commercial 연구용 composite다. 통합
라이선스는 `other`이며 F2 collection의 Apache-2.0 표기가 개별 upstream 권리를
재허가하지 않는다.

AutoRAG evaluation repository, query, qrel, corpus는 loader 입력으로 사용하지 않았다.
15-task exact audit에서 query/evaluation-text critical match는 0, AutoRAG exact match도
0이다. Ko-StrategyQA corpus와 같은 짧은 text hash가 query body에 1개 있어 이를
corpus-only exposure로 공개한다. clean zero-shot dataset이라고 주장하지 않는다.

## 구성

모든 source revision:
`codefuse-ai/F2LLM-v2@d520b8ad02c86d5e5611441c6196ff65d8888927`.

| File | Rows | Domain/language |
|---|---:|---|
| `fiqa.parquet` | 7,000 | finance, English |
| `amazon_qa.parquet` | 31,000 | commerce, English |
| `banking77.parquet` | 9,000 | banking intents, English |
| `multicpr_ecom.parquet` | 42,000 | e-commerce, Chinese |
| `lawzhidao.parquet` | 11,000 | legal QA, Chinese |
| 합계 | **100,000** |  |

각 row는 positive 1개와 explicit negative 7개를 갖는다.

## 사용법

```python
from datasets import load_dataset

ds = load_dataset(
    "LLM-OS-Models/korean-embedding-performance-v1-sionic-autorag-100k",
    split="train",
)
print(ds[0]["messages"][0]["content"])
```

JSONL은 ms-swift embedding schema인 `messages`, `positive_messages`,
`negative_messages` 세 필드로 구성된다. provenance는 row별 source/file/revision,
row SHA와 benchmark exposure를 보존한다.

## 전수 품질 감사

- rows 100,000; negative 7개/row
- row SHA mismatch 0; provenance index mismatch 0
- query body 4자 미만 0
- query styles: short/title 53,844; natural question 36,511; long 6,501;
  comma/keyword 3,144
- query length p50/p95: 24/138 characters
- positive length p50/p95: 83/734 characters
- upstream instruction variants 115
- exact duplicates beyond first: query 2,928; positive 2,468

## 15-task benchmark hash 감사

- checked: query full/body 각 100K, positive 100K, negative 700K
- query/evaluation-text critical matches: **0**
- AutoRAG query/corpus match: **0**
- corpus-only match: 고유 1 hash, Amazon QA query-body occurrence 1,
  Ko-StrategyQA corpus 위치

감사 결과에는 원문이 없고 SHA와 role/source/task 위치만 있다.

## 무결성

- train SHA:
  `9b636831e1f4c5eb5d453c0b5f18eb642115035ba13d75a4d70ffd9fb905b835`
- provenance SHA:
  `05006632636b7c619152dca259db1dd71b32fb9d3263bb30e024c702e34d0f01`
- quality audit: `metadata/training_data_quality_audit.json`
- overlap audit: `metadata/benchmark_overlap_audit.json`

Sionic 9 macro, 공식 MTEB Korean v1, clean 종합 회귀를 완료하기 전에는 AutoRAG나
Comsat 성능 향상을 주장하지 않는다. MIRACL을 포함한 다른 task도 모두 함께 평가한다.
