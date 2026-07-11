---
language:
- en
- zh
license: other
task_categories:
- sentence-similarity
- text-retrieval
pretty_name: Korean Embedding Sionic Health Multilingual 100K
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# Korean Embedding — Sionic health multilingual 100K

Qwen3-Embedding 계열의 한국어 PublicHealthQA와 multilingual medical retrieval을
보강하기 위한 100,000-row performance dataset이다. F2LLM-v2 collection의 영어 중심
medical QA/instruction/flashcard와 소량 중국어 WebMedQA를 query/positive/negative
contrastive schema로 묶었다.

## 사용 조건

`release_eligible: false`인 performance/non-commercial 연구용 composite다. 통합
라이선스 표기는 `other`이며 collection card의 Apache-2.0 표기가 각 upstream source의
권리·개인정보·의료 데이터 조건을 재허가하지 않는다. 상업·임상 사용을 위한 데이터가
아니며 의료 조언 시스템의 안전성을 보장하지 않는다.

PublicHealthQA test, Sionic 9, 공식 MTEB Korean evaluation repository는 loader 입력으로
사용하지 않았다. 이 데이터는 PublicHealthQA의 공식 train-family 데이터도 아니다.
다만 15-task exact-hash audit에서 114개 고유 text가 retrieval corpus와 겹쳤으므로
완전한 clean zero-shot 데이터로 공개하지 않는다. 평가 query/evaluation-text match는 0이다.

## 구성

모든 source revision은
`codefuse-ai/F2LLM-v2@d520b8ad02c86d5e5611441c6196ff65d8888927`이다.

| File | Rows |
|---|---:|
| `pubmedqa.parquet` | 25,000 |
| `healthcaremagic.parquet` | 25,000 |
| `medical_instruction.parquet` | 20,000 |
| `medical_flashcards.parquet` | 15,000 |
| `medmcqa.parquet` | 10,000 |
| `medqa_en.parquet` | 3,000 |
| `webmedqa.parquet` | 2,000 |
| 합계 | **100,000** |

F2 collection이 경고한 MKQA와 SIB200 train files는 포함하지 않았다. 각 row는 positive
1개와 seed 42로 고른 explicit negative 7개를 갖는다.

## 스키마와 사용법

```json
{
  "messages": [{"role": "user", "content": "Instruct: ...\nQuery: ..."}],
  "positive_messages": [[{"role": "user", "content": "positive"}]],
  "negative_messages": [
    [{"role": "user", "content": "negative 1"}],
    [{"role": "user", "content": "negative 2"}]
  ]
}
```

```python
from datasets import load_dataset

ds = load_dataset(
    "LLM-OS-Models/korean-embedding-performance-v1-sionic-health-100k",
    split="train",
)
print(ds[0]["messages"][0]["content"])
```

## 전수 품질 감사

- row hash mismatch 0, provenance index mismatch 0
- 100,000/100,000 rows가 negative 7개 보유
- query body 4자 미만 0
- query styles: natural question 54,149; keyword/comma 28,137; long 15,336;
  short/title 2,378
- query length p50/p95: 176/730 characters
- positive length p50/p95: 610/1,611 characters
- upstream instruction variants: 208
- exact duplicate beyond first: query 6,198; positive 9,952

## 15-task benchmark text-hash 감사

- checked: query full 100K, query body 100K, positive 100K, negative 700K
- query/evaluation-text critical matches: **0**
- retrieval-corpus unique matches: 114
- matched occurrences: positive 91, negative 266
- source occurrences: MedMCQA 226, MedQA-en 131
- task locations: Ko-StrategyQA 110 unique; MIRACL/MrTidy 4 unique
- PublicHealthQA query/corpus exact match: **0**

task 간 같은 hash는 중복될 수 있다. 감사 파일은 원문을 저장하지 않고 SHA, training
role/source count와 benchmark task 위치만 저장한다.

## 무결성

- train SHA-256:
  `6f9715bb130e1d58bac74f13d4b6d1996840bf45b1569ab281a92f632ac15302`
- provenance SHA-256:
  `cc9e41b7d4c7442ea7f78a4071ed9d94bb439e9374297ab54216b062d67054db`
- quality audit: `metadata/training_data_quality_audit.json`
- benchmark overlap audit: `metadata/benchmark_overlap_audit.json`

Sionic 9 macro, 공식 Korean v1, clean 종합 회귀를 끝내기 전에는 이 dataset build를
Comsat 우위나 SOTA 근거로 사용하지 않는다. MIRACL 등 corpus-only exposure도 모델
카드에서 target/domain-adapted 사실과 함께 공개한다.
