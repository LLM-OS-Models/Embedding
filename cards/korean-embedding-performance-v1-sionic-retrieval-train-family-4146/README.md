---
language:
- ko
license: other
task_categories:
- text-retrieval
pretty_name: Korean Sionic Retrieval Train-Family 4,146
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# Korean Sionic Retrieval Train-Family 4,146

F2LLM-v2가 공개한 Korean MIRACL, MrTidy, MLDR train-family row만 1M
decontaminated curriculum에서 lossless 추출한 target-adaptation dataset이다. 공개
evaluation query는 포함하지 않으며 current-student HN7 mining 전의 source artifact다.

## 구성과 목적

| source | rows | 역할 |
|---|---:|---|
| `f2_miracl_ko_train` | 700 | MIRACL Korean retrieval train-family |
| `f2_mrtidy_korean_train` | 1,200 | MrTidy Korean train |
| `f2_mldr_ko_train` | 2,246 | MLDR Korean long-document train-family |
| 합계 | **4,146** |  |

Sionic 9 카드에서 Qwen3-Embedding-8B가 Comsat에 뒤지는 폭은 MIRACL `0.0181`,
MrTidy `0.0066`, MLDR `0.0147`이다. 이 데이터는 public test를 생성·mining에 쓰지
않고 공개 train-family supervision의 비중을 짧게 높이는 specialist ablation용이다.

## Benchmark 노출과 contamination

이 데이터는 MIRACL/MrTidy/MLDR의 공식·provider train-family를 직접 사용하므로 해당
점수는 clean zero-shot이 아니다. Sionic 9 + 공식 MTEB Korean v1의 15-task text hash
감사 결과는 다음과 같다.

- critical query/evaluation-text unique match: **0**
- declared non-retrieval train-family match: 0
- shared retrieval corpus unique match: 13,973
- status: `pass_with_retrieval_corpus_exposure`

shared corpus exposure와 task-train 사용을 모델 카드에 반드시 표시한다.

## 출처

- parent dataset:
  `LLM-OS-Models/korean-embedding-performance-v1-performance-1m@5a2a3ab7`
- F2LLM-v2 source revision:
  `d520b8ad02c86d5e5611441c6196ff65d8888927`
- selection: exact `source_id` membership, parent row order 유지
- parent row index와 원 row SHA-256을 provenance에 보존

## 스키마와 사용

```python
from datasets import load_dataset

ds = load_dataset(
    "LLM-OS-Models/korean-embedding-performance-v1-sionic-retrieval-train-family-4146",
    split="train",
)
```

각 row는 ms-swift의 `messages`, `positive_messages`, `negative_messages` schema다.
실제 학습 queue는 current 1M winner로 candidate 24개를 찾고 `.95*s_pos` 아래의
score-quantile negative 7개를 고른 뒤, 같은 수의 1M general row와 50:50 replay한다.

## 무결성

- rows: `4,146`
- `data/train.jsonl` SHA-256:
  `6837367935ea56912375fe6a476360eb7dd0efcc0100459901e92a44029b7c60`
- `metadata/provenance.jsonl` SHA-256:
  `9d97802b378b6c2d3bd15824db2ab3a680315f9ee6fb95492b16778c093d015e`
- blocklist manifest SHA-256:
  `24f1eba04ec16436cab674c3709788c5dff2571106cd6159d75f5d711314ac1d`
- builder: `scripts/extract_training_source_subset.py`

## 사용 조건과 제한

`release_eligible: false`, 통합 라이선스 `other`인 연구·비상업 performance artifact다.
F2 composite 상단 조건이 upstream 권리를 자동으로 재허가한다고 주장하지 않는다.
일부 passage/negative가 매우 길어 trainer는 `max_length=512`와 right truncation을
명시한다. 4,146행을 여러 epoch 반복해 외우지 않고 50% general replay와 낮은 LR로
한 번의 짧은 specialist 실험만 수행한다.
