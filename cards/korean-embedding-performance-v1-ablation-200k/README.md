---
language:
- ko
- en
license: other
task_categories:
- sentence-similarity
- feature-extraction
pretty_name: Korean Embedding Performance v1 Ablation 200K
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# Korean Embedding Performance v1 — Ablation 200K

Qwen3-Embedding-8B의 한국어 retrieval continued fine-tuning에서 LoRA/DoRA/부분 및
full fine-tuning, loss, hard-negative 전략을 비교하기 위한 200,000-row 연구·비상업
성능 데이터다. `release_eligible: false`이며 통합 라이선스는 `other`다. upstream
source별 조건을 재허가하지 않는다.

## 구성

| 계열 | Rows | 역할 |
|---|---:|---|
| `nlpai-lab/ko-triplet-v1.0@1f5d72d` | 100,254 | 넓은 한국어 QA/retrieval core |
| F2 Korean QA/instruction | 68,000 | webfaq, mqa, koalpaca, realQA, komagpie |
| F2 retrieval task train-family | 4,146 | MIRACL 700, MrTidy 1,200, MLDR 2,246 |
| F2 PAWS-X Korean | 5,000 | paraphrase boundary |
| F2 ParaCrawl ko↔en | 10,000 | cross-lingual replay |
| KLUE YNAT/STS train | 9,000 | classification/semantic similarity |
| KorSTS train | 1,500 | Korean STS |
| Ko-StrategyQA train qrels | 2,100 | reasoning evidence retrieval |

F2 revision은 `d520b8ad02c86d5e5611441c6196ff65d8888927`, KLUE는
`349481ec73fff722f88e0453ca05c77a447d967c`, KorSTS는
`016f35f9b961daaaa7a352e927084e3da662ac1f`, Ko-StrategyQA는
`d243889a3eb6654029dbd7e7f9319ae31d58f97c`로 고정했다.

## Benchmark 노출

이 데이터는 official train/task-family source를 의도적으로 포함한다. 공식 Korean
MTEB v1의 KLUE-TC, KLUE-STS, KorSTS, Ko-StrategyQA, MIRACL retrieval/reranking에
직·간접 노출이 있다. Sionic 9에서는 MIRACL, MrTidy, MLDR, Ko-StrategyQA 4개
계열이 노출된다. 따라서 이 데이터로 학습한 모델을 완전한 zero-shot 모델이라고
부르면 안 된다. AutoRAG, PublicHealthQA, Belebele, SQuADKorV1, LawIRKo evaluation
row는 데이터 builder가 읽지 않는다.

2026-07-12 revision부터 15-task text-only blocklist를 build 안에 적용한다. Retrieval
evaluation query text는 source 선언과 무관하게 차단하고, KLUE/KorSTS처럼 blocklist가
공식 train split도 포함하는 non-retrieval task는 해당 source의 `trained_on_tasks`가 같은
task일 때만 expected train-family exposure로 허용한다. 첫 200K에서 retrieval eval-query
match row 12개를 제거하고 같은 source의 다른 row로 cap을 다시 채웠다.

최종 raw/ordered artifact 전수 audit 결과:

- query/evaluation-text critical match: **0**
- declared official train-family match: ordered 기준 고유 19,226 hash
- retrieval corpus-only match: ordered 기준 고유 17,357 hash
- 이 결과는 target/train-family exposed이며 clean zero-shot이 아니다.

## 스키마와 사용법

```json
{
  "messages": [{"role": "user", "content": "Instruct: ...\\nQuery: ..."}],
  "positive_messages": [[{"role": "user", "content": "positive"}]],
  "negative_messages": [[{"role": "user", "content": "negative"}]]
}
```

```python
from datasets import load_dataset

ds = load_dataset(
    "LLM-OS-Models/korean-embedding-performance-v1-ablation-200k",
    split="train",
)
```

```bash
swift sft \
  --model Qwen/Qwen3-Embedding-8B \
  --task_type embedding \
  --tuner_type lora \
  --dataset data/train.jsonl \
  --loss_type infonce \
  --lora_rank 64 \
  --lora_alpha 128 \
  --learning_rate 2e-5 \
  --bf16 true
```

공개 파일의 negative는 source-provided negative다. 최종 성능 단계에서는 base 및
current student로 top-24를 다시 mining하고, teacher 기준
`s_neg < 0.95 × s_pos` false-negative filter 뒤 4–7개를 표집한다. 쉬운 row를 더
오래 학습하는 것보다 loss-active row로 교체하는 것을 우선한다.

## Provenance와 무결성

`metadata/provenance.jsonl`은 200,000 train row와 같은 index로 source/revision/split,
row hash, `trained_on_tasks`, exposure를 보존한다. source별 accepted/rejected 수는
`metadata/manifest.json`에 있다.

- seed: `42`
- rows: `200,000`
- `data/train.jsonl` SHA-256:
  `087c543e97975115b826455318bdae37bce371e63c396e2242ad7ef5fbd4a3c2`
- `metadata/provenance.jsonl` SHA-256:
  `3114c455cf4a4604401a1ea0c723ff1fa5918478f97d0c70da72a9cff0bf9cd5`
- exact trainer order: 199,904 rows
- ordered train SHA:
  `8e2731ab25299ff558af675f067b253a6ce4375a850aa925acfe3b3117505e3c`
- ordered provenance SHA:
  `89f90133a95e5bbad2ddb392a1494c2a6480e94888100434c24504c8ac2cc0ea`
- blocklist manifest SHA:
  `24f1eba04ec16436cab674c3709788c5dff2571106cd6159d75f5d711314ac1d`
- build config: `configs/performance_data_mix_v1.json`
- phase: `ablation_200k`

## 제한

- build 완료 자체는 모델 성능 향상을 증명하지 않는다.
- 일부 source의 dataset license가 없거나 upstream 조건이 혼재한다.
- 개인정보·유해 콘텐츠에 대한 독립적인 전수 감사를 수행하지 않았다.
- 공개 leaderboard 결과에는 train-family 노출을 함께 표기해야 한다.
