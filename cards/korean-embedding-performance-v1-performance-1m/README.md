---
language:
- ko
- en
license: other
task_categories:
- sentence-similarity
- feature-extraction
pretty_name: Korean Embedding Performance v1 1M
size_categories:
- 1M<n<10M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# Korean Embedding Performance v1 — 1M

Qwen3-Embedding-8B의 한국어 retrieval data-scale 실험을 위한 정확히
1,000,000-row 연구·비상업 contrastive dataset이다. `release_eligible: false`, 통합
라이선스 `other`이며 upstream source 조건을 재허가하지 않는다.

## 구성

| 계열 | Rows | 비율 | 역할 |
|---|---:|---:|---|
| `nlpai-lab/ko-triplet-v1.0` | 600,254 | 60.03% | 넓은 한국어 QA/retrieval core |
| F2 Korean QA/instruction | 287,000 | 28.70% | webfaq, mqa, koalpaca, realQA, komagpie |
| F2 retrieval task train-family | 4,146 | 0.41% | MIRACL, MrTidy, MLDR |
| F2 PAWS-X Korean | 20,000 | 2.00% | paraphrase boundary |
| F2 ParaCrawl ko↔en | 40,000 | 4.00% | cross-lingual replay |
| KLUE YNAT/STS train | 19,000 | 1.90% | classification/STS |
| KorSTS train | 1,500 | 0.15% | Korean STS |
| Ko-StrategyQA train | 2,100 | 0.21% | reasoning evidence retrieval |
| KaLM multilingual replay | 26,000 | 2.60% | multilingual/semantic regression 완화 |

주요 revision:

- ko-triplet: `1f5d72d21ae8309b5221a588b13930b423385bff`
- F2LLM-v2: `d520b8ad02c86d5e5611441c6196ff65d8888927`
- KaLM fine-tuning data: `e9443ab6f5d4dc29c79cea03834e932428ed6ab1`
- KLUE: `349481ec73fff722f88e0453ca05c77a447d967c`
- KorSTS: `016f35f9b961daaaa7a352e927084e3da662ac1f`
- Ko-StrategyQA: `d243889a3eb6654029dbd7e7f9319ae31d58f97c`

## Benchmark 노출

공식 train/task-family source를 의도적으로 포함하므로 완전한 zero-shot 데이터가
아니다. 공식 Korean MTEB v1의 KLUE-TC, KLUE-STS, KorSTS, Ko-StrategyQA, MIRACL
retrieval/reranking에 직·간접 노출이 있다. Sionic 9에서는 MIRACL, MrTidy, MLDR,
Ko-StrategyQA가 노출된다. AutoRAG, PublicHealthQA, Belebele, SQuADKorV1, LawIRKo의
evaluation query/qrel/corpus는 builder가 읽지 않는다. KaLM replay도 여러 published
MTEB training task family를 포함하므로 multilingual score를 fully zero-shot이라고
표현하지 않는다.

## 스키마

```json
{
  "messages": [{"role": "user", "content": "Instruct: ...\\nQuery: ..."}],
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
    "LLM-OS-Models/korean-embedding-performance-v1-performance-1m",
    split="train",
)
```

## 권장 학습 순서

1. 50K/200K에서 loss, temperature, LoRA/DoRA/partial/full을 먼저 비교한다.
2. 1M source provenance로 16-row source-homogeneous microbatch를 만들고 trainer의
   추가 shuffle을 끈다.
3. base/current student top-24와 BM25 후보를 합친다.
4. positive와 같은 teacher scale에서 `s_neg < 0.95 × s_pos` false-negative filter를
   적용한다.
5. query당 4–7개 hard negative를 score quantile별로 표집한다.
6. 1M 한 epoch 이후 Sionic 9 전체, official Korean v1, clean comprehensive,
   multilingual regression을 측정한다.

공개 파일은 source-provided negative를 보존한다. 즉 이 파일 자체가 current-student
refresh를 이미 완료했다는 뜻은 아니다.

```bash
swift sft \
  --model Qwen/Qwen3-Embedding-8B \
  --task_type embedding \
  --tuner_type lora \
  --dataset data/train.homogeneous-b16.jsonl \
  --loss_type infonce \
  --lora_rank 64 \
  --lora_alpha 128 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --train_dataloader_shuffle false \
  --attn_impl flash_attention_2 \
  --bf16 true
```

## Provenance와 무결성

`metadata/provenance.jsonl`은 같은 row index로 source ID, repository, revision,
split/file, row SHA-256, `trained_on_tasks`, benchmark exposure를 보존한다.

- rows: `1,000,000`
- seed: `42`
- negatives per row: 최대 `7`
- `data/train.jsonl` SHA-256:
  `094d443e05cc27e4e764b5bfa253cf02c36ec769fbf7cd1e43fd937d73ec3c0a`
- `metadata/provenance.jsonl` SHA-256:
  `94334a0ef5dad83169fc8f00fc6705173c606f5976ef8365469fe1bc721b18c1`
- config: `configs/performance_data_mix_v1.json`
- phase: `performance_1m`
- builder: `scripts/build_performance_mix.py`
- homogeneous compiler: `scripts/build_homogeneous_batches.py`

## 제한

- 라이선스 미표기, custom/noncommercial, 여러 upstream 조건이 혼재한다.
- 개인정보·유해 콘텐츠의 독립적인 전수 감사가 완료되지 않았다.
- 1M이라는 규모 자체는 성능 향상을 보장하지 않는다. base에 쉬운 negative인 row는
  current-student mining으로 교체해야 한다.
- public benchmark 결과에는 train-family exposure와 per-task score를 함께 공개한다.
