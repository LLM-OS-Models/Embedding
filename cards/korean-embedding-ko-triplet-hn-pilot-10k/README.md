---
language:
- ko
license: other
task_categories:
- text-retrieval
- sentence-similarity
pretty_name: Korean Embedding Ko-Triplet Hard-Negative Pilot 10K
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
---

# Korean Embedding Ko-Triplet Hard-Negative Pilot 10K

`nlpai-lab/ko-triplet-v1.0`에서 결정론적으로 뽑은 한국어 retrieval train 10,000행과
validation 512행에 Qwen3-Embedding-8B dense hard negative 4개씩을 붙인 연구용
ms-swift embedding dataset이다.

## 사용 조건

원 source 카드에 명시적 라이선스가 없어 통합 라이선스는 `other`, manifest의
`release_eligible`은 `false`다. 연구·비상업 성능 실험용이며 이 카드가 원 source의
권리를 재허가하지 않는다.

## 출처와 sampling

- source: `nlpai-lab/ko-triplet-v1.0`
- pinned revision: `1f5d72d21ae8309b5221a588b13930b423385bff`
- source split: `train`
- seed: `42`
- train/validation: `10,000 / 512`
- source manifest SHA-256:
  `0c87d0e33ddce952c604b3669c4400505cf9d9c92b808c32974e4ae45cbf6f2f`

## Hard-negative mining

- miner: `Qwen/Qwen3-Embedding-8B@1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`
- normalized 4,096-dimensional embeddings, exact blockwise cosine search
- candidate pool 24, selected explicit negatives 4
- false-negative gate: `candidate_score < 0.95 × positive_score`
- own positive와 normalized exact query/document match 제외
- 동점은 normalized document SHA-256 오름차순
- train search: 10,000 × 10,000 = 100,000,000 exact dot products
- train selected negative cosine: mean `0.50020`, p95 `0.59108`
- validation selected negative cosine: mean `0.41864`, p95 `0.50179`

`metadata/*_mining_audit.jsonl`은 row별 score와 선택 근거를 보존하지만 document text는
포함하지 않는다.

## Benchmark contamination 감사

Sionic retrieval 9종과 공식 MTEB Korean v1 6종의 pinned text-only SHA-256 blocklist로
query, positive, negative 전체를 검사했다.

| split | rows | critical query/evaluation match | declared train-family | shared retrieval corpus |
|---|---:|---:|---:|---:|
| train | 10,000 | **0** | 0 | 0 |
| validation | 512 | **0** | 0 | 0 |

이 결과는 exact normalized text 중복이 없다는 뜻이지 의미적 유사성까지 없다는 뜻은
아니다. blocklist 원문은 공개하지 않고 hash와 task/split 위치만 사용한다.

## 스키마와 사용법

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
    "LLM-OS-Models/korean-embedding-ko-triplet-hn-pilot-10k",
)
print(ds["train"][0])
```

```bash
swift sft \
  --model Qwen/Qwen3-Embedding-8B \
  --model_type qwen3_emb \
  --task_type embedding \
  --tuner_type lora \
  --dataset data/train.jsonl \
  --val_dataset data/validation.jsonl \
  --loss_type infonce \
  --lora_rank 64 \
  --lora_alpha 128 \
  --max_length 512 \
  --truncation_strategy right
```

## 무결성

- train SHA-256:
  `3df507549ea801d9e1c4aba54d9bf95a88b6690b6b27a0f1e1a05b3c0c525adc`
- validation SHA-256:
  `f121f7eb3011ee2bfd796cb7622efd4b6f8f8ad80d09525cf083eeb18c7a9ede`
- train mining audit SHA-256:
  `fe1b25159067a6c33615c0bb0c950c897daa518fb34367534f01471f54fbefae`
- validation mining audit SHA-256:
  `366e4b46abae9871eed371070cc48db3c1882e5e13c02781bcb6008691c22c08`
- benchmark blocklist manifest SHA-256:
  `24f1eba04ec16436cab674c3709788c5dff2571106cd6159d75f5d711314ac1d`

## 제한

- 10K 규모와 Qwen3-mined negatives는 최종 SOTA 데이터가 아니라 loss/튜닝 검증용이다.
- miner와 student가 같은 base family라 model-specific bias가 있다.
- 명시적 라이선스가 없는 source이므로 상업·재배포 가능성을 주장하지 않는다.
- public benchmark를 checkpoint 선택에 사용하지 않으며 validation InfoNCE와 별도 clean
  holdout을 사용한다.

