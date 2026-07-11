---
language:
- ko
- en
license: other
task_categories:
- sentence-similarity
- feature-extraction
pretty_name: Korean Embedding Performance v1 Pilot 50K
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# Korean Embedding Performance v1 — Pilot 50K

Qwen3-Embedding 계열의 한국어 retrieval 성능 실험을 위한 50,000-row 연구용
contrastive dataset이다. 각 row는 instruction-aware query, positive passage 1개,
hard/easy negative passage 1–7개를 ms-swift embedding message schema로 저장한다.

## 사용 조건과 공개 범위

이 저장소의 통합 라이선스는 `other`다. `release_eligible: false`이며 연구·비상업
성능 실험용이다. 구성 source에는 라이선스 미표기, custom/noncommercial 조건,
CC-BY-SA-4.0, upstream 조건이 혼재한다. 이 카드가 upstream 권리를 재허가하지
않는다. 상업 사용이나 재배포 가능성을 주장하지 않으며 사용자는 아래 고정 source
각각의 최신 조건을 직접 확인해야 한다.

이 데이터는 공식 benchmark의 train/task-family 데이터를 의도적으로 포함한다.
따라서 이 데이터로 학습한 모델을 완전한 zero-shot 모델이라고 부르면 안 된다.

## 구성

| Source | Pinned revision | Rows | 알려진 평가 노출 |
|---|---|---:|---|
| `nlpai-lab/ko-triplet-v1.0` | `1f5d72d21ae8309b5221a588b13930b423385bff` | 25,147 | construction/decontamination 미공개 |
| `codefuse-ai/F2LLM-v2` Korean QA | `d520b8ad02c86d5e5611441c6196ff65d8888927` | 18,000 | 알려진 직접 노출 없음 |
| F2 MIRACL Korean train-family | same | 700 | MIRACL retrieval/reranking |
| F2 MrTidy Korean train | same | 1,200 | MrTidyRetrieval |
| F2 MLDR Korean train-family | same | 1,500 | MultiLongDocRetrieval |
| `klue/klue` YNAT train | `349481ec73fff722f88e0453ca05c77a447d967c` | 1,000 | KLUE-TC |
| `klue/klue` STS train | same | 1,000 | KLUE-STS |
| `dkoterwa/kor-sts` train | `016f35f9b961daaaa7a352e927084e3da662ac1f` | 500 | KorSTS |
| `taeminlee/Ko-StrategyQA` train qrels | `d243889a3eb6654029dbd7e7f9319ae31d58f97c` | 953 | Ko-StrategyQA |

합계는 50,000 rows다. F2 Korean QA는 `webfaq_kor` 8,000,
`mqa_ko` 3,000, `koalpaca` 3,500, `koalpaca_realqa` 3,500으로 구성된다.

Sionic 9개 retrieval task 중 MIRACL, MrTidy, MLDR, Ko-StrategyQA 4개는
train/task-family 노출이 있다. AutoRAG, PublicHealthQA, Belebele,
SQuADKorV1, LawIRKo의 evaluation row는 사용하지 않았다. 공식 MTEB Korean v1은
여섯 task 모두에 직·간접 train-family 노출이 있으므로 점수는 in-domain 결과로
표기해야 한다.

## 스키마

```json
{
  "messages": [{"role": "user", "content": "Instruct: ...\\nQuery: ..."}],
  "positive_messages": [[{"role": "user", "content": "positive passage"}]],
  "negative_messages": [
    [{"role": "user", "content": "negative passage 1"}],
    [{"role": "user", "content": "negative passage 2"}]
  ]
}
```

`metadata/provenance.jsonl`은 각 row의 source ID, repository, revision, split,
row hash, `trained_on_tasks`, benchmark exposure를 같은 row index로 연결한다.
`metadata/manifest.json`은 source별 accepted/rejected 수와 최종 SHA-256을 담는다.

## 불러오기

```python
from datasets import load_dataset

ds = load_dataset(
    "LLM-OS-Models/korean-embedding-performance-v1-pilot-50k",
    split="train",
)
print(ds[0]["messages"][0]["content"])
```

provenance는 별도 파일로 받는다.

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "LLM-OS-Models/korean-embedding-performance-v1-pilot-50k",
    "metadata/provenance.jsonl",
    repo_type="dataset",
)
```

## 학습 예시

이 repository의 JSONL은 ms-swift embedding trainer 형식이다.

```bash
swift sft \
  --model Qwen/Qwen3-Embedding-8B \
  --task_type embedding \
  --train_type lora \
  --dataset data/train.jsonl \
  --loss_type infonce \
  --lora_rank 64 \
  --lora_alpha 128 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --bf16 true \
  --gradient_checkpointing true
```

실제 성능 실험에서는 source-homogeneous batching, current-student hard-negative
refresh, positive-relative false-negative filtering, temperature/MRL ablation을
권장한다. 평가 query/qrel/corpus를 학습 데이터로 되먹이지 않아야 한다.

## 재현성과 무결성

- seed: `42`
- rows: `50,000`
- `data/train.jsonl` SHA-256:
  `b46a7be9842ab27e9dfd85e9831080d94410e5b38d956682072068ee7f18258a`
- `metadata/provenance.jsonl` SHA-256:
  `e8ccca33bb9ec73700895ab2ac17ae57e20875170be1ed0f7a3dbc20b11e6031`
- build config: `configs/performance_data_mix_v1.json`
- builder: `scripts/build_performance_mix.py`
- publisher: `scripts/publish_performance_dataset.py`

```bash
.venv-train/bin/python scripts/build_performance_mix.py \
  --phase pilot_50k \
  --output-dir outputs/data/performance-v1/pilot-50k

.venv-train/bin/python scripts/publish_performance_dataset.py
```

두 번째 명령은 기본적으로 hash/row/card 검증만 한다. 실제 Hub 업로드는 명시적으로
`--upload --public`을 넘겨야 한다.

## 제한

- dataset build 완료는 모델 성능 향상을 증명하지 않는다.
- 일부 negative는 현재 Qwen3-Embedding-8B에 너무 쉬울 수 있다.
- source별 upstream 권리와 개인정보·유해 콘텐츠를 독립적으로 감사하지 않았다.
- 공개 leaderboard 점수에는 학습 task 노출을 반드시 함께 표시해야 한다.
