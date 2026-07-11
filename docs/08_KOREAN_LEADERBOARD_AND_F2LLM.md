# MTEB Korean leaderboard와 상위 모델 감사

기준일: 2026-07-11. 공식 라이브 backend의 값은 native-prompt MTEB 결과로 사용하며, Sionic이 별도로 고른 retrieval 9종과는 섞지 않습니다.

## 결론

- `codefuse-ai/F2LLM-v2-8B`가 공식 `MTEB(kor, v1)`의 **Borda 1위**입니다.
- 단순 평균은 `PwC-Embedding_expr`가 더 높습니다: F2 `75.11`, PwC `77.01`.
- F2는 6개 중 MIRACL 2개를 학습한 66% zero-shot 모델입니다.
- PwC는 6개 중 5개 평가 계열을 학습한 16% zero-shot specialist입니다.
- 따라서 F2의 결과가 PwC보다 일반화 근거로는 강하지만, 둘 다 완전 zero-shot 비교는 아닙니다.
- Qwen3-Embedding-8B, Comsat, Harrier, Nemotron, 최신 KaLM은 현재 이 Korean 보드에 결과가 없습니다. F2 #1은 이 모델들을 직접 이겼다는 뜻이 아닙니다.

공식 원자료:

- [MTEB Korean live scores](https://mteb-leaderboard-backend.hf.space/v1/benchmarks/MTEB%28kor%2C%20v1%29/scores)
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [고정한 MTEB source](https://github.com/embeddings-benchmark/mteb/tree/193e3f66d2deac678065a43354c9c4efc57f507d)

## 보드가 실제로 보는 것

| Task | Type | Main metric | 규모 |
|---|---|---|---:|
| KLUE-TC | Classification | Accuracy | validation 2,048 |
| KLUE-STS | STS | cosine Spearman | 519 pairs |
| KorSTS | STS | cosine Spearman | 1,376 pairs |
| Ko-StrategyQA | Retrieval | NDCG@10 | 592 queries / 9,251 docs |
| MIRACLRetrieval | Retrieval | NDCG@10 | 213 queries / 1,486,752 docs |
| MIRACLReranking | Reranking | NDCG@10 | 212 queries / 약 21,201 candidates |

`Mean(Task)`는 6개 task를 각각 1/6로 평균하고, `Mean(TaskType)`은 Retrieval, STS, Reranking, Classification 네 유형을 각각 1/4로 평균합니다. `Rank`는 평균순이 아니라 task별 순위 점수를 합산하는 Borda입니다.

## 라이브 상위 10개

| Borda | Model | Mean(Task) | Mean(Type) | Retrieval | Rerank | STS | Class. | Zero-shot | Params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | F2LLM-v2-8B | 75.11 | 72.68 | 73.42 | 63.24 | 86.51 | 67.56 | 66% | 7.57B |
| 2 | F2LLM-v2-14B | 74.85 | 72.43 | 72.33 | 61.67 | 87.09 | 68.62 | 66% | 14.0B |
| 3 | PwC-Embedding_expr | 77.01 | 75.92 | 72.15 | 72.31 | 86.22 | 73.00 | 16% | 0.56B |
| 4 | F2LLM-v2-1.7B | 73.77 | 71.59 | 70.78 | 63.44 | 85.49 | 66.67 | 66% | 1.72B |
| 5 | F2LLM-v2-4B | 73.63 | 71.20 | 71.07 | 61.47 | 85.93 | 66.32 | 66% | 4.02B |
| 6 | multilingual-e5-large-instruct | 70.94 | 67.53 | 69.93 | 52.34 | 85.59 | 62.24 | 66% | 0.56B |
| 7 | F2LLM-v2-0.6B | 70.88 | 68.70 | 67.05 | 62.41 | 83.45 | 61.88 | 66% | 0.60B |
| 8 | jina-embeddings-v3 | 69.06 | 64.43 | 71.78 | 42.73 | 84.89 | 58.30 | 100% | 0.57B |
| 9 | snowflake-arctic-embed-l-v2.0 | 69.71 | 66.70 | 73.27 | 57.01 | 78.18 | 58.35 | 66% | 0.57B |
| 10 | gte-multilingual-base | 69.29 | 66.57 | 68.91 | 56.38 | 80.52 | 60.48 | 66% | 0.31B |

백엔드에는 137행이 있고 6개 task를 모두 완주한 모델은 101개입니다. 동적 보드와 캐시의 모델 수가 다를 수 있으므로 날짜와 raw endpoint를 함께 기록합니다.

## F2LLM-v2-8B는 어떻게 만들었나

F2는 CPT가 아니라 `Qwen/Qwen3-8B` 채팅 모델을 **full-parameter contrastive training**한 모델입니다. Qwen3-Embedding-8B를 이어 학습한 모델이 아닙니다.

- causal attention 유지
- EOS 위치 last-token pooling
- 4096차원, L2 normalization
- LoRA/QLoRA가 아닌 full fine-tuning
- 8B에는 pruning이나 knowledge distillation을 적용하지 않음

### 두 학습 단계

1. Stage 1: instruction 없는 retrieval 중심 27M examples
2. Stage 2: instruction-aware multi-task 18M examples, source당 최대 80K query

공개 composite는 157개 source, 60.1M rows, 약 564GB입니다. 60.1M 전체를 각 단계에 모두 넣었다는 뜻은 아닙니다. 논문상 한국어 라벨은 약 1.083M, 전체의 1.8%입니다.

대표 한국어 source에는 KoMagpie, KoAlpaca, WebFAQ Korean, MQA Korean, PAWS-X Korean, MIRACL Korean, MLDR Korean, MKQA Korean, ParaCrawl en↔ko가 있습니다.

### Loss와 negative

retrieval batch는 in-batch InfoNCE와 explicit hard-negative CE를 더합니다.

```text
L = L_in_batch + L_explicit_hard_negative
temperature = 0.05
```

- query마다 hard-negative 후보 24개를 저장하고 매 step 7개를 표집
- v2 negative miner는 Qwen3-Embedding-8B
- classification/clustering은 false negative를 줄이기 위해 explicit negatives만 사용
- 같은 batch에는 같은 source의 데이터만 넣는 homogeneous batching
- MRL prefix dimensions: 8, 16, 32, …, 2048, 4096

전작 F2의 공개 필터는 top-100에서 상위 5개를 빼고, cosine `<0.8`, negative score가 positive의 약 95% 미만인 후보 중 상위 24개를 고르는 방식입니다. v2가 동일 threshold를 썼는지는 공개 근거가 없습니다.

### 8B hyperparameters

- LR `6e-6`
- global batch 512
- 2 epochs
- AdamW beta `(0.9, 0.98)`, weight decay `.01`
- cosine scheduler, BF16, FlashAttention 2, gradient checkpointing, ZeRO-2
- 공개 예시의 train max length는 1024; model position은 40,960; MTEB 평가는 8,192이므로 서로 구분해야 함

원문과 코드:

- [F2LLM-v2 paper](https://arxiv.org/abs/2603.19223)
- [F2LLM predecessor](https://arxiv.org/abs/2510.02294)
- [F2LLM training code](https://github.com/codefuse-ai/CodeFuse-Embeddings/tree/main/F2LLM)
- [F2LLM-v2 data](https://huggingface.co/datasets/codefuse-ai/F2LLM-v2)

완전한 one-command 재현은 아닙니다. hard-negative mining/필터링, 157개 source 변환, decontamination 코드, 8B의 stage별 exact config와 GPU 시간은 공개되지 않았습니다. 코드 저장소 자체의 LICENSE도 불명확하고, composite의 Apache 표기가 upstream source별 권리를 덮어쓰지 않습니다.

## PwC-Embedding_expr는 어떻게 만들었나

계보는 `XLM-R-large → multilingual-e5-large-instruct → PwC`입니다.

- 24-layer encoder, mean pooling, 1024-d, cosine
- 모델 weight는 F32, 실효 text 길이 512 + special tokens
- 추가 학습 데이터로 KLUE-STS, KLUE-TC, KorSTS, KoTripletQA, KorNLI, MIRACL을 선언
- training code, 최종 data mix, paper는 비공개

과거 삭제된 trainer state에서 복구한 값:

- 4 GPU, per-device batch 32, global batch 128
- BF16 training, LR `1e-5`, linear scheduler, warmup 1%
- 약 92K examples/epoch, 계획 1.15 epoch
- gradient checkpointing, no-duplicates batch sampler
- LoRA 파일이 없고 optimizer가 약 4.47GB이므로 full fine-tuning으로 보는 것이 타당

Korean 보드 6개 중 KLUE-TC, KLUE-STS, KorSTS, MIRACL Retrieval/Reranking 5개가 in-domain입니다. base E5 대비 유일한 zero-shot task Ko-StrategyQA는 `79.91 → 79.69`로 오히려 0.23 point 하락했습니다. 그러므로 PwC를 한국어 zero-shot SOTA로 표현하면 안 됩니다.

## 우리 모델에 가져올 것

1. source별 homogeneous batch
2. in-batch InfoNCE + explicit-HN loss ablation
3. query별 24-candidate pool에서 매 step 4–7개 표집
4. BM25 + Qwen base/current student 후보와 positive-relative false-negative filtering
5. retrieval checkpoint와 STS/classification checkpoint를 분리한 뒤 clean validation으로 merge
6. 마지막 3–5개 checkpoint soup

Qwen3-Embedding-8B는 이미 대규모 embedding pretraining을 받았으므로 F2의 27M Stage 1부터 반복할 이유는 낮습니다. Korean high-quality Stage 2를 먼저 수행하는 편이 계산 효율적입니다.

## 주장 규칙

- `generalization track`: Korean MTEB 6개와 Sionic 9개 계열을 학습·mining에서 차단
- `leaderboard-adapted track`: train split 사용을 허용하되 task별 노출과 zero-shot 비율 공개
- Borda #1, Mean(Task) #1, retrieval #1은 서로 다른 주장으로 씀
- Korean MTEB와 Sionic 9-task 평균을 하나의 평균으로 합치지 않음
