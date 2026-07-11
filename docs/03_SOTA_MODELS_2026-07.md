# State of text embeddings — 2026-07-11

## 라이브 Multilingual v2

공식 MTEB backend를 2026-07-11에 조회한 결과입니다. 131개 task 전체 결과가 있는 모델을 기준으로 봅니다.

| Borda | Model | Params | Mean(Task) | Mean(Type) | Retrieval | Zero-shot |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `microsoft/harrier-oss-v1-27b` | 27.0B | 74.27 | 64.20 | 78.27 | 78% |
| 2 | `tencent/KaLM-Embedding-Gemma3-12B-2511` | 11.8B | 72.32 | 62.51 | 75.66 | 73% |
| 3 | `nvidia/llama-embed-nemotron-8b` | 7.5B | 69.46 | 61.09 | 68.69 | 99% |
| 4 | `Qwen/Qwen3-Embedding-8B` | 7.6B | 70.58 | 61.69 | 70.88 | 99% |
| 5 | `google/gemini-embedding-001` | closed | 68.37 | 59.59 | 67.71 | 99% |
| 6 | `Qwen/Qwen3-Embedding-4B` | 4.0B | 69.45 | 60.86 | 69.60 | 99% |
| 7 | `Octen/Octen-Embedding-8B` | 7.6B | 67.84 | 60.28 | 71.61 | 99% |
| 10 | `microsoft/harrier-oss-v1-0.6b` | 0.6B | 69.01 | 59.00 | 70.75 | 78% |

`Zero-shot`은 model author가 MTEB metadata에 신고한 training dataset과 benchmark task의 비중으로 계산되는 진단값이지 완벽한 contamination detector는 아닙니다. 그래도 Harrier/KaLM과 Qwen/Nemotron은 같은 조건이 아님을 보여줍니다.

## Harrier OSS v1 27B

확인된 내용:

- Gemma3-text 계열 27B, 5376-d
- causal decoder, last-token pooling, L2 normalization
- task별 instruction prompt
- 다국어 multi-task contrastive learning
- 카드상 32K 지원; config는 131,072 position을 가짐
- MIT 표기

공개되지 않은 내용:

- paper, 정확한 base relation, 데이터 목록/크기, training code
- loss 수식, hard-negative/teacher, hyperparameters, model merge 여부

작은 270M/0.6B에는 큰 embedding model의 knowledge distillation을 추가했다고 카드가 명시하지만 27B에는 해당되지 않습니다. 현재 점수만 보면 강하지만 어떤 기법을 복제해야 하는지 가장 불투명합니다.

## KaLM-Embedding-Gemma3-12B-2511

확인된 내용:

- base: `google/gemma-3-12b-pt`
- 11.76B, last-token, 3840-d
- 32K, MRL 3840/2048/1024/512/256/128/64
- 공개 fine-tuning dataset 약 6.34M rows, 각 query에 positive 1~4, negative 7
- custom Tencent KaLM community license
- 관련 논문: KaLM-Embedding 및 KaLM-Embedding-V2

핵심은 단순 모델 크기보다 공개된 대규모 multi-task pair/triplet, 7 hard negatives, Matryoshka 학습, 데이터 정제와 multi-stage training입니다. 다만 MTEB training overlap가 높아 pure zero-shot 비교에는 주의합니다.

## Llama-Embed-Nemotron-8B

가장 재현성이 좋은 상위 모델입니다.

- base: Llama-3.1-8B
- 모든 layer를 bidirectional attention으로 변환
- mean pooling, L2 normalization
- 총 16.1M pair: 7.7M public + 8.4M synthetic
- stage 1: web retrieval 약 70%, batch 2048, hard negative 1
- stage 2: retrieval/classification/STS/bitext, 4.3M high-quality mix, batch 128, hard negative 4
- InfoNCE temperature .02
- E5-Mistral + Qwen3로 negative mining, positive score의 95% 이상인 후보를 false-negative 위험으로 제외
- 서로 다른 data mix/hyperparameter의 6 checkpoint를 평균
- 공개 dataset, training code, technical report

최종 merge는 best single checkpoint 대비 Mean(Task) +0.84를 보고했습니다. 64×A100 80GB를 사용했기 때문에 그대로 복제할 수는 없지만, data recipe와 negative filtering, merge는 단일 H100 LoRA에도 이식 가능합니다.

## 어떤 모델을 베이스로 쓸 것인가

이번 목표에는 Qwen3-Embedding-8B를 유지합니다.

- Comsat과 동일 backbone이라 개선 원인을 분리하기 쉽습니다.
- 한국어 baseline이 이미 강합니다.
- global Mean과 Retrieval 모두 Nemotron보다 높습니다.
- Harrier 27B/KaLM 12B는 한 장 H100에서 반복 실험 비용이 큽니다.
- 상위 모델의 이득 중 재현 가능한 부분은 backbone 교체보다 data/negative/merge에 더 많이 있습니다.

후속 backbone ablation에서 Harrier 0.6B, Nemotron 8B, KaLM 12B를 비교합니다.
