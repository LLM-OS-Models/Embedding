# State of text embeddings — 2026-07-17

## 라이브 Multilingual v2

공식 MTEB backend를 2026-07-17에 다시 조회한 결과입니다. `MTEB(Multilingual,
v2)` 응답은 131 tasks, 454 rows, 131개 task가 모두 있는 complete rows 169개였고,
response SHA-256은
`35c3779288a0a630f8486329b34144a027b4a60b8782d4ce40597567cbb5d98e`다.
아래 순위는 complete row만 임의로 다시 정렬한 값이 아니라 backend가 준 Borda rank다.

| Borda | Model | Params | Mean(Task) | Mean(Type) | Retrieval | Zero-shot |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `microsoft/harrier-oss-v1-27b` | 27.0B | 74.27 | 64.20 | 78.27 | 78% |
| 2 | `tencent/KaLM-Embedding-Gemma3-12B-2511` | 11.8B | 72.32 | 62.51 | 75.66 | 73% |
| 3 | `nvidia/llama-embed-nemotron-8b` | 7.5B | 69.46 | 61.09 | 68.69 | 99% |
| 4 | `Qwen/Qwen3-Embedding-8B` | 7.6B | 70.58 | 61.69 | 70.88 | 99% |
| 5 | `google/gemini-embedding-001` | closed | 68.37 | 59.59 | 67.71 | 99% |
| 6 | `Qwen/Qwen3-Embedding-4B` | 4.0B | 69.45 | 60.86 | 69.60 | 99% |
| 7 | `Octen/Octen-Embedding-8B` | 7.6B | 67.84 | 60.28 | 71.61 | 99% |
| 8 | `codefuse-ai/F2LLM-v2-14B` | 14.0B | 68.74 | 59.45 | 66.50 | 88% |
| 9 | `codefuse-ai/F2LLM-v2-8B` | 7.6B | 68.09 | 58.99 | 66.15 | 88% |
| 10 | `microsoft/harrier-oss-v1-0.6b` | 0.6B | 69.01 | 59.00 | 70.75 | 78% |

`Zero-shot`은 model author가 MTEB metadata에 신고한 training dataset과 benchmark task의 비중으로 계산되는 진단값이지 완벽한 contamination detector는 아닙니다. 그래도 Harrier/KaLM과 Qwen/Nemotron은 같은 조건이 아님을 보여줍니다.

## Granite Multilingual R2의 위치

2026-04-29 공개된
[`ibm-granite/granite-embedding-311m-multilingual-r2`](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2)는
한국어를 명시적으로 튜닝한 52개 언어에 포함하고, ModernBERT·CLS pooling·768-d
MRL·model card 기준 32K context를 지원한다. Apache-2.0 weight이고 multi-teacher KD,
contrastive tuning, model merging을 사용했다고 공개했다. 공식 card의 별도 18-task
multilingual retrieval score는 65.2이며 512→128 dimension 축소 결과도 보고한다.

그러나 live 131-task Multilingual v2 complete board에서는 311M 모델이 Borda 68위,
Mean(Task) 55.96, Mean(Type) 49.35, Retrieval 65.21이고 97M 모델은 83위다. 따라서
이는 작은 parameter/768-d/장문 처리의 **효율 baseline**이지 현재 8B 절대 성능
backbone을 교체할 후보가 아니다. card는 permissive/public pair뿐 아니라 IBM 내부·생성
데이터도 사용했다고 밝히며 전체 training code와 row manifest는 공개하지 않는다. card의
32K와 MTEB metadata의 `maxTokens=8192` 표기도 구분해서 기록한다.

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

후속 backbone ablation에서 F2LLM-v2-8B, Nemotron 8B, KaLM 12B를 비교하고, Harrier
0.6B와 Granite 311M은 별도 효율 frontier에 둡니다.
