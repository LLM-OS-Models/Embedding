# F2LLM-v2-8B와 Comsat 학습 계보·레시피 감사

기준일: 2026-07-11 (Asia/Seoul)

대상:

- [`codefuse-ai/F2LLM-v2-8B`](https://huggingface.co/codefuse-ai/F2LLM-v2-8B)
- [`sionic-ai/comsat-embed-ko-8b-preview`](https://huggingface.co/sionic-ai/comsat-embed-ko-8b-preview)

이 문서는 모델 카드, 저자 공식 GitHub, 저자 논문, Hugging Face의 모델·데이터 API,
공식 MTEB live backend만 근거로 삼는다. 아래에서 `확인`은 primary source에 직접
적힌 내용, `코드 확인`은 공개 코드에서 직접 읽히는 동작, `추론`은 공개 artifact로부터
가능한 제한적 해석, `미공개`는 근거가 없는 항목을 뜻한다.

우리 평가 우선순위는 다음과 같이 고정한다.

1. Sionic Korean retrieval 9종
2. 공식 `MTEB(kor, v1)`
3. 독립적인 종합·다국어 평가

세 보드의 평균은 서로 합치지 않는다.

## 한 줄 결론

| 항목 | F2LLM-v2-8B | Comsat-ko-8B |
|---|---|---|
| 직접 출발점 | `Qwen/Qwen3-8B` | `Qwen/Qwen3-Embedding-8B` |
| Qwen3-Embedding 직접 파생 | **아님** | **맞음** |
| 학습 성격 | pair/tuple 기반 2-stage contrastive **full fine-tuning** | continued embedding fine-tune까지만 확인; loss·PEFT/full 여부 미공개 |
| raw-text LM CPT | 아님 | 했다는 근거 없음 |
| 공개 학습 규모 | Stage 1 약 27M + Stage 2 약 18M unique mixture | `1M+ Korean examples`만 공개 |
| `1M`의 단위 | 해당 없음 | examples; 문서 수나 token 수가 아님 |
| 논문 | [arXiv:2603.19223](https://arxiv.org/abs/2603.19223) | 연결된 논문 없음 |
| 학습 코드·데이터 | [코드](https://github.com/codefuse-ai/CodeFuse-Embeddings/tree/main/F2LLM), [60.147938M-row composite](https://huggingface.co/datasets/codefuse-ai/F2LLM-v2) | 미공개 |
| Sionic 9 평균 | `0.7621` | **`0.7930`** |
| 공식 Korean live | Borda **1위**, Mean(Task) `75.11` | 공식 제출 row 없음; 로컬 동일 protocol 측정은 별도 |

따라서 Sionic 9종만 먼저 이기는 목적에는 이미 `0.7825`인
`Qwen3-Embedding-8B`를 계속 학습하는 경로가 F2의 최종 weight `0.7621`에서
출발하는 것보다 유리하다. 반대로 F2의 **데이터 구성, source-homogeneous batch,
explicit hard negative, MRL, broad multi-task Stage 2**는 공식 Korean과 종합 성능을
올리는 레시피로 가져올 가치가 크다.

## 정확한 모델 계보

```text
Qwen/Qwen3-8B-Base
├── Qwen/Qwen3-8B
│   └── F2LLM-v2-8B-Preview     (Stage 1, instruction-free retrieval)
│       └── F2LLM-v2-8B         (Stage 2, instruction-aware multi-task)
└── Qwen/Qwen3-Embedding-8B
    └── comsat-embed-ko-8b-preview
```

이 계보는 각 모델 카드의 `base_model` metadata와 Hugging Face model tree가
가리키는 관계이다.

- F2 Stage 1 카드: [`base_model: Qwen/Qwen3-8B`](https://huggingface.co/codefuse-ai/F2LLM-v2-8B-Preview/blob/main/README.md)
- F2 Stage 2 카드: [`base_model: codefuse-ai/F2LLM-v2-8B-Preview`](https://huggingface.co/codefuse-ai/F2LLM-v2-8B/blob/main/README.md)
- Comsat 카드: [`base_model: Qwen/Qwen3-Embedding-8B`, relation `finetune`](https://huggingface.co/sionic-ai/comsat-embed-ko-8b-preview/blob/main/README.md)

두 모델 모두 Qwen3 decoder 구조라는 점만 같을 뿐, 출발 weight가 다르다. F2가
Qwen3-Embedding의 방법론을 참고하고 hard-negative miner로 Qwen3-Embedding-8B를
사용한 사실은 **weight lineage**를 Qwen3-Embedding으로 바꾸지 않는다.

## F2LLM-v2-8B 감사

### 무엇을 학습했나

F2는 raw 문서를 next-token prediction으로 계속 사전학습한 CPT가 아니다. 일반
`Qwen3-8B`를 `(query, positive, hard negatives)` 등의 embedding tuple로
full-parameter contrastive training한 모델이다.

확인된 representation contract는 다음과 같다.

- 표준 dense Qwen3 decoder와 causal attention을 유지한다.
- EOS token의 마지막 hidden state를 sequence embedding으로 사용한다.
- 8B hidden/embedding dimension은 `4096`이다.
- retrieval query에는 instruction을 붙이고 document에는 기본적으로 붙이지 않는다.
- embedding을 L2 normalize하므로 내적이 cosine similarity가 된다.
- model config의 position limit은 `40,960`이다.

공개 학습 코드가 optimizer에 `list(model.lm.parameters())` 전체를 넘기며 PEFT/LoRA
경로가 없으므로 8B는 full fine-tuning이다. 논문은 4B·8B·14B에 pruning이나 KD를
쓰지 않았다고 밝힌다. 8B에 LoRA를 썼다는 근거는 없다.

근거:

- [논문 §3, model architecture와 two-stage training](https://arxiv.org/abs/2603.19223)
- [전체 parameter를 optimizer에 전달하는 공식 코드](https://github.com/codefuse-ai/CodeFuse-Embeddings/blob/1c5291549b9cee9eeab1cd9de6a67be4d0295da0/F2LLM/run.py#L154-L164)
- [EOS last-token 추출 코드](https://github.com/codefuse-ai/CodeFuse-Embeddings/blob/1c5291549b9cee9eeab1cd9de6a67be4d0295da0/F2LLM/model.py#L27-L39)

### Stage 1: 약 27M instruction-free retrieval

논문 표의 공개 row 수를 합하면 정확히 `26,693,280`이며 논문과 Preview 카드는
이를 `27M`으로 반올림한다.

| Stage 1 source | Samples | 비중 |
|---|---:|---:|
| ParaCrawl | 10,684,184 | 40.03% |
| mMARCO | 5,470,174 | 20.49% |
| WebFAQ | 4,368,504 | 16.37% |
| CLIRMatrix | 3,275,561 | 12.27% |
| OpenCodeGeneticInstruct | 1,052,849 | 3.94% |
| CodeSearchNet | 936,813 | 3.51% |
| CodeSearchNet-CCR | 905,195 | 3.39% |
| **합계** | **26,693,280** | **100%** |

Stage 1에서는 raw pair/tuple을 쓰되 query instructional prefix는 사용하지 않는다.
이 단계의 결과가 [`F2LLM-v2-8B-Preview`](https://huggingface.co/codefuse-ai/F2LLM-v2-8B-Preview)다.
즉 저자들이 `semantic foundation` 또는 stage-1 embedding training이라고 부르는
것이지, raw-text language-model CPT가 아니다.

### Stage 2: 약 18M instruction-aware multi-task

Stage 2는 retrieval 외에 clustering, classification, STS, reranking, paraphrase,
bitext 등의 형식을 섞는다.

- 각 normalized data source에서 query를 최대 `80,000`개 표집한다.
- 최종 Stage 2 mixture는 약 `18M` examples이다.
- query에는 task-specific instruction을 붙인다.
- clustering, STS, bitext mining, paraphrase처럼 대칭인 task에서는 document와
  negative에도 `30%` 확률로 instruction을 붙인다.
- task schema는 retrieval, clustering, two-way classification 세 canonical
  형식으로 통일한다.

논문의 `157 public sources`와 코드의 `80K/source`에서 source의 단위를 주의해야
한다. 공개 dataset은 언어·subset별로 쪼개진 `500`개 parquet file이고, 공식
tokenizer가 file 단위로 `80K` cap을 적용한다. `157 × 80K = 12.56M`은 Stage 2의
18M보다 작으므로 여기서 cap의 source는 157개 upstream 이름만이 아니라
정규화된 언어/subset shard를 뜻하는 것으로 읽는 것이 코드와 숫자에 맞다.

### 60.1M composite와 실제 단계별 budget은 다르다

Hugging Face dataset server가 보고하는 공개 composite의 정확한 크기는 다음과 같다.

| 항목 | 값 |
|---|---:|
| Rows | `60,147,938` |
| Parquet files | `500` |
| Parquet bytes | `564,122,542,232` bytes |
| 논문상 upstream sources | `157` |
| 자연어 | `282` languages |
| 프로그래밍 언어 | `40+` |

출처: [F2LLM-v2 dataset](https://huggingface.co/datasets/codefuse-ai/F2LLM-v2),
[dataset size endpoint](https://datasets-server.huggingface.co/size?dataset=codefuse-ai/F2LLM-v2).

`60.1M`은 내려받을 수 있는 전체 composite 크기이지 8B가 한 stage에서 60.1M
전부를 보았다는 뜻이 아니다. 논문상 unique mixture budget은
`26.69328M + 약 18M = 약 44.69328M`이다. 8B hyperparameter 표의 `2 epochs`가
각 stage에 적용되었다고 읽으면 약 `89.39M` example presentations이지만, 저자들은
stage별 실제 optimizer step log와 중복 제거 후 exact exposure를 논문에 따로
보고하지 않았다. 그러므로 `89.39M`은 공개 숫자로 계산한 값이지 공식 보고
training counter는 아니다.

### 한국어 데이터 숫자

논문 language-distribution appendix의 한국어 label은 정확히 `1,083,205` samples다.
전체 공개 composite `60,147,938`의 약 `1.8009%`다. 이것은 Stage 2에서 실제로
사용한 한국어 row의 정확한 수와 동일하다는 뜻은 아니다.

명시적인 한국어-only source만 보면 다음과 같다.

| Source | Public composite rows |
|---|---:|
| KoMagpie | 428,780 |
| KoAlpaca | 21,126 |
| KoAlpaca-RealQA | 17,599 |
| **명시적 ko-only 소계** | **467,505** |

나머지 한국어 sample은 WebFAQ-ko, MQA-ko, MIRACL-ko, MrTidy-ko, MLDR-ko,
MKQA-ko, PAWS-X-ko, ParaCrawl en↔ko, CLIRMatrix hu-ko, SIB200 등의 multilingual
source에 분산된다. 특히 F2의 공개 training dataset list에는 Sionic 9종과 같은
계열인 MIRACL, MrTidy, MLDR가 모두 들어 있다. 그러므로 F2의 Sionic 9 결과를
완전한 zero-shot이라고 해석하면 안 된다.

### Loss와 hard negative: 코드로 확인되는 정확한 동작

retrieval sample의 기본 loss는 다음과 같다.

```text
L_retrieval = L_cross-GPU-in-batch + L_explicit-hard-negative
temperature = 0.05
```

- `L_in-batch`: 같은 global batch의 다른 positive document를 negative로 쓰는 CE.
- `L_hard`: 자기 positive 1개와 명시적 hard negative `K`개 사이의 CE.
- hard negatives는 **Qwen3-Embedding-8B로 mining**했다고 논문이 명시한다.
- 저장된 24개 후보 중 retrieval은 매 batch `7`개, clustering은 `9`개를 무작위
  표집한다.
- binary classification은 반대 label 하나만 explicit negative로 쓴다.
- clustering/classification에는 false negative를 피하기 위해 in-batch loss를
  쓰지 않고 explicit-negative loss만 쓴다.
- 각 DataLoader가 source별로 분리되어 한 batch는 한 source로만 구성된다.
  source 선택 확률은 남은 loader 길이에 비례한다. 즉 batch는 homogeneous지만
  source-uniform sampling은 아니다.

코드 근거:

- [temperature 0.05와 두 CE loss](https://github.com/codefuse-ai/CodeFuse-Embeddings/blob/1c5291549b9cee9eeab1cd9de6a67be4d0295da0/F2LLM/utils.py#L35-L114)
- [24개 pool에서 retrieval 7개·clustering 9개 표집](https://github.com/codefuse-ai/CodeFuse-Embeddings/blob/1c5291549b9cee9eeab1cd9de6a67be4d0295da0/F2LLM/run.py#L37-L59)
- [source별 DataLoader와 homogeneous batch](https://github.com/codefuse-ai/CodeFuse-Embeddings/blob/1c5291549b9cee9eeab1cd9de6a67be4d0295da0/F2LLM/run.py#L72-L127)

중요한 미공개 항목도 있다. v2 저자 repo에는 24개 후보를 실제로 검색·필터링하는
mining script가 없다. candidate retrieval depth, positive-relative threshold,
teacher score cutoff, duplicate/false-negative filter는 공개되지 않았다. 전작 F2의
threshold를 v2에 그대로 귀속하면 안 된다.

### MRL의 정확한 구현

8B는 `8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096` prefix dimension을
학습한다. 각 prefix를 다시 L2 normalize하고 loss를 계산한다. 공개 코드는 큰
dimension부터 순회하면서 각 term에 `1 / (n × sqrt(D / d))` 가중치를 준다.
따라서 단순히 10개 dimension loss를 같은 가중치로 평균하는 MRL은 아니다.

근거: [MRL dimension과 weighted loss 구현](https://github.com/codefuse-ai/CodeFuse-Embeddings/blob/1c5291549b9cee9eeab1cd9de6a67be4d0295da0/F2LLM/utils.py#L35-L114).

### 8B hyperparameters: 확정값과 예시값을 분리

논문 표가 8B에 직접 귀속한 값:

| Hyperparameter | 8B |
|---|---:|
| Total parameters | `7.568B` |
| Hidden dimension | `4096` |
| Layers / attention heads / KV heads | `36 / 32 / 8` |
| Learning rate | `6e-6` |
| Epochs | `2` |
| Global batch size | `512` |
| MRL | yes, minimum dimension `8` |
| KD teacher | none |

공식 current training code에서 확인되는 공통 설정:

- AdamW, beta `(0.9, 0.98)`, weight decay `0.01`
- cosine scheduler
- gradient checkpointing
- BF16, FlashAttention 2, DeepSpeed ZeRO-2
- seed `0` for training split/order, data-cap sampler seed `42`

다만 repo의 유일한 `configs/config.json`은 이름 그대로 **4B example**이며 LR
`8e-6`, per-device batch `16`, max length `1024`다. experiment 이름은 `16x32`를
기록하지만 함께 배포한 Accelerate YAML의 기본값은 8 processes라 CLI override 없이는
global batch 512가 되지 않는다. 이것을 8B final run의 exact config라고 쓰면 안 된다.
8B의 exact train max length,
node/GPU 수, GPU 종류, wall time, warmup steps, stage별 optimizer reset 여부는
공개 primary source에서 확인되지 않는다. model context `40,960`, MTEB evaluation
max length, training max length도 서로 다른 값이다.

### KD와 pruning을 8B에 잘못 귀속하지 말 것

F2LLM-v2 family 전체 설명에는 pruning과 knowledge distillation이 들어가지만
8B에 적용된 것은 아니다.

- 80M/160M/330M: Stage 1의 0.6B를 구조 pruning한 뒤 KD.
- 0.6B/1.7B: Stage 2에서 더 큰 F2 model로 embedding MSE KD.
- 4B/8B/14B: resource constraint 때문에 KD 미사용.

따라서 8B 성능을 설명하는 핵심은 대형 Qwen3 초기화, 약 27M retrieval Stage 1,
약 18M multi-task Stage 2, full FT, hard negatives, MRL이지 distillation이 아니다.

### F2의 재현 가능성과 남은 구멍

공개된 것:

- 논문 v1, 모델 weight, Stage-1 Preview, 일부 Stage-1 intermediate checkpoints
- `60,147,938` rows의 composite
- training loop, loss, MRL, KD, source-homogeneous loader
- model-size별 LR/epoch/global batch

완전히 공개되지 않은 것:

- 24-candidate hard-negative mining/false-negative filtering code
- final 8B의 exact stage별 config와 실행 log
- 157 upstream source를 500 normalized shard로 바꾸는 전체 converter
- benchmark query/corpus-level decontamination report
- GPU 수·종류·wall time·총 FLOPs
- final checkpoint 선택 또는 merge 절차

공개 dataset card도 2026-06에 MTEB가 MKQA와 SIB200 train split을 평가에 쓰므로
두 source를 제거하라고 사후 경고한다. 따라서 `fully open`은 상당 부분 맞지만
one-command bitwise reproduction과 contamination-free를 뜻하지는 않는다.

근거: [F2 dataset contamination note](https://huggingface.co/datasets/codefuse-ai/F2LLM-v2/blob/main/README.md).

## Comsat-ko-8B 감사

### 확인된 사실

Comsat은 `Qwen/Qwen3-Embedding-8B`의 직접 fine-tune이다. 이것은 모델 카드의
lineage metadata에 명시되어 있고, 공개 config도 Qwen3-Embedding-8B와 같은
36-layer, hidden-size 4096, 32 attention heads, 8 KV heads 구조를 유지한다.

| 항목 | 공개 값 |
|---|---|
| Base | `Qwen/Qwen3-Embedding-8B` |
| Relation | `finetune` |
| Training 규모 표현 | `over 1M Korean examples` |
| Embedding | 4096-d, last-token pooling, L2 normalization |
| Similarity | cosine; normalize 후 matrix inner product |
| Query prompt | Qwen3-Embedding-8B와 같은 web-search instruction |
| Document prefix | 없음 |
| 카드 권장 max length | 8,192 |
| Weight artifact | 단일 full-size BF16 safetensors 약 15.13GB; adapter file 없음 |
| License | CC BY-NC 4.0 |
| HF revision | `a5cc22b651c1b2e51cdd8bf671774ae93584f0ab` |

`1M+ Korean examples`는 모델 카드가 사용한 정확한 단어다. 다음으로 바꿔 쓰면
안 된다.

- 1M unique documents
- 1M tokens
- 1M triplets
- 1M query-positive pairs
- 1M synthetic QA

어떤 schema의 example인지 공개하지 않았기 때문이다.

### CPT인가 SFT인가

방어 가능한 분류는 **Qwen3-Embedding-8B의 opaque continued embedding
fine-tuning**이다. HF metadata는 `finetune` 관계를 확인하지만 objective는
공개하지 않는다.

- raw-text next-token CPT를 했다는 근거가 없다.
- contrastive InfoNCE, triplet loss, CachedMNRL, distillation 중 무엇인지 모른다.
- full FT인지, LoRA/DoRA를 학습한 뒤 merge했는지 모른다.
- 공개 repo에 adapter file이 없고 merged full weight만 있다는 사실은 학습 중
  full FT였다는 증거가 아니다.

retrieval model이므로 contrastive pair/triplet training이 가장 자연스러운 가설이지만,
이것은 **추론**일 뿐 모델 카드의 검증된 레시피로 인용하면 안 된다.

### 공개되지 않은 핵심 정보

- dataset 이름, source별 row 수, synthetic/translated/human 비율
- pair/triplet/multi-positive 구조와 token 수
- loss, temperature, batch size, learning rate, epochs
- hard-negative miner, pool 크기, false-negative 처리
- full FT/LoRA/DoRA/partial FT 여부
- benchmark train exposure와 decontamination
- checkpoint selection 기준
- GPU·시간·FLOPs
- MTEB package revision, raw result JSON, confidence interval
- training code와 paper

2026-07-11 현재 모델 카드에 연결된 paper, training GitHub, dataset repo가 없고,
공개 검색에서도 저자 primary paper를 확인하지 못했다. 모델 repo의 history도
하나의 `Squash history` commit으로 정리되어 있어 과거 training artifact를
복구할 수 없다. 즉 성능 weight는 공개됐지만 recipe는 재현 불가능하다.

## 두 모델의 Sionic 9 직접 비교

아래 값은 Sionic 모델 카드의 full-corpus NDCG@10 self-report다. 공식
`MTEB(kor, v1)` 값이 아니다.

| Task | Comsat | F2LLM-v2-8B | F2 − Comsat |
|---|---:|---:|---:|
| MIRACL | .6964 | .6311 | -.0653 |
| MrTidy | .6253 | .6162 | -.0091 |
| MLDR | .5183 | .3950 | -.1233 |
| AutoRAG | .8518 | .7678 | -.0840 |
| Ko-StrategyQA | .8394 | .8371 | -.0023 |
| PublicHealthQA | .8871 | **.9332** | **+.0461** |
| Belebele | .9853 | .9509 | -.0344 |
| SQuADKorV1 | .9168 | .8874 | -.0294 |
| LawIRKo | .8164 | **.8405** | **+.0241** |
| **Macro avg** | **.7930** | **.7621** | **-.0309** |

같은 표에서 Qwen3-Embedding-8B는 `.7825`다.

```text
Comsat − Qwen3-Embedding-8B = +.0105 NDCG = +1.05 point
Comsat − F2LLM-v2-8B       = +.0309 NDCG = +3.09 point
```

F2는 broad Korean embedding에서 강하지만 이 특정 retrieval 9종에서는 Qwen
embedding base보다도 `.0204` 낮다. 그러므로 **F2 최종 weight로 교체하면 Sionic
목표까지의 거리가 오히려 약 세 배가 된다.** F2는 PublicHealthQA와 LawIRKo에서는
Comsat보다 좋으므로 해당 도메인의 teacher·data ablation에는 가치가 있다.

Sionic 카드의 재현 주장 범위도 제한해야 한다.

- 9개 task를 같은 가중치로 macro-average한다.
- 모든 값은 한 회사의 모델 카드 self-report다.
- package commit, dataset revisions, raw result JSON이 없다.
- 현재 MTEB의 Belebele subset 이름과 카드의 `kor-kor` 표기가 다르므로 version
  drift 가능성이 있다.
- Comsat training exposure가 미공개라 zero-shot/contamination-free라고 할 수 없다.
- 반대로 공개 정보만으로 leakage를 단정할 수도 없다.

근거: [Comsat model card의 9-task 표](https://huggingface.co/sionic-ai/comsat-embed-ko-8b-preview#performance-mteb-korean-retrieval-ndcg10).

## 공식 MTEB Korean에서 F2가 의미하는 것

2026-07-11 live backend에서 F2LLM-v2-8B는 Borda rank 1이다.

| Official Korean item | Score |
|---|---:|
| Borda rank | **1** |
| Mean(Task), 6 tasks | **.7510995** |
| Mean(TaskType), 4 types | .726829 |
| Retrieval type | .734190 |
| STS type | .865091 |
| Reranking type | .632400 |
| Classification type | .675635 |
| Reported zero-shot rate | 66% |

| Task | Main score |
|---|---:|
| KorSTS | .836075 |
| KLUE-STS | .894107 |
| MIRACLRetrieval | .631280 |
| KLUE-TC | .675635 |
| Ko-StrategyQA | .837100 |
| MIRACLReranking | .632400 |

공식 backend는 F2가 MIRACLRetrieval과 MIRACLReranking을 학습한 것으로 표시한다.
그래서 zero-shot rate가 100%가 아니라 `66%`다. Borda 1위는 task별 rank를 합산한
순위이며 단순 Mean(Task) 1위라는 뜻은 아니다.

출처: [MTEB Korean live backend](https://mteb-leaderboard-backend.hf.space/v1/benchmarks/MTEB%28kor%2C%20v1%29/scores).

논문 snapshot은 2026-03-19에 Korean `75.11`, 당시 rank `(2)`로 기록했다. live
rank가 현재 1인 것과 모순이 아니라 leaderboard 행이 바뀐 결과다. 날짜 없는
`한국어 1등` 문구보다 score, 집계법, snapshot 날짜를 함께 써야 한다.

Comsat은 이 공식 Korean live board에 제출된 row가 없다. Sionic 9종의 `.7930`을
공식 Korean Mean(Task)나 Borda rank로 옮겨 쓰면 안 된다. 동일 protocol 로컬
측정은 `official-protocol local reproduction`으로 별도 표기해야 한다.

## F2의 broad 결과와 종합 보드 해석

F2 논문은 17개 MTEB benchmark, 총 430 task를 평가했다. 8B의 논문 snapshot은
다음과 같다.

| Benchmark | 8B mean | Benchmark | 8B mean |
|---|---:|---|---:|
| Multilingual (131) | 68.09 | English (41) | 72.86 |
| Code (12) | 80.16 | Medical (12) | 64.91 |
| European (73) | 69.22 | Scandinavian (28) | 69.94 |
| Indic (20) | 77.93 | German (19) | 66.81 |
| French (25) | 72.66 | Korean (6) | 75.11 |
| Polish (17) | 74.61 | Chinese (32) | 67.73 |
| Japanese (28) | 78.54 | Dutch (40) | 65.81 |
| Russian (23) | 70.57 | Persian (52) | 72.69 |
| Vietnamese (50) | 63.32 | **17-benchmark Avg** | **71.23** |

이 `71.23`은 논문이 17개 benchmark mean을 다시 평균한 값이지, 현재 official
MTEB multilingual leaderboard의 단일 score와 같은 통계가 아니다. 그래도
retrieval-only 9종에 치우치지 않고 430 task에서 범용성을 확인했다는 증거는
Comsat 카드보다 훨씬 강하다.

## 우리 학습 우선순위에 주는 결론

### 1순위: Sionic 9종

첫 backbone은 `Qwen/Qwen3-Embedding-8B`를 유지한다.

이유:

1. 이미 Sionic 9에서 `.7825`라 목표까지 `.0105`뿐이다.
2. Comsat과 prompt, 4096-d, causal last-token, normalization contract가 같다.
3. F2 8B는 `.7621`로 목표까지 `.0309`이며 final weight 교체가 불리하다.
4. F2가 27M Stage 1으로 새 embedding foundation을 만든 이유는 출발점이 일반
   Qwen3-8B였기 때문이다. 이미 약 150M weak/synthetic pair 계열 embedding
   training을 받은 Qwen3-Embedding-8B에 같은 Stage 1을 통째로 반복할 필요는 낮다.

F2에서 먼저 복제할 요소:

- Korean retrieval `(instruction, query, positive, hard negatives)` 중심 mixture
- Qwen3-Embedding-8B/current-student miner로 24-candidate pool 생성
- step마다 4·7 hard negative sampling ablation
- cross-batch/in-batch CE + explicit-HN CE
- temperature `.02`와 F2 `.05` 비교
- source-homogeneous batches와 source cap
- 8,192-token 평가는 유지하되 학습은 512/1024/2048 길이 curriculum 비교

사용자가 허용한 non-commercial performance track에서는 MIRACL/MrTidy/MLDR 등
target-family train split을 넣을 수 있다. 다만 task별 train exposure를 model card에
명시하고 `zero-shot`이라고 부르지 않는다. test query/qrel/corpus를 checkpoint
selection에 반복 사용하는 것은 점수의 설명력을 없애므로 별도 final-only gate로
유지한다.

### 2순위: 공식 Korean

Sionic retrieval-only tuning만 하면 KLUE-TC, KLUE-STS, KorSTS, reranking이
퇴행할 수 있다. 공식 Korean 단계에서 F2의 Stage 2를 축소 복제한다.

- retrieval + reranking + STS + classification replay
- symmetric task의 양쪽 instruction augmentation
- MRL 256/512/1024/2048/4096-d
- source-homogeneous batching
- retrieval checkpoint와 broad checkpoint를 clean dev에서 비교

공식 보드는 Borda이므로 Mean(Task) 하나만 최적화하지 말고 여섯 task의 rank를
동시에 본다. MIRACL train exposure가 있으면 F2처럼 zero-shot 비율을 함께 공개한다.

### 3순위: 종합·다국어

F2의 장점은 1.083M 한국어만이 아니라 60.1M composite에서 만든 broad replay다.
Sionic 9 추월 뒤에는 한국어 외 replay와 다양한 task type을 넣어 Qwen base 대비
회귀를 막는다.

- multilingual retrieval replay
- Korean legal/medical/public/finance/long-document holdout
- STS/classification/clustering/bitext replay
- short/long query와 long-document stress test
- 공개 leaderboard와 겹치지 않는 temporal/domain holdout

F2 paper의 430-task 범용성은 참고하되, 우리 종합 점수는 Sionic 9와 official
Korean을 다시 평균낸 홍보 score가 아니라 독립 holdout으로 유지한다.

## LoRA와 full fine-tuning 판단

F2가 8B full FT로 성공했다는 사실만으로 우리도 바로 full FT를 해야 하는 것은
아니다.

- F2: 일반 Qwen3-8B에서 embedding space를 처음 형성하고 약 44.7M unique
  stage-mixture를 두 epoch 학습했다.
- 우리: 이미 embedding-trained Qwen3-Embedding-8B에서 한국어 retrieval을
  적응시키는 문제다.
- Comsat: full/LoRA 여부가 미공개이므로 비교 근거가 아니다.

따라서 실제 비교 순서는 다음이 계산 효율적이다.

1. LoRA r64 + 강한 hard negatives
2. DoRA 또는 last 4 layers + norms
3. GaLore/full FT
4. 같은 data order·steps·effective batch로 Sionic-private dev와 official-Korean
   regression을 비교

LoRA가 지면 full FT로 간다. 반대로 easy-negative 10K에서 full FT를 먼저 돌리는
것은 F2의 성공 조건인 대규모 data diversity와 hard-negative signal을 재현하지
못하므로 유효한 비교가 아니다.

## 사실·주장·미공개 최종 체크리스트

| 문장 | 판정 |
|---|---|
| “F2LLM-v2-8B는 Qwen3-Embedding-8B fine-tune이다.” | **거짓**. `Qwen3-8B → F2 Preview → F2`다. |
| “F2 Stage 1은 CPT다.” | **거짓**. instruction-free contrastive retrieval training이다. |
| “F2는 60.1M을 두 stage에서 전부 학습했다.” | **근거 없음**. stage mixtures는 약 27M과 18M이다. |
| “F2 8B는 KD로 성능을 올렸다.” | **거짓**. 8B는 KD 미사용이다. |
| “F2 8B는 LoRA다.” | **거짓**. 공개 코드는 full parameter update다. |
| “F2 hard-negative pool은 24, step sample은 7이다.” | **코드 확인**. retrieval 기준이다. |
| “F2 v2 miner threshold까지 공개됐다.” | **거짓**. miner는 Qwen3-Embedding-8B지만 필터 code/threshold는 없다. |
| “Comsat은 Qwen3-Embedding-8B direct fine-tune이다.” | **확인**. |
| “Comsat은 1M 문서/1M token으로 학습했다.” | **거짓**. 카드는 1M+ `examples`라고만 한다. |
| “Comsat은 contrastive LoRA다.” | **미공개**. plausible inference일 뿐이다. |
| “Comsat은 공식 MTEB Korean 1위다.” | **거짓/미검증**. 공식 submitted row가 없다. |
| “F2는 공식 Korean live Borda 1위다.” | **2026-07-11 snapshot에서 확인**. |
| “F2는 Sionic 9에서 Comsat보다 낫다.” | **거짓**. `.7621 < .7930`. |

## 고정한 primary sources와 revisions

- [F2LLM-v2 paper, arXiv:2603.19223v1](https://arxiv.org/abs/2603.19223), 2026-03-19
- [F2 official code](https://github.com/codefuse-ai/CodeFuse-Embeddings/tree/main/F2LLM), audited commit `1c5291549b9cee9eeab1cd9de6a67be4d0295da0`
- [F2LLM-v2-8B model](https://huggingface.co/codefuse-ai/F2LLM-v2-8B), revision `e5725783762d69b4f8ba7e09a8872ce19a7a5ec3`
- [F2LLM-v2-8B Preview](https://huggingface.co/codefuse-ai/F2LLM-v2-8B-Preview), revision `ccb755c7592682817c30b8118e1aa07c196a85f2`
- [F2LLM-v2 dataset](https://huggingface.co/datasets/codefuse-ai/F2LLM-v2), audited revision `d520b8ad02c86d5e5611441c6196ff65d8888927`
- [Comsat model](https://huggingface.co/sionic-ai/comsat-embed-ko-8b-preview), revision `a5cc22b651c1b2e51cdd8bf671774ae93584f0ab`
- [Official MTEB Korean live scores](https://mteb-leaderboard-backend.hf.space/v1/benchmarks/MTEB%28kor%2C%20v1%29/scores), snapshot 2026-07-11
- [Qwen3 Embedding paper](https://arxiv.org/abs/2506.05176) and [official repository](https://github.com/QwenLM/Qwen3-Embedding)
