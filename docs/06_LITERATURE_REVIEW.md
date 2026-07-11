# Literature review

이 문서는 Qwen3-Embedding의 선행·인용·후속 연구를 방법론별로 정리합니다. 2026-07-11 기준 primary paper와 공식 code/model card만 최종 근거로 사용합니다.

## 중심 논문

### Qwen3 Embedding (2025)

- Qwen3 foundation model에서 last-token embedding
- 개선형 InfoNCE
- 약 150M synthetic weak pairs
- 약 12M filtered synthetic pairs를 supervised stage에 재사용
- public labeled data와 multi-task training
- 여러 checkpoint의 SLERP merge
- query instruction, document no-prefix

핵심 해석: 논문이 weakly supervised `pre-training`이라고 부르는 첫 단계도 causal LM CPT가 아니라 pair contrastive learning입니다.

## 직접적인 재현·후속 흐름

### Llama-Embed-Nemotron-8B (2025)

- Qwen3 논문을 명시적으로 인용하고 seed-document → query → hard-negative 흐름을 재현
- 16.1M public/synthetic pairs
- positive-relative hard-negative filtering
- six-checkpoint average
- 데이터, 코드, hyperparameter를 공개해 현재 가장 유용한 재현 참고점

### KaLM-Embedding / V2 (2025)

- 고품질 데이터 selection과 multi-stage training
- hard negatives, instruction, Matryoshka dimensions
- 6.34M 공개 fine-tuning dataset

## 반드시 포함할 기반 연구

- E5: weakly supervised contrastive pre-training → supervised fine-tuning
- multilingual-E5: 대규모 multilingual pair 학습
- Improving Text Embeddings with LLMs: 소량의 synthetic task/query/positive/hard-negative
- InPars-v2: document-to-query generation과 reranker filtering
- NV-Retriever: positive-aware hard-negative mining
- GradCache: 메모리를 크게 늘리지 않고 effective contrastive batch 확장
- BGE-M3: multilingual/multi-granularity/self-knowledge distillation
- Gecko/Gemini Embedding: LLM-generated data와 teacher distillation

Qwen3 논문을 실제로 인용한 2025 하반기~2026 논문의 선별 표는 조사 완료 후 이 파일에 추가합니다.
