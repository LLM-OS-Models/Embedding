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

Qwen3 논문을 실제로 인용한 2025 하반기~2026 논문 13편의 날짜, 출판 상태, 데이터, loss, Qwen 대비 의미와 원문 링크는 [상세 후속 연구 보고서](../qwen3_embedding_followup_papers_2026.md)에 정리했습니다.

그 조사에서 이번 프로젝트에 바로 반영할 결론은 다음과 같습니다.

1. broad CPT보다 reranker의 연속 relevance 분포를 증류하는 것이 첫 예산의 우선순위입니다.
2. top hard negative만 모으지 않고 teacher-score quantile 전 구간을 표집해야 OOD 일반화가 좋아집니다.
3. false negative와 생성기 문체·출처 shortcut을 별도 검증해야 합니다.
4. generic reasoning/RLVR checkpoint 자체는 자동 이득이 아니며 retrieval reward나 latent-CoT 증류처럼 목적에 직접 연결해야 합니다.
5. domain·task별 adapter/checkpoint를 clean validation으로 고른 뒤 merge하는 방법은 재현 가치가 큽니다.
6. 2,500억 토큰 multilingual CPT 사례는 존재하지만 Sionic의 1.05 NDCG point 차이를 넘기 위한 첫 실험으로는 과합니다.
