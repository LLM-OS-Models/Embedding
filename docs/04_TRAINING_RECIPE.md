# Training recipe

## CPT인가 SFT인가

용어를 구분합니다.

- **LM CPT/DAPT:** raw text에 next-token language modeling을 계속함
- **weakly supervised contrastive pre-training:** 대량의 noisy/synthetic pair를 InfoNCE로 학습
- **supervised contrastive fine-tuning:** 고품질 query-positive-negative로 InfoNCE를 학습
- **generative reranker SFT:** query-document를 함께 보고 `yes/no` next token을 학습

Qwen3-Embedding 논문의 embedding stage 1과 2는 둘 다 InfoNCE입니다. 논문이 stage 1을 pre-training이라고 부르지만 LM CPT가 아닙니다. 우리 주력도 continued contrastive fine-tuning입니다.

LM CPT는 다음 조건에서만 별도 ablation으로 둡니다.

- backbone tokenizer/언어 이해가 한국어 전문 문체를 심하게 놓침
- 수억~수십억 token의 권리 정리된 domain corpus가 있음
- CPT 뒤 embedding geometry를 다시 맞출 contrastive budget이 있음

## Stage A: short retrieval adaptation

- 300K~500K clean examples
- query 192/256 tokens, document 512
- positive 1, explicit negative 1
- BF16 LoRA r32/r64, all-linear
- InfoNCE tau: .01/.02/.05 소규모 sweep
- effective query batch 64~128 목표
- 0.5~1 epoch
- LR: r64 기준 2e-5/5e-5 비교, warmup 3%, cosine decay

단순 gradient accumulation은 microbatch 사이를 in-batch negative로 만들지 않습니다. ms-swift single-GPU 기본 loss만으로 global negatives가 부족하면 GradCache/CachedMNRL 구현을 별도 실험합니다.

## Stage B: mined negatives and long documents

- Stage A checkpoint와 base Qwen으로 candidate mining
- BM25 top-100 + dense top-100 + same-document sibling
- reranker teacher로 false negative 제거/secondary positive 승격
- 100K~200K 고품질 examples
- positive 1, explicit hard negatives 3~4
- LR 1e-5~2e-5, 0.5~1 epoch
- 75~80% 512, 15~20% 2K, 3~5% 4K/8K buckets
- evidence 위치를 head/middle/tail에 균등 배치

## Objective

첫 구현은 standard query-to-document InfoNCE입니다. 이후 Qwen exact denominator를 ablate합니다.

- explicit hard negatives
- in-batch query-to-document negatives
- optional query-query block
- optional document-document block
- false-negative mask

현재 pinned ms-swift 코드는 `INFONCE_INCLUDE_QQ`, `INFONCE_INCLUDE_DD`, `INFONCE_MASK_FAKE_NEGATIVE`를 지원합니다. 주의할 점은 Qwen 저장소의 과거 문서는 default temperature .01이라고 쓰지만, 현재 ms-swift 코드는 default .1입니다. 실험에서는 환경변수로 값을 반드시 명시합니다.

## Model merge

서로 다른 random seed만 평균내는 것이 아니라 역할이 다른 checkpoint를 만듭니다.

- general QA 중심
- PDF/OCR/domain 중심
- long-doc 중심
- multilingual replay 비중이 큰 checkpoint

각 delta 또는 full weight를 평균/SLERP하고 내부 dev에서 mixing coefficient를 고릅니다. public test로 coefficient를 고르지 않습니다.

## 단일 H100 예상

- 8B BF16 weights 약 15GB
- standard AdamW full FT state는 activation 전에도 대략 120GB라 80GB 한 장에 부적합
- BF16 LoRA는 현실적; QLoRA는 첫 1-point 경쟁에서 양자화 변수를 추가하므로 우선 사용하지 않음
- Stage A + mining + Stage B 예상 30~90 H100-hours, 실제 token length와 kernel에 따라 약 2배 변동 가능

## 최소 ablation

1. exact baseline/revision/prompt/max length
2. 50K LoRA smoke: positive + in-batch
3. 같은 data에 explicit hard negative
4. teacher false-negative filtering
5. 100K/300K/500K scale
6. 512-only vs long buckets
7. tau와 Qwen q-q/d-d blocks
8. r32 vs r64
9. checkpoint/domain adapter merge
