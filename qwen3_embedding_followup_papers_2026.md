# Qwen3-Embedding 이후 텍스트 임베딩 학습 연구 정리

> 조사 기준일: 2026-07-12
> 기준 논문: [Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](https://arxiv.org/abs/2506.05176)

## 1. 조사 범위와 판정 기준

이 문서는 단순히 Qwen3-Embedding을 downstream 시스템의 임베더로 사용한 논문이 아니라, **임베딩·검색 모델의 학습 방법 자체를 개선하거나 Qwen3-Embedding을 교사·초기화·비교 기준으로 삼은 후속 연구**만 추렸다.

포함 조건은 다음과 같다.

1. 2025년 6월 Qwen3-Embedding 논문 이후 공개되었다.
2. 원 논문의 참고문헌에서 `arXiv:2506.05176` 또는 해당 Qwen3-Embedding 논문을 실제로 인용한다.
3. contrastive learning, distillation, reranker supervision, hard-negative 구성, retrieval RL, long-context embedding, checkpoint merging 등 학습법에 새로운 내용이 있다.
4. 논문 원문에서 데이터와 목적함수 또는 학습 단계를 확인할 수 있다.

Semantic Scholar의 citation graph와 논문 검색을 후보 발굴에 사용했지만, 포함 여부는 **각 arXiv·ACL Anthology·OpenReview 원문의 본문과 참고문헌을 다시 확인**해 결정했다. 단순 응용, 벤치마크 시스템 보고, Qwen3 모델명만 언급하고 논문을 인용하지 않은 자료는 주 목록에서 제외했다.

출판 상태는 2026-07-11 현재 기준이다. `accepted`, `under review`, `submitted`를 구분해야 하며, arXiv 성능표만으로 동료평가 완료 연구처럼 취급하면 안 된다.

## 2. 결론부터: 2025년 하반기~2026년의 핵심 변화

Qwen3-Embedding 이후의 흐름은 “대규모 한국어 문서를 한 번 더 CPT하면 된다”로 요약되지 않는다. 실제 후속 연구에서 반복되는 개선축은 다음 여섯 가지다.

1. **교사의 풍부한 신호를 증류**한다. 정답/오답 이진 라벨만 쓰지 않고, embedding cosine, reranker의 연속 점수 분포, 중간 레이어 관계, token/trajectory 관계까지 전달한다.
2. **가장 어려운 negative만 고집하지 않는다.** teacher score 전 구간을 층화 표집하고 false negative를 제거하는 편이 OOD 일반화에 유리하다는 결과가 나왔다.
3. **reasoning은 직접 retrieval 목적에 연결해야 한다.** 일반 RLVR reasoning checkpoint를 contrastive fine-tuning의 시작점으로 쓰는 것만으로는 이득이 사라질 수 있다. 반대로 DCG 기반 retrieval reward 또는 explicit CoT를 latent embedding에 증류하면 reasoning-heavy retrieval에서 개선된다.
4. **긴 문서는 학습 목표 자체를 바꾼다.** 긴 토큰을 단순 투입하는 것보다 late chunking, local chunk loss와 global document loss, 실제 1K–4K 토큰 문서 쌍을 함께 사용한다.
5. **체크포인트 병합이 실전 도구로 자리 잡았다.** 서로 다른 task/domain/long-context 체크포인트를 SLERP로 합쳐 runtime ensemble 없이 배포한다.
6. **효율은 별도 연구축이다.** MRL, INT8 QAT, BitNet, 8B 교사에서 0.1–0.6B 학생으로의 증류가 품질/비용 절충을 만든다.

따라서 공개된 Sionic 9개 점수 평균만 넘기는 목적이라면, broad CPT보다 **한국어 task-aligned pair/triple 데이터 + 강한 reranker 점수 + score-stratified hard negatives + false-negative filtering + multi-task checkpoint merge**가 먼저 시도할 조합이다.

## 3. Qwen3-Embedding을 직접 인용한 학습법 후속 연구

| 공개일·상태 | 논문 | 핵심 학습법 | 데이터·목적함수 | Qwen3 대비 의미 |
|---|---|---|---|---|
| 2025-06-26 · ICLR 2026 Poster | [KaLM-Embedding-V2](https://openreview.net/forum?id=Y7qzhvWhcz) ([arXiv](https://arxiv.org/abs/2506.20923)) | causal mask 제거, mean pooling, 3단계 학습, focal형 sample weighting, online pair/list-wise negative 혼합, MRL | 약 4.7억 weak-supervised 예제 사전학습 → 600만 supervised 예제 → Qwen3-Embedding-8B의 similarity distribution 증류. InfoNCE + KL | 0.5B 학생도 데이터 규모·negative 설계·교사 증류를 잘하면 훨씬 큰 모델과 경쟁할 수 있다는 강한 재현 선례다. |
| 2025-09-18 · OpenReview, ICLR 2026 submitted | [DistillMoE](https://openreview.net/forum?id=VIYNWGb3TL) | pointwise·contrastive·pairwise 문장 증류 전문가를 둔 경량 MoE router, tokenizer가 다른 교사/학생 사이 DynamicCKA token alignment | Qwen3-Embedding-0.6B → BERT-base. Patent, SciTail, STS-B, ConTRoL-NLI, ANLI-R2 등의 classification/pair/STS 데이터와 여러 KD loss | cross-tokenizer 압축에 유용한 아이디어지만, full retrieval/MMTEB보다 좁은 평가이고 아직 accepted 논문이 아니다. |
| 2025-10-08 · arXiv | [Search-R3](https://arxiv.org/abs/2510.07048) | LLM이 reasoning을 생성한 뒤 특수 `<embed>` token의 hidden state로 검색. 2단계 SFT + retrieval GRPO, selective index refresh | TriviaQA, MS MARCO, CodeSearchNet/CoSQA, MIRACL, S2ORC, Qwen3-32B 생성 10만 hard-negative triples. CE + base-model KL + InfoNCE + triplet margin, 이후 cosine rank의 DCG reward로 GRPO | static embedding 추출을 넘어 retrieval reward가 embedding을 직접 훈련한다. reasoning latency와 corpus 재색인 비용은 대가다. |
| 2025-10-09 · ACL 2026 Long | [ReasonEmbed](https://aclanthology.org/2026.acl-long.54/) ([arXiv](https://arxiv.org/abs/2510.08252)) | ReMixer로 query 생성·source-excluded 후보 mining·reasoning relevance annotation. reasoning intensity에 따른 RI-InfoNCE weighting | 12개 domain, 8.2만 예제. Qwen3-235B trajectory로 Qwen3-8B annotator를 증류해 약 6만 pair를 주석. Qwen3 4B/8B base에 LoRA, query당 1,023 negatives | reasoning-heavy retrieval에서는 범용 Qwen3보다 데이터/negative의 질과 difficulty weighting이 큰 이득을 준다. 다만 BRIGHT류 특화 성능을 일반 MMTEB SOTA로 확대 해석하면 안 된다. |
| 2026-01-29 · arXiv | [Do Reasoning Models Enhance Embedding Models?](https://arxiv.org/abs/2601.21192) | 같은 backbone의 base와 SFT/RLVR reasoning 버전을 동일한 contrastive recipe로 통제 비교. HRSA로 representation·geometry·function 분석 | 11개 학습 데이터, Qwen3-Embedding-0.6B로 query당 hard negatives 3개 mining, positive-aware 95% margin. full-parameter InfoNCE. Multilingual MTEB, Code MTEB, BRIGHT 평가 | 일반 reasoning/RLVR 초기화의 장점이 contrastive 학습 뒤 사라지는 `manifold realignment`를 보고한다. 임베딩을 위해 generic RL부터 하는 전략에 반대되는 증거다. |
| 2026-02-11 · arXiv | [Diffusion-Pretrained Dense and Contextual Embeddings (pplx-embed)](https://arxiv.org/abs/2602.11151) | Qwen3 base의 causal mask 제거 후 diffusion-style continued pretraining, contextual late chunking, local+global contrastive loss, INT8 QAT, SLERP | 약 2,500억 multilingual tokens: FineWeb-Edu 50%, 29개 언어 FineWeb2 계열 50%. 60개 언어 pair mix. doc/query 양쪽 in-batch negatives와 false-negative masking. contextual checkpoint와 triplet checkpoint 병합 | Qwen3 위에서 **실제 CPT**가 의미 있음을 보여 주는 가장 직접적인 2026 예다. 단, 2,500억 토큰 규모라 비용이 매우 크며 Sionic을 좁은 벤치에서 넘기기 위한 첫 실험으로는 과하다. |
| 2026-02-17 · arXiv, v2 2026-04-28 | [jina-embeddings-v5-text](https://arxiv.org/abs/2602.15547) | 1단계 direct embedding distillation, 2단계 frozen backbone + task-specific LoRA adapters. 별도 long-context phase와 RoPE 조정 | Qwen3-Embedding-4B → EuroBERT-210M/Qwen3-0.6B. 300+ datasets, 30+ languages, 5만 step. 1K–4,096 token의 synthetic needle·책·긴 기사와 LLM query. cosine KD; retrieval adapter는 InfoNCE + KD + orthogonal regularizer | 교사 임베딩을 직접 맞춘 뒤 retrieval/similarity/classification별 adapter를 붙이는 실용적 압축법이다. 긴 문서도 별도 데이터 단계가 필요함을 보여 준다. |
| 2026-03-02 · arXiv under review | [LaSER](https://arxiv.org/abs/2603.01425) | explicit CoT view와 고정 길이 latent-thinking-token view를 공유 backbone으로 학습. output KL과 process/trajectory alignment로 reasoning을 latent embedding에 내재화 | Qwen3 base 0.6B/4B/8B. ReasonEmb 81,659 예제, 12 domain, GPT-4o-mini CoT. 두 view의 InfoNCE + output KL + trajectory alignment | Qwen3-Embedding 저자들이 참여한 개념적 직접 후속이다. autoregressive CoT 없이 reasoning retrieval 이득을 노리지만 범용 embedding보다 reasoning retrieval에 초점이 있다. |
| 2026-04-06 · SIGIR 2026 | [Beyond Hard Negatives](https://arxiv.org/abs/2604.04734) | teacher 점수의 quantile 전 구간에서 deterministic stratified sampling. top-K hard negatives 대신 score coverage·entropy·variance 보존 | MS MARCO 532K queries/8.8M docs. Qwen3-Embedding-8B top-100 + random 100 후보를 Qwen3-Reranker-8B가 점수화. 학생은 InfoNCE 뒤 KL 또는 MarginMSE KD | 가장 어려운 negative만 쓰는 것이 최선이 아니며, teacher의 전체 점수 분포가 OOD BEIR 일반화에 중요하다는 직접적인 reranker-distillation 지침이다. |
| 2026-05-31 · KDD 2026 accepted | [When Hard Negatives Hurt / CausalNeg](https://arxiv.org/abs/2606.01304) | positive relevance 조건을 CoT로 분해한 뒤 단 하나를 counterfactual하게 위반하는 synthetic negative. source artifact를 억제하는 entropy·mass-balance regularization | BGE-M3 Data 기반, mMARCO-zh·HotpotQA·NQ·TriviaQA에서 dataset당 약 1만 query. BM25 15개 + LLM 생성 3개 negative. Qwen3-0.6B full FT, InfoNCE + query-view entropy + source-mass balance | LLM이 만든 그럴듯한 negative가 source/style shortcut 때문에 오히려 성능을 해칠 수 있음을 보인다. 한국어 synthetic negative도 corpus 문체 일치와 출처 shortcut 검사가 필요하다. |
| 2026-06-17 · arXiv | [Querit-Reranker](https://arxiv.org/abs/2606.19037) | label-free distribution adaptation, teacher score-gap 비례 pairwise hinge, 여러 domain checkpoint의 sequential SLERP | Qwen3-Embedding-4B 초기화. 공개 352만 + 비식별 private 205만 ranking data, 이후 French/Russian/Chinese/Japanese 약 940만 synthetic 예제. teacher 점수 16 bins 표집 | Qwen 계열을 reranker로 특화하고 task/domain 체크포인트를 runtime ensemble 없이 합치는 후속 recipe다. private data와 targeted adaptation 때문에 완전 재현 및 leaderboard 과적합에는 주의해야 한다. |
| 2026-06-24 · arXiv under review | [BitNet Text Embeddings (BITEMBED)](https://arxiv.org/abs/2606.25674) | Qwen3/Gemma를 ternary-weight·quantized-activation encoder로 변환 후 continual contrastive pretraining. similarity·attention relation KD와 multi-bit output 학습 | Qwen3-0.6B/Gemma3-270M. 공개 BGE-en-ICL mixture; query/instruction/positive + hard negatives 7개. supervised InfoNCE + teacher similarity-distribution KL + one-layer attention-relation KL | 절대 최고 성능보다는 저장·추론 효율을 위한 후속이다. 품질 모델을 먼저 만든 뒤 저비트 배포 모델로 증류할 때 유용하다. |
| 2026-07 · ACL 2026 Long | [TALAS](https://aclanthology.org/2026.acl-long.1509/) | 교사의 최종 sentence embedding은 학생 상위 2–4개 layer에만 정렬. 하위층은 layer-aligned self-distillation, ASAM과 SimCSE 보조 | Qwen3-Embedding-0.6B/4B → MiniLMv2/BERT. Banking77, TweetEval, Emotion, MRPC, SciTail, WiC, SICK, STS12, STS-B | 용량 차가 큰 교사/학생의 모든 층을 강제로 맞추면 오히려 손해라는 결과다. 압축에는 유용하지만 평가가 retrieval보다 classification/pair/STS 중심이다. |

### 출판 상태에 대한 주의

- ICLR/ACL/KDD/SIGIR accepted 논문과 arXiv/under-review 논문의 근거 강도는 동일하지 않다.
- DistillMoE는 OpenReview에 `Submitted to ICLR 2026`으로 표시되며, 여기서는 채택 논문으로 간주하지 않았다.
- LaSER는 arXiv에 `Under Review`로만 표기되어 있다. 최종 proceedings metadata가 확인되기 전에는 채택 논문으로 부르면 안 된다.
- Querit-Reranker는 private data를 일부 포함하므로 결과 전체를 공개 데이터만으로 재현하기 어렵다.

## 4. 학습법별로 읽으면 무엇이 보이는가

### 4.1 Contrastive distillation

가장 흔한 기본형은 query (q), positive (d^+), negatives (d_i^-)에 대한 InfoNCE다.

\[
L_{\text{NCE}}=-\log\frac{\exp(s(q,d^+)/\tau)}{\exp(s(q,d^+)/\tau)+\sum_i \exp(s(q,d_i^-)/\tau)}
\]

2026년 후속 연구의 핵심은 여기에 단순 hard label이 아닌 교사의 **상대적 유사도 분포**를 추가하는 것이다.

\[
L=L_{\text{NCE}}+\lambda\,D_{\mathrm{KL}}(p_T(d\mid q)\Vert p_S(d\mid q))
\]

KaLM-V2와 BITEMBED는 이 방향을 사용한다. jina-v5는 학생 임베딩을 선형 투영해 Qwen3 교사 벡터와 cosine 거리를 직접 줄이고, TALAS는 교사의 최종 문장 표현을 학생의 적절한 상위층에만 전달한다. 공통 메시지는 “teacher의 top-1 정답만 복제하지 말고, geometry와 score distribution을 보존하라”는 것이다.

### 4.2 Reranker distillation

Qwen3-Reranker 같은 cross-encoder는 bi-encoder보다 느리지만 query-document 상호작용을 세밀하게 점수화한다. 실전 pipeline은 다음과 같다.

1. BM25와 현재 embedding model로 수백 개 후보를 모은다.
2. 강한 reranker가 후보 전체에 연속 relevance score를 부여한다.
3. student embedding model이 정답뿐 아니라 이 점수 순서와 간격을 배우게 한다.

`Beyond Hard Negatives`는 top-score negative만 고르는 관행보다 teacher score의 낮음·중간·높음 구간을 모두 보존하는 층화 표집이 더 낫다고 보고한다. Querit-Reranker도 score bin을 사용하고 score gap을 margin에 반영한다. Sionic 추월 실험에서 가장 즉시 적용하기 쉬운 결과다.

### 4.3 Reasoning과 RL

여기에는 상반된 두 결과가 있다.

- `Do Reasoning Models Enhance Embedding Models?`에서는 generic SFT/RLVR reasoning checkpoint의 이점이 이후 contrastive 학습에서 사라졌다.
- Search-R3는 rank/DCG를 직접 reward로 쓰는 GRPO를 적용했고, ReasonEmbed와 LaSER는 reasoning supervision을 sample weighting 또는 latent representation distillation에 직접 연결했다.

따라서 “reasoning model로 시작하면 embedding도 자동으로 좋아진다”는 주장은 근거가 약하다. **retrieval 성과를 직접 보상하거나 reasoning trace를 embedding geometry에 명시적으로 증류**해야 한다.

### 4.4 Long-context embedding

pplx-embed와 jina-v5에서 공통적으로 보이는 조건은 다음과 같다.

- causal mask를 제거하거나 bidirectional attention을 사용한다.
- 학습 중 실제 긴 문서와 그 문서를 구별해야 하는 query를 제공한다.
- chunk-level local contrastive objective와 document-level global objective를 같이 둔다.
- RoPE extrapolation만으로 끝내지 않고 1K–4K 이상의 길이를 실제로 학습한다.

MLDR 같은 long-document benchmark가 중요하면 이 축은 가치가 있다. 반대로 짧은 QA·STS·classification 비중이 큰 벤치마크를 먼저 넘기는 목적이라면 long-context CPT부터 시작할 이유는 약하다.

### 4.5 Model merging과 ensemble

Qwen3 계열 자체와 후속 연구는 task별 최고 체크포인트를 병합하는 전략을 적극 사용한다. SLERP는 두 weight vector 사이를 구면 보간해 단순 평균보다 norm과 방향을 보존하려는 방식이다. pplx-embed는 contextual/triplet checkpoint를, Querit-Reranker는 여러 domain checkpoint를 순차 SLERP한다.

이는 여러 모델의 embedding을 runtime에 합치는 ensemble과 다르다. 배포 모델은 하나지만, 병합 계수 자체가 validation benchmark에 과적합될 수 있으므로 hidden test와 domain holdout이 필요하다.

## 5. Sionic 점수를 넘기기 위한 우선순위

공개 표에서 `comsat-embed-ko-8b-preview`는 Qwen3-Embedding-8B보다 평균 `0.0105` 높고, 9개 중 8개 task에서 앞서며 LawIRKo만 `0.0007` 낮다. 이 정도 차이는 foundation model을 새로 만드는 문제라기보다 **공개 평가 분포에 맞는 supervised contrastive adaptation**으로 뒤집힐 가능성이 충분하다. 다만 그 결과를 곧바로 범용 한국어 또는 multilingual SOTA라고 부를 수는 없다.

권장 실험 순서는 다음과 같다.

### A. 먼저 해야 할 것

1. **벤치마크 재현과 leakage audit**
   - 동일 pooling, instruction prefix, max length, normalization, similarity function, chunking으로 Qwen3 원점수를 재현한다.
   - train corpus와 9개 test query/document의 exact·near-duplicate를 검사한다.
   - public dev를 tuning에 쓰더라도 별도의 domain holdout을 둔다.

2. **한국어 task-aligned supervised triples 구축**
   - 일반 웹 문서만 모으지 말고 QA, retrieval, long document, law, health, entailment/reading-comprehension에 맞춘 `(instruction, query, positive)`를 만든다.
   - 법률·의료·공공 데이터는 출처와 라이선스를 기록하고, benchmark 원문을 복제하지 않는다.

3. **hybrid candidate mining + reranker scoring**
   - BM25, 원본 Qwen3-Embedding, 이전 epoch 모델의 후보를 합친다.
   - Qwen3-Reranker-8B 또는 더 강한 multilingual cross-encoder로 100–200개 후보를 연속 점수화한다.
   - top hard negatives만 쓰지 말고 high/mid/low score bin에서 층화 표집한다.
   - false-negative 후보는 LLM 판정 하나가 아니라 lexical evidence, multiple teachers, positive-aware margin으로 걸러낸다.

4. **목적함수**
   - 기본은 cross-device in-batch InfoNCE.
   - reranker score distribution KL 또는 MarginMSE를 보조 loss로 둔다.
   - 긴 문서가 실제 병목이면 chunk-local + document-global objective를 추가한다.
   - MRL을 함께 쓰면 8B 모델을 256/512/1024차원 등 여러 embedding dimension으로 배포할 수 있다.

5. **task checkpoint와 병합**
   - retrieval, reasoning QA, long-document, semantic similarity/reading task를 하나의 sampling ratio로 억지로 맞추기보다 checkpoint 또는 adapter를 분리해 학습한다.
   - held-out 평균과 worst-task 성능을 같이 보며 soup/SLERP 계수를 정한다.

### B. 첫 예산으로는 미룰 것

- 2,500억 토큰급 broad CPT
- generic RLVR reasoning model 사전학습
- 목적 benchmark가 요구하지 않는 32K long-context 확장
- LLM 생성 hard negative를 검증 없이 대량 투입
- 공개 9개 test set 자체를 반복 tuning하는 leaderboard hill-climbing

### C. 현실적인 난이도 판정

- **공개 9개 평균을 0.7930보다 높이는 것:** 비교적 현실적이다. Qwen3-8B라는 강한 시작점과 target-aligned 한국어 데이터, reranker distillation이 있다면 작은 차이를 넘길 여지가 크다.
- **9개 모두에서 안정적으로 이기면서 hidden/OOD에서도 유지:** 중간 이상 난이도다. negative 구성, task sampling, leakage 방지가 더 중요하다.
- **multilingual MTEB 전체에서 새로운 8B SOTA:** 어렵다. 수십 언어·수백 데이터셋·대규모 synthetic/weak-supervised corpus와 상당한 평가/학습 비용이 필요하다.

## 6. 바로 구현 가능한 최소 실험 설계

| 단계 | 설계 |
|---|---|
| Base | `Qwen/Qwen3-Embedding-8B`, 원본 instruction format과 EOS pooling을 먼저 고정 |
| Data v1 | 공개·라이선스가 명확한 한국어 query-positive 약 20만–100만 pair; task별 source 분리 |
| Candidate pool | BM25 top-100 ∪ Qwen3 top-100 ∪ random/domain-near 20 |
| Teacher | Qwen3-Reranker-8B 또는 강한 multilingual reranker의 continuous score |
| Negatives | query당 7–31개, score quantile별 층화 + false-negative filter; generated negative는 일부만 사용 |
| Training | InfoNCE + score-distribution KL, cross-device negatives, 1–2 epoch부터 시작 |
| Ablation | `random vs top-hard vs stratified`, `InfoNCE only vs +KD`, `real vs generated negative`, `single vs SLERP` |
| Validation | 각 benchmark와 source가 겹치지 않는 Korean domain holdout, OOD retrieval, 긴 문서 slice |
| 성공 조건 | 공개 평균뿐 아니라 worst-task, OOD, duplicate-clean 성능과 latency/embedding dimension까지 함께 보고 |

이 최소 실험이 이기지 못한다면 그때 Korean continued pretraining, 더 큰 synthetic pair 생성, reasoning-specific objective를 순차적으로 추가하는 편이 원인 분석이 쉽다.

## 7. 주 목록에서 제외한 사례

- Qwen3-Embedding을 단순 feature extractor 또는 retrieval component로 사용한 downstream 응용 논문
- 모델 카드나 블로그에서 Qwen3를 비교했지만 학습법 논문이 아닌 자료
- Qwen3 모델명을 사용했으나 원 Qwen3-Embedding 논문을 참고문헌에서 명시적으로 인용하지 않은 자료
- SemEval 등의 단일 shared-task system paper는 synthetic hard-negative 아이디어가 있어도 일반 학습법 근거가 약하면 보조 사례로만 취급

예를 들어 [SemEval 2026 Team HITS](https://aclanthology.org/2026.semeval-1.338/)는 Qwen2.5-32B로 narrative hard negatives를 만들고 Qwen3-Embedding-8B를 multi-negative contrastive/self-distillation하는 사례지만, 단일 shared-task 최적화의 성격이 강해 핵심 13편에는 넣지 않았다.

### 7.1 7월 12일 최신 원문 재검색에서 확인한 인접 연구

아래 연구는 Qwen3 text embedder의 직접 후속 학습 recipe라는 주 목록 조건에는 맞지
않지만, 현재 실험 설계에는 분명한 함의가 있어 별도로 추적한다.

| 연구 | 확인된 내용 | 이번 프로젝트의 반영 |
|---|---|---|
| [Robustness Risk of Conversational Retrieval](https://arxiv.org/abs/2604.06176) | Qwen3-Embedding은 query prompt가 없을 때 구조화된 대화 filler·system artifact가 상위 검색 결과에 침투하는 현상이 나타났고, 가벼운 query prompt로 완화됐다. clean query benchmark만으로 잘 드러나지 않는 실패다. | 고정 Sionic prompt를 유지하고, clean 보드에 대화 filler/system header 0/1/5% 삽입과 prompt on/off paired slice를 추가한다. |
| [KV-Embedding](https://aclanthology.org/2026.acl-long.540/) | frozen decoder LLM의 마지막 token KV state를 prefix로 재배치하는 training-free 방식이며, Qwen/Mistral/Llama의 MTEB에서 기존 training-free baseline 대비 최대 10% 개선과 4,096-token 결과를 보고했다. | 현재 Qwen3-Embedding 후속학습 checkpoint와 저장 형식이 달라 주력 경로에는 넣지 않는다. `060`의 frozen-backbone architecture ablation 후보로만 둔다. |
| [LEAF](https://aclanthology.org/2026.acl-long.2008/) | 학생 벡터를 teacher 공간에 직접 정렬해 query는 작은 학생, corpus는 큰 teacher로 인코딩하는 비대칭 검색을 가능하게 한다. hard negative·judgment 없이 작은 batch로 학습 가능하며 teacher의 MRL/quantization robustness도 상속한다고 보고한다. | 최고 8B 모델 확정 후 query-side 저지연 학생을 만드는 `120_compression` 후보로 둔다. 교사와 학생의 벡터 호환성은 현재 Sionic 추월용 8B 품질 학습과 분리한다. |
| [Qwen3-VL-Embedding](https://arxiv.org/abs/2601.04720) | 텍스트·이미지·문서 이미지·비디오용 별도 multimodal 계열이다. large-scale contrastive pre-training → reranker distillation → MRL이라는 단계가 공개됐다. | text-only Korean 보드의 직접 비교 모델은 아니지만, reranker 연속 신호와 MRL을 뒤 단계에 두는 방향을 재확인한다. OCR 문서 이미지는 future multimodal track으로 분리한다. |

이 재검색으로 현재 우선순위는 바뀌지 않는다. 다만 **prompt/noise robustness**는 학습
데이터를 더 넣기 전에도 실패를 검출할 수 있으므로 clean 종합 보드의 필수 paired test로
승격한다.

## 8. 요약

Qwen3-Embedding 이후의 실질적 발전은 모델 크기보다 **데이터와 supervision의 구조**에 집중되어 있다. 특히 다음 세 가지가 가장 재현 가치가 높다.

1. reranker의 연속 점수 분포를 embedding student에 증류한다.
2. top hard negatives만 쓰지 않고 점수 전 구간을 표집하며 false negatives와 생성 출처 shortcut을 통제한다.
3. task별 checkpoint를 별도로 최적화한 뒤 validation으로 병합한다.

Sionic의 공개 평균을 넘기는 데는 거대한 CPT보다 이 세 가지가 비용 대비 성공 가능성이 높다. CPT는 한국어 tokenizer/표현 자체의 뚜렷한 결함이나 long-context/general-domain 부족이 확인된 뒤에 추가할 두 번째 단계다.
