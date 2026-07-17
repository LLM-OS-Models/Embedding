# 상위 모델 공식 근거 종합과 1×H100 최단 승리 레시피

기준일: 2026-07-15 (Asia/Seoul)

이 문서는 Qwen3-Embedding, Comsat, F2LLM-v2, PwC Embedding, Harrier,
KaLM, Llama-Embed-Nemotron, NV-Embed의 **저자 공식 논문·모델 카드·코드·데이터**만
대조해, 한 장에서 재현성과 실행 우선순위를 판단할 수 있게 만든다. Hugging Face
revision은 기준일에 API가 반환한 commit SHA다. 모델 카드 점수는 서로 다른
revision, prompt, task version, backend에서 생성될 수 있으므로 이 문서에서 새
leaderboard로 합산하지 않는다. 실제 local 수치의 단일 source는
[`09_EVALUATION_RESULTS.md`](09_EVALUATION_RESULTS.md)다.

## 결론

1×H100에서 가장 짧고 라이선스가 명확한 경로는
`Qwen/Qwen3-Embedding-8B@1d8ad4c...`를 고정하고, 권리가 확인된 한국어
retrieval 중심 데이터로 짧은 LoRA continued contrastive tuning을 하는 것이다.
첫 비교는 Nemotron식 `HN-only, tau=.02, HN 4`와 F2식
`in-batch + explicit-HN, tau=.05, HN 7`의 통제된 A/B다. clean private dev로
하나를 고른 뒤 200K 이내 1 epoch, hard-negative 1회 재채굴, 2~3개 checkpoint
average/SLERP 순서로 확장한다.

Comsat은 같은 Qwen3-Embedding-8B에서 한국어 1M+ examples를 학습한 직접적인
한국어 특화 기준선이지만 loss, negative, optimizer, full/LoRA 여부와 데이터가
비공개다. 따라서 **목표점**으로는 유효하지만 그대로 복제할 recipe는 아니다.
F2, Nemotron, NV-Embed에서 공개된 negative와 loss 설계를 가져오되, 비상업
가중치를 섞거나 composite dataset의 상단 license만 믿고 데이터를 가져오지 않는다.

## 1. 공식 artifact, 고정 revision, 라이선스

`last modified`는 논문 발표일이나 학습 완료일이 아니라 해당 Hub revision의
마지막 수정 시각이다. `모델 license`와 `학습 데이터 license`는 별개다.

| Artifact | 고정 revision · last modified (UTC) | 공식 근거 | 기준일 license와 사용 경계 |
|---|---|---|---|
| Qwen3-Embedding-8B | [`1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`](https://huggingface.co/Qwen/Qwen3-Embedding-8B/tree/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af) · 2025-07-07 | [논문](https://arxiv.org/abs/2506.05176), [공식 코드](https://github.com/QwenLM/Qwen3-Embedding) | 모델 `Apache-2.0`. 원 150M synthetic 및 labeled mixture는 공개되지 않았으므로 우리 데이터 권리는 별도 증명한다. |
| Comsat Korean 8B | [`a5cc22b651c1b2e51cdd8bf671774ae93584f0ab`](https://huggingface.co/sionic-ai/comsat-embed-ko-8b-preview/tree/a5cc22b651c1b2e51cdd8bf671774ae93584f0ab) · 2026-07-05 | [공식 모델 카드](https://huggingface.co/sionic-ai/comsat-embed-ko-8b-preview) | `CC-BY-NC-4.0`. 상업 목적 base, merge 또는 배포 후보로 사용하지 않는다. |
| F2LLM-v2-8B | [`e5725783762d69b4f8ba7e09a8872ce19a7a5ec3`](https://huggingface.co/codefuse-ai/F2LLM-v2-8B/tree/e5725783762d69b4f8ba7e09a8872ce19a7a5ec3) · 2026-05-27 | [논문](https://arxiv.org/abs/2603.19223), [공식 코드 `1c52915...`](https://github.com/codefuse-ai/CodeFuse-Embeddings/tree/1c5291549b9cee9eeab1cd9de6a67be4d0295da0/F2LLM) | 모델 `Apache-2.0`. 일반 Qwen3-8B에서 출발한 full-FT 모델이며 Qwen3-Embedding 파생 가중치가 아니다. |
| F2LLM-v2 data | [`d520b8ad02c86d5e5611441c6196ff65d8888927`](https://huggingface.co/datasets/codefuse-ai/F2LLM-v2/tree/d520b8ad02c86d5e5611441c6196ff65d8888927) · 2026-06-09 | [공식 dataset card](https://huggingface.co/datasets/codefuse-ai/F2LLM-v2) | 상단 metadata는 `Apache-2.0`. 157개 upstream의 원 license, 개인정보, benchmark overlap을 row별 감사하기 전에는 전체 ingest 금지. |
| PwC-Embedding_expr | [`6c5196980c685db45b58f67bd3be2f79d794351e`](https://huggingface.co/SamilPwC-AXNode-GenAI/PwC-Embedding_expr/tree/6c5196980c685db45b58f67bd3be2f79d794351e) · 2025-08-18 | [공식 모델 카드](https://huggingface.co/SamilPwC-AXNode-GenAI/PwC-Embedding_expr) | 모델 `Apache-2.0`. 학습 데이터 bundle과 전체 재현 recipe는 공개되지 않았다. |
| Harrier OSS v1 27B | [`0c0fc62f6d8af9e8604cb818c412301b103a0093`](https://huggingface.co/microsoft/harrier-oss-v1-27b/tree/0c0fc62f6d8af9e8604cb818c412301b103a0093) · 2026-03-30 | [공식 모델 카드](https://huggingface.co/microsoft/harrier-oss-v1-27b) | 모델 `MIT`. 학습 corpus와 launcher가 공개되지 않아 inference 기준선으로만 재현 가능하다. |
| KaLM Gemma3 12B | [`98c19ba34197906fbc93f6f1ef79402ca3a33956`](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511/tree/98c19ba34197906fbc93f6f1ef79402ca3a33956) · 2026-02-10 | [공식 모델 카드](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511), [license 원문](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511/blob/98c19ba34197906fbc93f6f1ef79402ca3a33956/LICENSE.txt) | metadata `other`. KaLM 조건은 EU 내 사용을 의도하지 않는다고 명시하며 Gemma Terms도 적용된다. 배포 전 별도 법무 검토가 필요하다. |
| KaLM fine-tuning data | [`e9443ab6f5d4dc29c79cea03834e932428ed6ab1`](https://huggingface.co/datasets/KaLM-Embedding/KaLM-embedding-finetuning-data/tree/e9443ab6f5d4dc29c79cea03834e932428ed6ab1) · 2025-11-27 | [공식 dataset card](https://huggingface.co/datasets/KaLM-Embedding/KaLM-embedding-finetuning-data) | 상단 metadata `MIT`. 6.34M composite의 upstream 조건은 별도로 보존·감사한다. |
| Llama-Embed-Nemotron-8B | [`aa3b43a495a9b280d1bdb716da37c54bb495d630`](https://huggingface.co/nvidia/llama-embed-nemotron-8b/tree/aa3b43a495a9b280d1bdb716da37c54bb495d630) · 2026-04-23 | [논문](https://arxiv.org/abs/2511.07025), [공식 모델 카드](https://huggingface.co/nvidia/llama-embed-nemotron-8b), [license 원문](https://huggingface.co/nvidia/llama-embed-nemotron-8b/blob/aa3b43a495a9b280d1bdb716da37c54bb495d630/LICENSE) | metadata `other`; NVIDIA license §3.3은 non-commercial research only로 제한한다. 제품 base나 상업 배포용 merge에 사용하지 않는다. |
| Nemotron training data | [`f457c3e2da4af3b9dd2818685d411b26298d7cbb`](https://huggingface.co/datasets/nvidia/embed-nemotron-dataset-v1/tree/f457c3e2da4af3b9dd2818685d411b26298d7cbb) · 2026-01-12 | [공식 dataset](https://huggingface.co/datasets/nvidia/embed-nemotron-dataset-v1) | pinned dataset에 단일 top-level license metadata가 없다. source별 조건을 확인하지 않은 재배포·학습은 보류한다. |
| NV-Embed-v2 | [`3fa59658547db50a1e8e3346cf057fd0c77ed6ef`](https://huggingface.co/nvidia/NV-Embed-v2/tree/3fa59658547db50a1e8e3346cf057fd0c77ed6ef) · 2025-07-21 | [논문](https://arxiv.org/abs/2405.17428), [공식 모델 카드](https://huggingface.co/nvidia/NV-Embed-v2) | 모델 `CC-BY-NC-4.0`. LoRA와 negative recipe의 근거로만 쓰고 배포 가중치에는 섞지 않는다. |

관련 논문의 확인 버전은 Qwen3-Embedding arXiv v3(2025-06-11), F2LLM-v2
v1(2026-03-19), KaLM-Embedding-V2 v5(2025-10-14),
Llama-Embed-Nemotron v1(2025-11-10), NV-Embed v3(2025-02-25)다. 논문의
arXiv license는 모델·데이터 license가 아니다.

## 2. 공개 학습법과 재현성 비교

`높음`은 코드·데이터·주요 hyperparameter가 공개됐다는 뜻이지, 해당 artifact가
상업적으로 사용 가능하다는 뜻이 아니다. `낮음`인 셀은 공개 근거가 없으므로
score나 config에서 역추론하지 않는다.

| 계열 | Base·pooling·길이 | 데이터·단계 | Loss·HN·MRL·학습 방식 | 한국어·다국어 해석 | 재현성 |
|---|---|---|---|---|---|
| Qwen3-Embedding-8B | Qwen3 dense, causal, EOS last-token, 4096-d, 카드상 32K, query instruction | Qwen3-32B가 약 150M weak pair 생성; cosine `> .7`인 약 12M synthetic와 labeled SFT source를 사용 | improved InfoNCE: explicit HN, in-batch query/document, same-tower 항, false-negative mask; MRL; 여러 SFT checkpoint SLERP. 원 run의 full/LoRA, optimizer, batch, temperature, stage별 길이는 미공개 | 100+ languages를 겨냥한 가장 강한 공개 출발점. 한국어 특화 전후에는 같은 prompt의 local paired 평가가 필요 | **부분**: weights·논문·continued-tuning 예제는 공개, 원 data·launcher는 미공개 |
| Comsat | Qwen3-Embedding-8B 파생, causal last-token, 4096-d, 카드상 32K | 카드가 `1M+ Korean examples`만 공개 | continued embedding fine-tune까지 확인. loss, HN, MRL schedule, optimizer, full/LoRA, prompt 생성법 모두 미공개 | 공식 카드의 한국어 retrieval 특화 목표점. 선택된 한국어 task의 우세를 종합·다국어 우세로 확대 해석하지 않음 | **낮음**: weights와 추론 contract만 재현 가능 |
| F2LLM-v2-8B | 일반 Qwen3-8B → Preview → v2, causal EOS last-token, 4096-d; train max 1024 | Stage 1 instruction-free retrieval 약 27M; Stage 2 instruction-aware multi-task 약 18M. 공개 composite 60.1M, 157 sources, 282 natural + 40 code languages | **full-parameter FT**. retrieval은 cross-GPU in-batch CE + explicit-HN CE, `tau=.05`; Qwen miner pool 24→7 표집. clustering 9/classification 1 HN, 두 task는 in-batch 제외. MRL `8…2048 + 4096`. 8B LR `6e-6`, global batch 512, 2 epochs, AdamW `(0.9,0.98)`, wd `.01`, cosine, BF16/FA2/checkpointing | 한국어와 MIRACL·MrTyDi·MLDR 계열을 포함하므로 해당 계열 결과를 순수 zero-shot으로 부르지 않음. broad multilingual replay와 source-homogeneous batch의 강한 근거 | **높음/부분**: code·data·주요 설정 공개. exact miner filter, stage row manifest, decontamination은 불완전 |
| PwC | multilingual-e5-large-instruct, bidirectional mean pool, 1024-d, max 514 | 카드가 curated/augmented STS·balanced data 범위만 설명 | exact loss, HN, optimizer, steps, merge, code 미공개 | 강한 한국어 in-domain specialist 후보. 학습 계열과 겹치지 않는 retrieval·long·multilingual gate가 필수 | **낮음**: 기술 보고서·학습 bundle·launcher 없음 |
| Harrier 27B | Gemma3 계열 27B, last-token, 5376-d, 32K | 94 languages의 대규모 multilingual multi-task contrastive mixture라는 설명만 공개 | exact loss, temperature, HN, optimizer, batch, merge 미공개 | 종합·다국어 inference 상한 비교용. 27B 비용과 비공개 recipe 때문에 1×H100 첫 학습 경로가 아님 | **낮음**: weights·inference만 공개 |
| KaLM current 12B | Gemma3-12B, last-token, 3840-d, 32K, MRL dimensions | 카드가 6.34M fine-tuning data를 연결 | current 12B의 stage, loss, KD weight, HN, optimizer, merge 미공개 | multilingual 비교 모델. 아래 V2 논문의 0.5B recipe를 current 12B의 실제 recipe로 귀속하면 안 됨 | **낮음/부분**: model·linked data 공개, exact current training 미공개 |
| KaLM V2 논문 | **별도 모델** Qwen2-0.5B, bidirectional mean pool, 896-d, max 512 | 470M weak pairs → 6M supervised → 같은 6M distillation | focal reweight `gamma=.5`, online HN, rank 50~100 HN 7, MRL `896/512/256/128/64`; Qwen3-Embedding-8B teacher, `0.3 CL + 0.7 KL` | focal/KD ablation의 근거이지 Gemma3-12B 결과의 인과 설명이 아님 | **높음/부분**: 논문 설정은 상세하나 470M 전체 provenance와 current 12B 경로는 별개 |
| Nemotron 8B | Llama-3.1-8B causal→bidirectional, global average pooling, 4096-d; train 512, inference 32K | 논문 16.1M pairs: public 7.7M + synthetic 8.4M. 현재 card의 16.4M 표기와 dataset viewer row 수가 달라 revision별 manifest 고정 필요 | full end-to-end FT. HN-only InfoNCE `tau=.02`, in-batch/same-tower 없음; pretrain HN 1, fine-tune HN 4; `s_neg < .95*s_pos`; 6개 model equal average. 보고 compute 64×A100-80GB, 25h + 21.5h | multilingual source와 한국어 retrieval source를 포함. base license 때문에 우리 상업 배포 backbone이 아니라 negative·soup 근거로 사용 | **높음**: paper·code·data·compute 공개. count/format 차이와 비상업 license는 별도 제약 |
| NV-Embed-v2 | Mistral-7B, bidirectional + latent-attention pooling, max 512 | 2-stage supervised contrastive tuning | **LoRA** `r=16`, `alpha=32`, dropout `.1`; LR `2e-5`→`1.5e-5`, batch 128, HN 7, 20K→18K steps. Stage 1 in-batch+HN, Stage 2 in-batch off; `.95×positive` filter; MRL 없음 | 사실상 English 중심 recipe라 한국어 data 선택의 근거는 아님. 1×H100 LoRA와 positive-aware filter의 직접 근거 | **중간/높음**: 논문 설정은 상세, 전체 data pipeline과 상업 사용은 제한 |

### 공개 근거에서 공통으로 남는 것

- hard negative는 단순 top-hard보다 **positive-relative false-negative 제거**와 함께
  써야 한다. `.95`는 NV/Nemotron의 시작값이지 모든 domain의 자연법칙은 아니다.
- in-batch negative는 항상 이롭지 않다. Qwen/F2는 사용하지만, Nemotron은 쓰지
  않고 NV는 두 번째 stage에서 끈다. 그러므로 동일 data·budget의 A/B가 필요하다.
- 강한 모델들은 한 번에 모든 것을 학습하기보다 weak/broad stage 뒤에
  high-quality stage를 두고, 마지막에는 checkpoint/model soup을 사용한다.
- MRL은 배포 차원 절감에는 유용하지만 full 4096-d 품질 목표를 먼저 이긴다는
  보장은 없다. 첫 승리 run에서는 loss 복잡도를 늘리지 않는다.
- 32K inference 지원은 32K로 학습했다는 뜻이 아니다. F2와 Nemotron의 공개
  train length는 각각 1024와 512다.

## 3. 1×H100 최단 승리 레시피

목표는 공개 benchmark 한 평균만 올리는 specialist가 아니라, 같은 local protocol에서
Comsat을 넘고 clean retrieval, STS·분류, long-context, robustness, multilingual,
비용 gate를 함께 통과하는 8B 모델이다.

### 단계 0: 평가와 오염 차단을 먼저 고정

1. base와 Comsat의 model revision, exact query prompt, document prompt, pooling,
   dtype, attention backend, batch, max length, task revision을 manifest에 고정한다.
2. Sionic 9와 공식 Korean 평가의 query, corpus, qrel, title·text normalized hash,
   document ID와 near-duplicate fingerprint를 생성·mining·학습에서 차단한다.
3. public test가 아닌 source-document-held-out private dev를 checkpoint 선택에 쓴다.
   target-family train을 쓰는 specialist 실험은 별도 track과 이름으로 격리한다.
4. 작은 차이는 batch/backend가 다른 결과로 판정하지 않는다. 동일 raw result를
   보존하고 strict low-batch cross-check까지 통과해야 승리로 인정한다.

### 단계 1: 가장 싼 loss A/B

backbone은 Qwen3-Embedding-8B pinned revision, representation은 EOS last-token,
L2 normalize, full dimension 4096을 유지한다. 공통 설정은 BF16, FlashAttention 2,
gradient checkpointing, LoRA `r=16`, `alpha=32`, dropout `.1`, LR `2e-5`, effective
batch 64~128, max length 512, 10K~20K clean rows의 동일 순서다.

| Pilot | Objective | Negative | 목적 |
|---|---|---|---|
| A: Nemotron형 | explicit HN-only InfoNCE, `tau=.02` | query당 HN 4 | false-negative와 heterogeneous batch에 안전한 최소 loss |
| B: F2형 | in-batch CE + explicit-HN CE, `tau=.05` | pool 24에서 HN 7 표집, false-negative mask | 더 넓은 denominator의 이득 검증 |

LoRA rank, optimizer, data, seed, step 수는 두 pilot에서 바꾸지 않는다. 둘 다
underfit이면 그때만 `r=32/64` 또는 batch를 늘린다. 승자는 public leaderboard가
아니라 private clean composite와 worst-task regression으로 고른다.

### 단계 2: 200K 이내 high-quality run

첫 mixture는 다음을 기본값으로 하되 source cap으로 한 corpus의 지배를 막는다.

| 비중 | 데이터 | 역할 |
|---:|---|---|
| 70% | 권리 확인 한국어 retrieval: 일반·법률·공공·금융·상거래·건강, 검색창형과 조건/예외/수치 query | Comsat 추월의 직접 신호 |
| 20% | 권리 확인 multilingual/general retrieval replay | Qwen의 다국어·일반 표현 보존 |
| 10% | STS, paraphrase, classification·clustering형 symmetric pair | retrieval-only 과적합 방지 |

각 row는 `source_revision`, 원 license, document ID, transformation, generator/teacher
revision, positive evidence span, benchmark-overlap 결과를 가진다. 짧은 512-token
1 epoch로 시작하고 clean dev가 계속 개선될 때만 winner를 1024 tokens로 재실행한다.

### 단계 3: hard-negative를 한 번 제대로 만든다

1. BM25, pinned base Qwen, 현재 checkpoint의 top candidates를 합쳐 query당 pool을
   저장한다. 한 retriever의 top-hard만 쓰지 않는다.
2. 같은 pinned teacher가 positive와 candidate를 함께 점수화한다.
3. 시작 filter는 `s_neg < .95 * s_pos`다. `.90/.98`은 pilot에서 false-negative와
   난이도를 보고 바꾼다. partial positive와 ambiguous는 hard negative에서 제외한다.
4. pool 24를 유지하고 선택한 objective에 맞게 4개 또는 7개를 epoch마다 표집한다.
5. 첫 200K best checkpoint로 candidate를 **한 번만** 재채굴해 짧은 second pass를
   한다. 계속 재채굴해 benchmark-style shortcut을 증폭하지 않는다.

### 단계 4: full-dimension 승리 뒤에만 확장

1. 서로 다른 clean-dev peak 2~3개를 equal average하고, 같은 lineage·license 안에서만
   SLERP를 비교한다.
2. 4096-d가 종합 gate를 통과한 뒤 MRL `1024/512/256`을 별도 ablation한다.
3. 512/1024 winner가 short·broad gate를 통과한 뒤 실제 2K~8K source-document
   held-out data로 long-context phase를 연다. position extrapolation만으로 long
   retrieval 성능을 주장하지 않는다.
4. 마지막 한 후보만 고정 Korean retrieval, broad Korean, multilingual, long,
   prompt/noise robustness, throughput·VRAM·저장비용 순으로 전체 평가한다.

2026-07-17 frontier는 LoRA capacity 상한을 그대로 두지 않는다. 동일 200K Qwen/Comsat
LoRA를 clean-only로 비교한 뒤 승자 계보의 raw base 하나에서 상위 4개 transformer block과
final norm을 동일 3,123-step/global batch 64로 update한다. 실제 microbatch 8/HN4 backward가
H100 80GB에서 먼저 통과해야 하며, input/base/completion SHA contract가 없으면 full artifact로
package하지 않는다. 이 last4 challenger까지 포함한 clean winner가 1M stage의 시작점이다.

## 4. 금지사항

- `.env`, shell history, process argument, log, manifest, notebook output에 GitHub/HF
  token을 출력하지 않는다. `.env`를 `source`한 채 `env`, `set`, `printenv`를 실행하지
  않고, token이 포함된 remote URL이나 command를 문서·commit에 남기지 않는다.
- `.env`, credential, cache, 원 checkpoint, gated data를 GitHub에 commit하지 않는다.
  GitHub에는 code/docs/small manifest만, 모델·대용량 dataset은 license gate 후
  Hugging Face의 전용 repo와 LFS/Xet 경로로 올린다.
- Comsat, NV-Embed-v2, Nemotron의 non-commercial weight를 상업 배포 후보에 merge,
  distill 또는 base로 사용하지 않는다. KaLM은 EU 및 Gemma 조건 검토 없이 사용하지
  않는다.
- F2/KaLM composite의 상단 `Apache-2.0`/`MIT` metadata를 모든 upstream row의
  권리로 간주하지 않는다. `unknown`은 허용이 아니라 보류다.
- 평가 test/query/qrel/corpus나 그 near duplicate로 학습·mining·checkpoint selection을
  하지 않는다. target-family train specialist 결과를 종합 zero-shot SOTA라고 부르지
  않는다.
- 모델 카드의 숫자를 revision, prompt, task version, max length, backend가 다른 local
  숫자와 평균 내지 않는다. Comsat card의 선택된 한국어 우세를 다국어·long·robustness
  우세로 확대하지 않는다.
- KaLM V2의 Qwen2-0.5B bidirectional mean-pool recipe를 current Gemma3-12B
  last-token 모델이 실제로 사용했다고 쓰지 않는다.
- 1×H100 첫 run부터 full FT, 8K/32K training, MRL multi-loss, KD, online mining을 한꺼번에
  켜지 않는다. 원인을 분리할 수 없는 run은 승리 근거가 아니다.
- `trust_remote_code=True` 모델은 pinned revision의 Python 파일을 정적 감사하기 전에
  실행하지 않는다. model revision만 고정하고 remote code revision을 떠 있게 두지
  않는다.
- tiny gain 하나로 승격하지 않는다. 동일 protocol 재실행, worst-task regression,
  clean private set, broad·long·multilingual gate를 모두 통과해야 한다.

## 5. 재현 manifest 최소 계약

모든 download, train, merge, evaluation run은 다음을 남긴다.

- model ID, exact model SHA, remote-code SHA, weight-index hash, license snapshot
- dataset ID/SHA, 실제 사용 row ID 또는 shard hash, source별 license와 제외 사유
- code commit, container/lockfile hash, CUDA·PyTorch·Transformers·MTEB version
- exact query/document prompt를 whitespace까지 포함한 JSON, pooling, normalize, max length
- seed, LoRA target/rank/alpha/dropout, optimizer, LR schedule, batch/accumulation,
  temperature, loss term과 weight, negative 수와 sampling seed
- miner/teacher ID와 SHA, candidate pool, positive-relative threshold, 재채굴 횟수
- hardware, peak VRAM, wall time, tokens/examples, checkpoint와 merge coefficient

2026-07-17 구현에서는 Nemotron의 equal-average 근거와 독립 adapter factor의 basis ambiguity를
함께 반영했다. 서로 다른 specialist LoRA의 A/B를 직접 평균하지 않고 safe-merged full model
weight를 tensor별 FP32로 누적한다. immediate parent retention 2종, general/combined 2종,
specialist 3종의 일곱 fixed coefficient만
사전에 등록하며, source shard와 model evidence SHA 및 ST contract가 모두 같은 경우에만
BF16 sharded soup를 생성한다. 실제 clean 결과 전에는 개선을 주장하지 않는다.
- evaluation task/data revision, backend, dtype, batch, raw result path와 실패/OOM 기록

이 계약을 만족하지 않는 외부 점수는 참고값이며, 우리 모델의 승격이나 README의
성능 주장에 사용하지 않는다.
