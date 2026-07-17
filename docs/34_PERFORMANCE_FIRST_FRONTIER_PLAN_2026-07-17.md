# 2026-07-17 성능 최우선 임베딩 방법론과 전면 복구 계획

> **Validation correction:** legacy 512 query-positive pair가 active 200K에 512/512
> 포함돼 active Qwen의 eval loss는 완료/finite 신호로만 사용한다. 모든 archived
> checkpoint는 legal v2 text-strict 10K에서 재선택하고, Comsat 이후 run은 v2에서 파생한
> 독립 512를 사용한다. 세부 감사는
> [validation 정정 문서](35_VALIDATION_LEAKAGE_CORRECTION_2026-07-17.md)를 따른다.

기준일: **2026-07-17 (Asia/Seoul)**

## 결론

이 프로젝트의 최종 목표는 라이선스가 허용하는 상업 배포 모델을 먼저 만드는 것이
아니다. **비상업 연구용 데이터와 가중치까지 허용해 한국어 retrieval 및 한국어가 포함된
broad embedding 평가에서 가능한 최고 성능을 달성하는 단일 모델**을 먼저 만든다.
권리가 깨끗한 공개 모델은 이 최고 성능 모델에서 검증된 방법을 다시 적용하는 두 번째
track이다.

현재 가장 근거가 강한 방법은 새로운 loss 하나가 아니라 다음 파이프라인의 결합이다.

1. `Qwen/Qwen3-Embedding-8B` clean lineage와
   `sionic-ai/comsat-embed-ko-8b-preview` Korean warm-start lineage를 같은 조건으로 비교한다.
2. 대규모 일반 retrieval pair로 embedding geometry를 먼저 안정화하고, instruction이 붙은
   한국어·다국어 multitask/target mix로 후속 학습한다.
3. 작은 microbatch의 in-batch negative에 머물지 않고 cached large-batch 또는 memory
   queue를 A/B해 effective negative pool을 키운다.
4. 현재 student, BM25, 강한 외부 embedder의 후보를 합치고 `Qwen3-Reranker-8B`로 false
   negative를 제거한다.
5. reranker 상위 hard negative만 쓰지 않는다. teacher score 전 구간에서 quantile로
   뽑은 후보를 함께 사용해 listwise KL 또는 MarginMSE로 점수 분포를 전달한다.
6. 법률·금융·건강처럼 조건이 여러 개인 질의에는 한 조건만 틀린 counterfactual
   negative를 추가한다. 자유 생성 negative는 그대로 학습하지 않는다.
7. general replay를 유지한 400K target curriculum을 학습한 뒤 마지막 5개 checkpoint
   평균과 specialist adapter soup을 clean dev에서 비교한다.

공개 benchmark의 dev/test query, qrel, corpus를 학습·negative mining·checkpoint 선택에
사용하지 않는 원칙은 유지한다. 이것은 권리 문제가 아니라 점수를 실제 성능으로 해석하기
위한 최소 조건이다. target-family의 공개 **train split**은 사용할 수 있지만 노출을 모델
카드에 표시한다.

## 1. 재시작 후 실제 상태

2026-07-17 재점검·복구 결과, 이전 local checkpoint는 소실됐지만 원격에 고정한 공개
data/model과 법률 원문, 학습·평가 환경은 NFS에 다시 복원됐다.

| 항목 | 현재 실제 상태 |
|---|---|
| Git | 복구/방법론/capacity/soup/runtime storage guard 변경을 `origin/main`에 지속 push; credential은 one-shot helper에서만 사용 |
| submodule | 5개 모두 pinned commit으로 복원 완료; 최신 conditional method 참고용 CLEAR 포함 |
| 로컬 data/cache/output | 학습/검증 dataset 15개, comprehensive text용 dataset 13개, core/teacher 4개, 외부 비교 5개, 법률 312,581문서를 exact 복원·검증 |
| Python 환경 | NFS `.venv-hf-tools`, `.venv-train-fa2`, `.venv-mteb` 복원; 8B backward와 전체 test 225/225 통과 |
| valid model | 0개; 새 Qwen 200K active, 성공 종료 후 Comsat 200K와 나머지 frontier campaign 직렬 queue 대기 |
| GPU | H100 80GB 1장, Qwen production 100% utilization |
| NFS | `/home/ubuntu/data`, 49TB 중 48TB 가용, 사용률 3%, inode 1% |
| system disk | `/`, 2.0TB 중 1.6TB 가용, 사용률 22% |

모든 새 경로는 `/home/ubuntu/data/Embedding` 아래로 고정한다.

최종 completion은 clean selection 파일만으로 성립하지 않는다. final-selected 한 모델의
Sionic9·공식 Korean6·comprehensive7/414 완주, exact training manifest, 격리 staging으로 만든
private final model의 전체 remote file/LFS 검증 report와 campaign-result Git push가 모두
성공해야 한다. 하나라도 실패하면 queue는 nonzero로 멈추고 완료라고 기록하지 않는다.

```text
/home/ubuntu/data/Embedding/
├── .cache/huggingface/hub/       # model/dataset cache
├── .cache/pip/                   # package cache
├── .venv-*/                      # 격리 환경
├── data/raw/                     # Git/원본 data
├── data/processed/               # 변환 data
├── outputs/                      # train/eval/checkpoint
└── artifacts/                    # 최종 package
```

`TMPDIR`, `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `PIP_CACHE_DIR`, `TORCH_HOME`,
`XDG_CACHE_HOME`도 모두 위 NFS 아래로 지정한다. root overlay에 model shard나 optimizer를
두지 않는다. 학습 전후 `df -h`와 `df -ih`를 기록하고, NFS 사용률 80%에서 새 download를
중단하며 90%에서는 학습 시작을 금지한다. 8B 학습 진입점은 추가로 workspace
500GiB/100만 inode와 `/tmp` 50GiB/10만 inode를 fail-closed 최소 headroom으로
검사한다. 현재 NFS 약 48TB와 root 약 1.6TB가 가용하다.

장기 stage 사이의 검사만으로는 공용 볼륨의 외부 급증을 잡지 못하므로 runtime watchdog도
둔다. workspace 500GiB/100만 inode, root 100GiB/20만 inode, `/tmp` 50GiB/10만 inode를
30초 간격으로 확인하며, 일시적인 `df` 실패를 피하려고 2회 연속 실패만 emergency로 본다.
emergency에서는 시작 시 자기 자신이 PGID leader임을 확인했고 신호 직전에도 같은 identity인
우리 Qwen training/watcher/frontier group에만 TERM, 30초 뒤 KILL을 보낸다. 다른 사용자 PID나
경로를 탐색해 종료하지 않는다.

## 2. 참고 코드와 법률 원문 위치

### 고정 Git 코드

| local path | upstream | pinned commit |
|---|---|---|
| `Qwen3-Embedding/` | [QwenLM/Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding) | `44548aa5f0a0aed1c76d64e19afe47727a325b8f` |
| `third_party/ms-swift/` | [modelscope/ms-swift](https://github.com/modelscope/ms-swift) | `3d61b9318b27fdd5659e530cd36db7f4ce740fd7` |
| `third_party/mteb/` | [embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb) | `193e3f66d2deac678065a43354c9c4efc57f507d` |
| `third_party/CodeFuse-Embeddings/` | [codefuse-ai/CodeFuse-Embeddings](https://github.com/codefuse-ai/CodeFuse-Embeddings) | `1c5291549b9cee9eeab1cd9de6a67be4d0295da0` |
| `third_party/CLEAR/` | [dltmddbs100/CLEAR](https://github.com/dltmddbs100/CLEAR) | `68f5916a0ae3206f1c0655c7e0d25877193ad47a` |

### 법률 Markdown source

법률 source 정의는 `configs/legal_data_sources_v1.json`, 변환기는
`scripts/prepare_legal_embedding_data.py`, 기존 감사는
`docs/17_LEGAL_AND_KO_DATA_SOURCE_AUDIT.md`에 있다. 복원 경로와 revision은 다음과 같다.

| source | local path / glob | pinned revision | 문서 수 |
|---|---|---|---:|
| 법령 | `data/raw/legal_source_audit/legalize-kr/kr/**/*.md` | `db3cd760c14042ee04fd9166e1bdbb662fc999bc` | 5,725 |
| 행정규칙 | `data/raw/legal_source_audit/admrule-kr/**/본문.md` | `64a5a272909ab5bc077b0ad9519ef31de8febb46` | 20,390 |
| 판례 | `data/raw/legal_source_audit/precedent-kr/*/*/*.md` | `40cd00e54df19d98562abb170c8ff51fd6fe2c2e` | 124,116 |
| 자치법규 | `data/raw/legal_source_audit/ordinance-kr/**/본문.md` | `6443e5dd5833d863219064cd362111f516430bec` | 162,350 |
| 합계 | `data/raw/legal_source_audit/` | 위 네 snapshot | **312,581** |

고정 extractor의 source-native 후보는 2,756,363개다. 법령/행정규칙/자치법규의
`title 또는 article heading → article body`, 판례의 `판시사항 → 판결요지`를 사용한다.
법률 250K 공개 shard를 먼저 정확히 복원하고, 성능 개선이 확인되면 전체 후보에서
중복·너무 짧은 positive·parsing failure를 제거해 1M 이상으로 확장한다.

## 3. 2026년 7월 17일까지의 1차 근거

### 3.1 직접 채택하는 방법

| 연구/공식 자료 | 확인된 핵심 | 이 프로젝트에서의 채택 |
|---|---|---|
| [Qwen3-Embedding 공식 코드](https://github.com/QwenLM/Qwen3-Embedding) | Qwen3 8B causal backbone, last-token/EOS 계열 embedding, instruction-aware retrieval/reranking | 원본 clean lineage, tokenizer/pooling/prompt 계약, Qwen reranker teacher |
| [ML-Embed, ICML 2026](https://arxiv.org/abs/2605.15081) | 121개 공개 source·50M sample, 26.7M retrieval pre-stage 뒤 source당 최대 100K인 8.3M instruction multitask stage; retrieval은 in-batch+HN7, 분류/클러스터링은 HN-only; 8B는 length 1024, LR `7e-6`; 마지막 5 checkpoint 평균 | two-stage curriculum, task-type별 negative 정책, HN7, 1024 specialist, last-5 평균을 채택. 3D Matryoshka는 최고점 확정 뒤 효율 stage로 보류 |
| [Seed1.5 공식 기술 공개](https://seed.bytedance.com/en/blog/bytedance-s-seed1-5-embedding-model-achieves-sota-in-retrieval-training-details-unveiled) | 대규모 pair/in-batch-only 1단계, supervised+synthetic 2단계; retrieval은 in-batch+hard, 그 외 HN-only; current target model 재채굴과 hard/in-batch false-negative filter | general→task stage, current-student mining, 두 종류 false-negative filter를 채택 |
| [Seed1.6 공식 기술 공개](https://seed.bytedance.com/en/blog/built-on-seed1-6-flash-seed-1-6-embedding-launched) | text pair InfoNCE→multimodal→instruction mixed fine-tuning의 3단계, 어려운 task에 난이도별 negative mining | text-only 최종 모델에는 stage 1/3 구조와 난이도별 negative만 참고 |
| [Scaling and Stabilizing Large-Scale EBR, SIGIR 2026](https://arxiv.org/abs/2607.10096) | cross-batch negative, current model ANN+cross-encoder+metadata hybrid offline mining, 작은 legacy teacher에서 큰 student로 KL warm-start. 각각 progressive ablation에서 추가 이득 | 1 GPU에서는 cross-GPU 대신 queue4096, current-student ANN+Qwen reranker를 채택. 현재 clean-selected 8B→1M continual run은 별도 legacy-score KL warm-start가 아니며, 새 backbone 교체 때만 그 ablation을 연다. product metadata rule은 provenance가 request/audit에 실제 결속되기 전에는 채택했다고 주장하지 않음 |
| [Beyond Hard Negatives, SIGIR 2026](https://arxiv.org/abs/2604.04734) | query당 positive+200 후보를 teacher로 채점하고 score quantile을 고르게 덮는 deterministic stratified sampling. Top-K보다 KL/ MarginMSE와 in/out-domain에서 안정적 | top hard만 저장하지 않고 score quantile HN7/16 cache, KL을 기본 KD로 채택 |
| [CausalNeg, KDD 2026](https://arxiv.org/abs/2606.01304) | relevance를 여러 요구조건으로 분해하고 정확히 한 조건만 위반하는 counterfactual negative; 자유 생성의 style/source shortcut을 QEM으로 억제 | 법률 조문 번호·기관·시점·예외, 금융 상품/조건, 건강 대상/증상/처치에서 one-condition negative 생성. style reference와 source-balance regularizer를 A/B |
| [IKEA negative mining, SIGIR 2026](https://arxiv.org/abs/2605.00353) | LLM judge 1–5에서 positive≥4, negative≤2, 3 제외; 같은 category의 속성 위반과 한 조건 위반 negative가 유효. 더 많은 query expansion은 더 작은 HNM보다 낮았음 | teacher 경계가 애매한 후보 제외, raw synthetic 수량보다 실제 질의 intent와 one-condition quality 우선 |
| [SemEval HITS, 2026](https://aclanthology.org/2026.semeval-1.338/) | Qwen3-Embedding-8B, multi-negative InfoNCE, false-negative mask, self-distillation KL. 1회 KD는 test를 높였지만 2회는 `0.705→0.690`으로 하락 | self-distillation은 1회씩 clean dev gate를 통과할 때만 반복; 반복 자체를 기본값으로 두지 않음 |

### 3.2 조건부 또는 후속 채택

| 연구 | 가치 | 지금 주 모델에 바로 넣지 않는 이유 |
|---|---|---|
| [LEAF, ACL 2026](https://aclanthology.org/2026.acl-long.2008/) | teacher embedding 자체를 L2로 맞추며 hard negative 없이 작은 batch로 학습; 비대칭 query-student/document-teacher 가능 | 23M 압축 모델에 특히 적합하다. 8B 최고점 모델에서는 reranker score KD보다 우선할 근거가 부족해 후속 query encoder 압축에 사용 |
| [Relevance-Based Embeddings, ICML 2026](https://arxiv.org/abs/2607.03515) | support item/query에 대한 heavy-ranker relevance vector를 MLP embedding으로 바꾸며 복잡한 ranker를 근사; `l2-greedy` support 선택 | 새 query마다 support에 대한 heavy-ranker call이 필요하고 corpus 의존성이 있어 범용 text encoder의 drop-in replacement가 아님. 폐쇄형 법률 catalog의 별도 후보 생성기로 평가 |
| [ReasonEmbed](https://arxiv.org/abs/2510.08252) | reasoning 요구가 있는 data synthesis와 sample weighting으로 BRIGHT를 개선 | 공개 data가 영어 reasoning domain 중심이다. 방법만 가져와 한국어 법률·금융 source-grounded query에 적용 |
| [3D Matryoshka/ML-Embed](https://arxiv.org/abs/2605.15081) | dimension, layer, token embedding rank를 함께 중첩 학습해 효율 개선 | full 4096-d/전체 layer 최고 성능을 먼저 확정해야 한다. 여러 prefix/layer loss는 제한된 H100에서 primary gradient를 분산시킬 수 있어 효율 후보에만 적용 |
| [Dewey Long Context Embedding](https://arxiv.org/abs/2503.20376) | local chunk와 global document embedding을 distillation으로 함께 정렬하는 chunk-alignment training; 영어 LongEmbed와 128K 지원을 보고 | 현재 512-token general gradient에 혼합하면 task 비중이 흔들린다. 최종 general winner에서 한국어 법률/QA 장문만 사용한 single-vector specialist로 A/B하고 MPE multi-vector track과 별도 비교 |
| [Multi-Prefix Embedding](https://arxiv.org/abs/2606.23642) | 한 번의 causal forward에서 EOS-separated chunk별 prefix embedding을 꺼내 MaxSim하고 문서-level label만으로 학습; 장문 evidence 위치도 반환 | 문서당 여러 vector와 새 MaxSim index가 필요해 현재 단일-vector 모델의 같은 점수표가 아니다. 최종 weight 고정 뒤 Korean MLDR/법률 장문용 별도 serving ablation |
| [EvoEmbedding](https://arxiv.org/abs/2606.21649) | EvoTrain-180K로 latent memory+raw segment를 공동 학습하고 queue로 recurrent collapse를 억제; segment batching 3.8×와 장문/agentic-memory 이득 보고 | query embedding이 session history에 의존하는 stateful architecture라 현재 stateless SentenceTransformers 계약·재색인 방식과 비호환. 별도 agentic-memory 제품 track으로 보류 |
| [CLEAR, ACL 2026](https://aclanthology.org/2026.acl-long.13/) · [공식 코드](https://github.com/dltmddbs100/CLEAR) | 영어 passage를 bridge로 삼고 source-language query, 번역 query, target-language negative query를 reverse-training. 고정 코드 `68f5916a`는 forward q→d CE, reverse d+→번역 q/negative-query CE, optional distribution KL과 GradCache를 제공하며 기본 `alpha=0.4`, `beta=0.2`, query-negative 5개, cosine scale 20이다. 논문은 저자 실험에서 low-resource cross-lingual retrieval 최대 15% 개선을 보고 | 현재 1M의 ko↔en pair만으로는 `cross_anchor`와 target-language hard-negative-query 계약이 없다. exact compiler를 추가한 뒤 한국어↔영어 clean CLIR dev에서 general winner의 별도 specialist로 A/B하고 한국어 monolingual 회귀를 함께 gate |
| [Situated Embedding Models, ACL 2026](https://aclanthology.org/2026.acl-short.5/) | 짧은 evidence chunk를 더 넓은 원문 context에 조건화해 local retrieval과 document context를 함께 보존; 저자 book-plot benchmark에서 1B가 일부 7B급 baseline을 상회 | chunk encoder 입력·색인 단위가 현재 standalone 문장 embedding과 다르고 영어 단일 benchmark 근거다. 법률 원문/판례의 heading→본문 장문 track에서 동일 chunk/qrel로 late chunking·MPE와 분리 비교 |
| [Representation Sharpening, EACL 2026](https://aclanthology.org/2026.eacl-long.173/) | target corpus의 유사 문서와 구별되는 정보를 document representation에 더하는 training-free zero-shot 방법; 20개 이상 다국어 dataset과 BRIGHT 개선 및 indexing-time approximation 보고 | corpus마다 재색인하는 retrieval-system 기법이지 배포 가능한 단일 model weight 개선이 아니다. 최종 모델 공개 점수와 섞지 않고 폐쇄형 법률/사내 corpus의 serving A/B로 평가 |

## 4. 어떤 backbone과 teacher를 쓸 것인가

### 4.1 본선 lineage

| 역할 | 고정 자산 | 이유 |
|---|---|---|
| clean base | `Qwen/Qwen3-Embedding-8B@1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` | 기존 200K/1M manifest와 exact 호환, 공개 architecture와 code |
| performance warm start | `sionic-ai/comsat-embed-ko-8b-preview@a5cc22b651c1b2e51cdd8bf671774ae93584f0ab` | 이전 동일 protocol에서 Korean retrieval 9종 macro 0.7930으로 Qwen 0.7825보다 높았던 현재 한국어 기준점; CC-BY-NC가 성능 track에서 허용됨 |
| relevance teacher | `Qwen/Qwen3-Reranker-8B@77d193c791ed757ca307ee72715aa132723da912` | query-document 직접 상호작용의 연속 `P(yes)`로 false negative와 KD target 생성 |
| embedding ensemble miner/teacher | pinned F2LLM-v2-8B, Qwen3-Embedding-8B, Comsat; 필요 시 KaLM 12B/Nemotron 8B | 각 모델의 nearest-neighbor blind spot을 합쳐 candidate recall 증가. 이 점수 평균을 최종 label로 간주하지는 않음 |
| external comparison | F2LLM-v2-8B, PwC, Harrier 27B, KaLM 12B, Nemotron 8B | 같은 evaluator에서 최종 후보의 broad/worst-task 회귀 확인 |

Comsat을 바로 최종 base로 고정하지 않는다. 동일한 200K와 동일 token budget으로 다음을
먼저 비교한다.

- `Qwen + LoRA r64`: 기존 exact baseline 복구
- `Comsat + LoRA r64`: Korean warm-start가 같은 data에서도 유지되는지 확인
- 승자가 유의미한 차이를 내면 partial/full tuning 소규모 A/B

Comsat lineage의 원 학습 data는 완전히 공개되지 않았으므로 그 결과는 `warm-start,
noncommercial, upstream-exposure-unknown`으로 표시한다. clean lineage를 삭제하지 않는다.

### 4.2 왜 F2/ML-Embed를 본선 base로 바로 쓰지 않는가

F2/ML-Embed는 공개 code/data와 매우 좋은 다국어 recipe를 제공하지만, 논문 snapshot에서
8B Korean 6-task 평균은 74.84이고 당시 Korean top은 77.01이었다. 우리 이전 Korean
retrieval 9종에서도 Comsat이 더 높았다. 따라서 F2는 data recipe·miner·teacher·비교군으로
우선 사용하고, Comsat/Qwen 본선보다 좋아지는 local evidence가 생길 때만 base로 승격한다.

## 5. 어떤 데이터를 쓸 것인가

### 5.1 즉시 복원할 exact artifact

| 목적 | dataset@revision | 실제 학습 row |
|---|---|---:|
| baseline | `LLM-OS-Models/korean-embedding-performance-v1-ablation-200k@f605128d3233e7cc488dc741b8f2af9ecf68b6fa` | 199,904 ordered |
| general scale | `LLM-OS-Models/korean-embedding-performance-v1-performance-1m@5a2a3ab7f0928c6570929cc231eaefdd3fa203e1` | 999,936 ordered |
| SQuAD train family | `...sionic-squad-train-60k@8fbc6d6d5c93c3493456079d930921ac90ec6801` | 60,000 |
| health | `...sionic-health-100k@5fc4bb817f6970a710be53376f35e0225201d2e2` | 100,000 |
| finance/commerce/legal | `...sionic-autorag-100k@9140e9e02bb3f40ac1c22a6e595d58208770f696` | 100,000 |
| retrieval train family | `...sionic-retrieval-train-family-4146@c9513a66ad64e5eab586969f6fdde7f9c8abd922` | 4,146 |
| legal | `LLM-OS-Models/korean-legal-retrieval-source-native-250k@ec2f09a220dc5aa326c5d63b8e49adbf3a5524bc` | 250,000 |
| finance supplement | `BCCard/BCAI-Finance-Kor-Embedding-Triplet@f63d59969dba9916bd34c86c82112331890b11da` | train 43,394 |
| clean selector | `LLM-OS-Models/korean-legal-source-heldout-retrieval-v1@ee1300f04ea03d66bb51e23bbbda34376fece3f0` | 10,000 query/corpus/qrel |
| blocklist | `LLM-OS-Models/korean-embedding-benchmark-blocklist-v1@5e876f26606830cd4d663cd62806d1f4c36387c9` | SHA-256 only, 약 547MB |

`scripts/restore_hf_assets.py`가 exact revision, file SHA-256, row count를 검증한다. 공개
asset download에는 인증 token을 쓰지 않는 mode가 기본이어야 한다. private checkpoint나
최종 Hub upload가 필요한 별도 과정만 process memory에서 credential을 읽고 값은 출력하지
않는다.

### 5.2 1M general의 현재 구성

기존 1M은 한국어/다국어 retrieval과 broad task를 섞은 decontaminated curriculum이다.
핵심 source family는 다음과 같다.

- Korean triplet 약 100K
- F2 공개 WebFAQ, multilingual QA, KoAlpaca, RealQA, Ko-Magpie
- MIRACL, MrTyDi, MLDR의 공개 train family
- PAWS-X Korean, ParaCrawl ko↔en
- KLUE YNAT/STS, KorSTS, Ko-StrategyQA train
- 명시 hard negative 1–7개와 source-homogeneous length bucket order

1M을 그대로 한 번 학습한 뒤, 현재 student로 다시 채굴한 1M-v2와 비교한다. 1M의
`critical overlap=0`은 유지하며, 50K diagnostic은 평가 query hash 4개 때문에 대표 모델
학습에 쓰지 않는다.

### 5.3 target 400K의 고정 시작점

| role | rows | 비중 |
|---|---:|---:|
| SQuADKor train-family | 40,000 | 10% |
| health/medical | 40,000 | 10% |
| AutoRAG finance/commerce/legal | 40,000 | 10% |
| MIRACL/MrTyDi/MLDR train-family | 4,128 | 1.032% |
| Korean legal/public | 60,000 | 15% |
| general 1M replay | 215,872 | 53.968% |
| 합계 | **400,000** | **100%** |

이 curriculum은 score를 보고 비중을 바꾸기 위한 종착점이 아니라 첫 비교점이다. 이후
변경은 공개 test가 아니라 clean legal, 별도 target-dev, broad regression으로만 결정한다.

### 5.4 성능이 오르면 확장할 데이터

1. `codefuse-ai/F2LLM-v2`의 60.1M 공개 composite에서 한국어 전체와 ko↔en,
   retrieval/STS/classification을 우선 추출한다.
2. ML-Embed처럼 첫 stage는 MMARCO/WebFAQ/CLIRMatrix/ParaCrawl/OCGI/CodeSearch 계열,
   둘째 stage는 source-language당 최대 100K를 기본 cap으로 둔다.
3. 312,581개 법률 Markdown의 2.75M source-native 후보를 사용하되, 한 source가 batch를
   독점하지 않게 source/task/query-style cap을 둔다.
4. BCAI finance train 43,394를 exact/MinHash dedup하고, validation/test는 학습에서 봉인한다.
5. 실제 한국어 질의 형태를 늘리기 위해 keyword, 자연질문, 장문 조건, 부정/예외,
   기간/시점, 다중조건 query를 source-grounded하게 생성한다.
6. reasoning data는 영어 ReasonEmbed raw를 무조건 번역하지 않고 한국어 원문에서
   query 요구조건과 근거를 함께 보존해 생성한다.

원 데이터 수가 늘어나는 것만으로 승격하지 않는다. IKEA 연구처럼 더 큰 synthetic query
expansion이 더 작은 HNM mix보다 나쁠 수 있으므로, 각 확장은 동일 token budget A/B에서
효과를 확인한다.

## 6. loss와 negative의 최종 설계

### 6.1 기본 retrieval loss

첫 baseline은 기존과 동일하다.

```text
L_base = InfoNCE(query, positive, in-batch positives, explicit HN)
tau = 0.02
explicit HN = 4 (200K exact baseline), 이후 7 또는 16 A/B
```

retrieval batch에만 in-batch negative를 쓴다. clustering, classification, STS처럼 다른
row의 positive가 실제 positive일 수 있는 task는 source/label-aware mask 또는 HN-only를
쓴다.

### 6.2 큰 negative pool

한 H100에서는 Walmart의 multi-GPU differentiable all-gather를 그대로 구현할 수 없다.
대신 다음 두 구현을 같은 example/token budget으로 비교한다.

- **Cached large-batch**: query/document embedding을 microbatch로 forward하고 similarity
  gradient를 cache/replay해 effective batch 256→512→1024를 단계적으로 키운다.
- **Memory queue**: 최근 document embedding을 stop-gradient queue로 저장하고 current
  microbatch denominator에 추가한다. queue staleness 때문에 refresh 간격과 age를 기록한다.

먼저 cached 방식이 strict parity와 finite gate를 통과하면 이를 기본으로 한다. queue는
속도 이득이 충분하고 clean dev 회귀가 없을 때만 채택한다. batch 확장은 false negative를
함께 늘리므로 normalized exact duplicate, positive family, high teacher score에 mask를
적용한다.

### 6.3 hybrid candidate pool

각 query에서 positive를 반드시 포함하고 최대 200개의 distinct candidate를 만든다.

```text
BM25 top-50
∪ current student ANN top-100
∪ Qwen/Comsat/F2 disagreement top-50
∪ same-domain / same-category structural negatives
∪ one-condition counterfactual negatives
```

Qwen reranker가 모든 candidate의 `P(yes)`를 채점한다. 다음은 제외하거나 positive로
재분류한다.

- exact/near duplicate of positive
- source metadata상 같은 answer/span/family
- teacher가 positive와 사실상 동급으로 본 후보
- benchmark blocklist와 겹치는 text

negative 7개를 쓸 때 전부 top-hard에서 뽑지 않고 teacher score quantile의 서로 다른
구간을 덮는다. score range가 좁은 query는 억지로 7개를 채우지 않고 row를 drop하거나
가용 후보만 쓴다.

### 6.4 dual-teacher distillation

후속 loss 후보는 다음이다.

```text
L = L_InfoNCE
  + lambda_r * KL(student candidate distribution || Qwen reranker distribution)
  + lambda_l * KL(student replay distribution || legacy Comsat/Qwen distribution)
```

- `lambda_r`: reranker의 fine-grained relevance 전달
- `lambda_l`: general/Korean geometry를 target fine-tuning 중 보존
- teacher/student distribution temperature는 manifest에 고정
- KL을 기본으로 하고 MarginMSE는 stratified candidate에서만 A/B

모든 기능을 한 run에 동시에 넣지 않는다. `filter-only → stratified KL → MarginMSE →
legacy warm-start` 순으로 한 요소씩 비교한다.

## 7. 1×H100 실행 순서

### R0 — 복구와 무결성

1. submodule exact commit 초기화 — **완료**
2. 공개 HF asset을 unauthenticated exact revision으로 복원 — **dataset 13개와 core/teacher 8B 4개 완료**
3. 법률 Git 4개를 NFS에 shallow가 아닌 pinned commit으로 복원 — **완료; exact HEAD와 312,581 Markdown 확인**
4. model shard/file SHA, dataset row/file SHA 검증 — **HF asset과 법률 inventory 완료**
5. NFS-only virtualenv와 cache 환경 생성 — **학습·MTEB/FAISS 평가 환경 완료**
6. Qwen/Comsat baseline smoke encoding과 evaluator parity 확인

### R1 — 과거 baseline 재현

- Qwen 200K LoRA r64, 199,904 rows, 3,123 optimizer steps를 처음부터 재학습
- 동일 recipe로 Comsat 200K warm-start를 학습
- clean Grade-I legal과 별도 held-out loss에서 두 lineage를 비교

2026-07-17 11:46 KST에 Qwen run을 시작했고,
[`run_frontier_200k_pair_queue.sh`](../scripts/run_frontier_200k_pair_queue.sh)가 Qwen의
정상 3,123-step 종료를 검증한 뒤 Comsat 전용 5+5-step backend report, 별도 private
watcher, 동일 3,123-step run을 직렬로 시작하도록 대기 중이다. 두 8B job은 겹치지 않는다.

Comsat이 정상 종료되면 같은 queue가 watcher를 명시적으로 정리한 뒤
`post-training-eval-20260717-frontier/clean-first-selection.json`을 반드시 생성한다. Qwen과
Comsat 후보는 Grade-I legal holdout과 noise robustness에서 먼저 비교하며, selection 파일이
없으면 1M으로 넘어가지 않는다. 이 첫 비교는 `SELECTION_ONLY=1`로 public benchmark와
publication을 건너뛴다. 승자 evidence의 exact base가 Qwen인지 Comsat인지 fail-closed로
결정한 뒤, production microbatch 8/HN4 last4 backward probe와 동일 200K·3,123-step partial-full
challenger 하나를 실행한다. probe OOM이면 skip evidence를 남기며, 성공하면 LoRA 두 계보와
last4를 다시 clean-first 선택한다. 이 post-capacity selection이 없으면 1M으로 넘어가지 않는다.
두 selection-only winner는 public score 없이 전체 모델을 `LLM-OS-Models2` private repo에 올리고,
remote manifest exact 재검증과 Hub commit report가 없으면 다음 단계로 넘어가지 않는다.
selection gate를 통과하면 선택된 lineage에서 1M general,
Qwen3 reranker KD A/B, retrieval/SQuAD/health/AutoRAG, 법률 25% + general replay 75%, combined 400K를 차례대로
실행한다. 각 큰 경계에서 workspace 500GiB/100만 inode와 `/tmp` 50GiB/10만 inode를 다시
검사한다. 존재하지 않는 `.venv-train` 하드코딩은 없애고, 명시적 `TRAIN_ENV` →
`.venv-train` → 복원된 `.venv-train-fa2` 순서로 runtime을 결정한다.

legal/combined까지 끝나면 queue는 종료하지 않고
`final-frontier-selection-20260717/clean-first-selection.json`을 반드시 생성한다. 이 최종
stage는 200K, 1M, KD 세 variant, retrieval/SQuAD/health/AutoRAG, legal, combined의 실제
존재 run과 fallback을 모두 순회하며 single best와 same-trajectory FP32 last-available-5
평균을 같은 Grade-I clean/robustness gate에서 비교한다. 모델 이름에 맞는 실제 training
manifest가 없으면 publish하지 않고, final selection 파일이 없으면 campaign을 실패시킨다.
첫 Qwen run의 full resume checkpoint 보존 한도는 3이지만 private watcher가 검증·정제된
adapter-only snapshot을 training version별로 별도 보존하므로 최종 last-5 FP32 평균에는
동일 궤적의 최신 5개를 사용할 수 있다. 이 archive는 optimizer/trainer state와 데이터를
포함하지 않으며, 이후 run은 full checkpoint도 5개를 유지한다.

2026-07-17 step 1000 재감사에서 최초 watcher의 declared train SHA 오타를 발견했다. 해당
private repo는 superseded로 표시하고 승격에서 제외했으며, 실제 file SHA를 쓰는 `-candidates-v2`에
step 250/500/750/1000을 full-payload와 remote allowlist까지 재검증해 보존했다. 회전으로 full
checkpoint가 사라진 step 250은 기존 immutable private manifest+local sanitized archive를 함께
검증하는 lineage-only correction 도구로 이관했고 adapter/config bytes는 바꾸지 않았다.

이 stage가 끝나기 전에는 과거 중단 run의 loss나 문서상 결과를 새 checkpoint 결과로
간주하지 않는다.

### R2 — tuning capacity

200K winner에서 같은 token budget으로 `LoRA r64 vs higher-rank/DoRA vs partial/full FT`를
비교한다. H100 80GB 한 장에서 full 8B optimizer가 OOM이면 optimizer offload/ZeRO를
사용하되 wall time과 numeric parity를 기록한다. 성능 차이가 near-tie면 더 안정적인
LoRA lineage를 유지한다.

2026-07-17 구현은 최단 성능 경로로 범위를 좁혔다. Qwen/Comsat LoRA clean 승자 계보의
raw pinned base 하나에 대해 trainable 771.790M인 last4+final norm을 동일 global batch 64와
token budget으로 비교한다. DoRA r32는 r64 LoRA보다 trainable capacity가 작고, GaLore/full은
단일 H100 시간·OOM 비용이 크므로 이 challenger 결과가 명백히 부족할 때의 다음 ablation이다.
`capacity_run_manifest.json`은 input SHA와 base revision을 `armed`로 고정하고 exact 3,123-step
종료 뒤 train/logging SHA를 넣어 `complete`로 바뀐다. full model packaging은 이 contract의
현재 파일 SHA, checkpoint containment, shuffle-off/global-batch 조건까지 다시 확인한다.

### R3 — general 1M과 큰 batch

1. exact 1M + 기존 in-batch/HN7
2. exact 1M + cached effective batch
3. winner로 1M candidate를 current-student 재채굴
4. Qwen reranker filter-only

각 run의 checkpoint를 clean selector로만 비교하고 public Sionic 9를 selector로 쓰지 않는다.

### R4 — score-distribution KD

먼저 50K–100K subset에서 query당 200 candidate를 채점한다.

- top-HN7
- random-HN7
- quantile-HN7
- quantile-HN16
- filter-only InfoNCE
- quantile KL
- quantile MarginMSE

quantile KL이 clean/worst-domain 모두 좋아질 때만 1M 전체 또는 target 400K를 채점한다.
reranker cache는 resumable shard로 만들고 model revision, prompt, yes/no token ID, raw logits,
normalized score, input SHA를 저장한다.

2026-07-17 구현 상태는 다음과 같다. 아직 아래 A/B의 **성능 결과는 없다**.

- 1M merged student로 전체 corpus를 한 번 encode하고, seed 고정 10K query에서 ANN top-200과
  positive를 `teacher-requests.jsonl`로 보존한다. pilot 이득이 확인되면 50K→100K로 늘린다.
- pinned `Qwen3-Reranker-8B@77d193c`가 local-only/FA2로 positive 포함 201개를 점수화한다.
  cache는 shard atomic resume와 content/runtime fingerprint를 그대로 사용한다.
- positive-relative `.95`, absolute margin `.02`, positive score `.5` gate 뒤 eligible pool의
  양 끝을 포함한 rank-quantile 15개를 고른다. 원 yes probability와 raw logits는 버리지 않는다.
- Swift external plugin은 `0.3 hard InfoNCE + 0.7 listwise KL`을 기본으로 하며
  filter-only, KL, KL+stop-gradient queue4096을 같은 token budget으로 비교한다. MarginMSE는
  명시적 선택 ablation이다. queue의 positive 근접 문서는 student score로 mask한다.
- 현재 teacher request는 query/document text, candidate ID, retriever score를 SHA-bound하지만
  source/task/document provenance를 아직 운반하지 않는다. 따라서 이 stage를
  `metadata-hybrid mining`이라고 부르지 않는다. 후속 structural-HN 실험은 provenance를
  request→score cache→compiler audit까지 결속하고, source별 false-negative/선택률을 먼저
  보고한 뒤에만 candidate로 추가한다.
- 1M 원본과 세 KD 후보는 Grade-I legal/robustness에서만 비교한다. 이 단계에서는 Sionic 9와
  공식 Korean 6을 호출하지 않으며, clean winner만 target/legal queue의 general base가 된다.
- exact KD train/audit/request/score artifacts는 SHA/row/admissibility를 다시 확인하고
  `LLM-OS-Models2/korean-embedding-qwen3-reranker-kd-pilot-v1` private dataset으로 background
  upload한다.
- clean selector의 exact winner는 model shard SHA, merge parity, clean/robustness summary+ranks,
  selection policy, 실제 training manifest를 다시 결속하고 optimizer/credential을 차단한 뒤
  `LLM-OS-Models2/qwen3-embedding-8b-ko-reranker-kd-clean-winner-v1-private`에 worker 1개로
  background upload한다. public benchmark 결과가 없는 intermediate backup임을 카드에 명시한다.

구현 진입점은 `scripts/run_reranker_kd_ablation_queue.sh`, loss는
`scripts/listwise_distillation.py`, Swift plugin은
`experiments/030_teacher_distillation/listwise_kd_plugin.py`, strict compiler는
`scripts/compile_reranker_kd_dataset.py`다.

### R5 — target 400K와 specialist

general winner에서 400K target curriculum을 학습한다. general replay 53.968%를 유지하며
각 specialist의 catastrophic forgetting을 막는다. 이후 법률 250K→1M, finance,
reasoning/counterfactual, length 1024/2048 specialist adapter를 별도로 만든다.

### R6 — merge와 final selection

1. 동일 run 마지막 5 checkpoint의 FP32 arithmetic mean
2. clean dev near-tie 안에서 general/target/legal specialist의 basis-safe full-weight linear soup
3. merge 전후 embedding cosine/rank parity
4. clean Korean suite로 한 model 선택
5. 선택된 **한 모델**에 Sionic 9, 공식 MTEB Korean v1, broad/long/noise 평가

public benchmark 결과를 보고 merge coefficient나 checkpoint를 되돌려 고르지 않는다.
raw-score 최고 모델, clean-lineage 최고 모델, release-safe 모델은 서로 다를 수 있으므로
각각 이름을 붙여 공개한다.

2026-07-17에는 1번을 실제 코드와 post-training queue에 연결했다.
`average_lora_checkpoints.py`는 anchor와 같은 Trainer version의 최신 최대 5개만 선택하고,
config/tensor contract와 finite를 확인한 뒤 FP32로 평균한다. source/output/config SHA가 묶인
`average_report.json`을 atomic하게 생성하며, `merge_embedding_adapter.py`가 report와 실제
adapter 파일의 hash를 다시 검증한다. single best와 평균 모델은 둘 다 Grade-I clean/robustness
후보가 될 뿐 자동 승격되지 않는다. 이미 시작된 Qwen run의 Trainer live 보존은
`save_total_limit=3`이지만 private watcher archive가 모든 주기 checkpoint를 별도로 보존하므로
마지막 최대 5개를 사용할 수 있다. Comsat/1M/KD/target run도 5개를 보존한다. 실제
평균 모델 성능 결과는 아직 없다. 2번 specialist soup은 독립 LoRA A/B factor를 평균하지
않고 safe-merged full weight를 FP32로 누적하는 방식으로 구현했다. general winner와 immediate
local parent 75:25·50:50, general/combined 50:50·25:75, general 50%+specialist 5개 각 10%,
general/combined 각 25%+specialist 각 10%, combined 50%+specialist 각 10%의 일곱 coefficient는
평가 전에 고정한다. architecture/ST contract,
source/output shard SHA, key/shape/dtype/finite와 weight 합을 fail-closed 검증한다. 아직 실제
soup 성능 결과는 없고, adaptive/greedy coefficient 탐색은 일반-domain clean dev 확장 전까지
보류한다.

200K Qwen/Comsat base 대결의 checkpoint 탐색 폭은 동일하게 고정한다. corrected 512는
전체 clean 10K에서 source-document-held-out으로 뽑은 균형 부분집합이므로 train과는 exact
분리됐지만 clean 10K와 독립된 별도 corpus는 아니다. 이에 두 200K run 모두 private
archive의 무결성 검증된 모든 checkpoint를 동일 clean 10K와 paired noise 6조건으로
재평가한다. 이후 1M/KD/전문가 run은 corrected 512로 single checkpoint를 정하고 같은
trajectory의 last-available-5 FP32 평균만 추가해 clean-first 비교한다.

1M 뒤 reranker-KD A/B에는 KD를 거치지 않은 1M 원본도 항상 후보로 넣는다. 그러므로 KD가
개선되지 않으면 원본을 선택하는 것이 정상이다. 다만 A/B 선택 파일 또는 선택 model·weights
SHA에 정확히 결속된 private remote file set과 immutable commit이 없으면 이후 전문가·법률
stage는 시작하지 않는다. 검증되지 않은 fallback base로 장시간 학습을 계속하지 않는다.

frontier 끝단에도 같은 queue를 재호출해 전 stage의 single-best/평균 후보를 한 번 더
clean-first 선택하도록 연결했다. 따라서 “평균 코드는 있지만 200K에만 쓰이는” 상태는
아니다. 최종 clean winner에만 Sionic 9/공식 Korean/comprehensive evaluation과 private
candidate publication을 실행하며 public score는 selector 입력에 들어가지 않는다.
1M, retrieval/SQuAD/health/AutoRAG, legal, combined queue는
`ENABLE_PUBLIC_INTERMEDIATE_EVAL=0`을 기본값으로 검증하고, frontier가 scale/legal 경계에서
이를 명시적으로 전달한다. 따라서 과거 intermediate public summary 파일이 남아 있어도 새
frontier 실행에서는 호출하거나 publication evidence로 재사용하지 않는다.
각 stage가 새로 만든 derived dataset은 Models2 upload가 완료돼야 queue 완료로 인정한다.
publisher는 immutable commit을 다시 조회해 요청 visibility, 전체 file allowlist, LFS SHA/size,
metadata download SHA와 upload 전후 local source identity를 전부 확인한다.
retrieval/SQuAD/health/AutoRAG/legal/combined 중 하나라도 학습 fallback까지 실패하면 다음
stage로 조용히 건너뛰지 않는다. 그 직전에 `run_model_soup_queue.sh`가 여섯 target 모델과
general의 eligible local parent를 필수 확인하고 fixed full-weight soup 일곱 종을 모두 만들어 같은 selector의
명시적 allowlist에 추가한다. soup가 winner이면 `soup_report.json`을 immutable local revision과
publication evidence로 사용한다.

## 8. 승격 기준

성능 최우선이어도 “loss가 내려갔다”는 승격 기준이 아니다.

| gate | 최소 조건 |
|---|---|
| artifact | finite weights, exact base/data/code revision, merge/package parity |
| local retrieval | clean legal NDCG@10 개선; `0.002` 이하는 near-tie |
| target | Sionic 9 worst-task의 큰 회귀 없음, task-family exposure 표시 |
| broad | Korean 6-task와 multilingual diagnostic에서 명백한 collapse 없음 |
| robustness | prompt/noise/OCR 조건의 worst-case 회귀 제한 |
| final claim | 모든 task raw result와 실패/OOM까지 보존 |

사용자 목표에 따라 raw leaderboard 성능을 1순위로 보고하되, test leakage나 upstream
exposure를 숨겨서 만든 숫자는 최고 성능으로 인정하지 않는다.

## 9. token, commit, push, upload 정책

공용 컴퓨터에서는 token 값을 command line, shell history, process list, log, Markdown,
Git remote URL, commit에 넣지 않는다.

- public Git/HF download는 가능한 한 unauthenticated로 수행한다.
- `.env`가 없으면 임의로 만들거나 token을 요청·출력하지 않는다.
- credential을 읽어야 하는 upload는 값이 아니라 존재 여부만 확인하고 process memory에서
  사용한다. persistent `login`을 만들지 않는다.
- 학습·mining·평가 parent에서는 `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`을 unset한다. dataset
  uploader는 짧은 background subshell에서 mode-0600 `.env`의 HF key 하나만 안전하게 파싱해
  export하고 GitHub key와 token alias를 제거한다. model publisher/watcher도 같은 파일에서 HF
  key 하나만 process memory로 읽는다. `.env` 전체를 source하지 않는다.
- Git에는 code, configs, docs, small manifest/report만 push한다.
- model, dataset, checkpoint는 Hugging Face의 해당 model/dataset repository에 upload하고
  Git repository에는 넣지 않는다.
- 완성되지 않았거나 manifest/SHA/evaluation gate를 통과하지 않은 artifact는 대표
  모델명으로 공개하지 않는다. 장기 학습 checkpoint가 필요하면 private recovery artifact로
  분리한다.

새 Hugging Face artifact의 owner는 **`LLM-OS-Models2`**로 고정한다. 기존
`LLM-OS-Models/...` artifact는 exact 복구용 read-only source이며 추가 upload하지 않는다.
derived dataset, private candidate, 최종 model은 모두 `LLM-OS-Models2/...`로 생성한다.

2026-07-17 실제 권한 검증으로 private model repo
[`LLM-OS-Models2/embedding-upload-permission-test-20260717`](https://huggingface.co/LLM-OS-Models2/embedding-upload-permission-test-20260717)를
생성하고 `README.md` content write까지 성공했다. 검증 commit은
`ce762703eea2d174342e131db122b8ff1fadfc9c`다. test repo에는 model weight, dataset,
credential이 없다. `.env`는 ignored 상태로 존재하지만 public restore에는 계속
`token=False`를 명시하며, `restore_hf_assets.py`와
`restore_comprehensive_eval_assets.py` 모두 `--use-token`을 명시한 경우에만 credential을
읽는다. upload process만 token을 메모리에서 읽는다.

## 10. 즉시 해야 할 일

1. 이 문서와 README의 재시작 상태를 commit/push한다. — **완료; `origin/main@eade122`**
2. `scripts/restore_hf_assets.py`를 public unauthenticated download가 기본이 되도록 수정하고
   test한다. — **완료, 전용 test 3개 통과**
3. submodule, pinned dataset, Qwen/Comsat/reranker core model을 NFS에 복원한다. — **완료**;
   F2LLM-v2-8B, PwC, Harrier 27B, KaLM 12B, Nemotron 8B도 full 40-hex revision으로 익명
   복구·hard local-only 재검증했고, comprehensive text 진단용 8개 full snapshot과 modality가
   맞지 않는 시각 문서 5개의 query/qrels/metadata snapshot도 13/13 local-only 재검증했다.
4. 법률 Git 4개를 정확한 commit으로 복원하고 312,581 Markdown inventory를 재검증한다. — **완료**
5. Qwen 200K exact 5+5-step backend probe와 schema/SHA preflight를 통과한다. — **완료; SDPA 선택**
6. 200K Qwen과 Comsat 두 run을 처음부터 시작한다. — **Qwen active; step 1500까지 private
   immutable commit·LFS/manifest/lineage 독립 재검증 완료, Comsat 및
   post-eval→1M→target/legal 직렬 queue armed**
7. winner에 1M, current-student wide mining, reranker stratified KD를 순서대로 적용한다. — **코드·자동 queue·clean selector 완료, 실제 실행 대기**

이 순서는 저장공간을 먼저 보호하면서도, 이전 결과를 복구했다는 착각 없이 가장 빠르게
새 valid checkpoint와 최고 성능 후보를 만드는 경로다.
