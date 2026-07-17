# 논문·모델별 데이터와 학습 방법 매트릭스

조사 기준일: 2026-07-17. 이 문서는 이 저장소에서 검토한 논문, 공식 모델 카드, 공식 학습 저장소를 하나의 실행 가능한 의사결정표로 합친다. 논문의 arXiv 라이선스, 모델 가중치 라이선스, 학습 데이터 라이선스는 서로 다른 항목이다. 아래에서 `공개`는 곧 `상업적 재배포 가능`을 뜻하지 않는다.

## 결론

Sionic의 공개 9-task 평균 `0.7930`을 넘기기 위한 첫 선택은 raw-text next-token CPT가 아니다. `Qwen3-Embedding-8B`의 기존 다국어 표현을 보존하면서 권리가 정리된 한국어 `(instruction, query, positive, negatives)`를 만들고, 다음 네 가지를 순서대로 검증하는 것이 비용 대비 근거가 가장 강하다.

1. BM25, 원본 Qwen, 현재 checkpoint를 합친 candidate pool과 positive-relative false-negative 제거
2. top-hard만 고르지 않는 reranker-score quantile 표집
3. hard-label InfoNCE와 reranker score-distribution KD의 결합
4. general, domain, long-context specialist checkpoint를 clean dev로 선택한 뒤 평균/SLERP

Qwen3, E5, F2LLM, Nemotron, KaLM의 첫 대규모 단계가 논문에서 `pre-training`으로 불리기도 하지만, 대부분은 raw text next-token CPT가 아니라 **pair 기반 contrastive training**이다. 이 프로젝트의 기본 분류는 다음과 같다.

| 용어 | 입력 | 목적함수 | 이 프로젝트의 위치 |
|---|---|---|---|
| LM CPT/DAPT | raw text | next-token 또는 diffusion token loss | 명확한 언어/도메인 결함이 확인된 뒤에만 ablation |
| weak contrastive pre-training | noisy/synthetic pair | InfoNCE/NCE | Qwen에 이미 대규모 수행됨; 처음부터 반복하지 않음 |
| supervised contrastive tuning | query-positive-negative | InfoNCE, ranking loss | 주력 |
| contrastive distillation | teacher가 점수화한 candidate list | KL, MarginMSE, representation KD | 두 번째 주력 |
| generative reranker SFT | query-document 한 context | yes/no next-token CE | teacher 생성용; 최종 bi-encoder와 구분 |

## 근거 강도와 공개성 표기

- **확인:** 논문 본문, 공식 카드 또는 공식 코드에 직접 명시됨
- **부분:** 모델/데이터는 공개됐으나 전체 생성·필터·학습 경로 일부가 빠짐
- **미공개:** 저자가 공개하지 않은 항목이며, 점수나 config로 역추론하지 않음
- **release-safe 아님:** 다운로드 가능하더라도 upstream별 라이선스, 개인정보, benchmark overlap을 통과하지 못한 상태

특히 composite dataset 상단의 단일 라이선스 표시는 구성 source 각각의 조건을 자동으로 소거하지 않는다. 모든 행은 `source_revision`, `upstream_license`, `document_id`, `attribution`, `transformation`, `benchmark_overlap`을 별도로 기록해야 한다.

## 1. 기반 모델과 대규모 학습 recipe

| 모델/논문 | Base·attention·pooling | 공개된 데이터와 단계 | Loss·negative·teacher·merge | 공개/누락 | 우리 결정 |
|---|---|---|---|---|---|
| [Qwen3 Embedding](https://arxiv.org/abs/2506.05176) · [공식 repo](https://github.com/QwenLM/Qwen3-Embedding) | Qwen3 dense 0.6B/4B/8B, causal attention, EOS last-token, 1024/2560/4096-d, L2, MRL, query instruction | Qwen3-32B로 약 150M multi-task·multilingual weak pairs 생성; cosine `>0.7`인 약 12M synthetic pair를 SFT에 재사용하고 public labeled data를 혼합 | improved InfoNCE: explicit HN, in-batch query/document와 same-tower 항, false-negative mask; 여러 SFT checkpoint를 SLERP | 가중치 Apache-2.0. synthetic corpus, source별 수량, 전체 hyperparameter와 원 학습 launcher는 미공개. repo의 SWIFT 문서는 **continued tuning 예제**이지 원 150M recipe 재현 코드가 아님 | backbone과 input contract를 유지. `010`, `030`, `050`; 150M 단계를 반복하지 않음 |
| [F2LLM-v2](https://arxiv.org/abs/2603.19223) · [code](https://github.com/codefuse-ai/CodeFuse-Embeddings/tree/main/F2LLM) · [data](https://huggingface.co/datasets/codefuse-ai/F2LLM-v2) | `Qwen3-8B` chat base, causal, EOS last-token, 4096-d, L2; Qwen3-Embedding에서 시작하지 않음 | Stage 1 instruction-free retrieval 27M; Stage 2 instruction-aware multi-task 18M, source cap 80K. 공개 composite 60.1M/157 sources/약 564GB/282 languages이며 한국어 약 1.083M | retrieval은 in-batch InfoNCE + explicit-HN CE, `tau=.05`; HN pool 24에서 7개 표집; homogeneous source batches; MRL 8…4096; 8B는 KD 없이 full FT. LR `6e-6`, global batch 512, 2 epochs, ZeRO-2 | dataset card는 Apache-2.0이지만 upstream 조건은 별도 감사 필요. v2의 정확한 miner threshold, stage별 row manifest와 decontamination은 미공개/불완전 | homogeneous batching, pool-then-sample, dual HN loss를 `020`에 채택. 이미 embedding-pretrained인 Qwen에 Stage 1 27M을 재현하는 것은 기각. `060`, `070`에서 full FT 비교 |
| [Llama-Embed-Nemotron-8B](https://arxiv.org/abs/2511.07025) · [model](https://huggingface.co/nvidia/llama-embed-nemotron-8b) · [data](https://huggingface.co/datasets/nvidia/embed-nemotron-dataset-v1) · [code](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/retrieval/bi_encoder/llama_embed_nemotron_8b) | Llama-3.1-8B, causal mask 제거, bidirectional attention, mean pooling, 4096-d, L2 | 논문 16.1M pairs: public 7.7M + open-weight LLM synthetic 8.4M. Stage 1 web retrieval 11.8M, Stage 2 high-quality retrieval/classification/STS/bitext 4.3M | InfoNCE `tau=.02`; Stage 1 HN 1/global batch 2048, Stage 2 HN 4/batch 128; E5-Mistral/Qwen3 mining; `s_neg < .95*s_pos`; 서로 다른 6 checkpoint 평균, best single 대비 Mean(Task) `+0.84` | 데이터·코드·보고서 공개로 상위권 중 재현성이 높음. paper 16.1M과 현재 card 16.4M 표기가 달라 revision 고정 필요. 모델/데이터는 NVIDIA custom non-commercial 계열 조건 확인 | positive-relative filter와 checkpoint soup를 `020`, `050`에 직접 채택. backbone은 `060`에서 비교하되 가중치 배포 조건을 분리 |
| [KaLM-Embedding-V2/V2.5](https://arxiv.org/abs/2506.20923) | **논문 모델은** Qwen2-0.5B, causal mask 제거, mean pooling, 896-d, max 512, full FT | 470M weak pairs/20+ categories → 6M supervised/100+ categories → 같은 6M로 distillation. 550K persona synthetic, 40K instructions 포함 | CL `tau=.01`; focal-style weight `gamma=.5`; rank 50–100에서 HN 7, online pair/list HN 혼합; MRL 896/512/256/128/64; Qwen3-Embedding-8B teacher, `0.3 CL + 0.7 KL`, KD `tau=.05` | appendix에 source·규모·hyperparameter 공개. 그러나 source별 라이선스는 제각각이고 470M 전체를 release-safe로 볼 수 없음 | focal weighting, online HN, cached teacher distribution을 `020`, `030`에 ablate. 0.5B 결과를 12B 카드에 그대로 귀속하지 않음 |
| [KaLM-Embedding-Gemma3-12B-2511](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511) | **현재 leaderboard 모델은** `gemma-3-12b-pt`, 11.76B, last-token, 3840-d, 32K, MRL 3840…64 | 카드가 공개 `KaLM-embedding-finetuning-data` 6.34M을 연결. query당 positive 1–4/negative 7 구조 | 카드가 V2 논문을 인용하지만 Gemma-12B의 정확한 stage, loss weight, optimizer, batch, merge는 명시하지 않음 | Tencent custom model license. 논문의 Qwen2-0.5B/mean-pool recipe와 현재 Gemma12B/last-token 모델은 **동일 실험이 아님** | 강한 비교 모델로 `060`. 12B 성능을 특정 V2 요소의 인과 증거로 쓰지 않음 |
| [Harrier OSS v1 27B](https://huggingface.co/microsoft/harrier-oss-v1-27b) | Gemma3-text 계열 27B, causal decoder, last-token, 5376-d, L2, 32K, instruction | 카드는 large-scale multilingual multi-task contrastive mixture만 밝힘. 270M/0.6B는 큰 embedder로 KD, 27B에는 그 설명 없음 | 정확한 loss, temperature, batch, HN, teacher, merge 모두 미공개 | 가중치 MIT. paper, data list/scale, training code와 base lineage 세부가 미공개 | 점수/추론 기준선으로만 `060`; recipe를 추정하여 복제하지 않음 |
| [E5](https://arxiv.org/abs/2212.03533) | MiniLM/BERT base/large shared bidirectional encoder, mean pooling, query/passage prefix, cosine | CCPairs: Reddit, StackExchange, Wikipedia, S2ORC, Common Crawl/news 등 semi-structured pairs를 consistency filter해 약 270M. 이후 NLI, MS MARCO, NQ | Stage 1 large-batch in-batch InfoNCE (`tau=.01`). Stage 2 mined HN + cross-encoder soft-label KL; NLI contradiction HN | paper/code/model 공개. CCPairs 원문 bundle과 source별 권리 manifest는 완전 공개되지 않음 | 대량 pair → 소량 high-quality/KD라는 원형만 채택. 웹 scrape를 그대로 수집하지 않음. `010`, `030` |
| [multilingual E5](https://arxiv.org/abs/2402.05672) | XLM-R 계열 encoder, mean pooling, query/passage prefix | 1B pairs: Wikipedia 150M, mC4 160M, CC News 160M, NLLB 160M, Reddit 160M, S2ORC 50M, StackExchange 50M, xP3 80M, SBERT 10M. Stage 2 labeled 1.6M; instruct판은 GPT-3.5/4 synthetic 추가 | Stage 1 in-batch InfoNCE; Stage 2 mined HN + cross-encoder KD | source와 sampled count 공개, 개별 upstream 조건은 혼합. MIRACL·MrTyDi 등 benchmark train overlap이 존재 | 10–15% multilingual replay와 task diversity의 근거. clean zero-shot 후보에는 target benchmark 계열을 넣지 않음 |
| [BGE-M3](https://arxiv.org/abs/2402.03216) · [code](https://github.com/FlagOpen/FlagEmbedding) | XLM-R-large 계열, 8192 tokens, CLS dense + lexical sparse + ColBERT-style multi-vector를 한 backbone에서 출력 | RetroMAE: Pile/Wudao/mC4 184M texts/105 languages; weak pair stage 1.2B pairs/194 languages; labeled EN/ZH/MIRACL/MrTyDi; GPT-3.5 MultiLongDoc | 각 retrieval head InfoNCE; dense/sparse/multi-vector score를 통합한 self-knowledge distillation; length-bucket split batch | model/code 공개, 대규모 corpus와 labeled source의 조건은 제각각. hybrid index까지 써야 paper의 전체 기능을 재현 | long-data와 length bucket은 `040`, hybrid teacher/diagnostic은 `030`; 현재 single-vector 제품 목표에 sparse/multi-vector head를 즉시 추가하지 않음 |
| [Gecko](https://arxiv.org/abs/2403.20327) | 비공개 1.2B pretrained transformer, mean pooling, 256/768-d MRL, task feature | community QA + web title/body pre-finetune. FRet 6.6M: real web passage에서 LLM이 task/query 생성 → embedder candidate retrieval → 같은 LLM이 positive/HN 재라벨. NQ/HotpotQA/FEVER/MedMCQA/NLI/classification/MIRACL 혼합 | pre-stage in-batch NCE; fine-stage explicit HN, in-batch document와 same-tower query negatives; LLM query-likelihood + relevance classification을 RRF | model/data/backbone과 전체 규모가 비공개인 폐쇄 recipe. FRet에서 최초 source가 positive가 아닌 비율 약 15%라는 분석은 공개 | source-as-positive를 검증 없이 쓰지 않는 원칙을 data factory와 `020`에 채택. 폐쇄 데이터 자체는 사용 불가 |
| [Gemini Embedding](https://arxiv.org/abs/2503.07891) | Gemini 초기화 bidirectional transformer, mean pool + projection, 3072-d 계열, MRL 768/1536 | billion-scale title/passage pre-finetuning; task/language/code fine-tune mixtures; Gemini synthetic retrieval/classification, filtering, HN grading | in-batch NCE, explicit HN optional, classification FN mask; same-tower negatives는 오히려 제외; 한 source/dataset homogeneous batch; 여러 run parameter average/model soup | API/가중치와 data count·source 세부는 비공개. 논문이 recipe 개념과 ablation만 공개 | synthetic 생성→grader filtering, homogeneous batch, soup을 채택. closed teacher 의존은 기본 경로에서 제외; `020`, `050` |

### 1.1 2026-07 최신 인접 연구와 적용 경계

| 연구 | 방법·데이터 공개 범위 | 직접 적용 여부 |
|---|---|---|
| [Robustness Risk of Conversational Retrieval](https://arxiv.org/abs/2604.06176) | Qwen3-Embedding의 prompt 없는 대화 검색에서 filler/system artifact 침투를 분석한 평가 연구. 구조 노이즈 비율별 NDCG@5와 최고 noise rank를 보고하고 query prompt 완화 효과를 보임 | 학습 recipe가 아니므로 data mix 근거로 쓰지 않는다. clean 보드의 prompt on/off·noise 0/1/5% paired test로 채택 |
| [KV-Embedding](https://aclanthology.org/2026.acl-long.540/) | frozen decoder LLM의 마지막-token layer별 KV를 prefix로 re-route하는 training-free 방식. Qwen/Mistral/Llama, 최대 4,096 tokens | 현재 SentenceTransformer/Qwen3-Embedding weight contract와 다르므로 `060` frozen architecture ablation에만 보류 |
| [LEAF](https://aclanthology.org/2026.acl-long.2008/) | judgment/HN 없이 teacher vector space에 학생을 정렬해 asymmetric query-student/corpus-teacher 검색과 MRL·quantization 특성 상속 | 8B 최고 교사가 확정된 뒤 `120_compression`에서 사용. 현재 Sionic 추월 8B 학습에는 미적용 |
| [Qwen3-VL-Embedding](https://arxiv.org/abs/2601.04720) | 2B/8B multimodal 계열; contrastive pre-training→reranker distillation→MRL, 32K, 텍스트·이미지·문서 이미지·비디오 | text-only 보드의 비교 모델은 아님. OCR 이미지용 별도 multimodal track에만 참고 |
| [Dewey Long Context Embedding](https://arxiv.org/abs/2503.20376) | localized chunk embedding과 global document embedding을 distillation으로 함께 정렬하는 chunk-alignment training; 공개 보고는 영어 MTEB v2·LongEmbed와 128K context 중심 | single-vector 장문 정보 손실을 직접 겨냥하므로 최종 8B weight 고정 뒤 한국어 법률/QA 장문 specialist A/B에 사용. 영어 결과를 한국어 우위로 간주하지 않고 chunk/global clean holdout을 별도 구성 |
| [Multi-Prefix Embedding](https://arxiv.org/abs/2606.23642) | 한 장문을 EOS 경계 chunk로 나누되 causal forward는 한 번만 수행하고 각 prefix 경계 embedding을 꺼내 chunk-level MaxSim. MLDR-en·BrowseComp-Plus·LongEmbed에서 single-vector/독립 chunk 대비 경쟁력과 evidence attribution을 보고 | 문서당 단일 4096-d vector라는 현재 index/API 계약과 다르다. 주 모델 weight를 바꾸지 않고도 적용 가능한 별도 multi-vector serving/index track으로 두며, Korean MLDR raw result 없이 주 모델 우위로 합치지 않음 |
| [EvoEmbedding](https://arxiv.org/abs/2606.21649) | segment를 순차 처리하며 latent memory를 갱신해 동일 query도 context history에 따라 다른 embedding을 생성. EvoTrain-180K, collapse 방지 memory queue, segment batching을 사용하고 장문·agentic-memory에서 Qwen3-Embedding-8B/KaLM-12B 대비 우위를 보고 | stateless text→vector 함수가 아니라 session state를 갖는 새 architecture다. corpus 재색인·serving state·평가 protocol이 모두 달라 현재 SentenceTransformers release와 merge하지 않고 agentic-memory track으로 격리 |

### 모델·코드·데이터 공개성과 권리 상태

아래 상태는 조사일의 공식 card/repository 기준이다. 논문 PDF의 CC 표기는 논문 본문의 재사용 조건일 뿐 model/data에 전이되지 않는다. `unknown` asset은 public release 학습에서 기본 제외한다.

| 계열 | 모델/서비스 | 학습 코드 | 학습 데이터 | release 판단 |
|---|---|---|---|---|
| Qwen3 Embedding | open weights, Apache-2.0 | 공식 repo와 SWIFT continued-tuning 경로 공개; 원 150M launcher는 미공개 | 150M/12M synthetic와 전체 labeled mix 미공개 | base/adapter 배포 가능성은 높지만 **우리 데이터** 권리를 별도 증명 |
| F2LLM-v2 | model card Apache-2.0 | 학습 repo 공개 | composite 60.1M card Apache-2.0 | upstream 157개 source의 원 조건과 benchmark overlap을 row별 재감사하기 전 ingest 금지 |
| Nemotron | NVIDIA custom model license, non-commercial/research 제한 | NeMo AutoModel recipe 공개 | dataset 공개, dataset card license metadata는 조사 시 `unknown` | 방법 재현용. 우리 public/commercial derived weight의 base로는 법무 확인 전 제외 |
| KaLM V2/current 12B | current model은 Tencent KaLM community license | 논문 pseudo/상세 recipe는 있으나 current 12B exact code 없음 | current linked 6.34M dataset은 MIT 표기 | MIT 상단 표기와 upstream 데이터 권리를 구분; current model derived-weight 조건 확인 |
| Harrier | MIT weights | 학습 코드 미공개 | 목록·규모 미공개 | inference baseline만 가능; 데이터 방법론 재현 불가 |
| E5/mE5 | Microsoft 공개 weights/code 계열 | UniLM code 공개 | source/count는 공개, CCPairs 완성 bundle·권리 manifest는 불완전 | 방법과 public labeled subset만 source별 승인 후 사용 |
| BGE-M3 | `BAAI/bge-m3` MIT | FlagEmbedding 공개 | source 목록은 공개, 1.2B 전체 provenance bundle은 불완전 | model 비교 가능; source별 허용 row만 재구성 |
| Gecko | 공개 API/model family이나 논문 1.2B base와 FRet 원 data는 비공개 | 전체 training code 미공개 | FRet 6.6M 미공개 | 아이디어만 사용 |
| Gemini Embedding | closed API | 미공개 | 규모·source 세부 미공개 | 선택적 teacher API도 입력 권리/보안 승인 때만 사용 |
| NV-Retriever | NVIDIA 공개 모델 계열, card별 custom 조건 | paper recipe 공개, 구현 공개 범위는 revision별 확인 | 15개 public retrieval source이나 조건 혼합 | positive-aware algorithm만 독립 구현 |
| ReasonEmbed | paper가 recipe를 상세 공개 | FlagEmbedding 기반 설정 공개; artifact 공개 범위 별도 확인 | source corpus와 synthetic annotations의 배포 license 미확인 | source-exclusion 방법만 채택; data 직접 ingest 금지 |
| Beyond Hard Negatives | 연구용 student/teacher는 공개 모델 | 논문 algorithm은 재현 가능 | MS MARCO terms 적용; 별도 Korean data 아님 | quantile sampler를 독립 구현 |
| CausalNeg | 논문 code 공개 | GitHub/Zenodo 공개 | BGE/mMARCO/NQ/Hotpot/Trivia source별 조건 혼합 | code 참고 가능; 생성/원 data는 권리 재감사 |
| jina-v5 | small model card CC-BY-NC-4.0 | 논문 hyperparameter 공개 | 300+ dataset 전체 row manifest/license는 미공개 | non-commercial 조건과 data 불투명성 때문에 teacher/idea로만 우선 사용 |
| pplx-embed | 0.6B/4B model card MIT | 논문 recipe 공개 | FineWeb 계열은 식별되나 contrastive/synthetic 전체 row bundle은 미공개 | model ablation 가능; 250B corpus 재사용은 source별 조건 확인 |

### Qwen3 원 논문에서 정확히 가져올 것

Qwen의 denominator가 단순 `(positive + explicit negatives)`만 포함한다고 가정하면 안 된다. 원 논문은 hard negatives, 다른 query, 다른 positive/document를 함께 비교하고 mask를 둔다. 현재 pinned ms-swift의 `INFONCE_INCLUDE_QQ`, `INFONCE_INCLUDE_DD`, `INFONCE_MASK_FAKE_NEGATIVE`는 이 축의 ablation 수단이다. 다만 문서와 framework version 사이 default temperature가 다르므로 모든 run에서 `tau`와 세 환경변수를 manifest에 명시한다.

Qwen의 150M/12M synthetic data는 공개되지 않았으므로 “같은 데이터로 재현”할 수 없다. 우리 실험은 원 논문의 scale claim이 아니라 **현재 공개 8B checkpoint 위의 continued contrastive tuning**이다.

### F2에서 가져올 것과 가져오지 않을 것

가져올 항목:

- source-homogeneous batch
- query당 후보 pool을 먼저 저장하고 epoch마다 일부 HN을 표집하는 방식
- in-batch와 explicit-negative objective의 동시 사용
- task/source cap으로 큰 dataset이 전체 gradient를 지배하지 않게 하는 방식
- MRL은 품질 후보가 확정된 뒤 deployment dimension ablation으로 사용

가져오지 않을 항목:

- Qwen3 chat base에서 27M Stage 1을 다시 시작하는 것
- 공개 composite 60.1M을 upstream 권리와 benchmark overlap 검증 없이 전부 ingest하는 것
- F2가 full FT였다는 이유만으로 작은 한국어 budget에서도 full FT가 우월하다고 가정하는 것

### KaLM 논문과 현재 12B 카드를 분리해야 하는 이유

KaLM-V2/V2.5 논문의 재현 가능한 표는 `Qwen2-0.5B + bidirectional + mean pool + 896-d`에 관한 것이다. 현재 상위 모델은 `Gemma3-12B + last-token + 3840-d`다. 현재 카드가 같은 6.34M fine-tuning dataset과 V2 논문을 연결하지만, Gemma 모델의 전체 optimizer, stage별 데이터, KD weight, online HN, checkpoint merge를 공개하지 않았다. 따라서 다음 두 문장은 구분한다.

- **근거 있음:** focal reweighting과 CL+KL은 논문 0.5B 모델 ablation에서 이득이 있었다.
- **근거 없음:** 현재 Gemma12B의 leaderboard 점수가 바로 그 설정들 때문에 나왔다.

## 2. negative, distillation, reasoning, long-context 후속 연구

| 연구 | 검증된 핵심 | 데이터·학습 세부 | 채택/기각 및 실험 연결 |
|---|---|---|---|
| [NV-Retriever](https://arxiv.org/abs/2407.15831) | positive score를 anchor로 false negative 제거 | E5-large-unsup/Mistral-7B 실험. `TopK-PercPos`: `s_neg < .95*s_pos`; top-10에서 4개 표집. scale 실험은 15 retrieval datasets/728,160 examples, E5-Mistral teacher | `020`: `.90/.95/.98` ablation. 0.95를 보편 상수로 고정하지 않음 |
| [ReasonEmbed](https://arxiv.org/abs/2510.08252) | source 문서를 자동 positive로 삼지 않고, source-excluded mining 뒤 reasoning relevance annotation; difficulty-weighted RI-InfoNCE | 12 source domains, raw 95,960 → valid 81,659 queries; query당 평균 positive 12/negative 86. Qwen3 4B/8B 또는 Llama, EOS, `tau=.02`, LoRA r64/alpha32, max512, query당 총 1,023 negatives | `020`: source exclusion과 annotation. reasoning weighting은 clean general model 후 별도 specialist로만; generic QA 전체에 적용하지 않음 |
| [Beyond Hard Negatives](https://arxiv.org/abs/2604.04734) | teacher score의 quantile 전 구간을 deterministic stratified sample하면 top-hard보다 OOD가 안정적 | MS MARCO 532K queries/8.8M docs. Qwen3 Embed top100 + random100, Qwen3 Reranker로 positive 포함 201개 점수. CL batch1024 후 KL/ MarginMSE KD batch16 | `030`의 핵심: high/mid/low quantile 7/15개와 KL을 우선. MarginMSE는 잘못된 sampling에서 collapse했으므로 보조 ablation |
| [CausalNeg / When Hard Negatives Hurt](https://arxiv.org/abs/2606.01304) · [code](https://github.com/mzhangzhicheng/CausalNeg) | 생성 negative가 문체/source shortcut을 만들 수 있음. relevance 조건 중 하나만 위반하는 counterfactual 생성 + query-view entropy/source balance | mMARCO-zh, HotpotQA, NQ, TriviaQA에서 dataset당 약 10K query. BM25 HN 15 + generated 3. Qwen3-0.6B full FT, InfoNCE + entropy/mass-balance | `020`: synthetic negative 최대 비중 제한, real/generated source classifier와 embedding cluster audit. 검증 없는 LLM HN 대량 혼합은 기각 |
| [jina-embeddings-v5-text](https://arxiv.org/abs/2602.15547) | 큰 teacher geometry를 먼저 직접 증류하고 frozen student에 retrieval/STS/clustering/classification별 LoRA를 붙임 | Qwen3-Embedding-4B → Qwen3-0.6B/EuroBERT. 300+ datasets, 30+ languages, 50K steps. student→teacher projection cosine KD; retrieval은 InfoNCE + KD + global orthogonal regularizer. LoRA r32/a32, adapter checkpoint average. 별도 1K–4096 long phase | `030`, `040`, `050`, `070`: task adapter와 KD. 우리 8B→8B에서는 vector direct-KD보다 reranker score KD를 먼저 사용 |
| [pplx-embed](https://arxiv.org/abs/2602.11151) | 실제 LM continued pretraining의 대규모 사례. causal mask 제거/diffusion CPT 뒤 pair, contextual, triplet branch와 SLERP | Qwen3 base 0.6B/4B. FineWeb-Edu 50% + 29언어 FineWeb2 계열 50%, 4096×global1024×60K step ≈250B tokens. pair mix 60 languages; local chunk + global doc objective; INT8 QAT | `040`, `050`, `060`의 장기 ablation. 250B CPT는 첫 Sionic 추월 예산에서 기각; 한국어 표현 결함이 측정될 때만 축소 실험 |
| [Do Reasoning Models Enhance Embedding Models?](https://arxiv.org/abs/2601.21192) | generic SFT/RLVR reasoning 초기화 이득이 동일 contrastive training 뒤 사라질 수 있음 | 11 train datasets, Qwen3-Embedding-0.6B miner HN 3, positive-aware 95%, full FT InfoNCE | generic reasoning checkpoint 선행은 기각. reasoning은 retrieval objective와 직접 연결할 때만 후속 stage |
| [Search-R3](https://arxiv.org/abs/2510.07048) | reasoning 생성 후 `<embed>` state, retrieval DCG reward GRPO | TriviaQA/MS MARCO/code/MIRACL/S2ORC, Qwen3-32B synthetic 100K; CE+base KL+InfoNCE+triplet, 이후 GRPO | online reasoning latency와 재색인 비용이 커서 범용 candidate 뒤의 별도 연구로 미룸 |
| [LaSER](https://arxiv.org/abs/2603.01425) | explicit CoT와 latent thinking-token view의 output/process distillation | ReasonEmb 81,659, Qwen3 0.6B/4B/8B, InfoNCE+KL+trajectory alignment | reasoning specialist가 필요할 때 `030` 이후. 현재 under-review 근거로 일반 recipe를 바꾸지 않음 |
| [Querit-Reranker](https://arxiv.org/abs/2606.19037) | teacher score-gap pairwise loss, 16 score bins, domain checkpoint sequential SLERP | public 3.52M + private 2.05M ranking, 추가 약 9.4M multilingual synthetic | score bin은 `030`, domain SLERP는 `050`; private component 때문에 전체 결과 재현 claim 금지 |
| [BITEMBED](https://arxiv.org/abs/2606.25674) | ternary/activation quantization과 similarity/attention-relation KD | Qwen3-0.6B/Gemma3-270M, BGE-en-ICL mix, HN 7 | 최고 품질 모델 뒤 compression stage. 현재 경쟁 모델 학습에는 넣지 않음 |
| [TALAS](https://aclanthology.org/2026.acl-long.1509/) | 큰 교사의 최종 sentence vector는 학생 상위 2–4 layer에만 정렬 | Qwen3 embedding teacher → MiniLM/BERT; classification/pair/STS 중심 | 0.1–0.6B 학생 배포 연구에만 사용; 8B retrieval 주력과 분리 |

## 3. 어떤 데이터를 왜 쓸 것인가

### 3.1 데이터 선택 원칙

성능을 올리는 단위는 raw document 수가 아니라 **정답 관계와 어려운 비정답 관계가 검증된 training row**다. 문서가 1M개 있어도 query와 relevance label이 없으면 embedding objective를 직접 학습하지 못한다. 반대로 rights-safe 문서 20만 개에서 서로 다른 검색 의도, 길이, evidence 위치를 생성하면 여러 pair를 얻을 수 있다.

release candidate의 최소 row schema는 다음과 같다.

```json
{
  "instruction": "Given a Korean web search query, retrieve relevant passages that answer the query",
  "query": "...",
  "positive": [{"text": "...", "document_sha256": "...", "evidence": "..."}],
  "candidates": [
    {"text": "...", "source": "bm25", "retriever_score": 0.0, "teacher_score": 0.0,
     "label": "negative", "verification": "..."}
  ],
  "source_id": "...",
  "source_revision": "...",
  "upstream_license": "...",
  "attribution": "...",
  "generator_model": "...",
  "generator_prompt_revision": "...",
  "split": "train",
  "benchmark_overlap": false
}
```

### 3.2 rights-safe Korean 1M v1 mix

기존 목표 비중 `30/18/15/12/10/10/5`를 한국어 85%에 적용하고, 원 Qwen 능력 보존용 multilingual/general replay를 15%로 고정한 실행 수치다. 아래 bucket은 주 sampling label이고, `pdf`, `ocr`, `long`, `synthetic`, `evidence_position`은 별도 orthogonal tag로도 기록한다.

| 주 bucket | Rows | 비중 | 허용 source 예 | 필요한 이유 |
|---|---:|---:|---|---|
| 일반 factoid·백과 | 255,000 | 25.50% | Korean Wikipedia/Wikisource의 허용 revision, 공공누리 제0/1유형 문서 | MIRACL/MrTyDi/SQuAD류의 일반 QA와 lexical-semantic 균형 |
| 공공·금융·상거래 PDF/OCR | 153,000 | 15.30% | 공공누리 제0/1유형 보고서·통계·공공기관 FAQ; 권리가 확인된 표/문단 | AutoRAG와 실제 RAG의 layout/OCR/noise. OpenDART 원문은 별도 승인 전 제외 |
| long-document retrieval | 127,500 | 12.75% | 위 허용 문서 중 1K–8K 문서, 장/절 metadata 보존 | MLDR와 long RAG; evidence head/middle/tail를 균등화 |
| 법령·행정 | 102,000 | 10.20% | 공식 법령/조례/행정자료 중 저작권·API 조건을 확인한 원문 | LawIR와 조건·예외·관할 구분. 상업 해설/판례 DB 무단 수집 제외 |
| 건강·공중보건 | 85,000 | 8.50% | 공공누리 보건자료, 미국 연방기관 작성 public-domain 부분, PMC OA 논문별 CC whitelist | PublicHealth와 수치/위험/권고 검색. 제3자 그림·표와 NC 자료 제외 |
| multi-hop evidence | 85,000 | 8.50% | 서로 독립적으로 허용된 2–3개 문서를 연결해 생성 | Ko-StrategyQA/복합 질의. 각 hop evidence와 정답 가능성을 저장 |
| query style·오탈자·구어체 | 42,500 | 4.25% | 위 승인 pair에서 생성한 짧은 검색어, typo, spacing, paraphrase | 실제 검색 query 분포와 robustness. 원 query와 변환 관계를 저장 |
| multilingual/general replay | 150,000 | 15.00% | source별 permissive/조건부 허용 pair, target benchmark와 비중복 | Qwen의 다국어·STS·classification 회귀 방지 |
| 합계 | **1,000,000** | **100%** |  |  |

이 비율은 성능을 알고 정한 정답이 아니라 사전 등록할 첫 mixture다. `50K → 200K → 500K → 1M` scale curve에서 domain별 marginal gain과 worst-task를 보고 수정한다. 공개 9-task test 점수로 비율을 고르지 않는다.

### 3.3 source와 라이선스 gate

| Source 계열 | 초기 상태 | release candidate 조건 |
|---|---|---|
| 공공누리 제0/1유형 | 조건부 허용 | 파일별 유형, 출처표시 문구, 제공기관, revision을 manifest에 보존. 제2/3/4유형은 NC/ND 조건 때문에 기본 제외 |
| Korean Wikipedia/Wikisource | 조건부 허용 | page revision과 attribution 보존, share-alike/재배포 의무를 dataset card와 배포 방식에 반영 |
| 법제처·정부 API | 조건부 허용 | 법령 본문 자체와 편집·해설·DB 권리를 구분하고 API 이용약관/호출 결과 저장 허용을 확인 |
| OpenDART | 기본 제외 | 구조화 공시의 이용 조건과 제출기업 원문 권리가 승인된 subset만 승격 |
| 미국 연방정부/CDC | 조건부 허용 | 연방정부 직원 작성 public-domain 부분만; logo, 사진, 외부 인용, contractor 작성물 제외 |
| PMC Open Access | 조건부 허용 | OA라고 일괄 허용하지 않고 article별 CC license whitelist. NC/ND는 release 모델에서 기본 제외 |
| `nlpai-lab/ko-triplet-v1.0` | smoke 전용 | 744,862 rows이지만 명시적 dataset license가 없고 upstream이 혼합됨. public weight 학습에는 사용하지 않음 |
| F2/KaLM/Nemotron composite | 연구·비교용 | 상단 license만 믿지 않고 row별 upstream이 release policy를 통과할 때만 재구성하여 사용 |
| Sionic 9와 MTEB Korean task corpus/query/qrel | 평가 전용 | train/dev/test 이름과 무관하게 release-clean 후보의 생성·mining·학습에서 차단 |

이 표는 법률 자문이 아니라 엔지니어링 gate다. `unknown`은 허용이 아니라 보류를 뜻한다. 원문을 번역·요약·OCR하거나 synthetic query를 생성해도 원 source 의무가 사라지지 않는다.

### 3.4 논문들의 데이터에서 가져올 구조

| 선례 | 가져올 구조 | 그대로 가져오지 않는 이유 |
|---|---|---|
| E5/mE5 | semi-structured title-body, QA, citation/parallel pair와 broad replay | Common Crawl/Reddit 대량 scrape는 권리·개인정보·품질 통제가 약함 |
| Qwen3 | task/language/length/difficulty/persona를 명시한 synthetic generation | 150M data와 정확한 filter가 비공개라 재현할 수 없음 |
| F2 | source cap, homogeneous batch, pool 24→sample 7 | MIRACL/MLDR 등 target 관련 source와 composite upstream을 그대로 학습하면 clean claim이 약해짐 |
| Nemotron/NV | positive-relative false-negative filter, 단계별 HN 수 증가 | `.95` 하나를 모든 teacher/domain의 절대 규칙으로 사용하지 않음 |
| Gecko/Gemini | source passage를 자동 positive로 믿지 않고 후보를 teacher가 재라벨 | 폐쇄 LLM/data에 의존하며 exact corpus를 재현할 수 없음 |
| Beyond Hard Negatives | top100 + random100의 넓은 candidate pool과 quantile score coverage | MS MARCO 자체를 한국어 target data로 쓰지 않고 방법만 이전 |
| CausalNeg | 단 하나의 relevance requirement를 위반하는 near-miss와 source-style audit | LLM 생성문을 corpus negative보다 많이 넣으면 shortcut 위험 |
| jina/pplx/BGE-M3 | 실제 1K–4K+ 문서, length bucket, 별도 long phase | 짧은 retrieval 목표까지 처음부터 8K로 학습하면 batch/negative 수가 급감 |

## 4. candidate mining과 label 생성 pipeline

### 4.1 생성 전에 benchmark를 차단한다

1. Sionic 9, MTEB Korean 6개 task의 query, qrel, positive, corpus ID와 원문 fingerprint를 고정한다.
2. source ingest에서 normalized exact hash, URL/title/article ID, 5-gram MinHash, embedding near-duplicate 순으로 제거한다.
3. 차단된 문서에서 query를 생성한 뒤 query만 삭제하는 방식은 허용하지 않는다. source document가 이미 supervision을 누설하기 때문이다.
4. `blind-temporal`은 2026년 이후 또는 snapshot 이후의 새 문서와 사람이 작성한 query로 별도 유지한다.

### 4.2 query와 positive

문서마다 검색창형, 완전한 질문, 조건·예외·수치형, multi-hop형을 생성한다. generator는 query뿐 아니라 answer와 exact evidence span을 출력한다. Gecko/ReasonEmbed의 교훈에 따라 seed document를 무조건 positive로 지정하지 않는다. hybrid retrieval candidate를 teacher가 다시 평가하여 더 적합한 문서가 있으면 multi-positive로 승격하고, seed가 답을 충분히 지지하지 못하면 row를 폐기한다.

### 4.3 candidate pool

각 query에 다음을 합쳐 중복 제거한다.

- BM25 top 100
- pinned base Qwen top 100
- 현재 checkpoint top 100
- same-entity/same-section sibling 20
- domain-random 20
- 허용된 경우 counterfactual generated candidates 1–3

모든 candidate를 곧바로 training negative로 쓰지 않는다. positive와 candidate를 동일 retriever/teacher로 점수화하고 다음 label 중 하나를 부여한다.

- `positive`: query를 충분히 답함
- `partial_positive`: 일부 조건을 만족; graded relevance/KD에는 보존
- `hard_negative`: topical하지만 필수 조건을 만족하지 않음
- `easy_negative`: 명백히 무관
- `ambiguous`: 학습에서 제외하고 audit queue로 이동

### 4.4 false-negative와 score distribution

`020`은 `s_neg < alpha*s_pos`, `alpha ∈ {.90,.95,.98}`를 비교한다. 이는 Nemotron/NV의 positive-aware filter를 옮긴 것이며, 서로 다른 model의 raw cosine을 한 threshold로 섞지 않는다.

teacher KD용 candidate는 top-hard만 남기지 않는다. query별 teacher score를 정규화하고 7개 또는 15개의 evenly spaced quantile anchor에 가장 가까운 서로 다른 candidate를 고른다. positive/partial-positive는 negative bin에서 제거하되 teacher의 원 점수와 grade는 저장한다. 첫 비교는 다음 네 개다.

1. random/in-batch only
2. retriever top-hard
3. positive-aware top-hard
4. positive-aware + teacher-score stratified

### 4.5 generated negative 안전장치

CausalNeg에 따라 generated negative는 전체 explicit negative의 최대 10%로 시작한다. 생성 prompt는 query가 요구하는 entity, time, quantity, jurisdiction, causality 조건을 분해하고 **한 조건만** 변경한다. corpus의 실제 문체를 style reference로 제공하되 개인정보를 prompt로 보내지 않는다.

각 batch/run에서 다음 진단을 남긴다.

- real/generated를 맞히는 source classifier accuracy
- source별 query similarity histogram과 teacher score histogram
- embedding cluster의 source purity
- lexical length, punctuation, heading, disclaimer 빈도
- real-only, generated-only, mixed validation 성능

source classifier가 쉽게 구분하거나 generated cluster가 분리되면 더 많은 생성문을 넣지 않고 style/filter를 수정한다.

## 5. 학습 단계와 현재 experiments 연결

### Stage A — clean short retrieval (`010`, `070`)

- Qwen3-Embedding-8B의 causal/EOS/prompt/L2 contract 고정
- rights-safe 50K pilot 후 300K–500K
- BF16 LoRA r32/r64, all-linear; `tau=.01/.02/.05`
- positive 1, explicit negative 1부터 시작하고 effective query batch 64–128
- 같은 token budget에서 LoRA, top-layer partial, memory-efficient full FT 비교

목적은 새 geometry를 무리하게 만드는 것이 아니라 한국어 query/document boundary를 이동시키면서 원 Qwen 능력을 보존하는 것이다. 현재 288-row smoke는 negative가 너무 쉬워 loss가 즉시 0에 수렴했으므로 품질 근거가 아니다.

### Stage B — negative curriculum (`020`)

- base/current/BM25 hybrid mining
- positive-relative `.90/.95/.98`
- HN 1 → 4 → 7, pool 24 또는 candidate 200에서 매 epoch 재표집
- top-hard 대 score-stratified 비교
- source-homogeneous batching과 synthetic source audit

성공 기준은 train margin 증가가 아니라 clean dev NDCG, false-negative audit, worst-domain 개선이다.

### Stage C — teacher distillation (`030`)

- Qwen3-Reranker-8B가 positive + candidates 100–200개에 연속 점수 부여
- `InfoNCE only`, `filter only`, `InfoNCE + KL`, `InfoNCE + MarginMSE` 비교
- 첫 조합은 KaLM을 참고해 `0.3 CL + 0.7 KL`, CL `tau=.01/.02`, KD `tau=.05`를 **starting point**로 사용
- base Qwen embedding distribution/replay loss로 catastrophic forgetting 감시

KaLM의 coefficient는 0.5B 학생의 값이므로 8B에 그대로 최적이라고 가정하지 않는다. `{0.3,0.5,0.7}` KD weight와 teacher temperature를 clean dev에서 제한적으로 sweep한다.

### Stage D — long-context (`040`)

- 75–80% 512, 15–20% 2K, 3–5% 4K/8K
- evidence head/middle/tail 균등
- 512-only 대 length buckets
- 기본 single-vector InfoNCE와 chunk-local + document-global objective 비교
- short retrieval regression과 tokens/sec/VRAM을 같이 보고

BGE-M3, jina-v5, pplx 모두 position extension만으로 끝나지 않고 실제 long pair를 별도 학습했다. 따라서 32K config 표기만으로 long retrieval 능력을 주장하지 않는다.

### Stage E — specialist와 merge (`050`)

다음 역할별 adapter/checkpoint를 만든다.

- general factoid/QA
- public/PDF/OCR/domain
- legal/health/multi-hop
- long-document
- multilingual/general replay

single best, last-N average, domain-delta average, base interpolation, SLERP를 비교한다. Qwen/Nemotron/Gemini/jina/pplx의 공통 교훈을 사용하되 coefficient는 `dev-clean + blind-temporal + regression`으로 정하고 public 9에서는 고르지 않는다.

### Stage F — backbone (`060`)

Qwen 0.6B/4B/8B, Nemotron 8B, KaLM Gemma12B, Harrier 0.6B/27B의 동일 evaluator·동일 data budget 비교다. 체크할 것은 단순 leaderboard score가 아니라 다음이다.

- 같은 data를 학습했을 때의 delta
- max length별 throughput/VRAM
- instruction sensitivity와 pooling contract
- 가중치·derived model 배포 라이선스
- Korean clean/OOD와 multilingual regression

Harrier와 current KaLM의 recipe가 불투명하므로 이 실험은 architecture/initialization 효과만 측정한다.

### Stage G — update strategy (`070`)

- LoRA r32/r64, DoRA, top 4/8 blocks, full FT를 동일 token budget으로 비교
- full FT는 standard AdamW 1-step feasibility와 state-efficient 방법을 분리
- 품질, peak allocated/reserved VRAM, tokens/s, checkpoint/optimizer size 기록

F2/Nemotron/KaLM은 full FT 선례이고 ReasonEmbed/jina는 LoRA 선례다. 어떤 방식이 낫다는 결론은 같은 clean data와 hard negatives에서의 quality/VRAM/GPU-hour Pareto로만 낸다.

## 6. 모든 논문을 experiments 010–070에 추적한 표

| 근거 | 010 | 020 | 030 | 040 | 050 | 060 | 070 | 070 이후 |
|---|---|---|---|---|---|---|---|---|
| Qwen3 | prompt/EOS/InfoNCE | FN mask, q-q/d-d ablation | base replay | 32K는 평가만 | SLERP | size family | full/LoRA 지원 | MRL release dimension |
| F2LLM | — | homogeneous batch, HN pool | — | length 설정 참고 | — | Qwen-chat 대비 | full FT 상한 | source cap + MRL |
| Nemotron | — | `.95*s_pos`, 2-stage HN | miner ensemble 참고 | — | 6-checkpoint average | backbone | full FT 비용 | public recipe scale-up |
| KaLM V2 | — | focal/online HN | CL+KL teacher distribution | — | — | 0.5B 비교 | full FT 선례 | MRL/focal production |
| current KaLM 12B | — | 공개 7 HN data 형식만 | 정확 recipe 미상 | 32K eval | 미상 | black-box baseline | feasibility | 공개되지 않은 항목 추정 금지 |
| Harrier | — | 미상 | 미상 | 32K eval | 미상 | black-box baseline | feasibility | recipe 공개 전 복제 금지 |
| E5/mE5 | clean pair | in-batch→mined HN | CE soft-label KD | — | — | mE5 baseline | — | multilingual replay |
| BGE-M3 | — | — | multi-head self-KD 참고 | long bucket/MultiLongDoc | — | hybrid baseline | — | optional sparse/multi-vector |
| Gecko/Gemini | query-positive relabel | LLM grader/FN mask | teacher labeling | — | model soup | closed baseline | — | rights-safe data factory |
| NV-Retriever | — | positive-aware mining | — | — | — | — | — | iterative re-mining |
| ReasonEmbed | LoRA 선례 | source-excluded mining | reasoning label | — | specialist only | backbone ablation | r64 | reasoning-only adapter |
| Beyond HN | — | score pool 구축 | stratified KL/MarginMSE | — | — | — | — | candidate cache service |
| CausalNeg | — | generated HN/source audit | entropy reg. 후보 | — | — | — | — | counterfactual generator |
| jina-v5 | — | hard triples | vector KD/GOR | 별도 1K–4K phase | adapter average | student baseline | task LoRA | compressed release |
| pplx-embed | — | FN threshold mask | — | local+global long loss | branch SLERP | diffusion backbone | full CPT 비용 | CPT go/no-go 이후 |
| Search-R3/LaSER | — | — | retrieval-linked reasoning | — | specialist merge | — | — | latency 허용 제품만 |
| Querit | — | score bin | score-gap KD | — | sequential SLERP | — | — | private-data 재현 제한 |
| BITEMBED/TALAS | — | — | layer/relation KD | — | — | small student | — | 품질 확정 뒤 압축 |

### 070 이후 제안 stage

아직 실험 폴더를 만들기 전의 예약 설계다. 앞 단계가 통과할 때만 승격한다.

| Stage | 목적 | 진입 조건 |
|---|---|---|
| `080_clean_data_factory` | 1M rights-safe manifest, benchmark blocklist, query/evidence verifier | 50K data audit pass |
| `090_score_distribution_kd` | reranker cache, quantile sampler, KL/MarginMSE | `020`이 top-hard 대비 clean/OOD 이득 |
| `100_multitask_mrl` | retrieval/STS/classification replay와 256–4096-d MRL | Sionic 9 이득과 broad regression 동시 통과 |
| `110_release_soup` | role별 adapter/checkpoint clean merge | specialist 2개 이상이 서로 다른 slice에서 우월 |
| `120_compression` | 0.6B/4B student KD, INT8/binary ablation | 8B teacher 품질 후보 확정 |
| `130_cpt_ablation` | 작은 Korean domain CPT 또는 bidirectional conversion | tokenizer/long/domain probe가 contrastive tuning으로 해결되지 않는 결함 확인 |

## 7. 채택할 것과 명시적으로 기각할 것

### 즉시 채택

- revision이 고정된 Qwen3-Embedding-8B와 원 prompt/pooling contract
- rights-safe Korean task-aligned pair/triple과 15% replay
- source-homogeneous batching과 source cap
- hybrid candidate pool, positive-aware filtering, score-stratified sampling
- hard-label InfoNCE + continuous teacher-distribution KL
- general/domain/long specialist와 clean checkpoint soup
- LoRA/full FT의 동일 token-budget 비교

### 조건부 채택

- MRL: full-dimension quality가 먼저 통과한 뒤
- long local/global loss: MLDR와 real long slice가 병목일 때
- counterfactual generated negatives: source shortcut audit가 통과할 때
- reasoning weighting/latent reasoning: general retrieval이 아니라 reasoning slice에서만
- bidirectional attention/CPT: architecture ablation과 충분한 raw-text budget이 있을 때

### 첫 예산에서 기각

- target benchmark의 train/dev/test 또는 near-duplicate로 학습
- `ko-triplet-v1.0`처럼 license가 비어 있는 composite로 public model 배포
- top-1 dense result를 검증 없이 negative로 사용
- 생성 negative를 real corpus보다 많이 섞기
- 공개되지 않은 Harrier/KaLM-Gemma recipe를 점수만 보고 추정
- 250B-token broad CPT, generic RLVR 선행, 이유 없는 32K 확장
- public Sionic 9 점수로 data mix, merge coefficient, checkpoint를 반복 선택

## 8. promotion과 실패 판정

한 run이 다음 단계로 올라가려면 모두 만족해야 한다.

1. `dev-clean` retrieval macro와 worst-domain이 base보다 개선
2. `blind-temporal`에서 개선 방향 유지
3. broad Korean, multilingual/STS/classification regression이 사전 한도 이내
4. exact/MinHash/embedding benchmark overlap audit 통과
5. data manifest의 license/attribution/transform provenance 누락 0건
6. max length별 peak VRAM, throughput, embedding dimension/storage 보고
7. Sionic 9와 MTEB Korean은 최종 확인용이며 hyperparameter selection에 사용하지 않음

실패도 결과로 남긴다. 특히 training loss가 빨리 0이 되는 것은 성공이 아니라 negative가 너무 쉽다는 신호일 수 있다. full FT가 더 높은 train margin을 내더라도 clean/OOD나 regression을 잃으면 승격하지 않는다.

## 9. 이 문서가 내리는 최종 recipe

첫 performance candidate는 다음 조합이다.

```text
Base          Qwen/Qwen3-Embedding-8B, pinned revision
Update        LoRA r64 우선; top-layer partial과 full FT를 070에서 비교
Data A        rights-safe Korean 300K–500K + multilingual/general replay 15%
Data B        hybrid-mined, teacher-verified 100K–200K
Negatives     pool 24–200, positive-aware filter, score-quantile 4/7 candidates
Teacher       Qwen3-Reranker-8B continuous scores
Loss          InfoNCE(tau .01/.02) + KL(tau .05), KD weight .3/.5/.7 ablation
Length        512 중심 → 2K/4K specialist를 별도 학습
Selection     dev-clean + blind-temporal + regression
Merge         general/domain/long adapter average 또는 SLERP
Release       rights manifest와 benchmark decontamination을 통과한 weight만 public
```

이 recipe가 Sionic 9를 넘지 못하면 순서는 `더 큰 clean data → online re-mining → full/partial update → MRL/multi-task → 작은 CPT ablation`이다. 바로 broad CPT로 뛰지 않는다. 공개 9개 평균을 넘는 것과 multilingual/한국어 범용 SOTA를 주장하는 것은 별도 목표이며, 후자는 공식 MTEB와 clean comprehensive suite까지 동시에 통과해야 한다.

## 10. 원문 목록

- [Qwen3 Embedding](https://arxiv.org/abs/2506.05176), [official repository](https://github.com/QwenLM/Qwen3-Embedding)
- [F2LLM-v2](https://arxiv.org/abs/2603.19223), [F2LLM code](https://github.com/codefuse-ai/CodeFuse-Embeddings/tree/main/F2LLM), [F2LLM-v2 dataset](https://huggingface.co/datasets/codefuse-ai/F2LLM-v2)
- [Llama-Embed-Nemotron-8B](https://arxiv.org/abs/2511.07025), [official model card](https://huggingface.co/nvidia/llama-embed-nemotron-8b)
- [KaLM-Embedding-V2](https://arxiv.org/abs/2506.20923), [current Gemma3-12B model card](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511)
- [Harrier OSS v1 27B model card](https://huggingface.co/microsoft/harrier-oss-v1-27b)
- [E5](https://arxiv.org/abs/2212.03533), [multilingual E5](https://arxiv.org/abs/2402.05672)
- [BGE-M3](https://arxiv.org/abs/2402.03216), [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [NV-Retriever](https://arxiv.org/abs/2407.15831)
- [Gecko](https://arxiv.org/abs/2403.20327), [Gemini Embedding](https://arxiv.org/abs/2503.07891)
- [ReasonEmbed](https://arxiv.org/abs/2510.08252)
- [Beyond Hard Negatives](https://arxiv.org/abs/2604.04734)
- [CausalNeg](https://arxiv.org/abs/2606.01304)
- [jina-embeddings-v5-text](https://arxiv.org/abs/2602.15547)
- [pplx-embed](https://arxiv.org/abs/2602.11151)
- [Do Reasoning Models Enhance Embedding Models?](https://arxiv.org/abs/2601.21192)
- [Search-R3](https://arxiv.org/abs/2510.07048), [LaSER](https://arxiv.org/abs/2603.01425)
- [Querit-Reranker](https://arxiv.org/abs/2606.19037), [BITEMBED](https://arxiv.org/abs/2606.25674), [TALAS](https://aclanthology.org/2026.acl-long.1509/)
