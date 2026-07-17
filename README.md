# Korean Embedding Lab

한국어 검색 임베딩 모델을 연구하되 `sionic-ai/comsat-embed-ko-8b-preview`나 Sionic 한
보드를 넘는 데서 멈추지 않고, **비상업 연구 자산까지 허용한 성능 최우선 모델**과 그
방법을 재현 가능한 clean-release 모델로 다시 만드는 두 track의 작업 공간입니다.
Korean retrieval·broad text·다국어·긴 문맥/context·noise 강건성을 함께 보며, 실질적으로
미미한 차이는 near-tie로 취급하고 실패 축과 오염을 숨기지 않습니다.

기준일: **2026-07-18 (Asia/Seoul)**

## 한 줄 결론

- 최우선 목표는 **비상업 연구 자산까지 사용한 한국어 embedding 최고 성능**이다. 같은 방법을 권리가 확인된 데이터로 재학습하는 clean-release track은 그 다음이다.
- 2026-07-17 19:06 KST에 Qwen 200K가 `1875/3123`에서 외부 종료됐다. 마지막 exact-resumable checkpoint는 `1750`이다. 2026-07-18 07:33 KST에 `checkpoint-1750`에서 exact resume를 재기동했고(resume validation pass, matched_sdpa 재선택), frontier queue·checkpoint watcher·storage watchdog을 함께 복구했다. 복구 세부와 당일 문헌 점검은 [docs/37](docs/37_RESUME_RECOVERY_AND_LITERATURE_2026-07-18.md)에 있다. 2026-07-18 00:26 KST에 `Nemotron-3-Embed-8B-BF16@2b29550c`의 고정 Sionic 9 전체 평가를 완료했으며 macro NDCG@10은 **`0.732212`**로 Comsat `0.7930`보다 `-0.060788`이다. 원본 그대로의 교체는 탈락이다. 태스크별 값은 MIRACL `.64994`, Mr.TyDi `.49324`, MLDR `.33463`, AutoRAG `.88550`, Ko-StrategyQA `.79387`, PublicHealthQA `.82497`, Belebele `.95209`, SQuADKorV1 `.92018`, LawIRKo `.73549`다. MLDR nominal batch/cache key 64는 유지하면서 OOM 1회 뒤 실제 encoder microbatch 32로 완료했다.
- 현재 valid performance candidate는 **0개**다. 자동 chain은 Nemotron/Qwen/Comsat의 독립 legal 10K·finance/knowledge 1.9K 비교를 진행 중이다. raw deficit이 `.020`을 초과하고 multidomain가 guard에서 탈락해 Nemotron은 성능상 채택 조건을 충족하지 못했다. `outputs/evaluation/nemotron3-base-decision.json`의 `decision`는 `resume_qwen_checkpoint_1750_and_reselect`로 고정되어 있어, 현재 베이스 재개는 Qwen `checkpoint-1750`이 기준이다. clean 비교 결과는 Nemotron을 teacher/miner로 쓸지 판단하는 근거로 보존한다.
- 공개 학습용 첫 권리안전 artifact는 [`LLM-OS-Models2/ko-legal-embedding-training-v1`](https://huggingface.co/datasets/LLM-OS-Models2/ko-legal-embedding-training-v1)이다. 한국 법령·행정규칙·판례·자치법규 source-native pair 250,000행이며 모든 행의 source/revision/license/redistribution 권리를 검증했고, 고정 benchmark blocklist exact overlap은 query/evaluation/corpus 모두 0이다. public immutable commit은 `faf431f53a9d8e8bbfa4d57903012a5d786f8716`이다.
- Nemotron-3 공개 적응은 decision gate가 허용할 때만 `scripts/train_nemotron3_public_lora.py`가 실행한다. base architecture/mean-pooling/public manifest SHA를 검증하고 PEFT LoRA·cached all-negative loss·optimizer checkpoint를 재개한다. source JSONL에 저장된 기존 `Instruct/Query`를 한 번 제거한 뒤 학습 query에만 Sionic 고정 비교와 동일한 Qwen 검색 지시문을 붙이고 positive/negative는 무접두 source-native text로 유지해 이중 prompt 없이 train/eval 입력 계약을 맞춘다. contract-only와 단위 검증은 통과했다.
- 현재 자동 chain은 `run_nemotron3_base_decision.sh` → `run_nemotron3_post_decision_probe.sh` → `run_nemotron3_public_pipeline.sh` → `run_nemotron3_post_training_release.sh` 순서로 대기한다. Nemotron gate를 통과하면 저장된 기존 query instruction을 제거하고 학습·평가와 같은 query-only 고정 prompt로 public HN을 mining해 dataset 배포, 300-step LoRA, public checkpoint watcher, winner 병합·전체 final gate·public 최종 모델 검증까지 이어진다. gate가 Qwen을 선택하면 Nemotron mutation/release를 건너뛰고 Qwen exact-resume 대상으로 남긴다.
- 위 chain은 아직 **검증된 public adapter checkpoint에 도달하지 않았다**. decision이 Nemotron 적응을 허용하는 경우에만 same-step heldout loss로 checkpoint를 고르고, masked-mean/normalize/prompt 계약을 보존해 병합한 뒤 legal·multidomain guard → Sionic 9 `>0.7930` → 공식 Korean 6을 통과한 단일 모델을 public Hub repo에 visibility/file-set/LFS SHA까지 재검증한다.
- `scripts/select_nemotron3_public_checkpoint.py`는 예정된 step 50/100/150/200/250/300 모두의 adapter/config/trainer/optimizer/scheduler 완결성과 base·public training-manifest SHA를 확인하고, public benchmark가 아닌 같은 step의 독립 512 heldout `eval_loss` 최솟값만 선택한다. 하나라도 없거나 non-finite면 병합으로 진행하지 않는다.
- `scripts/merge_nemotron3_adapter.py`는 selector가 고른 exact adapter SHA만 pinned Nemotron base에 PEFT safe-merge한다. Qwen용 last-token 병합기를 재사용하지 않고 bidirectional `Ministral3Model`·4,096차원 masked mean·L2 normalize·query-only 고정 prompt를 검증하며, adapter 적용 전후와 병합 후 row cosine/pairwise-score parity가 통과한 경우에만 sibling staging을 최종 디렉터리로 원자 rename한다.
- `scripts/gate_nemotron3_final_candidate.py`는 merged weight SHA로 legal·multidomain·Sionic summary의 model/revision을 다시 묶고, Qwen/Comsat 최고 clean reference 대비 legal/multidomain macro `-0.010`, finance/knowledge 각각 `-0.015` guard와 Sionic macro strict `>0.7930`을 모두 통과시킨다. public score는 checkpoint 선택에 쓰지 않았음을 report에 고정한다.
- 최종 `publish_best_embedding_model.py`는 pinned upstream 계보가 Nemotron이면 `masked_mean+Ministral3Model`, Qwen이면 `last_token`을 각각 fail-closed로 검증하고 모델 카드에도 실제 pooling 계약을 기록한다. 따라서 Nemotron winner를 Qwen 계약으로 잘못 포장하거나 반대 경우를 허용하지 않는다.
- `scripts/approve_nemotron3_public_release.py`는 사용자의 2026-07-17 “모델도 public” 지시를 authorization basis로 남기되, exact merged weight SHA·rights-safe manifest SHA·final gate·Sionic/공식6/comprehensive/clean/robustness summary SHA가 모두 맞을 때만 특정 public repo용 approval을 원자적으로 만든다.
- Hugging Face 산출물은 **학습/파생 데이터, 중간 adapter checkpoint, 병합 모델, 최종 모델 모두 public이 기본**이다. 공개 repo에는 source revision·license·변환·dedup·benchmark overlap·모델 계보를 카드와 manifest로 함께 싣고 업로드 전후 visibility/file-set/SHA를 재검증한다. 권리 불명확·재배포 금지 source가 섞인 기존 performance track은 공개하지 않고 rights-safe 데이터로 다시 만든다. 점수 선택 오염을 막기 위한 고정 소규모 holdout만 비공개 예외다.
- 최종 mined manifest는 `training_track=rights-safe-release`, `use_policy=public-redistributable-training`을 명시한다. 이 필드와 row별 권리·visibility·release blocker·SHA가 모두 맞아야 public checkpoint와 최종 model publisher가 받는다.
- 최종 모델 카드는 이 manifest의 `artifact_id`를 우선 사용해 실제 학습 artifact인 `LLM-OS-Models2/ko-legal-embedding-training-nemotron3-hn-v1`만 연결한다. 과거 Qwen target-adapted 데이터 repo를 문자열 추정으로 잘못 계승하지 않는다.
- 법률 공개 source 적응 표기는 canonical `target-adapted-legal-public-source`로 고정해 모델 카드가 LawIRKo/AutoRAG 관련 점수를 clean zero-shot으로 오인하지 않도록 경고한다.
- 본선은 `Qwen clean lineage`와 `Comsat Korean warm-start lineage`를 같은 200K 조건으로 비교하고, 승자 계보의 원본 base에서 동일 200K/token budget인 last4 partial-full challenger를 거친 뒤, 1M general → current-student wide ANN pool → Qwen reranker score-quantile KD/queue A/B → 400K target → 모든 stage의 single-best 대 동일-trajectory last-available-5 FP32 평균을 최종 legal/multidomain/robustness gate로 재선택하는 순서다.
- checkpoint는 public score가 아니라 Grade-I legal 10K guard와 고정 비공개 finance/knowledge 1.9K에서 먼저 고릅니다. 기존 512 validation의 200K 전량 중복을 발견해 Qwen/Comsat 200K는 양쪽 모두 보존된 모든 checkpoint를 동일한 legal·multidomain·noise 조건으로 다시 고릅니다. 이후 1M/KD/전문가 run은 source-document-held-out 512 내부 신호와 last-available-5 FP32 평균을 같은 최종 gate에서 비교합니다. Sionic 9와 공식 Korean 6은 local winner에 final-once로 실행합니다.
- Comsat의 `1M+`는 문서나 토큰이 아니라 출처와 형식이 공개되지 않은 **Korean training examples**입니다.
- Comsat의 `0.7930`은 일반 MTEB SOTA가 아니라, 자체 선택한 한국어 retrieval 9종의 macro `NDCG@10`입니다. Qwen3-Embedding-8B 대비 차이는 `+0.0105`입니다.
- 가장 직접적인 학습법은 raw-text LM CPT가 아니라 `query / positive / hard negatives`를 이용한 **continued contrastive fine-tuning(InfoNCE)** 입니다.
- 현재 첫 실험은 `Qwen/Qwen3-Embedding-8B + BF16 LoRA + InfoNCE`로 시작합니다. 공개 평가 9종은 학습·negative mining·checkpoint selection에서 차단합니다.

새 성능 우선 판단과 재복구 순서는
[2026-07-17 frontier plan](docs/34_PERFORMANCE_FIRST_FRONTIER_PLAN_2026-07-17.md), 기존
전체 배경은 [요약 문서](docs/00_EXECUTIVE_SUMMARY.md)부터 읽으면 됩니다.
새 Nemotron-3 비교와 중단 재개 명령은
[Nemotron-3 한국어 base 결정](docs/36_NEMOTRON3_KOREAN_BASE_DECISION_2026-07-17.md)에
고정합니다.

## 성능 보드 3종

세 표는 평가 대상과 집계법이 달라서 **서로 평균내지 않습니다**. 공식 Korean 보드는 한국어 전반의 6-task Borda/평균, Sionic 보드는 검색 9종 NDCG@10, 종합 보드는 오염을 차단한 자체 holdout과 효율을 봅니다. `—`는 0점이 아니라 미제출·미측정입니다.

### 1. 공식 MTEB Korean v1

공식 보드의 값을 그대로 신뢰하는 snapshot입니다. Comsat은 공식 제출 행이 없어서 같은 MTEB `2.18.0`/6-task protocol로 로컬 재현 중이며, 완료 전에는 순위를 부여하지 않습니다.

| 구분 | 모델 | Borda | Mean(Task) | Mean(Type) | Retrieval | Zero-shot | 출처 |
|---|---|---:|---:|---:|---:|---:|---|
| 공식 #1 | `codefuse-ai/F2LLM-v2-8B` | 1 | **75.11** | 72.68 | 73.42 | 66% | MTEB live |
| 공식 #2 | `codefuse-ai/F2LLM-v2-14B` | 2 | 74.85 | 72.43 | 72.33 | 66% | MTEB live |
| 공식 #3 | `SamilPwC-AXNode-GenAI/PwC-Embedding_expr` | 3 | **77.01** | **75.92** | 72.15 | 16% | MTEB live |
| 로컬 재현 | `sionic-ai/comsat-embed-ko-8b-preview` | **6 if inserted** | **73.32** | **70.06** | **76.77** | 별도 감사 | 동일 protocol, 6/6 완료 |
| 로컬 측정 대기 | `Qwen/Qwen3-Embedding-8B` | — | — | — | — | registry 감사 | registered-loader 6-task 자동 queue |
| 우리 모델 | smoke LoRA r32 | — | 미측정 | 미측정 | 미측정 | 100% | pipeline 검증 전용 |

`F2LLM-v2-8B`는 Borda 1위지만 단순 평균 1위는 PwC입니다. PwC는 6개 중 5개 평가 계열을 학습한 in-domain specialist이므로 zero-shot 일반화와 구분합니다. 전체 task별 값과 방법론 감사는 [Korean leaderboard 문서](docs/08_KOREAN_LEADERBOARD_AND_F2LLM.md)에 있습니다.

Comsat 행은 공식 제출이 아니라 pinned local reproduction을 2026-07-12 live 137-row
board에 가상 삽입한 값이다. 공식 rank 재계산은 137/137 일치했고, complete official
row는 101개였다.

### 2. Sionic Korean retrieval 9종

9개 retrieval task의 macro NDCG@10입니다. `AutoRAG`의 `실측` 표시는 같은 고정 evaluator로 full corpus를 직접 실행한 값이고, Avg/나머지 값은 Sionic 카드의 공개 표입니다.

| 모델 | 9-task Avg | AutoRAG NDCG@10 | 측정 상태 | Comsat 대비 Avg |
|---|---:|---:|---|---:|
| `sionic-ai/comsat-embed-ko-8b-preview` | **0.7930** | **0.85261** 실측 | 9개 카드 + 1개 canonical 재현 | 기준 |
| `Qwen/Qwen3-Embedding-8B` | 0.7825 | 0.82442 실측 | 9개 카드 + 1개 canonical 재현 | -0.0105 |
| `codefuse-ai/F2LLM-v2-8B` | 0.7621 | 0.76789 실측 | 9개 카드 + 1개 canonical 재현 | -0.0309 |
| `nvidia/Nemotron-3-Embed-8B-BF16@2b29550c` | 0.732212 실측 | 0.88550 실측 | 9개 canonical 전체 직접 측정 | -0.060788 |
| `SamilPwC-AXNode-GenAI/PwC-Embedding_expr` | — | 0.78473 실측 | AutoRAG native max 512 | — |
| 우리 smoke LoRA r32 | — | 미측정 | 성능 주장 금지 | — |
| 우리 공개 후보 목표 | **> 0.7930** | 회귀 없음 | 9개 전부 직접 측정 | **> 0** |

현재 근거로 Comsat은 “별로인 모델”이 아니라 Qwen 대비 한국어 retrieval에 잘 특화된 모델입니다. 다만 선택된 9개 task만으로 일반 한국어·다국어 SOTA라고 할 수는 없습니다. 위 AutoRAG canonical 값은 두 모델 모두 BF16, FA2, batch 192, max length 8192로 다시 잰 값입니다. 과거 batch-2 값(Qwen `0.82765`, Comsat `0.85222`)과 섞어 비교하지 않습니다. raw run과 revision은 [평가 로그](docs/09_EVALUATION_RESULTS.md)에 기록합니다.

### Nemotron-3 한국어 평가 정리(판단용)

Nemotron raw 점수는 Sionic 9에서만 봤을 때 목표 `0.7930`에서 큼직하게 밀립니다 (`0.732212`, -0.060788).
legal/multidomain 보조 guard를 기준으로 보면:

| 모델 | Legal 10K NDCG@10 | Finance NDCG@10 | Knowledge NDCG@10 | multidomain macro | Sionic9 macro |
|---|---:|---:|---:|---:|---:|
| `nvidia/Nemotron-3-Embed-8B-BF16` | **0.982399** | **0.858991** | **0.645951** | **0.752471** | **0.732212** |
| `Qwen/Qwen3-Embedding-8B` | 0.978809 | 0.872074 | 0.697344 | 0.784709 | 0.7825 |
| `sionic-ai/comsat-embed-ko-8b-preview` | 0.981363 | 0.875759 | 0.708021 | 0.791890 | 0.7930 |

Nemotron은 legal에서 Qwen 대비 +0.0036 정도 우세했지만, finance/knowledge/전체 multidomain에서 함께 떨어져 guard를 통과하지 못했습니다.
그래서 현재 베이스선은 `Qwen checkpoint-1750` 재개 + `reselect` 경로가 유효합니다.

### 3. Clean Korean 종합 보드

우리의 실제 모델 선택 보드입니다. public leaderboard test를 보며 튜닝하지 않고, 데이터 provenance와 중복 차단이 확인된 holdout에서만 checkpoint를 고릅니다. 첫 10K 법률 source-document-held-out set은 build·독립 검증·공개를 완료했고 baseline/model 수치는 후속 queue에서 채웁니다.

| 모델 | Clean retrieval | Broad semantic | Long-context | Noise/OCR robustness | Peak VRAM | 상태 |
|---|---:|---:|---:|---:|---:|---|
| Qwen3-Embedding-8B | 예정 | 예정 | 예정 | 예정 | 예정 | 기준선 |
| Comsat-embed-ko-8b-preview | 예정 | 예정 | 예정 | 예정 | 예정 | 비교군 |
| 우리 smoke LoRA r32 | 평가 제외 | 평가 제외 | 평가 제외 | 평가 제외 | **17.07 GiB** 학습 | pipeline-only |
| 우리 release candidate | 예정 | 예정 | 예정 | 예정 | 예정 | 권리 확인 데이터로 학습 예정 |

종합 보드는 clean retrieval, STS/분류, 긴 문맥의 evidence 위치, OCR·띄어쓰기·질의체 변화, 처리량·VRAM·차원/저장비용을 각각 보고합니다. 설계와 승격 기준은 [종합 평가 설계](docs/10_COMPREHENSIVE_SUITE.md)에 고정합니다.
현재 candidate 상태, Grade-I-not-Z 선택, public final-once와 7-task/414-subset diagnostic의
정확한 역할은 [종합 최고 모델 선택·평가 계약](docs/33_COMPREHENSIVE_SELECTION_AND_EVALUATION.md)에 고정합니다.

#### Clean 법률 source-held-out 10K 실측

<!-- CLEAN_LEGAL_RESULTS_START -->
아직 완료된 clean 법률 baseline이 없습니다.
<!-- CLEAN_LEGAL_RESULTS_END -->

### 자동 캠페인 실측 결과

아래 block은 최종 local winner가 Sionic 9와 공식 Korean 6을 final-once로 끝낸 뒤
자동 갱신·push한다. 중간 stage의 public 평가는 기본적으로 비활성화한다.

<!-- CAMPAIGN_RESULTS_START -->
아직 완료된 성능 후보가 없습니다.
<!-- CAMPAIGN_RESULTS_END -->

## 현재 상태

> **재시작 정정(2026-07-17):** 아래 dataset/card와 과거 run 기록은 원격 공개 artifact와
> 역사적 실측을 뜻한다. 재시작 직후에는 `data/`, `outputs/`, 기존 `.venv-*`, model cache가
> 없었으나, 현재 submodule 5개, 학습/검증 dataset 15개, comprehensive text용 dataset
> 13개, core/teacher 4개와 외부 비교 모델 5개 cache, H100 학습 환경을 NFS에 exact 복원했다.
> valid candidate는 아직 0이다. Qwen 200K는 step 1875에서 중단됐다. Nemotron-3
> full Sionic9은 `0.732212`로 완료됐고 legal·multidomain base decision이 active다.
> 정확한 재개 지점과 명령은 docs/36에 있다.
> cache/env/data/checkpoint는 모두 `/home/ubuntu/data/Embedding`의 NFS 아래에 둔다.

| 항목 | 상태 | 위치 |
|---|---|---|
| Hugging Face 새 publish namespace | `LLM-OS-Models2` private model repo 생성+README write 실검증 완료; 기존 `LLM-OS-Models`는 source read-only | [`embedding-upload-permission-test-20260717`](https://huggingface.co/LLM-OS-Models2/embedding-upload-permission-test-20260717) |
| Qwen3-Embedding 공식 저장소 | pinned submodule 복원 완료 (`44548aa5`) | [`Qwen3-Embedding/`](Qwen3-Embedding/) |
| 공식 후속학습 프레임워크 `ms-swift` | pinned submodule `3d61b931`, NFS `.venv-train-fa2`, CUDA 12.6/PyTorch 2.5 import+8B backward 통과 | [`third_party/ms-swift/`](third_party/ms-swift/) |
| MTEB/FAISS 평가 환경 | NFS `.venv-mteb` 복원; MTEB 2.18.0, FAISS 1.14.3, NumPy 1.26.4, Transformers 5.12.1 import gate·전체 test 225/225 통과 | [`bootstrap_mteb_env.sh`](scripts/bootstrap_mteb_env.sh) |
| 상위 비교 모델 local cache | F2 8B, PwC, Harrier 27B, KaLM 12B, Nemotron 8B exact revision 익명 복구·hard-offline config/tokenizer load 완료 | [상위 모델 평가 매트릭스](docs/20_TOP_MODEL_LOCAL_EVAL_MATRIX.md) |
| Sionic 벤치마크 감사 | 1차 완료 | [docs/02_COMSAT_AUDIT.md](docs/02_COMSAT_AUDIT.md) |
| 2026-07 라이브 MTEB 및 상위 모델 감사 | 완료, 새 결과는 날짜 고정 갱신 | [docs/03_SOTA_MODELS_2026-07.md](docs/03_SOTA_MODELS_2026-07.md) |
| 외부 상위 모델·종합 평가 자산 복구 | F2LLM-v2-8B, PwC, Harrier 27B, KaLM 12B, Nemotron 8B exact revision과 comprehensive text용 13개 dataset snapshot을 `token=False`로 복구하고 hard local-only 재검증. 총 HF cache 147GB; 48TB NFS에 저장. frontier 종료 뒤 7모델 Sionic 동등 비교 queue와 별도 storage watchdog 예약 | [상위 모델 평가 매트릭스](docs/20_TOP_MODEL_LOCAL_EVAL_MATRIX.md) |
| 데이터 manifest / 오염 차단 | 15/15 task exact SHA-256 blocklist 빌드·공개 완료 | [`LLM-OS-Models/korean-embedding-benchmark-blocklist-v1`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-benchmark-blocklist-v1) |
| 100만 행 공개 가능 데이터 공장 | source·수량·검수 gate 설계 완료 | [docs/13_RIGHTS_SAFE_DATA_FACTORY.md](docs/13_RIGHTS_SAFE_DATA_FACTORY.md) |
| Legalize-KR 데이터 | NFS exact HEAD 4개·Markdown 312,581개 재검증 완료, 2,756,363 source-native 후보 추출 가능 | [docs/17_LEGAL_AND_KO_DATA_SOURCE_AUDIT.md](docs/17_LEGAL_AND_KO_DATA_SOURCE_AUDIT.md) |
| Ko-triplet exhaustive HN 10K/512 | Qwen3 exact dense top-24→HN4, ratio .95; train/validation 15-task exact overlap 모두 0, manifest/audit 공개 | [`LLM-OS-Models/korean-embedding-ko-triplet-hn-pilot-10k`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-ko-triplet-hn-pilot-10k/tree/0865276985dd2eae5efec33a4fa181ee3086bd5f) |
| 성능 우선 50K 데이터 | raw/ordered 모두 eval-query hash 4개 확인; diagnostic 전용으로 카드 경고·감사 공개, 대표 모델 선택 금지 | [`LLM-OS-Models/korean-embedding-performance-v1-pilot-50k`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-performance-v1-pilot-50k/tree/77ed4e8d30c89722b262d500f38b4818b359eaf4) |
| 성능 우선 200K 데이터 | critical eval-query row 12개 교체, exact 199,904-row order, query/eval critical overlap 0·품질/overlap 감사 공개 | [`LLM-OS-Models/korean-embedding-performance-v1-ablation-200k`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-performance-v1-ablation-200k/tree/f605128d3233e7cc488dc741b8f2af9ecf68b6fa) |
| SQuADKorV1 train-family 60K | 원본 KorQuAD train 질문→문맥 60K; 평가 query/evaluation-text match 0, Wikipedia shared eval-corpus 6,426 고유 hash 공개 | [`LLM-OS-Models/korean-embedding-performance-v1-sionic-squad-train-60k`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-performance-v1-sionic-squad-train-60k/tree/8fbc6d6d5c93c3493456079d930921ac90ec6801) |
| PublicHealth health-domain 100K | F2 medical QA/instruction/flashcard 100K; critical eval-text overlap 0, PublicHealthQA exact overlap 0·공개 완료 | [`LLM-OS-Models/korean-embedding-performance-v1-sionic-health-100k`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-performance-v1-sionic-health-100k/tree/5fc4bb817f6970a710be53376f35e0225201d2e2) |
| AutoRAG domain 100K | F2 finance/banking/commerce/legal 100K; critical eval-text 0, AutoRAG query/corpus exact overlap 0·공개 완료 | [`LLM-OS-Models/korean-embedding-performance-v1-sionic-autorag-100k`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-performance-v1-sionic-autorag-100k/tree/9140e9e02bb3f40ac1c22a6e595d58208770f696) |
| MIRACL·MrTidy·MLDR train-family 4,146 | F2 공개 train-family lossless 추출; critical eval-query 0, shared corpus 13,973; 2K HN7 specialist queue 연결 | [`LLM-OS-Models/korean-embedding-performance-v1-sionic-retrieval-train-family-4146`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-performance-v1-sionic-retrieval-train-family-4146/tree/c9513a66ad64e5eab586969f6fdde7f9c8abd922) |
| Sionic combined target 400K | SQuAD/health/AutoRAG 각 10% + retrieval-family 1.032% + legal 15% + general 53.968%; multidomain audit queue 구현 | [combined 모델 설계](docs/28_SIONIC_COMBINED_TARGET_MODEL.md) |
| 법률 source-native 250K | 4개 source 균형 shard + bootstrap 한계·질의 분포 전수 감사/카드 공개 | [`LLM-OS-Models/korean-legal-retrieval-source-native-250k`](https://huggingface.co/datasets/LLM-OS-Models/korean-legal-retrieval-source-native-250k/tree/ec2f09a220dc5aa326c5d63b8e49adbf3a5524bc) |
| 성능 우선 1M 데이터 | critical row 2,839 교체, exact 999,936-row order, final critical overlap 0·raw/ordered 감사 공개 | [`LLM-OS-Models/korean-embedding-performance-v1-performance-1m`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-performance-v1-performance-1m/tree/5a2a3ab7f0928c6570929cc231eaefdd3fa203e1) |
| derived dataset publication | 새 1M HN7/KD/전문가/법률/combined curriculum은 `LLM-OS-Models2`만 허용. 요청 visibility, immutable 40-hex commit, 전체 remote file allowlist, 모든 LFS SHA/size와 metadata download SHA를 다시 검증하고 source가 upload 중 바뀌지 않았을 때만 완료 | [`publish_derived_training_dataset.py`](scripts/publish_derived_training_dataset.py) |
| 평가 오염 방지 blocklist | Sionic 9 + 공식 Korean 6, 원문 없는 SHA-256 547MB | [`LLM-OS-Models/korean-embedding-benchmark-blocklist-v1`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-benchmark-blocklist-v1) |
| Clean 법률 retrieval 10K v1 | training document overlap 0, benchmark exact overlap 0이나 다른 원문 문서의 동일 법률 text 98행이 legal 250K와 exact match; 역사 기록 전용, 모델 선택 금지 | [`LLM-OS-Models/korean-legal-source-heldout-retrieval-v1`](https://huggingface.co/datasets/LLM-OS-Models/korean-legal-source-heldout-retrieval-v1/tree/ee1300f04ea03d66bb51e23bbbda34376fece3f0) |
| 법률 holdout 후보 snapshot | v2 재생성용 pinned 중간 증거 242,675행·JSONL 16개·추출 manifest 16개를 private 보존. snapshot manifest `dca58ecb…`, 원격 visibility/allowlist/모든 콘텐츠 SHA 재검증 | [`LLM-OS-Models2/...-shards12-15@18cbfef7`](https://huggingface.co/datasets/LLM-OS-Models2/korean-legal-holdout-candidates-v1-shards12-15/tree/18cbfef7162fe07470d5377e198062301698ef33) |
| Clean 법률 retrieval 10K v2 text-strict | 242,675 candidate의 선언 train-role text 교집합 248개를 차단; 최종 query/positive training-text 0, source-document 0, benchmark exact 0, 10K 고유 문서. 독립 verify 통과, 원격 SHA/allowlist/private 재검증 | [`LLM-OS-Models2/...-v2-text-strict@ce9d3bb5`](https://huggingface.co/datasets/LLM-OS-Models2/korean-legal-source-heldout-retrieval-v2-text-strict/tree/ce9d3bb57ca4dc5144753f6d0f8b4a2256851e97) |
| Trainer validation 정정 | legacy 512 query-positive pair가 active 200K에 512/512 포함됨을 확인. active Qwen eval loss는 완료/finite 신호로만 사용하고 모든 archived checkpoint를 clean v2 10K로 재선택; 이후 run은 Grade-I text-strict 512·HN4·전체 예정 train 역할 exact overlap 0 사용. 원격 SHA/allowlist/private 재검증 | [`LLM-OS-Models2/...-512@8fdd1cad`](https://huggingface.co/datasets/LLM-OS-Models2/korean-embedding-legal-validation-v2-text-strict-512/tree/8fdd1cad0007a9bfadf328d1702dcf6973c3c03d) |
| 고정 비공개 다영역 selector | finance 900 + knowledge 1,000; 선택 query training overlap 0, knowledge query/corpus training overlap 0, 공개 benchmark blocklist overlap 0. finance corpus 1,373건 노출을 target-dev로 명시. private visibility·전체 원격 파일/SHA exact 검증 | [`LLM-OS-Models2/...-heldout-v1@d261e1e3`](https://huggingface.co/datasets/LLM-OS-Models2/korean-embedding-multidomain-selection-heldout-v1/tree/d261e1e3ff64e13828e73017fe2c312aae575709) |
| 대화형 noise robustness | prompt on/off × noise 0/1/5%, exact rank·cache·모델 카드 자동화; baseline 실행 대기 | [종합 평가 설계](docs/10_COMPREHENSIVE_SUITE.md) |
| 200K 학습 backend | 2026-07-17 exact homogeneous-order 5+5-step: SDPA 11.96, FA2 11.53 s/step(1.0373x); FA2 탈락, exact 검증된 `.venv-train-fa2 + SDPA` 선택 | [진행 현황](docs/14_PROGRESS_AND_BOTTLENECKS.md) |
| runtime storage watchdog | workspace 500GiB/100만 inode, root 100GiB/20만 inode, `/tmp` 50GiB/10만 inode를 30초마다 검사. 2회 연속 실패 때만 시작 시 검증한 우리 campaign PGID에 TERM→30초→KILL; 다른 프로세스는 신호하지 않음 | [`watch_storage_headroom.sh`](scripts/watch_storage_headroom.sh) |
| 200K production·capacity | 2026-07-17 11:46 KST Qwen 시작; 199,904행·3,123-step, 양쪽 shuffle off, offline/token-free. legacy validation loss는 선택에서 제외하고 Qwen/Comsat 양쪽의 모든 archived checkpoint를 같은 legal 10K·다영역 1.9K·robustness로 평가. 종료 뒤 계보 선택 → 승자 raw base last4 partial-full 200K → 1M/KD/전문가/수프/최종 선택을 자동 실행 | [2026-07-17 frontier plan](docs/34_PERFORMANCE_FIRST_FRONTIER_PLAN_2026-07-17.md) |
| last4 partial-full capacity challenger | Qwen/Comsat clean 승자 계보 하나만 동일 199,904행·3,123-step·global batch 64로 비교. 실제 microbatch 8/HN4 메모리 probe 실패 시 OOM 근거를 남기고 skip; 성공 시 상위 4 block+final norm 771.790M parameter update. input/completion log SHA와 exact base revision이 complete contract에 묶여야 package 가능 | [tuning strategy](experiments/070_tuning_strategy/) |
| checkpoint watcher / resume | 기존 Qwen step-250…1750은 당시 정책대로 private 보존했다. 후속 run은 public checkpoint repo가 기본이며, allowlist 3파일·LFS/manifest SHA·요청 visibility를 commit 전후 검증한다. 재시작은 complete optimizer checkpoint의 exact contract가 같을 때만 resume | [checkpoint watcher](docs/31_PRIVATE_CHECKPOINT_WATCHER.md) |
| public clean-winner full model | clean selector의 exact winner만 격리 staging에 hardlink/copy하고 모델 로딩 allowlist 밖 파일을 거부한다. 원본 모델은 변경하지 않으며 evidence의 로컬 절대경로·credential을 제거한다. 업로드 뒤 public visibility, 전체 remote file set, 모든 safetensors LFS SHA/size와 metadata SHA를 검증해야 다음 continual run이 시작된다 | [`publish_private_clean_candidate.py`](scripts/publish_private_clean_candidate.py) |
| final evaluation/publication completion | 최종 local winner 한 모델에만 Sionic9→공식 Korean6→comprehensive7/414를 실행한다. 어느 평가·legal/multidomain/ranks evidence·training manifest·token-file·private upload·campaign-result push라도 실패하면 campaign은 완료 처리되지 않는다. 최종 모델도 격리 staging과 전체 remote file/LFS exact report를 통과해야 한다 | [종합 선택 계약](docs/33_COMPREHENSIVE_SELECTION_AND_EVALUATION.md) |
| clean-guard multidomain model selection | valid performance candidate 0; 법률 Grade-I 최고 `-0.005` guard → finance/knowledge macro 최고 `-0.002` → robustness `-0.002` → intrusion `+0.001`. public Sionic/official score는 selector 입력에서 제외하며 모든 stage에 mandatory gate로 실행 | [다영역 선택 계약](docs/35_FIXED_MULTIDOMAIN_SELECTION_HOLDOUT.md) |
| text-only comprehensive diagnostic | 7 tasks·414 selected subsets; K-HATERS는 unsupported registered task, visual-document 5 assets는 modality 불일치로 명시 제외; public medium/high contamination diagnostic | [종합 선택 계약](docs/33_COMPREHENSIVE_SELECTION_AND_EVALUATION.md) |
| Qwen3 reranker teacher scorer/KD | `Qwen3-Reranker-8B@77d193c`; official yes/no logits, 5개 약 16GB LFS content SHA 전수 검증 후 local-only load. wide pool200→quantile15 compiler, hard InfoNCE+listwise KL/MarginMSE, queue4096 A/B와 1M 원본을 clean-first 비교. 선택·가중치 SHA·private exact file set·immutable commit이 모두 일치하지 않으면 전문가/법률 stage로 진행하지 않음. 아직 실제 score/KD 성능 결과 없음 | [teacher scorer](experiments/030_teacher_distillation/) |
| last-available-5 FP32 LoRA 평균 | 같은 Trainer version의 최신 최대 5개만 config/key/shape/dtype/finite gate 뒤 FP32 평균·atomic 저장; safe merge parity 후 single best와 동일 clean selector에서 비교. 첫 active Qwen도 검증 adapter archive로 5개를 보존하고 이후 run은 full checkpoint 5개도 유지 | [model merge](experiments/050_model_merge/) |
| basis-safe full-weight soup | 독립 LoRA factor 평균 금지; safe-merged general/parent/retrieval/SQuAD/health/AutoRAG/legal/combined full weight를 FP32 누적→BF16 sharded 출력. 6개 target 모델·local parent·Models2 derived-data upload를 필수 gate로 두고, parent retention 2종, general↔combined 2종, specialist 3종의 사전 고정 7개 coefficient를 모두 최종 clean selector에 추가; 실제 성능 결과 대기 | [model merge](experiments/050_model_merge/) |
| pinned model lineage / card | Qwen·Comsat 직접 base의 40-hex Hub commit을 LoRA/full→local continual→specialist→soup까지 `upstream_base_models`로 재귀 승계한다. 누락·비고정 revision·복수 evidence는 fail-closed이며, 혼합 soup는 Hugging Face의 복수 `base_model` metadata와 Comsat `CC-BY-NC-4.0` 비상업 조건을 카드에 자동 공개한다 | [frontier plan](docs/34_PERFORMANCE_FIRST_FRONTIER_PLAN_2026-07-17.md#pinned-hub-계보와-모델-카드) |
| 첫 8B LoRA smoke | 학습·저장·재로딩 검증 통과, 성능 주장은 없음 | [experiments/010_qwen3_8b_ko_lora/](experiments/010_qwen3_8b_ko_lora/) |
| smoke adapter HF artifact | private 업로드 완료, raw data/optimizer 제외 | [`LLM-OS-Models/qwen3-embedding-8b-ko-smoke-20260711`](https://huggingface.co/LLM-OS-Models/qwen3-embedding-8b-ko-smoke-20260711) |
| LoRA vs full tuning | 메모리·품질 비교 진행 중 | [experiments/070_tuning_strategy/](experiments/070_tuning_strategy/) |
| 10K exhaustive HN + LoRA r64 | 160 steps 완료, best step 80; train 10,000 + validation 512 모두 15-task exact text overlap 0; FP32 strict-parity 재병합 대기 | [진행 현황](docs/14_PROGRESS_AND_BOTTLENECKS.md) |
| 50K LoRA r64 | step 480 best loss 0.00350491; step 600까지 개선 없음. trainer data의 eval-query hash 4개 때문에 diagnostic 전용·public selection 자동 제외 | [진행 현황](docs/14_PROGRESS_AND_BOTTLENECKS.md) |

## 문서 지도

1. [전체 결론과 의사결정](docs/00_EXECUTIVE_SUMMARY.md)
2. [임베딩 모델과 MTEB가 실제로 재는 것](docs/01_EMBEDDINGS_AND_MTEB.md)
3. [Comsat 주장·점수·오염 가능성 감사](docs/02_COMSAT_AUDIT.md)
4. [2026년 7월 최고 모델과 방법론](docs/03_SOTA_MODELS_2026-07.md)
5. [논문 기반 학습 레시피](docs/04_TRAINING_RECIPE.md)
6. [데이터, 라이선스, contamination 정책](docs/05_DATA_AND_GOVERNANCE.md)
7. [Qwen3 논문과 후속 연구](docs/06_LITERATURE_REVIEW.md)
8. [실행 및 재현 runbook](docs/07_RUNBOOK.md)
9. [MTEB Korean 상위 모델과 F2LLM/PwC 감사](docs/08_KOREAN_LEADERBOARD_AND_F2LLM.md)
10. [실제 평가 결과 로그](docs/09_EVALUATION_RESULTS.md)
11. [Clean Korean 종합 평가 설계](docs/10_COMPREHENSIVE_SUITE.md)
12. [공식 MTEB Korean v1 로컬 재현 protocol](docs/11_MTEB_KOREAN_V1_PROTOCOL.md)
13. [논문·모델별 데이터와 학습 방법 매트릭스](docs/12_PAPER_DATA_METHOD_MATRIX.md)
14. [100만 행 공개 가능 데이터 공장](docs/13_RIGHTS_SAFE_DATA_FACTORY.md)
15. [진행 현황, 병목과 다음 의사결정](docs/14_PROGRESS_AND_BOTTLENECKS.md)
16. [성능 우선 50K→1M 데이터 믹스](docs/15_PERFORMANCE_DATA_MIX.md)
17. [F2LLM-v2-8B와 Comsat 계보·레시피 정밀 감사](docs/16_F2LLM_COMSAT_RECIPE_AUDIT.md)
18. [Legalize-KR·LLM-Ko-Datasets 원본 감사와 데이터 설계](docs/17_LEGAL_AND_KO_DATA_SOURCE_AUDIT.md)
19. [LoRA adapter 병합·평가·공개 절차](docs/18_ADAPTER_MERGE_AND_EVAL.md)
20. [근거 기반 합성 query·hard-negative 공장](docs/19_GROUNDED_SYNTHETIC_QUERY_FACTORY.md)
21. [상위 모델 로컬 평가 매트릭스](docs/20_TOP_MODEL_LOCAL_EVAL_MATRIX.md)
22. [Qwen3 임베딩 vLLM·TEI·FA2 서빙](docs/21_QWEN3_EMBEDDING_SERVING.md)
23. [1M scale 학습·평가 실행 계약](docs/22_SCALE_1M_EXECUTION.md)
24. [250K–1M FAISS hard-negative mining](docs/23_SCALABLE_HARD_NEGATIVE_MINING.md)
25. [법률·공공 source-document-held-out 종합 평가](docs/24_LEGAL_SOURCE_HELDOUT_RETRIEVAL.md)
26. [Sionic SQuADKorV1 train-family 60K target adaptation](docs/25_SIONIC_SQUAD_TARGET_ADAPTATION.md)
27. [Sionic PublicHealthQA multilingual health-domain adaptation](docs/26_SIONIC_PUBLIC_HEALTH_ADAPTATION.md)
28. [Sionic AutoRAG finance/commerce/legal domain adaptation](docs/27_SIONIC_AUTORAG_DOMAIN_ADAPTATION.md)
29. [Sionic 9 combined target-domain final candidate](docs/28_SIONIC_COMBINED_TARGET_MODEL.md)
30. [MIRACL·MrTidy·MLDR train-family specialist](docs/29_SIONIC_RETRIEVAL_FAMILY_ADAPTATION.md)
31. [상위 모델 공식 근거 종합과 1×H100 최단 승리 레시피](docs/30_TOP_MODEL_RECIPE_SYNTHESIS.md)
32. [200K private checkpoint 증분 업로드 watcher](docs/31_PRIVATE_CHECKPOINT_WATCHER.md)
33. [Qwen reranker teacher와 금융·시간성 추가 데이터](docs/32_NEXT_STAGE_TEACHER_AND_DATA.md)
34. [종합 최고 모델 선택·평가 계약](docs/33_COMPREHENSIVE_SELECTION_AND_EVALUATION.md)
35. [2026-07-17 성능 최우선 frontier 방법론과 전면 복구 계획](docs/34_PERFORMANCE_FIRST_FRONTIER_PLAN_2026-07-17.md)
36. [고정 비공개 finance/knowledge 다영역 모델 선택 보드](docs/35_FIXED_MULTIDOMAIN_SELECTION_HOLDOUT.md)
37. [Nemotron-3 한국어 base 결정·중단 재개](docs/36_NEMOTRON3_KOREAN_BASE_DECISION_2026-07-17.md)
38. [2026-07-18 중단 복구와 방법론 문헌 점검](docs/37_RESUME_RECOVERY_AND_LITERATURE_2026-07-18.md)

## 실험 지도

실험 번호는 실행 순서가 아니라 비교 축을 나타냅니다. 각 폴더에 가설, 데이터 revision, config, 로그, 결과를 남깁니다.

| 폴더 | 비교할 것 |
|---|---|
| [`000_baseline`](experiments/000_baseline/) | Qwen/Comsat 및 평가 파이프라인 재현 |
| [`010_qwen3_8b_ko_lora`](experiments/010_qwen3_8b_ko_lora/) | 첫 clean Korean contrastive LoRA |
| [`020_hard_negative`](experiments/020_hard_negative/) | BM25/dense/reranker negative와 false-negative filtering |
| [`030_teacher_distillation`](experiments/030_teacher_distillation/) | reranker/강한 embedder의 soft-label distillation |
| [`040_long_context`](experiments/040_long_context/) | 512/2K/4K/8K 길이·evidence 위치 curriculum |
| [`050_model_merge`](experiments/050_model_merge/) | checkpoint/domain adapter 평균·SLERP |
| [`060_backbone_ablation`](experiments/060_backbone_ablation/) | Qwen 0.6B/4B/8B, Nemotron, Gemma 계열 비교 |
| [`070_tuning_strategy`](experiments/070_tuning_strategy/) | LoRA/DoRA/부분학습/full FT의 품질·VRAM·속도 비교 |
| [`080_f2_recipe`](experiments/080_f2_recipe/) | F2형 dual loss와 exact MRL을 기본 InfoNCE와 비교 |
| [`090_sionic_squad_adaptation`](experiments/090_sionic_squad_adaptation/) | KorQuAD train 60K current-student HN + general replay와 broad 회귀 |
| [`100_sionic_health_adaptation`](experiments/100_sionic_health_adaptation/) | F2 medical 100K current-student HN + general replay와 PublicHealth/broad 회귀 |
| [`110_sionic_autorag_adaptation`](experiments/110_sionic_autorag_adaptation/) | F2 finance/commerce/legal 100K current-student HN + general replay와 AutoRAG/broad 회귀 |
| [`120_sionic_combined_target`](experiments/120_sionic_combined_target/) | 네 target domain + general replay를 한 400K 최종 후보로 결합 |

## 원칙

- 공개 test 점수를 반복해서 보고 checkpoint를 고르지 않습니다. Grade-I legal guard와 고정 비공개 finance/knowledge, robustness로 먼저 고른 한 winner에 Sionic 9와 공식 Korean 6을 final-once 실행합니다.
- 법률 최고에서 `0.005` 이내인 후보만 허용하고, 그 안에서 다영역 macro `0.002`, worst-condition robustness `0.002`, noise intrusion `0.001` near-tie 순서를 적용합니다.
- 모든 데이터 행에 `source`, `revision/date`, `license`, `sha256`, `generator`, `prompt_version`을 남깁니다.
- 현재 자체 법률 holdout은 같은 repository 안에서 source document를 분리한 Grade I이며, Grade Z 또는 `clean-zero-shot`으로 부르지 않습니다. 공개 benchmark exact blocklist와 추가 near-duplicate 감사 결과는 분리해 보고합니다.
- benchmark train split을 쓰는 별도 실험은 `supervised/in-domain`으로 명시합니다.
- 평균 점수뿐 아니라 per-task 변화, bootstrap confidence interval, 원 Qwen 다국어 성능 회귀, 속도/메모리를 함께 봅니다.

## 환경

- GPU: NVIDIA H100 80GB 1장
- Python: 3.10
- 첫 backbone: `Qwen/Qwen3-Embedding-8B`
- 프레임워크: Qwen 공식 가이드가 연결하는 `ms-swift`
- 계산: last-token pooling, L2 normalization, cosine/dot-product retrieval

실제 명령과 pinned revision은 [runbook](docs/07_RUNBOOK.md)에 기록합니다.

Sionic 9-task 비교는 [고정 protocol](configs/sionic9_protocol.json)과 [평가 스크립트](scripts/evaluate_sionic9.py)를 사용합니다. 공식 `MTEB(kor, v1)` 리더보드 결과와 이 9-task retrieval 평균은 서로 다른 표로 유지합니다.
