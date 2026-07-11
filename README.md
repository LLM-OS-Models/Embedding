# Korean Embedding Lab

한국어 검색 임베딩 모델을 연구하고, `sionic-ai/comsat-embed-ko-8b-preview`를 **오염 없이 재현 가능하게** 넘어서는 것을 목표로 하는 작업 공간입니다.

기준일: **2026-07-12 (Asia/Seoul)**

## 한 줄 결론

- 현재 최적화 우선순위는 **Sionic retrieval 9종 → 공식 MTEB Korean v1 → clean 종합 보드**이며, 첫 공개 후보는 성능 우선 비상업 모델입니다.
- Comsat의 `1M+`는 문서나 토큰이 아니라 출처와 형식이 공개되지 않은 **Korean training examples**입니다.
- Comsat의 `0.7930`은 일반 MTEB SOTA가 아니라, 자체 선택한 한국어 retrieval 9종의 macro `NDCG@10`입니다. Qwen3-Embedding-8B 대비 차이는 `+0.0105`입니다.
- 가장 직접적인 학습법은 raw-text LM CPT가 아니라 `query / positive / hard negatives`를 이용한 **continued contrastive fine-tuning(InfoNCE)** 입니다.
- 현재 첫 실험은 `Qwen/Qwen3-Embedding-8B + BF16 LoRA + InfoNCE`로 시작합니다. 공개 평가 9종은 학습·negative mining·checkpoint selection에서 차단합니다.

자세한 판단은 [요약 문서](docs/00_EXECUTIVE_SUMMARY.md)부터 읽으면 됩니다.

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
| `sionic-ai/comsat-embed-ko-8b-preview` | **0.7930** | **0.85222** 실측 | 9개 카드 + 1개 재현 | 기준 |
| `Qwen/Qwen3-Embedding-8B` | 0.7825 | 0.82765 실측 | 9개 카드 + 1개 재현 | -0.0105 |
| `codefuse-ai/F2LLM-v2-8B` | 0.7621 | 0.76611 실측 | 9개 카드 + 1개 재현 | -0.0309 |
| `SamilPwC-AXNode-GenAI/PwC-Embedding_expr` | — | 0.78329 실측 | AutoRAG만 측정 | — |
| 우리 smoke LoRA r32 | — | 미측정 | 성능 주장 금지 | — |
| 우리 공개 후보 목표 | **> 0.7930** | 회귀 없음 | 9개 전부 직접 측정 | **> 0** |

현재 근거로 Comsat은 “별로인 모델”이 아니라 Qwen 대비 한국어 retrieval에 잘 특화된 모델입니다. 다만 선택된 9개 task만으로 일반 한국어·다국어 SOTA라고 할 수는 없습니다. raw run과 revision은 [평가 로그](docs/09_EVALUATION_RESULTS.md)에 기록합니다.

### 3. Clean Korean 종합 보드

우리의 실제 모델 선택 보드입니다. public leaderboard test를 보며 튜닝하지 않고, 데이터 provenance와 중복 차단이 확인된 holdout에서만 checkpoint를 고릅니다. 첫 10K 법률 source-document-held-out set은 build·독립 검증·공개를 완료했고 baseline/model 수치는 후속 queue에서 채웁니다.

| 모델 | Clean retrieval | Broad semantic | Long-context | Noise/OCR robustness | Peak VRAM | 상태 |
|---|---:|---:|---:|---:|---:|---|
| Qwen3-Embedding-8B | 예정 | 예정 | 예정 | 예정 | 예정 | 기준선 |
| Comsat-embed-ko-8b-preview | 예정 | 예정 | 예정 | 예정 | 예정 | 비교군 |
| 우리 smoke LoRA r32 | 평가 제외 | 평가 제외 | 평가 제외 | 평가 제외 | **17.07 GiB** 학습 | pipeline-only |
| 우리 release candidate | 예정 | 예정 | 예정 | 예정 | 예정 | 권리 확인 데이터로 학습 예정 |

종합 보드는 clean retrieval, STS/분류, 긴 문맥의 evidence 위치, OCR·띄어쓰기·질의체 변화, 처리량·VRAM·차원/저장비용을 각각 보고합니다. 설계와 승격 기준은 [종합 평가 설계](docs/10_COMPREHENSIVE_SUITE.md)에 고정합니다.

#### Clean 법률 source-held-out 10K 실측

<!-- CLEAN_LEGAL_RESULTS_START -->
아직 완료된 clean 법률 baseline이 없습니다.
<!-- CLEAN_LEGAL_RESULTS_END -->

### 자동 캠페인 실측 결과

아래 block은 각 장기 stage가 Sionic 9와 공식 Korean 6을 모두 끝낸 뒤 자동 갱신·push한다.

<!-- CAMPAIGN_RESULTS_START -->
아직 완료된 성능 후보가 없습니다.
<!-- CAMPAIGN_RESULTS_END -->

## 현재 상태

| 항목 | 상태 | 위치 |
|---|---|---|
| Qwen3-Embedding 공식 저장소 | clone 완료 | [`Qwen3-Embedding/`](Qwen3-Embedding/) |
| 공식 후속학습 프레임워크 `ms-swift` | commit 고정, 격리 환경 설치 완료 | [`third_party/ms-swift/`](third_party/ms-swift/) |
| Sionic 벤치마크 감사 | 1차 완료 | [docs/02_COMSAT_AUDIT.md](docs/02_COMSAT_AUDIT.md) |
| 2026-07 라이브 MTEB 및 상위 모델 감사 | 완료, 새 결과는 날짜 고정 갱신 | [docs/03_SOTA_MODELS_2026-07.md](docs/03_SOTA_MODELS_2026-07.md) |
| 데이터 manifest / 오염 차단 | 15/15 task exact SHA-256 blocklist 빌드·공개 완료 | [`LLM-OS-Models/korean-embedding-benchmark-blocklist-v1`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-benchmark-blocklist-v1) |
| 100만 행 공개 가능 데이터 공장 | source·수량·검수 gate 설계 완료 | [docs/13_RIGHTS_SAFE_DATA_FACTORY.md](docs/13_RIGHTS_SAFE_DATA_FACTORY.md) |
| Legalize-KR 데이터 | 312,581문서 감사, 2,756,363 source-native 후보 추출 가능 | [docs/17_LEGAL_AND_KO_DATA_SOURCE_AUDIT.md](docs/17_LEGAL_AND_KO_DATA_SOURCE_AUDIT.md) |
| 성능 우선 50K 데이터 | 50,000 rows + 현재 run의 exact 49,904-row order/provenance·품질 감사/카드 공개 | [`LLM-OS-Models/korean-embedding-performance-v1-pilot-50k`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-performance-v1-pilot-50k/tree/da0b0ff09ea0b14d2281a88671d2d346a45ebfbe) |
| 성능 우선 200K 데이터 | 200,000 rows + exact 199,904-row length-bucketed train/provenance·품질 감사/카드 공개 | [`LLM-OS-Models/korean-embedding-performance-v1-ablation-200k`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-performance-v1-ablation-200k/tree/0a3a0e38fa766ba99fdd8f82ee49862e25f0aaf4) |
| SQuADKorV1 train-family 60K | 원본 KorQuAD train 질문→문맥 60K; 평가 query/evaluation-text match 0, Wikipedia shared eval-corpus 6,426 고유 hash 공개 | [`LLM-OS-Models/korean-embedding-performance-v1-sionic-squad-train-60k`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-performance-v1-sionic-squad-train-60k/tree/8fbc6d6d5c93c3493456079d930921ac90ec6801) |
| 법률 source-native 250K | 4개 source 균형 shard + bootstrap 한계·질의 분포 전수 감사/카드 공개 | [`LLM-OS-Models/korean-legal-retrieval-source-native-250k`](https://huggingface.co/datasets/LLM-OS-Models/korean-legal-retrieval-source-native-250k/tree/ec2f09a220dc5aa326c5d63b8e49adbf3a5524bc) |
| 성능 우선 1M 데이터 | 1,000,000 rows + exact 999,936-row train/provenance + 품질 감사·카드 공개 | [`LLM-OS-Models/korean-embedding-performance-v1-performance-1m`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-performance-v1-performance-1m/tree/ac3e806f2c01f2aa9f45686207b822e992889da2) |
| 평가 오염 방지 blocklist | Sionic 9 + 공식 Korean 6, 원문 없는 SHA-256 547MB | [`LLM-OS-Models/korean-embedding-benchmark-blocklist-v1`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-benchmark-blocklist-v1) |
| Clean 법률 retrieval 10K | training document overlap 0, benchmark exact overlap 0, 독립 verifier pass | [`LLM-OS-Models/korean-legal-source-heldout-retrieval-v1`](https://huggingface.co/datasets/LLM-OS-Models/korean-legal-source-heldout-retrieval-v1/tree/ee1300f04ea03d66bb51e23bbbda34376fece3f0) |
| 대화형 noise robustness | prompt on/off × noise 0/1/5%, exact rank·cache·모델 카드 자동화; baseline 실행 대기 | [종합 평가 설계](docs/10_COMPREHENSIVE_SUITE.md) |
| 격리 FA2 학습 후보 | NVIDIA PyTorch 2.5 + flash-attn 2.4.2 import/CLI pass; 실제 8B backward 전에는 미승격 | [튜닝 전략](experiments/070_tuning_strategy/) |
| 첫 8B LoRA smoke | 학습·저장·재로딩 검증 통과, 성능 주장은 없음 | [experiments/010_qwen3_8b_ko_lora/](experiments/010_qwen3_8b_ko_lora/) |
| smoke adapter HF artifact | private 업로드 완료, raw data/optimizer 제외 | [`LLM-OS-Models/qwen3-embedding-8b-ko-smoke-20260711`](https://huggingface.co/LLM-OS-Models/qwen3-embedding-8b-ko-smoke-20260711) |
| LoRA vs full tuning | 메모리·품질 비교 진행 중 | [experiments/070_tuning_strategy/](experiments/070_tuning_strategy/) |
| 10K exhaustive HN + LoRA r64 | 160 steps 완료, best step 80; FP32 strict-parity 재병합 대기 | [진행 현황](docs/14_PROGRESS_AND_BOTTLENECKS.md) |
| 50K LoRA r64 | 360/800 검증 완료, H100 100%, trainer 59.30GiB/device 약 61.9GiB; best step 200 loss 0.00351495 | [진행 현황](docs/14_PROGRESS_AND_BOTTLENECKS.md) |

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

## 원칙

- 공개 test 점수를 반복해서 보고 checkpoint를 고르지 않습니다.
- 모든 데이터 행에 `source`, `revision/date`, `license`, `sha256`, `generator`, `prompt_version`을 남깁니다.
- 공개 9개 benchmark의 query, qrel, corpus 및 near-duplicate를 차단한 `clean-zero-shot` 결과를 주 결과로 냅니다.
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
