# 진행 현황, 병목과 다음 의사결정

기준일: 2026-07-12 (Asia/Seoul). 이 문서는 “코드가 실행됨”, “평가 재현됨”, “모델 성능이 개선됨”을 구분한다. 숫자가 없는 항목을 완료로 표현하지 않는다.

## 현재 한 줄 상태

평가와 학습 plumbing은 검증됐지만 **Comsat을 이긴 우리 성능 모델은 아직 없다**. 최적화·보고 순서는 **Sionic retrieval 9종 → 공식 MTEB Korean v1 → clean 종합 보드**다. Comsat 공식 Korean 6-task, 10K exhaustive hard-negative mining, 첫 10K LoRA r64 학습을 완료했고 현재 50K LoRA r64가 실행 중이다. 이후 같은 자동 queue가 200K→F2 dual/MRL→1M→법률 replay로 확대한다. 첫 후보는 비상업 공개가 가능한 `performance` 트랙이며, 권리가 정리된 `clean/release` 트랙은 별도로 유지하되 performance 학습을 막지 않는다.

## 두 개의 모델 트랙

| 트랙 | 우선 목표 | 허용 데이터 | 주장 방식 |
|---|---|---|---|
| `performance/non-commercial` | Sionic 9 및 Korean/MTEB 최고 성능 | 공개 다운로드 가능한 unknown/NC/custom-license composite, benchmark train split 허용 | 사용 source와 in-domain task를 공개하고 zero-shot SOTA라고 부르지 않음 |
| `clean/release` | 배포 가능한 일반화 모델 | provenance/license/attribution 및 decontamination gate 통과 row만 | clean zero-shot, 권리·회귀까지 통과한 경우만 public release |

두 트랙 모두 evaluation test query/qrel의 직접 학습, test score를 본 checkpoint 반복 선택, 모델 카드에서의 데이터 노출 은폐는 허용하지 않는다. 권리 조건을 완화하는 것과 test leakage를 허용하는 것은 다른 결정이다.

## 완료된 것

| 영역 | 결과 | 근거/판정 |
|---|---|---|
| GitHub | `LLM-OS-Models/Embedding` main에 중간 commit 지속 push | secrets 제외, submodule/revision 고정 |
| Qwen 학습 환경 | `.venv-train`, Torch 2.13/CUDA 13, ms-swift 4.5 dev | H100 BF16/SDPA 정상 |
| MTEB 환경 | `.venv-mteb`, MTEB 2.18.0/commit `193e3f66` | task/split/dataset SHA 고정 |
| Sionic 9 evaluator parity | AutoRAG Qwen `0.82765`, Comsat `0.85222` | 카드와 각각 `0.00005`, `0.00042` 차이 |
| 비교 모델 AutoRAG | F2 `0.76611`, PwC `0.78329` | 같은 full-corpus NDCG@10 |
| 8B LoRA smoke | 20 steps/43.81s/peak 17.07GiB, reload pass | 4096-d, adapter SHA와 positive margin 검증 |
| smoke HF artifact | private repo 업로드 | raw data/optimizer/log 제외, public 전환 금지 |
| 공식 Korean protocol | 정확한 6 task와 prompt fallback 구현 | local result를 official submission과 구분 |
| 논문/데이터 감사 | Qwen/F2/Nemotron/KaLM/Harrier 및 2026 후속 matrix | 공개 사실, 누락, 채택/기각 분리 |
| hard-negative miner | exact blockwise dense mining, `.95*s_pos`, pool24 | dry-run/fake encoder/strict validator 통과 |
| benchmark seal | Sionic 9 + 공식 Korean 6의 ID/text/qrel fingerprint | deterministic gzip/manifest 빌더 검증 통과 |
| 공개 가능 데이터 공장 | KOGL·법률·Wikipedia·PMC·CDC 1,000,000행 계획 | source/revision/license 및 생성·검수 gate 고정 |
| 10K private pilot 입력 | train 10,000 / validation 512, hash 검증 | source license 미명시로 public release 불가 |
| performance 50K mix | 계획 수량 전체 build·strict validation 완료 | train SHA `b46a7be…258a`, provenance SHA `e8ccca…6031` |
| performance 200K mix | 200,000 rows build·strict validation·공개 업로드 | train SHA `379694…e480`, provenance SHA `7243e6…17b8` |
| 법률 source-native mix | 4개 pinned repository에서 균형 250,000 rows build·공개 업로드 | train SHA `1d8136…4c90`, provenance SHA `a1b3cd…de3e`; bootstrap negative 표시 |
| 데이터 공개 | 50K, 200K, 법률 250K, 성능 우선 1M, benchmark blocklist | `LLM-OS-Models` HF organization의 public dataset 5개, 원격 API에서 `private=false`와 파일 확인 |
| vLLM 환경 | 별도 `.venv-vllm`, vLLM 0.24/Torch 2.11 설치 | Ko-Strategy parity/처리량 측정 완료; 이 workload에서는 FA2가 더 빠름 |
| adapter 병합/공개 | safe merge, 6-probe parity, ST contract, 카드/대용량 upload 코드 | tiny Qwen 실제 LoRA merge에서 max pair delta `4.68e-8` |
| homogeneous batching | provenance source별 16-row microbatch compiler | 50K `49,904`, 200K `199,904` rows; 모든 emitted batch 단일 source |
| performance 1M mix | 1,000,000 rows build·strict validation·public HF upload | train SHA `094d44…3c0a`, provenance SHA `94334a…18c1` |
| performance 1M homogeneous | 999,936 rows / 62,496 source-homogeneous batches | ordered train SHA `ac39ea…0169`; source remainder 총 64 rows |
| scalable hard-negative miner | resumable float32 embedding memmap + FAISS IVFFlat + exact selected-score recompute | 250K legal dry-run, index persist/resume, false-negative filter test 통과 |
| public model artifact contract | model card, 사용법, data/evaluation manifest와 Sionic/official raw summary 동봉 | post-training/1M/legal 각 캠페인에 공개 upload stage 연결 |
| benchmark blocklist | Sionic 9 + 공식 Korean 6의 exact hash artifact 15/15 | 547,245,091 bytes, 104 files, public revision `5e876f266068`; 원문·raw ID 없음 |
| 10K exhaustive HN | 10,000×10,000 exact cosine, 4 negatives/row, drop 0 | 91.75초; negative mean `0.50020`, p95 `0.59108`; train SHA `3df507…5adc` |

## 현재 실행 중

Comsat의 공식 `MTEB(kor, v1)` 6개를 모두 직접 측정했다.

| Task | Local official-protocol score |
|---|---:|
| KLUE-TC | 0.5213867 |
| Ko-StrategyQA | 0.8401600 |
| KLUE-STS | 0.8631865 |
| KorSTS | 0.7943686 |
| MIRACLReranking | 0.6846700 |
| MIRACLRetrieval | 0.6952600 |

마지막 retrieval은 H100 1장, FlashAttention 2, batch 224에서 1,486,752 documents를
완료했다. exact float32 embedding cache는 MTEB 50K corpus chunk마다 약 819MB를 atomic
저장했고, 실측 GPU 메모리는 corpus batch에 따라 약 47–60GiB였다. 6-task Mean(Task)은
73.3172, Mean(Type)은 70.0636이다. 2026-07-12 live 137-row board에 가상 삽입하면 Borda
6위이며 공식 row 자체가 아니다. 완료 즉시 GPU는 10K hard-negative mining으로 넘어갔고,
mining도 완료됐다.

`Qwen3-Embedding-8B + LoRA r64 + InfoNCE(batch negatives + explicit HN 4개)`
10K run은 160 steps/626.7초에 완료됐다. BF16/SDPA, microbatch 16, accumulation 4이며
최선 checkpoint는 80-step(`eval_loss=0.00338515`)이다. peak allocated VRAM은
22.17GiB, trainable parameters는 174.588M이었다. BF16 직접 fold는 adapter 대비 probe
minimum cosine `0.992115767`로 엄격 gate `0.999`를 통과하지 못했다. 따라서 점수를
만들지 않고, 다음 merge부터 FP32 fold로 자동 재시도해 같은 parity gate를 통과한 경우만
평가한다.

현재 50K LoRA r64는 평균 tokenized length `118.44±124.54`, 512-token cap에서
H100 100%, 약 56.38GiB, 약 22.8초/step이며 초기 예상은 약 5시간이다. 200K와 1M
ordered curriculum은 source-homogeneous 계약을 유지하면서 length bucket을 적용했다.
200K 문자 길이 proxy 기준 random batching 대비 padding이 `160,181,088 → 85,258,880`,
즉 46.77% 줄었다.

`performance_1m` 1,000,000-row base mix와 999,936-row/62,496-batch homogeneous 파생
파일은 build를 마쳤다. 50K/200K/1M 원본 dataset과 법률 250K는 Hugging Face에
공개됐고, GPU campaign은 완료된 manifest를 자동 감지해 scale run에 사용한다.
benchmark decontamination blocklist는 Sionic 9와 공식 Korean 6의 15/15 task build를
완료해 Hugging Face에 공개했다. 이것은 평가 전용이며 어떤 query/text/qrel도 학습 데이터
생성이나 checkpoint 선택에 사용하지 않는다.

## 아직 성능 결과가 아닌 것

- 288-row LoRA의 loss는 첫 step부터 거의 0이었다. negative가 너무 쉬워 pipeline 검사 외 의미가 없다.
- adapter probe의 positive margin `0.44580`은 세 문장 무결성 검사이지 retrieval benchmark 점수가 아니다.
- 10K hard-negative mining과 LoRA r64 학습은 완료됐다. validation InfoNCE loss와 merge probe만으로 Comsat 우위를 주장하지 않으며 Sionic 9 전체가 끝나야 한다.
- vLLM Ko-StrategyQA는 `0.83830`, 기존 FA2는 `0.84016`으로 `-0.00186` 차이였다. 65K-token 설정은 약 200 docs/s로 FA2보다 느렸고, 131K-token/1024-seq/95% VRAM은 75.85GiB에서 OOM이 나 공식 full run에는 쓰지 않는다.
- clean comprehensive suite는 설계만 고정됐고 rights-safe holdout 수치는 아직 없다.

## 주요 병목

### 1. 대규모 performance mix 변환과 균형

학습 데이터가 없는 것이 아니다. 즉시 쓸 수 있는 `ko-triplet-v1.0` 744,862 rows, F2LLM-v2 composite 60.1M/한국어 약 1.083M, KaLM fine-tuning 6.34M, Nemotron 약 16.1–16.4M과 target 계열 train split이 있다. 추가로 `legalize-kr`의 법령·행정규칙·판례·자치법규와 `LLM-Ko-Datasets`가 가리키는 한국어 원천을 감사 중이다. 현재 병목은 이를 같은 schema로 변환하고 source가 큰 gradient를 독점하지 않게 균형화하며, 실제로 어려운 negative를 보존하는 일이다.

해소 조건:

- 10K → 50K → 200K → 744K scale curve 자동화
- F2 Korean과 KaLM/Nemotron multilingual replay의 공통 query/positive/HN schema
- source cap과 homogeneous batch sampler
- target train 사용 여부를 row/task manifest에 기록
- 1M–2M performance mix에서 domain/length/query-style 분포 고정

권리/provenance는 clean/release 트랙의 병목으로 남지만 performance/non-commercial 트랙의 진입을 차단하지 않는다. 원 source와 license 상태는 나중에 제거할 수 있도록 계속 기록한다.

### 2. negative 품질과 false negative

기존 triplet의 negative는 base Qwen에 너무 쉽다. 반대로 dense top-1을 무조건 negative로 쓰면 실제 정답을 오답으로 학습할 수 있다.

해소 조건:

- base/current/BM25 candidate 합집합
- `s_neg < {.90,.95,.98}*s_pos` 비교
- Qwen reranker 연속 점수와 positive/partial-positive 검수
- top-hard뿐 아니라 score quantile 전 구간 표집
- real/generated source shortcut audit

### 3. 8B full-corpus 평가 시간

MIRACL Korean corpus만 약 149만 문서다. 일반 SentenceTransformers batch 2는 비현실적으로 느리고, 현재 FA2/batch224로 최적화해도 한 모델당 상당한 시간이 든다. 기본 MTEB result cache는 task 완료 전 embedding을 보존하지 않으므로 이 저장소가 encode 호출별 exact NPY cache를 추가했다.

해소 조건:

- 현재 Comsat run은 중단하지 않고 완료
- backend마다 짧은 throughput/parity gate를 먼저 실행하고 vLLM이 실제로 빠른 모델·길이에서만 continuous batching 사용
- Comsat MIRACL은 FA2 batch 224를 사용하고 peak/처리량에 따라 208/192 fallback
- 각 50K document chunk의 input/options/model namespace와 float32 array를 atomic NPY로 저장하고 재시작 때 exact hit만 재사용
- official board에 이미 신뢰할 값이 있는 모델은 불필요하게 재실행하지 않음

### 4. LoRA 대 full FT 결정

F2는 45M example full FT 선례지만 우리 budget과 base는 다르다. standard full은 update capacity가 크지만 메모리와 회귀 위험이 크고, 작은 microbatch 때문에 retrieval의 true in-batch negatives가 줄 수 있다.

현재 근거:

| 방식 | Trainable | Peak/예상 VRAM | 상태 |
|---|---:|---:|---|
| LoRA r32 | 87.294M | **17.07GiB 실측** | pipeline pass |
| LoRA r64 | 174.588M | 10K **22.17GiB**, 50K long mix **56.38GiB** 실측 | 10K 완료, 50K 실행 중 |
| DoRA r32 | 약 88.695M | 17–19GiB 예상 | 대기 |
| 마지막 4층 + norm | 771.790M | 20–25GiB 예상 | 대기 |
| GaLore full | encoder full update | 35–45GiB 예상 | 대기 |
| standard full AdamW | 약 7.567B encoder | 60–75GiB 예상 | OOM 가능 1-step만 먼저 |

결정 gate는 동일 hard-negative data와 token budget에서 clean 품질/회귀/VRAM/GPU-hour Pareto다. 현재 기본 선택은 LoRA r64이며, 성능이 막힐 때 partial/GaLore/full 순으로 승격한다.

### 5. clean selection 보드 부재

Sionic 9와 공식 MTEB를 반복해 checkpoint를 고르면 leaderboard overfitting이 된다. 아직 rights-safe temporal/domain holdout이 없어 public score와 독립적인 선택 근거가 부족하다.

해소 조건:

- 정부·법률·보건·금융·일반의 source/time 분리 holdout
- long evidence 위치 및 OCR/띄어쓰기 paired slice
- Qwen/Comsat baseline 고정
- bootstrap CI와 worst-domain gate

## 실행 queue

| 순서 | 작업 | 진입 조건 | 완료 조건 |
|---:|---|---|---|
| 1 | Comsat official Korean MIRACLRetrieval | 실행 중 | 6-task summary + raw cache |
| 2 | live Borda 가상 삽입 | 6-task complete | backend rank 137/137 재현 + local 위치 |
| 3 | README 공식 Korean row 갱신/push | 1–2 완료 | local reproduction 표기와 task별 숫자 |
| 4 | 10K train/validation dense HN mining | GPU free | pool/score/filter manifest + strict JSONL |
| 5 | 10K/50K/200K LoRA 및 F2 loss ablation | mined data | non-zero learning signal, reload, VRAM |
| 6 | LoRA/DoRA/last4/GaLore/full 1-step memory probes | ablation 뒤 GPU free | 실제 peak VRAM/속도/OOM 기록 |
| 7 | 후보별 Sionic 9 전체 평가와 최선 모델 공개 | checkpoint 검증 | 9-task summary + model/data revision + model card |
| 8 | 최고 후보 공식 Korean v1 | Sionic 선택 완료 | 6-task raw/summary 및 README 반영 |
| 9 | 1M homogeneous LoRA scale | 1M manifest 완료 | 7,812 steps, Sionic 9/official, public model |
| 10 | 법률 250K target-adaptation | 1M stage 종료 | FAISS HN, provenance projection, Sionic 9/official, public model |
| 11 | top-model Sionic 동등 평가 | target stage 종료 | Comsat/Qwen/F2/PwC/Harrier/KaLM/Nemotron raw results |
| 12 | partial/DoRA/GaLore/full 품질 비교 | memory probe 통과 | 동일 200K/token budget Pareto |
| 13 | rights-safe 50K→500K clean model | source gate 완료 | license/provenance/blocklist audit pass |

## 주장 gate

`performance/non-commercial` 모델에서 “Sionic 9 평균을 이겼다”는 표현은 다음을 만족할 때 사용한다.

1. Sionic 9개 전부 동일 protocol로 직접 실행
2. macro NDCG@10이 `0.7930`보다 높음
3. task별 score와 model/data revision 공개
4. evaluation test query/qrel 직접 학습 없음
5. benchmark train/source 노출과 zero-shot 비율 공개
6. broad/multilingual 회귀와 효율 동시 보고

“clean/general Korean SOTA” 또는 public release는 여기에 benchmark overlap audit, clean holdout, data license/provenance gate를 추가로 통과해야 한다.

이 문서는 각 장시간 평가·학습과 중요한 실패 뒤 갱신하고, 해당 commit을 GitHub에 push한다.

## 야간 자동 GPU queue

[`scripts/run_night_gpu_queue.sh`](../scripts/run_night_gpu_queue.sh)는 실행 중인 대형
baseline PID가 끝난 뒤 MIRACL fallback/summary/Borda, 10K hard-negative mining,
LoRA r64, 준비된 50K/200K 성능 mix, F2 dual-loss/MRL, LoRA·DoRA·partial·full memory
probe를 순차 실행한다. 각 LoRA/F2 run의 best-validation checkpoint는 즉시 safe merge하고
대형 MIRACL/MrTidy를 제외한 Sionic 7-task early screen을 실행한다. 이 result와 exact
embedding cache는 후속 9-task pass가 그대로 재사용한다. 각 stage는 시작·종료 시각과
exit status를 남기며, 한 ablation의
실패가 뒤의 유효한 실험을 막지 않는다. 동일 GPU에서 두 stage를 동시에 실행하지 않는다.

그 다음 [`scripts/run_post_training_eval_queue.sh`](../scripts/run_post_training_eval_queue.sh)가
각 run의 minimum-eval-loss checkpoint를 선택해 safe merge, Sionic 9종 전체, 최고
후보 공식 Korean v1, 공개 model card/HF upload를 수행한다. 마지막으로
[`scripts/run_top_model_sionic_queue.sh`](../scripts/run_top_model_sionic_queue.sh)가
Comsat/Qwen/F2/PwC/Harrier/KaLM/Nemotron을 같은 Sionic protocol로 측정한다.

실제 장기 실행은 [`scripts/run_full_campaign_queue.sh`](../scripts/run_full_campaign_queue.sh)가
위 세 queue, 1M scale, 법률 250K FAISS target-adaptation을 한 프로세스에서 순차
호출한다. watcher PID를 여러 개 수동 연결하지 않아 앞 단계 종료와 다음 단계 시작
사이의 race를 없앤다.
