# 070 — LoRA, 부분학습, full fine-tuning 비교

## 질문

Qwen3-Embedding-8B를 한국어 retrieval에 적응시킬 때 LoRA의 낮은 메모리 비용이 충분한가, 아니면 full-parameter update의 추가 품질이 H100 80GB와 배포 비용을 정당화하는가?

## 현재 실측 기준선

| Run | Trainable | Peak VRAM | 20 steps | 저장 크기 | 판정 |
|---|---:|---:|---:|---:|---|
| LoRA r32, all-linear, BF16 | 87.294M / 8.276B (1.0548%) | **17.07 GiB** | **43.81 s** | adapter 349 MB, optimizer 699 MB | pipeline pass, 데이터가 너무 쉬워 품질 판정 불가 |

표의 메모리는 실제 H100 80GB 학습 로그 값이다. 짧은 평균 길이 약 39 tokens의 smoke run이므로 512/2K sequence의 production peak로 해석하면 안 된다.

## 비교할 설정

| ID | Update | 목적 |
|---|---|---|
| A | LoRA r32 / r64, all-linear | 가장 빠른 반복과 adapter merge |
| B | DoRA r32 | 비슷한 trainable budget에서 magnitude 적응 여부 |
| C | 상위 4/8 transformer block + projection/norm | 한국어 specialization과 base 회귀 사이 절충 |
| D | full FT + 표준 AdamW | F2LLM과 가장 가까운 품질 상한; 80GB feasibility probe |
| E | full FT + optimizer/state 절감 또는 GaLore/ZeRO-offload | 단일 H100에서 가능한 full-update 대안 |

## 공정 비교 조건

- 동일한 clean train/dev manifest, mined-negative pool, token budget를 사용한다.
- optimizer step 수가 아니라 본 token 수와 wall-clock/GPU-hours도 맞춰 비교한다.
- 짧은 pilot은 peak VRAM·처리량만 판단하고 모델 품질 결론에는 쓰지 않는다.
- 품질은 clean retrieval, Sionic 9 최종 평가, broad/multilingual 회귀, long/noise를 함께 본다.
- trainable parameter 수, optimizer 종류, activation checkpoint, sequence-length histogram, peak allocated/reserved VRAM을 기록한다.

## 사전 판단

현재 PyTorch 2.13 fused AdamW가 이 환경에서 BF16 moment를 만드는 것을 확인했다. 따라서 FP32 master/moment를 가정한 120GB 추정보다 작지만, encoder weight·gradient·BF16 moments만 약 56.4GiB이고 activation/temporary/allocator가 추가된다. 표준 full FT는 8B 단일 H100 80GB에서 여유가 작으므로 1-step 안전 probe로 확인하고 바로 긴 학습을 걸지 않는다.

Qwen3-Embedding checkpoint에는 `lm_head.weight`가 없지만 ms-swift loader는 약 621M parameter의 임의 BF16 head를 만든 뒤 embedding forward에서 사용하지 않는다. full/partial/GaLore에서는 이 미사용 head를 반드시 `--freeze_parameters lm_head`로 동결한다.

학습 queue는 격리 train 환경에서 `flash_attn` import를 먼저 확인하고, 설치돼 있으면
FlashAttention 2, 없으면 PyTorch SDPA를 production probe와 LoRA/F2 quality run에
사용한다. 평가용 MTEB 환경의 FlashAttention 2 설치 여부와 혼동하지 않는다.
bitsandbytes, Q-GaLore, DeepSpeed는 기본 의존으로 두지
않는다. 비양자 GaLore는 ms-swift 내부 구현으로 별도 package 없이 쓸 수 있다. 단일
GPU에서는 ZeRO/FSDP 통신 이득도 없다. memory probe는 더 이상 64-token smoke로
낙관 측정하지 않고 mined training row, max length 512, FA2에서 실행한다.

첫 실전 후보는 `LoRA r64 → DoRA r32 → 마지막 4층 partial FT → GaLore full` 순서다. 마지막 4층+final norm은 771.790M trainable parameters이고, all-linear LoRA r64는 174.588M이다. 충분히 어려운 hard negatives에서 앞 설정이 Comsat 격차를 닫지 못할 때 full update를 승격한다. F2LLM의 full FT 결과만 보고 우리 데이터 규모에서도 full FT가 자동으로 낫다고 가정하지 않는다.

모든 1-step 명령은 [`probe_memory.sh`](probe_memory.sh)에 고정했다. `standard_full`은 OOM 가능성을 명시적으로 감수하는 마지막 probe이며 저장을 끄므로 15GB checkpoint를 만들지 않는다.

probe를 통과한 뒤 동일 200K homogeneous data에서 quality run은
[`train_quality.sh`](train_quality.sh)로 실행한다.

```bash
experiments/070_tuning_strategy/train_quality.sh last4
experiments/070_tuning_strategy/train_quality.sh galore
experiments/070_tuning_strategy/train_quality.sh lisa8
```

기본 global batch는 4×accumulation16=64, steps 3,123, LR `6e-6`, FA2/max512다.
모든 emitted microbatch는 단일 source이며 trainer shuffle을 끈다. standard full은
memory probe가 실제로 통과한 경우에만 `ALLOW_STANDARD_FULL=1`로 명시 실행한다.

```bash
ALLOW_STANDARD_FULL=1 experiments/070_tuning_strategy/train_quality.sh standard_full
```

full/partial checkpoint는 LoRA adapter가 아니므로 `merge_embedding_adapter.py`에 넣지
않는다. ms-swift가 저장한 SentenceTransformers checkpoint에서 optimizer를 제외한
배포 artifact는 `scripts/package_full_embedding_checkpoint.py`가 만들고 last-token,
Normalize, 4096-d, 실제 positive-margin probe를 확인한다. 이 gate를 통과한 모델만
Sionic 9/MTEB 평가와 공개 대상으로 승격한다. 야간 queue는 last4 production-length
memory probe가 통과하면 동일 200K data의 quality run을 자동 실행한다.

## 성공 기준

- 같은 token budget에서 quality/VRAM/GPU-hour Pareto frontier를 작성한다.
- Comsat 대비 Sionic 9 평균 개선과 Qwen 대비 multilingual 회귀를 함께 통과한다.
- production 길이별 512/2K/4K peak VRAM을 재측정한다.
