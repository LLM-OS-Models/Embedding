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

표준 mixed-precision AdamW full FT는 weights, gradients, FP32 optimizer state와 구현에 따른 master weights 때문에 8B 단일 H100 80GB에서 여유가 거의 없거나 OOM일 가능성이 높다. 따라서 1-step 안전 probe로 사실을 확인하고, 바로 긴 학습을 걸지 않는다.

첫 실전 후보는 LoRA r64와 top-layer partial tuning이다. 충분히 어려운 hard negatives에서 이 둘이 Comsat 격차를 닫지 못할 때 memory-efficient full FT를 승격한다. F2LLM의 full FT 결과만 보고 우리 데이터 규모에서도 full FT가 자동으로 낫다고 가정하지 않는다.

## 성공 기준

- 같은 token budget에서 quality/VRAM/GPU-hour Pareto frontier를 작성한다.
- Comsat 대비 Sionic 9 평균 개선과 Qwen 대비 multilingual 회귀를 함께 통과한다.
- production 길이별 512/2K/4K peak VRAM을 재측정한다.
