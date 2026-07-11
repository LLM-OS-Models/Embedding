# 1M scale 실행 계약

`performance_1m`은 1,000,000 rows를 실제 한 epoch 상당으로 학습하는 performance
curriculum이다. 50K/200K에서 여러 loss와 update를 먼저 비교하고, Sionic 9에서 선택된
safe-merged winner가 있으면 그 모델에서 이어 학습한다. selection 또는 merge evidence가
없을 때만 pinned Qwen3-Embedding-8B로 fallback한다. 따라서 기본 실행 결과는 순수한
base-only data-size ablation이 아니라 최종 성능을 우선한 연속 학습 결과다.

## 데이터

- ko-triplet 600,254
- F2 Korean QA/instruction/hard-negative/cross-lingual 351,146
- official Korean train/task-family 22,600
- KaLM multilingual replay 26,000

합계 1,000,000이다. source cap, revision, exposure는
`configs/performance_data_mix_v1.json`과 build manifest에 고정한다. 이 stage는 법률
250K를 아직 섞지 않는다. 법률/합성은 1M base curve 뒤 별도 adapter로 비교해 일반
성능 회귀와 LawIR/AutoRAG target adaptation을 분리한다.

실제 critical-hash decontaminated build SHA-256은 train
`056abaf6b72d7521f9f60483e7ee1267011e3fe4803ee0429e19db4df323d2fa`, provenance
`81dac008cc9523cb64983aaa8f623b7cb81c5075f6cc045ac6fd74e04e6bd4f4`다.
homogeneous compiler는 999,936 rows/62,496 batches를 내고 source remainder 64 rows를
제외했다. ordered train SHA-256은
`7f2641a0a0256e89b2abb3462aa2c8b415b3e605fd4ca413178d2ad4abfc9009`다. 이 revision은
source-homogeneous length bucketing으로 random order 대비 padding proxy를 줄인 것이다.

15-task critical text-hash gate가 이전 mix의 row 2,839개를 제거하고 같은 source의 다른
row로 cap을 재충전했다. 원인은 retrieval evaluation query match 26 occurrences와
undeclared KLUE-STS-family text 8,435 occurrences다. final ordered audit는 critical
query/evaluation-text unique match 0, declared train-family 29,226, shared retrieval-corpus
23,379를 기록한다. 따라서 clean zero-shot은 아니지만 직접 critical overlap은 차단했다.

기본 performance run은 이 원본 order를 즉시 학습하지 않는다. post-training winner로
1M query와 unique positive corpus를 encode하고 FAISS IVFFlat(`nlist=1024`,
`nprobe=32`, `search_k=256`, training points 50K, CPU threads 64)에서 24개 후보를 찾은 뒤 `.95*s_pos`보다 낮은 7개를
current-student negative로 다시 고른다. 선택된 score는 float32 exact dot으로
재계산하고 own positive/query exact match를 제외한다. mining/provenance projection/
homogeneous compiler 중 하나라도 실패할 때만 위 원본 999,936-row curriculum으로
fallback하며 log와 model training manifest가 실제 선택을 구분한다. 이 경로는 target
train-family 노출이 있으므로 clean zero-shot이 아니라 `performance target-adapted`다.

## 학습

| 항목 | 값 |
|---|---|
| base | post-training Sionic winner의 safe merge; 없으면 `Qwen/Qwen3-Embedding-8B@1d8ad4c...` |
| tuner | LoRA r64, alpha128 |
| loss | ms-swift InfoNCE, tau .02, current-student mined explicit HN 7(원본 fallback 4), fake-negative mask |
| attention | 격리 NVIDIA PyTorch 환경의 실제 8B backward probe 성공 시 FA2, 아니면 SDPA |
| max length | 512 |
| global batch | 16 × accumulation 8 = 128 |
| steps | homogeneous manifest의 `floor(output_rows / 128)`, 약 1 epoch |
| LR | **1e-5** cosine, warmup 5%; 50K의 2e-5가 step 80 이후 악화된 실측 반영 |
| checkpoint | **250 steps**, minimum validation loss 선택; 초기 최적점 누락 방지 |

source별 row를 먼저 shuffle하고 16-row source-homogeneous microbatch로 나눈 뒤
microbatch 순서만 전역 shuffle한다. source별 16 미만 remainder는 manifest에 기록하고
제외하며 trainer의 추가 shuffle을 끈다. batch16 OOM 시 batch8/accumulation16으로
같은 global batch를 유지한다. 학습 완료 후
adapter reload, safe merge parity, Sionic 9종 전체, 공식 Korean v1 전체를 실행한다.
결과가 나쁘더라도 숨기지 않고 별도 1M 모델/manifest에 연결한다.
current-student quantile-HN 파생 curriculum을 사용한 경우 exact train/provenance/mining
audit/manifest는 학습 종료 직후 GPU 평가와 겹쳐 백그라운드로
`LLM-OS-Models/korean-embedding-performance-1m-quantile-hn7-v1`에 공개한다.
업로드 전에 final ordered train/provenance를 전수 감사하고 report input SHA, row 수,
row-hash/batch contract가 실제 publisher 입력과 일치해야 한다. report도 dataset card와
`metadata/training_data_quality_audit.json`에 포함한다.

```bash
WAIT_PID=<post-training-eval-pid> bash scripts/run_scale_1m_queue.sh
```

## 다음 단계

1. 50K/200K/1M data-size curve를 task별로 비교한다.
2. 1M이 회귀하면 쉬운 ko-triplet 비중을 줄이고 current-student loss-active row와
   F2/법률 hard-negative row로 교체한다.
3. 법률 250K는 bootstrap negative 그대로 쓰지 않고 dense/reranker refresh한다.
4. MRL, F2 dual CE, checkpoint soup은 같은 1M token budget에서 비교한다.
5. Sionic 9 우승 후보만 clean comprehensive와 multilingual regression gate로 보낸다.
