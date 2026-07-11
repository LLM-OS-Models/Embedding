# 1M scale 실행 계약

`performance_1m`은 1,000,000 rows를 실제 한 epoch 상당으로 학습하는 scale
experiment다. 50K/200K에서 방법론을 먼저 비교하고, 그 결과와 독립적으로 base
Qwen3-Embedding-8B의 1M data-scale 효과를 측정한다.

## 데이터

- ko-triplet 600,254
- F2 Korean QA/instruction/hard-negative/cross-lingual 351,146
- official Korean train/task-family 22,600
- KaLM multilingual replay 26,000

합계 1,000,000이다. source cap, revision, exposure는
`configs/performance_data_mix_v1.json`과 build manifest에 고정한다. 이 stage는 법률
250K를 아직 섞지 않는다. 법률/합성은 1M base curve 뒤 별도 adapter로 비교해 일반
성능 회귀와 LawIR/AutoRAG target adaptation을 분리한다.

실제 build SHA-256은 train `094d443e05cc27e4e764b5bfa253cf02c36ec769fbf7cd1e43fd937d73ec3c0a`,
provenance `94334a0ef5dad83169fc8f00fc6705173c606f5976ef8365469fe1bc721b18c1`다.
homogeneous compiler는 999,936 rows/62,496 batches를 내고 source remainder 64 rows를
제외했다. ordered train SHA-256은
`ac39ea777f112ce9c2211c0e2410ddd399c7feb5e487629730d4d086f5b90169`다.

## 학습

| 항목 | 값 |
|---|---|
| base | `Qwen/Qwen3-Embedding-8B@1d8ad4c...` |
| tuner | LoRA r64, alpha128 |
| loss | ms-swift InfoNCE, tau .02, explicit HN 4, fake-negative mask |
| attention | FlashAttention 2 |
| max length | 512 |
| global batch | 16 × accumulation 8 = 128 |
| steps | homogeneous manifest의 `floor(output_rows / 128)`, 약 1 epoch |
| LR | 2e-5 cosine, warmup 5% |
| checkpoint | 500 steps, minimum validation loss 선택 |

source별 row를 먼저 shuffle하고 16-row source-homogeneous microbatch로 나눈 뒤
microbatch 순서만 전역 shuffle한다. source별 16 미만 remainder는 manifest에 기록하고
제외하며 trainer의 추가 shuffle을 끈다. batch16 OOM 시 batch8/accumulation16으로
같은 global batch를 유지한다. 학습 완료 후
adapter reload, safe merge parity, Sionic 9종 전체, 공식 Korean v1 전체를 실행한다.
결과가 나쁘더라도 숨기지 않고 별도 1M 모델/manifest에 연결한다.

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
