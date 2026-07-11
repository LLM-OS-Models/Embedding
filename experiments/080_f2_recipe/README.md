# 080 — F2LLM-v2 recipe ablation

목적은 F2LLM-v2의 이름만 빌리는 것이 아니라, 공개 코드의 학습 목적함수를
`Qwen3-Embedding-8B` continued tuning에 정확히 이식했을 때 이득이 있는지 같은
데이터와 token budget으로 검증하는 것이다.

## F2 코드에서 직접 확인한 차이

- F2LLM-v2-8B의 출발점은 `Qwen3-8B`이며 `Qwen3-Embedding-8B`가 아니다.
- 한 source 안에서 batch를 만들고, query와 다른 row의 positive만 사용하는
  in-batch cross entropy를 계산한다.
- 별도로 자기 positive와 explicit hard negatives만 넣은 cross entropy를 계산해
  두 loss를 더한다.
- 데이터에는 24개 hard-negative 후보가 있고 매 step 7개를 표집한다.
- 기본 temperature는 `0.05`다.
- MRL은 `8, 16, ..., 4096`차원을 모두 쓰며 각 항의 가중치는
  `sqrt(dim / 4096) / 10`이다.

ms-swift 기본 `infonce`는 모든 row의 positive와 explicit negatives를 하나의 큰
denominator에 넣는다. [`f2_dual_loss_plugin.py`](f2_dual_loss_plugin.py)는 F2처럼
`in-batch-positive CE + own-hard-negative CE`로 분리한다. 두 목적함수 중 어느 쪽이
한국어 검색에 좋은지는 사전에 단정하지 않고 아래처럼 비교한다.

| Run | Loss | Temperature | MRL |
|---|---|---:|---|
| A | ms-swift combined InfoNCE | 0.02 | off |
| B | F2 dual CE | 0.05 | off |
| C | F2 dual CE | 0.02 | off |
| D | F2 dual CE | 0.05 | exact F2 weights |

모두 같은 mined JSONL, LoRA r64, microbatch, optimizer step 수를 사용한다. 10K에서
학습 신호와 private dev를 먼저 보고, public Sionic 9는 최종 후보만 측정한다.

## 실행

```bash
.venv-train/bin/python experiments/080_f2_recipe/test_f2_dual_loss.py

# B
experiments/080_f2_recipe/train_pilot_f2_dual_lora_r64.sh

# C
F2_DUAL_TEMPERATURE=.02 \
RUN_NAME=qwen3-embedding-8b-ko-hn10k-f2dual-t002-lora-r64 \
  experiments/080_f2_recipe/train_pilot_f2_dual_lora_r64.sh

# D
USE_F2_MRL=1 \
RUN_NAME=qwen3-embedding-8b-ko-hn10k-f2dual-mrl-lora-r64 \
  experiments/080_f2_recipe/train_pilot_f2_dual_lora_r64.sh
```

이 코드는 단일 H100에서 먼저 검증한다. 분산 실행 시 positive embedding을 rank 간
gather하고 로컬 rank의 tensor만 gradient를 보존하며, 서로 다른 rank batch 크기는
즉시 오류로 처리한다.

