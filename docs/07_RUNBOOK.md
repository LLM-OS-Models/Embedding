# Runbook

## 2026-07-17 현재 재개 지점

현재 GPU 작업은 Qwen 학습이 아니라 Nemotron-3 Sionic9 full 평가다. Qwen 200K는
`1875/3123`에서 외부 종료됐고 exact-resumable 지점은 `checkpoint-1750`이다. 남아 있던
`run_frontier_200k_pair_queue.sh` polling PGID는 다음 stage 오기동을 막기 위해 정상
종료했다. Nemotron 평가의 exact restart command, cache와 Qwen legacy validation SHA는
[`36_NEMOTRON3_KOREAN_BASE_DECISION_2026-07-17.md`](36_NEMOTRON3_KOREAN_BASE_DECISION_2026-07-17.md)에
고정한다.

재개 순서는 다음과 같다.

1. Nemotron-3 Sionic9 full summary를 끝낸다.
2. legal v2 10K와 fixed multidomain 1.9K를 Nemotron/Qwen/Comsat에 같은 protocol로 잰다.
3. raw Nemotron이 `0.7930`을 넘고 clean guard 안이면 rights-safe 공개 데이터의 짧은
   target adaptation만 한다.
4. 그렇지 않으면 Qwen `checkpoint-1750`을 원 legacy validation으로 exact resume한다.

새 Hugging Face 산출물은 dataset·checkpoint·clean winner·final model 모두 public이
기본이다. 공개 training manifest는 `release_eligible=true`, 빈 `release_blockers`, public
visibility와 row별 `source/revision/license/redistribution_allowed=true`를 가져야 한다.
모델은 pinned upstream과 base license가 모두 확인돼야 한다. 이 gate를 못 통과한 기존
performance track은 공개 업로드하지 않는다. 고정 model-selection holdout만 비공개다.

## Pinned repositories

| Repository | Commit | 용도 |
|---|---|---|
| QwenLM/Qwen3-Embedding | `44548aa5f0a0aed1c76d64e19afe47727a325b8f` | 공식 inference/evaluation/training guide |
| modelscope/ms-swift | `3d61b9318b27fdd5659e530cd36db7f4ce740fd7` | 실제 embedding LoRA/InfoNCE trainer |

Qwen 저장소에는 원 논문의 150M/12M 생성·필터링 전체 학습 코드가 없습니다. 후속학습은 ms-swift에 위임됩니다.

## Hardware

```text
NVIDIA H100 80GB HBM3 x 1
Python 3.10.12
CUDA visible: 1 GPU
```

## Environment

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -e third_party/ms-swift
```

2026-07-11 실제 실행에서는 시스템 PyTorch 2.5가 최신 ms-swift의 FSDP2 import와 호환되지 않았습니다. 학습은 별도 격리 환경 `.venv-train`에 최신 PyTorch를 고정하고, 첫 smoke에서는 외부 FlashAttention ABI 대신 PyTorch SDPA를 사용합니다. 정확한 버전은 성공한 run manifest에 기록합니다.

기본 학습/MTEB 환경을 변경하지 않는 FA2 학습 후보는 다음으로 준비한다.

```bash
scripts/bootstrap_train_fa2_env.sh
```

이 명령은 NVIDIA system PyTorch와 기존 FlashAttention을 상속한 별도
`.venv-train-fa2`만 만든다. import pass는 성능·정확도 pass가 아니다. 장기 1M 및 법률
queue는 시작 직전 Qwen3-Embedding-8B LoRA 1-step backward probe까지 성공해야 이
환경과 `flash_attention_2`를 선택한다. 그렇지 않으면 `.venv-train + sdpa`로 자동
복귀한다. 활성 학습 도중에는 backend나 environment를 바꾸지 않는다.
Python overlay는 [`train-fa2-overlay.txt`](../requirements/train-fa2-overlay.txt)에
고정하고, system torch/flash-attn의 실제 version은 probe log에 함께 남긴다.

설치가 끝나면 package/version snapshot을 `artifacts/environment/`에 저장합니다.

평가와 대규모 ANN candidate mining은 `.venv-mteb`로 분리한다. 2026-07-11 실제
환경은 MTEB 2.18.0/pinned checkout `193e3f66`, FAISS CPU 1.14.3, NumPy 1.26.4다.
FAISS 최신 wheel이 NumPy 2.x를 끌어오면 이 H100 이미지의 RAPIDS/ModelOpt `<2`
제약과 충돌하므로 extras 파일의 두 버전을 함께 설치한다.

```bash
scripts/bootstrap_mteb_env.sh
```

이 bootstrap은 host에 `ensurepip`가 없는 재시작 환경에서도 repository-local
`virtualenv` fallback을 사용하고, 마지막에 MTEB/FAISS/NumPy exact version과
SentenceTransformers/Transformers/Torch/FlashAttention import gate를 실행한다.

10K mining은 exact blockwise dot product를 사용한다. 250K–1M은 FAISS IVF/HNSW로
candidate pool만 만들고, 최종 positive-relative filter와 teacher score는 별도 정확
단계에서 계산한다. approximate ANN score만으로 false negative를 확정하지 않는다.

## Important explicit settings

현재 ms-swift의 InfoNCE default temperature는 `.1`입니다. 과거 Qwen training doc의 `.01`과 다르므로 모든 run에서 다음 값을 명시합니다.

```bash
export INFONCE_TEMPERATURE=0.02
export INFONCE_USE_BATCH=true
export INFONCE_HARD_NEGATIVES=1
export INFONCE_MASK_FAKE_NEGATIVE=true
export INFONCE_FAKE_NEG_MARGIN=0.1
```

Qwen exact denominator ablation에서만 다음을 켭니다.

```bash
export INFONCE_INCLUDE_QQ=true
export INFONCE_INCLUDE_DD=true
```

## Execution order

1. dependency import/GPU smoke
2. model download and inference parity
3. dataset manifest and target blocklist
4. 100-sample overfit test
5. 5K smoke run
6. 50K pilot
7. 300K~500K Stage A
8. hard-negative mining and verifier
9. Stage B
10. merge and blind evaluation

각 실행 명령은 해당 `experiments/<id>/commands.sh`와 `run_manifest.json`에 남깁니다.
