# 050 — Checkpoint and adapter merging

서로 다른 data mix와 전문성을 가진 checkpoint/LoRA delta를 합치기 위한 stage다. 현재
production에 연결된 방법은 **동일 Trainer trajectory의 마지막 최대 5개 LoRA checkpoint를
FP32 arithmetic mean**한 뒤, 기존 safe PEFT merge/parity gate를 통과시키는 것이다.

## 구현된 last-available-5 평균

`scripts/average_lora_checkpoints.py`는 best-validation checkpoint를 anchor로 사용하되,
anchor가 속한 정확히 같은 version directory에서 최신 checkpoint를 최대 5개 고른다.
별도 retry/version directory를 섞지 않는다. 모든 adapter config, tensor key·shape·dtype가
같아야 하고 모든 floating tensor가 finite여야 한다. 각 tensor는 FP32로 누적·평균하고
FP32 safetensors로 저장한다. output은 임시 sibling directory에서 완성된 뒤 atomic rename된다.

```bash
.venv-train-fa2/bin/python scripts/average_lora_checkpoints.py \
  --run-dir outputs/<run> \
  --anchor-checkpoint outputs/<run>/<version>/checkpoint-<best-step> \
  --output-dir artifacts/adapters/<run>-last-available5-fp32-average \
  --last-n 5 --minimum-checkpoints 2
```

`average_report.json`은 실제 step 목록, source/output SHA-256, config SHA-256, dtype와 tensor
수를 기록한다. `merge_embedding_adapter.py`가 이 report를 weights/config에 다시 결속해
검증하고, LoRA 대 merged embedding parity가 통과한 경우만 full model을 만든다.

`run_post_training_eval_queue.sh`는 single best와 이 averaged model을 모두 Grade-I clean
legal 및 noise robustness에 넣는다. 평균 모델이 자동 승격되는 것은 아니며, 두 모델 모두
동일한 clean near-tie gate를 거친다. 같은 queue는 200K lineage 비교 때 한 번, 1M/KD 및
retrieval/SQuAD/health/AutoRAG/legal/combined가 끝난 뒤 전체 run을 대상으로 다시 한 번
실행된다. 두 번째 mandatory gate가 전 stage의 best-vs-average clean winner를 고른다.
Sionic 9와 공식 Korean v1은 각 clean selection의 winner 한 개에만 실행한다. 2026-07-17
재시작 후 이미 실행 중이던 첫 Qwen run은 3개 보존 설정으로 시작했기 때문에 최대 3개만
평균될 수 있고, Comsat/1M/KD/target 이후 run은 5개를 보존한다.

## 다음 비교

- single best checkpoint
- last-available-5 FP32 average — 구현·queue 연결 완료, 실제 성능 대기
- domain specialist full-weight linear soup — 구현·queue 연결 완료, 실제 성능 대기
- base-to-finetune interpolation
- SLERP

서로 다른 LoRA run은 A/B factor를 직접 평균하지 않는다. basis symmetry 때문에 factor 평균이
effective delta 평균과 같지 않다. [`merge_full_model_soup.py`](../../scripts/merge_full_model_soup.py)는
각 adapter를 이미 fold한 full model만 받아 source shard/evidence hash, architecture,
SentenceTransformers metadata, tensor key/shape/dtype/finite를 확인한다. tensor는 FP32로
가중 누적한 뒤 BF16 reference shard layout으로 저장하고 `soup_report.json`을 atomic하게 만든다.

[`run_model_soup_queue.sh`](../../scripts/run_model_soup_queue.sh)의 coefficient는 평가 전에
고정돼 있다.

- general `.5` + combined `.5`
- general `.5` + retrieval/SQuAD/health/AutoRAG/legal 각 `.1`
- general `.25` + combined `.25` + specialist 5개 각 `.1`
- combined `.5` + specialist 5개 각 `.1`

public Sionic/MTEB는 soup 생성이나 coefficient 선택에 사용하지 않는다. 네 candidate는
single-best/checkpoint-average와 같은 최종 Grade-I clean/robustness selector에 들어간다.
source가 하나라도 없으면 그 balanced variant를 만들지 않으며, weight 합이 `1±1e-9`가
아니거나 shard index/payload가 어긋나면 실패한다. adaptive/greedy coefficient는 더 넓은
clean general-domain dev가 생기기 전까지 보류한다.
