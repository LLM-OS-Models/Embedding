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
동일한 clean near-tie gate를 거친다. Sionic 9와 공식 Korean v1은 clean winner 한 개에만
final-once로 실행한다. 2026-07-17 재시작 후 이미 실행 중이던 첫 Qwen run은 3개 보존 설정으로
시작했기 때문에 최대 3개만 평균될 수 있고, Comsat/1M/KD/target 이후 run은 5개를 보존한다.

## 다음 비교

- single best checkpoint
- last-available-5 FP32 average — 구현·queue 연결 완료, 실제 성능 대기
- domain specialist delta average/linear soup
- base-to-finetune interpolation
- SLERP

서로 다른 LoRA run은 A/B factor를 직접 평균하지 않는다. basis symmetry 때문에 factor 평균이
effective delta 평균과 같지 않으므로, specialist soup은 full delta 또는 merged weight 공간에서
별도 구현·검증한다. mix coefficient는 internal clean dev에서만 선택하며 public benchmark
점수를 보고 되돌려 고르지 않는다.
