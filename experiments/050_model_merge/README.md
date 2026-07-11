# 050 — Checkpoint and adapter merging

서로 다른 data mix와 전문성을 가진 checkpoint/LoRA delta를 평균 또는 SLERP합니다.

비교:

- single best checkpoint
- last-N average
- domain specialist average
- base-to-finetune interpolation
- SLERP

mix coefficient는 internal dev에서만 선택합니다.
