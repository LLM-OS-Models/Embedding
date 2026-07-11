# 100 — Sionic PublicHealthQA health-domain adaptation

F2LLM-v2 medical QA/instruction/flashcard 100K를 current-student quantile hard negative로
재채굴하고 general 1M과 50:50 replay하는 격리 실험이다.

- 직접 PublicHealthQA test/train-family 데이터는 사용하지 않는다.
- raw shard의 15-task query/evaluation-text exact overlap은 0이다.
- 영어 중심 health signal이 Qwen의 multilingual alignment를 통해 한국어에 전달되는지
  검증한다.
- PublicHealthQA 하나가 아니라 Sionic 9 macro, 공식 Korean, clean 회귀로 승격한다.
- 결과 model/data는 `health-domain-adapted`, `release_eligible=false`로 공개한다.

실행 wrapper는 [`scripts/run_sionic_health_adaptation_queue.sh`](../../scripts/run_sionic_health_adaptation_queue.sh),
공통 engine은 [`scripts/run_sionic_squad_adaptation_queue.sh`](../../scripts/run_sionic_squad_adaptation_queue.sh),
근거와 수치는 [`docs/26_SIONIC_PUBLIC_HEALTH_ADAPTATION.md`](../../docs/26_SIONIC_PUBLIC_HEALTH_ADAPTATION.md)에
고정한다.
