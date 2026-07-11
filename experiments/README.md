# Experiments

모든 실험은 다음 최소 산출물을 가집니다.

- `README.md`: 가설, 변경점, 성공/실패 기준
- `config/`: 실행 configuration
- `scripts/`: 데이터/학습/평가 명령
- `runs/<run_id>/manifest.json`: git commit, model/data revision, env, seed
- `runs/<run_id>/metrics.json`: per-task metric
- `runs/<run_id>/notes.md`: 실패와 관찰

실험 사이에 결과 파일을 복사해 덮어쓰지 않습니다. 공통 데이터는 root `data/`, 모델 결과는 `artifacts/`에 두고 manifest로 참조합니다.

현재 비교 축은 `000`–`090`이다. LoRA와 full-parameter tuning의 비용·품질 비교는
[`070_tuning_strategy`](070_tuning_strategy/), F2 dual-loss/MRL은
[`080_f2_recipe`](080_f2_recipe/), SQuADKorV1 train-family adaptation은
[`090_sionic_squad_adaptation`](090_sionic_squad_adaptation/), health-domain adaptation은
[`100_sionic_health_adaptation`](100_sionic_health_adaptation/), AutoRAG domain adaptation은
[`110_sionic_autorag_adaptation`](110_sionic_autorag_adaptation/)에 기록한다.
