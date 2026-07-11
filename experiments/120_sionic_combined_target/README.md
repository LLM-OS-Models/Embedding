# 120 — Sionic combined target model

SQuAD 10% + health 10% + AutoRAG domains 10% + Korean legal 15% + general replay
55%를 한 400K curriculum으로 학습한다.

- base: 1M general winner
- input negatives: 각 current-student pool24 quantile HN7
- batch: source-homogeneous 16-row microbatch
- selection: Sionic 9 macro + official Korean + clean regression
- disclosure: `target-adapted-sionic-combined-v1`

실행: [`scripts/run_sionic_combined_adaptation_queue.sh`](../../scripts/run_sionic_combined_adaptation_queue.sh)

설계: [`docs/28_SIONIC_COMBINED_TARGET_MODEL.md`](../../docs/28_SIONIC_COMBINED_TARGET_MODEL.md)
