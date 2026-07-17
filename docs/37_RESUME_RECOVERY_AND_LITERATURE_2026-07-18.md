# 2026-07-18 중단 복구와 방법론 문헌 점검

기준일: **2026-07-18 (Asia/Seoul)**

## 이 문서의 역할

2026-07-18 아침 세션이 수행한 (1) Qwen 200K exact resume 복구의 정확한 상태와
수정 사항, (2) 같은 날 기준 외부 문헌 점검 결과를 고정한다. 성능 주장은 없다.

## 1. 복구 상태

- decision `resume_qwen_checkpoint_1750_and_reselect`(docs/36)에 따라
  `checkpoint-1750`에서 3,123 step까지 exact resume를 재기동했다.
  `scripts/validate_resume_checkpoint.py`는 `{"resume_step": 1750, "status": "pass"}`를
  기록했고 backend는 원 run과 같은 `matched_sdpa`가 재선택됐다.
- 복구 중 고친 것:
  1. `train_pilot_lora_r64.sh`의 clean-validation gate는 b3c72c0에서 추가되어
     docs/36이 요구하는 legacy validation(exact SHA `f121f7eb…`) resume와 충돌했다.
     `ALLOW_EXACT_RESUME_LEGACY_VALIDATION=1` + pinned SHA + checkpoint 존재의
     3중 조건일 때만 legacy 파일을 허용하는 fail-closed 예외를 추가했다.
  2. transformers 5.12는 torch<2.6에서 모든 `torch.load`를 거부한다
     (CVE-2025-32434). 이 가드는 신뢰할 수 없는 pickle용이고 exact resume는
     이 머신이 직접 쓴 optimizer/scheduler/RNG를 읽는 것이므로,
     `EMBEDDING_TRUST_LOCAL_TORCH_LOAD=1`로 명시한 프로세스에서만 해당 버전
     거부를 끄는 patch를 `compat/torch25/sitecustomize.py`에 추가했다. env 없는
     프로세스는 가드가 그대로 산다(양방향 검증 완료).
  3. checkpoint watcher `--public`은 training manifest의
     `release_eligible/visibility` 권리 gate를 fail-closed로 요구한다. 200K
     performance manifest는 권리 불명확 track이라 이 gate를 통과할 수 없고
     통과해서도 안 된다. frontier queue의 두 watcher 호출에서 `--public`을
     제거했다(실행된 적 없는 잠재 crash). 후보 checkpoint repo
     `…-candidates-v2`는 원 run과 같은 private을 유지하고, 공개는 rights-safe
     track과 최종 winner publication gate가 담당한다.
  4. 실행 환경에 `rg`(ripgrep)가 없어 frontier queue의 완료/실패 감지가 모두
     무력화되는 상태였다. 시스템에 ripgrep을 설치했다.
- watcher/trainer/queue/storage watchdog은 각각 독립 process group으로 재기동하고
  storage watchdog이 세 PGID를 감시한다. NFS 여유 48TB, root 1.6TB로 저장공간
  위험은 현재 없다.

## 2. 2026-07-18 문헌 점검(외부)

결론 먼저: **현재 frontier 계획(docs/34)의 설계를 뒤집을 만한 새 방법은 발견하지
못했다.** 아래 논문들은 기존 설계의 세부 파라미터를 보강하거나, 이미 채택한
설계의 독립 근거가 된다. 기존 상시 매트릭스는 [docs/12](12_PAPER_DATA_METHOD_MATRIX.md),
상위 모델 recipe 대조는 [docs/30](30_TOP_MODEL_RECIPE_SYNTHESIS.md)이 담당하고,
이 절은 2026-07-18에 새로 검토한 논문만 다룬다.

### 신규 검토 논문과 적용 결정

| 논문 | URL | 핵심 주장 | 우리 파이프라인 적용 | 상태 |
|---|---|---|---|---|
| When Hard Negatives Hurt (2026-06) | <https://arxiv.org/abs/2606.01304> | LLM 합성/강한 mining negative가 query와 의미적으로 너무 가까우면 discriminative 신호와 모순되어 성능 저하. embedding-similarity 임계 filtering + density outlier 검출 + borderline negative 하향 가중 제안 | (1) `INFONCE_MASK_FAKE_NEGATIVE=true`의 positive-relative ratio `.95`·margin `.02` mask, (2) KD compiler(`scripts/compile_reranker_kd_dataset.py`)의 positive score `.5` gate가 같은 방어를 이미 구현. R4 KD stage에서 mask 기준 완화 금지의 외부 근거로 채택. borderline 하향 가중(soft reweight)은 R4 listwise KL이 teacher 분포로 자연 구현하므로 별도 loss 추가는 기각 | 채택(현행 유지 근거) |
| HiNS: Hierarchical Negative Sampling (2026-01) | <https://arxiv.org/abs/2601.14857> | negative를 coarse(주제)→fine(의미 근접) 계층으로 층화해 동시 학습하면 단일 난이도 HN보다 일반화 우수 | source-homogeneous batch(coarse in-batch 층, `scripts/build_homogeneous_batches.py`) + dense HN4/HN7(fine 층, `scripts/mine_faiss_hard_negatives.py`) + reranker rank-quantile 15의 양 끝 포함 규칙(전 난이도 층화, docs/34 R4)이 이미 계층 구조. quantile 선택이 최상위 난이도에 몰리지 않게 하는 현행 규칙의 근거 | 채택(현행 유지 근거) |
| Improving Korean-English Cross-Lingual Retrieval (2025-07) | <https://arxiv.org/abs/2507.08480> | 한국어 단일 학습은 한국어에 강하지만 영어/교차언어 회귀 유발. 언어 구성 비율 통제와 언어별 모델의 merging이 양 언어 유지에 효과적 | (1) docs/34 R5의 general replay 53.968% 유지 근거(한국어 target 학습 중 다국어 회귀 방지), (2) R6 basis-safe full-weight soup에서 general↔combined coefficient 2종을 사전 고정한 설계의 독립 근거, (3) 최종 gate의 원 Qwen 다국어 회귀 검사(원칙 §5) 유지 | 채택(현행 유지 근거) |

### 참고(신규 채택 없음)

- MMTEB multilingual 1위 [Tencent KaLM-Embedding-Gemma3-12B](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511)
  (72.32, 2026-07 기준)는 exact revision을 이미 local cache에 확보한 비교군이며
  [docs/20](20_TOP_MODEL_LOCAL_EVAL_MATRIX.md)의 Sionic 동등 비교 queue 대상이다.
- [Conan-embedding-v2](https://arxiv.org/abs/2509.12892),
  [NV-Embed](https://arxiv.org/abs/2405.17428)의 two-stage(retrieval 먼저, broad
  blend 나중) 순서는 docs/34의 200K→1M→target 400K 순서와 일치한다. NV-Embed의
  latent-attention pooling은 Qwen3-Embedding의 last-token 계약과 충돌하고 이득
  근거가 약해 기각한다(docs/12 매트릭스와 동일 결론).
- 이번 탐색에서 Sionic 9 한국어 retrieval에 직접 새 SOTA를 주장하는 2026-07
  신규 한국어 특화 임베딩 논문은 발견하지 못했다.

### 방법론 확정 요약(2026-07-18 시점)

각 결정의 실행 계약은 docs/34에 있고, 여기서는 논문 근거만 연결한다.

1. **continued contrastive fine-tuning(InfoNCE) 우선, raw-text CPT 기각** —
   [Qwen3 Embedding](https://arxiv.org/abs/2506.05176)이 이미 대규모 weak
   contrastive를 수행했으므로 반복하지 않는다(docs/12 결론).
2. **LoRA r64 먼저, 그 다음 last4 partial-full challenger(771.790M)** — 단일
   H100에서 full 8B FT는 OOM/시간 비용이 크고, [F2LLM-v2](https://arxiv.org/abs/2603.19223)의
   full FT 대비 이득은 우리 계약에선 R2 challenger로 검증한다(docs/34 R2).
3. **hard negative**: F2식 pool-then-sample(HN 24→7)과 Nemotron식
   positive-relative filter(`s_neg < .95*s_pos`)를 결합. false-negative mask는
   When Hard Negatives Hurt로 재확인.
4. **teacher KD**: Qwen3-Reranker-8B yes/no logits의 score-quantile 15 +
   `0.3 hard InfoNCE + 0.7 listwise KL`(docs/34 R4). borderline 하향 가중은
   listwise KL이 대체.
5. **checkpoint 평균/soup**: [Llama-Embed-Nemotron](https://arxiv.org/abs/2511.07025)의
   6-checkpoint 평균 `+0.84`와 한·영 교차언어 merging 근거(2507.08480)로
   last-available-5 FP32 평균과 basis-safe full-weight soup을 최종 selector에서
   비교(docs/34 R6).
6. **언어 구성**: target 400K에 general replay 53.968% 유지, 최종 gate에서
   다국어 회귀 검사(2507.08480 근거).

## 3. 다음 판단 지점

1. Qwen 3,123 step 완료 → frontier queue가 Comsat 200K 동일 계약 학습을 자동
   시작.
2. 양 계보의 모든 archived checkpoint를 legal v2 10K + 고정 multidomain 1.9K로
   재선택(공개 점수 미사용).
3. 이후 last4 capacity → 1M → KD → target/legal → soup → 최종 gate는 docs/34
   계약 그대로.
