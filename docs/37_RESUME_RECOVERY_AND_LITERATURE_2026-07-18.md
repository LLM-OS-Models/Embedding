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
못했다.** 아래 두 편은 기존 설계의 세부 파라미터를 보강하는 근거로 쓴다.

### When Hard Negatives Hurt (arXiv 2606.01304, 2026-06)

- LLM 합성/강한 mining hard negative가 query와 의미적으로 너무 가까우면
  discriminative 학습 신호와 모순되어 성능을 해친다는 실증. embedding similarity
  임계 filtering + density 기반 outlier 검출 + borderline negative 하향
  가중(loss reweighting)을 제안.
- 우리 대응: 이미 `INFONCE_MASK_FAKE_NEGATIVE=true`(positive-relative ratio
  `.95`, margin `.02`) false-negative mask와 KD compiler의 positive score `.5`
  gate가 같은 방향의 방어를 구현한다. R4 KD stage에서 rank-quantile 15의 양 끝
  선택 시 teacher yes-probability 상위 borderline negative의 mask 기준을 완화하지
  말 것(현행 유지)의 근거로 기록.

### HiNS: Hierarchical Negative Sampling (arXiv 2601.14857, 2026-01)

- negative를 coarse(주제 수준)→fine(의미 근접) 계층으로 층화해 동시에 학습하면
  단일 난이도 hard negative보다 일반화가 좋다는 보고.
- 우리 대응: source-homogeneous batch(coarse in-batch negative 층) + dense HN4/HN7
  (fine 층) + reranker score-quantile 15(전 난이도 층화)가 이미 계층 구조를
  형성한다. 1M/KD stage에서 quantile 선택이 최상위 난이도에 몰리지 않게 양 끝
  포함 규칙을 유지한다.

### 기타

- MMTEB multilingual 1위는 Tencent `KaLM-Embedding-Gemma3-12B`(72.32,
  2026-07 기준)로 이미 local cache에 exact revision을 확보한 비교군이다.
- Conan-embedding-v2, NV-Embed 계열의 two-stage(retrieval 먼저, broad blend
  나중) 순서는 docs/34의 200K→1M→target 400K 순서와 일치한다.

## 3. 다음 판단 지점

1. Qwen 3,123 step 완료 → frontier queue가 Comsat 200K 동일 계약 학습을 자동
   시작.
2. 양 계보의 모든 archived checkpoint를 legal v2 10K + 고정 multidomain 1.9K로
   재선택(공개 점수 미사용).
3. 이후 last4 capacity → 1M → KD → target/legal → soup → 최종 gate는 docs/34
   계약 그대로.
