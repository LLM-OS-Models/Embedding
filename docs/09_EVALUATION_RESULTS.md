# Evaluation log

이 파일에는 실제로 실행한 결과만 기록합니다. 모델 카드의 성능표는 여기의 raw result와 revision을 기준으로 생성합니다.

## 2026-07-15 200K training backend admission

같은 격리 NVIDIA PyTorch 2.5 / flash-attn 2.4.2 환경에서 Qwen3-Embedding-8B
LoRA r64, batch 16, accumulation 4, max length 512를 SDPA와 FA2로 각각 5
optimizer-step 실행했다. 199,904행 원본을 source와 length strata별로 투영한 같은
320행 subset을 사용했고, subset train SHA-256은
`155ce90a20fb9f4dacce3244a43962bd9a96f8fc765365d54295d16f2cc503b9`다.

| Backend | Seconds / optimizer-step | Peak VRAM | Process |
|---|---:|---:|---|
| SDPA | 29.30 | 63.44 GiB | pass |
| FlashAttention 2 | **27.10** | **61.52 GiB** | pass |

FA2 speedup은 **1.08118x**로 고정 입장 기준 1.05x를 통과했다. 원본 ordered train
SHA-256은 `8e2731ab25299ff558af675f067b253a6ce4375a850aa925acfe3b3117505e3c`이며,
양쪽 loss도 반올림 기준 각 step에서 일치했다. 따라서 200K r64 run은 FA2를 쓰고,
이 admission의 backend·data lineage가 달라지면 재사용하지 않는다. 첫 실패 시도의
report/log도 `attempts/20260715T082837Z/`에 보존했으며 성능 수치로 사용하지 않는다.

## 2026-07-11 training/adapter pipeline verification

Run: `qwen3-embedding-8b-ko-smoke-r32/v1-20260711-210119`, checkpoint 20, Qwen base revision `1d8ad4ca`, H100 80GB, BF16 SDPA.

| Check | Measured |
|---|---:|
| Trainable / total | 87.294M / 8.276B (1.0548%) |
| 20-step wall time | 43.81 s |
| Peak training VRAM | 17.07 GiB |
| Fresh-process shape | 3 × 4096 |
| Probe positive cosine | 0.66728 |
| Probe negative cosine | 0.22148 |
| Probe margin | **0.44580** |
| Adapter SHA-256 | `18965a6d8b1b04c26b9d4651b6d759577a082c7e98c4f33fa7498b44ae54a9de` |

BF16 embedding norms were `0.99856–1.00044`, within the explicit `5e-3` normalization tolerance. Reload status is `pass`. The 32-row validation loss was effectively zero from the start because the supplied negative passages were too easy; this run validates plumbing only and is excluded from all performance leaderboards.

검증된 checkpoint-20 adapter는 `LLM-OS-Models/qwen3-embedding-8b-ko-smoke-20260711`에 private artifact로 업로드했다(HF commit `0f949faf5d01edc549fb11745bd26da3af7addc3`). allowlist에는 adapter weight/config, sanitized verification/manifest/model card만 포함되며 raw examples, optimizer/scheduler, trainer/RNG state, logs, local path는 포함하지 않는다. 데이터 라이선스가 미명시이므로 public 전환은 금지한다.

## 2026-07-11 evaluator parity — AutoRAG

Protocol: `sionic9-fixed-prompt-v1`의 AutoRAG slice, full corpus, NDCG@10, query-only fixed prompt, MTEB `2.18.0` commit `193e3f66`, H100 80GB, normalized cosine/IP.

| Model | Revision | Measured | Card reference | Difference |
|---|---|---:|---:|---:|
| Qwen3-Embedding-8B | `1d8ad4ca` | **0.82765** | 0.8276 | +0.00005 |
| Comsat-embed-ko-8b-preview | `a5cc22b6` | **0.85222** | 0.8518 | +0.00042 |
| F2LLM-v2-8B | `e5725783` | **0.76611** | 0.7678 | -0.00169 |
| PwC-Embedding_expr | `33358978` | **0.78329** | 없음 | — |

Qwen 수치가 반올림까지 일치하고 Comsat/F2도 0.002 이내이므로 task/split/prompt/corpus 설정이 Sionic 표와 대체로 일치합니다. 남은 차이는 모델 revision, MTEB/Transformers version, numeric kernel 차이로 설명 가능한 범위입니다.

### vLLM backend gate

공식 Korean `Ko-StrategyQA`에서 Comsat을 별도 vLLM 0.24 backend로 측정한 값은
`0.83830`이고, pinned SentenceTransformers+FA2 값은 `0.84016`이다. vLLM 65K-token
설정은 약 200 docs/s로 이 workload의 FA2보다 느렸다. 131K-token/1024-seq/95%
설정은 75.85GiB 사용 뒤 추가 activation 할당에서 OOM이 났다. 따라서 속도와
protocol parity 모두에서 현재 Comsat full-corpus 기본값은 FA2이며, vLLM 결과는
별도 backend 실험으로만 보존한다.

Raw cache와 query-level predictions는 `outputs/evaluation/sionic9/` 아래에 보존되며 Git에는 대용량 artifact를 넣지 않습니다. 최종 모델 카드에는 전체 9개가 완료된 run의 hash와 공개 artifact URL을 연결합니다.

## 2026-07-15 restored evaluator compute-profile audit

과거 AutoRAG raw output이 Git/HF에 남아 있지 않아 문서 숫자만으로는 당시 compute
profile을 완전히 감사할 수 없었다. 동일 model/dataset/MTEB revision으로 batch와 attention
backend를 분리 재실행했다.

| Model | Dtype / attention | Batch | AutoRAG NDCG@10 | Raw result SHA-256 |
|---|---|---:|---:|---|
| Qwen3-Embedding-8B | BF16 / SDPA | 2 | 0.82804 | `b0d48954263f14ae01658654a52d41984518f1f8908112b0d382905c65a0ef2c` |
| Qwen3-Embedding-8B | BF16 / FA2 | 2 | 0.82776 | `9c46313e98222dfb4b550d56c8f9a9e3923f42496fed161f7f1e7016d69fb2d2` |
| Comsat-embed-ko-8b-preview | BF16 / FA2 | 2 | **0.85222** | `1b5b371a5791fc6e32f99129696d782d22be3280f97682b7b56e3bc8d588a5ed` |
| Qwen3-Embedding-8B | BF16 / FA2 | 192 | **0.82442** | `79a43fceba481cbf7067eed3c099cc019bf134cc67a5381fb876ae0edcef5681` |
| Comsat-embed-ko-8b-preview | BF16 / FA2 | 192 | **0.85261** | `01a01b8c1cb263151f6fe01d309296b13eb3cc9be6ed90c61cc342182eac5c59` |
| F2LLM-v2-8B | BF16 / FA2 | 192 | 0.76789 | `408a51894578c1ffca9a4d3bfc5ec9b791b6d246e901795b7a2f7587fb1bf1e4` |
| PwC-Embedding_expr | BF16 / SDPA, max 512 | 192 | 0.78473 | `73b822eccf26b91a2f33f089abe1812ba290190ba3e899dbd73c86005bf83796` |

Qwen의 batch 2와 192 차이는 `-0.00334`이므로 BF16 retrieval 결과에서 batch는
단순 처리량 설정이 아니라 결과 계약의 일부다. 데이터 revision은 두 Qwen run 모두
AutoRAG `fd7df84ac089bbec763b1c6bb1b56e985df5cc5c`, model revision은
`1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`로 동일하다. prompt도 최초 고정
protocol의 `Query:` 뒤에 공백을 추가하지 않았다.

Comsat batch-2 재실행은 과거 문서값 `0.85222`를 정확히 복구했다. Qwen batch-2
FA2의 `+0.00011` 차이는 실질 차이로 취급하지 않는다. 이 결과로 legacy 수치의
dataset/prompt parity는 복구됐지만, campaign 승패는 아래 고정 profile끼리만 정한다.

이후 campaign 선택은 **BF16 + FA2 + batch 192 + max length 8192**끼리만 비교한다.
과거 `0.82765`/`0.85222`는 legacy batch-2 parity로 보존하고 새 후보 선택에 사용하지
않는다. evaluator는 runtime contract와 profile hash를 기록하며, 완료 task가 있는
output directory에 다른 batch/backend가 들어오면 fail-closed한다. FP32 safe merge
후보는 FA2가 지원하지 않으므로 evaluator가 SDPA로 전환하고 별도 profile로 기록한다.

F2는 campaign profile을 그대로 적용했다. PwC는 XLM-R position table의 native limit가
512이므로 max 8192/FA2 강제 run은 position-index device assertion으로 실패했고 점수에
사용하지 않았다. 성공 행은 current Hub revision `6c5196980c685db45b58f67bd3be2f79d794351e`,
max 512, SDPA를 사용했다. 서로 다른 architecture에는 각 모델의 지원 범위를 적용하되,
그 runtime 차이를 숨기지 않는다.

## 2026-07-12 Comsat 공식 MTEB Korean v1 로컬 재현

Model revision `a5cc22b651c1b2e51cdd8bf671774ae93584f0ab`, MTEB `2.18.0`,
FlashAttention 2, H100 80GB, task-specific query/document prompt contract로 6/6을
완료했다.

| Task | Type | Score |
|---|---|---:|
| KLUE-TC | Classification | 0.521387 |
| MIRACLReranking | Reranking | 0.684670 |
| MIRACLRetrieval | Retrieval | **0.695260** |
| Ko-StrategyQA | Retrieval | 0.840160 |
| KLUE-STS | STS | 0.863187 |
| KorSTS | STS | 0.794369 |

- Mean(Task): **73.3172** leaderboard points
- Mean(Type): **70.0636** leaderboard points
- Retrieval type mean: **0.767710**
- live board에 가상 삽입한 Borda rank: **6** (`755` points)
- official row rank 재현: `137/137`; complete official rows `101`
- 공식 제출 행이 아니라 exact protocol의 local reproduction

가상 삽입 시 task rank는 Ko-StrategyQA 2, MIRACLReranking 2, KorSTS 20, KLUE-TC
32, KLUE-STS 16, MIRACLRetrieval 1이었다. raw result는
`outputs/evaluation/mteb_korean_v1/`에, live comparison response hash와 neighbors는
`outputs/evaluation/mteb_korean_v1/comsat-live-comparison.json`에 보존한다.

### PwC 길이 표기 주의

공식 MTEB metadata는 514 tokens로 보이지만 XLM-R position table에서 실제 tokenizer max length는 content 512와 special tokens를 구분해야 합니다. 직접 `max_seq_length=514`로 덮어쓰면 position index 오류가 발생했고, 공식 model revision에 512를 적용한 run은 정상 완료했습니다. 성공 결과만 위 표에 사용했습니다.

## 아직 주장하지 않는 것

- AutoRAG 한 task만으로 전체 평균 우승을 주장하지 않습니다.
- 우리 smoke adapter는 아직 Sionic 9-task 성능 모델이 아닙니다.
- Sionic/다른 모델 카드 값을 공식 Korean MTEB 결과로 바꾸어 쓰지 않습니다.
