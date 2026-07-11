# Evaluation log

이 파일에는 실제로 실행한 결과만 기록합니다. 모델 카드의 성능표는 여기의 raw result와 revision을 기준으로 생성합니다.

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

Raw cache와 query-level predictions는 `outputs/evaluation/sionic9/` 아래에 보존되며 Git에는 대용량 artifact를 넣지 않습니다. 최종 모델 카드에는 전체 9개가 완료된 run의 hash와 공개 artifact URL을 연결합니다.

### PwC 길이 표기 주의

공식 MTEB metadata는 514 tokens로 보이지만 XLM-R position table에서 실제 tokenizer max length는 content 512와 special tokens를 구분해야 합니다. 직접 `max_seq_length=514`로 덮어쓰면 position index 오류가 발생했고, 공식 model revision에 512를 적용한 run은 정상 완료했습니다. 성공 결과만 위 표에 사용했습니다.

## 아직 주장하지 않는 것

- AutoRAG 한 task만으로 전체 평균 우승을 주장하지 않습니다.
- 우리 smoke adapter는 아직 Sionic 9-task 성능 모델이 아닙니다.
- Sionic/다른 모델 카드 값을 공식 Korean MTEB 결과로 바꾸어 쓰지 않습니다.
