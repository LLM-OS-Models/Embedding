# Sionic MIRACL·MrTidy·MLDR train-family adaptation

기준일: 2026-07-12 (Asia/Seoul)

## 목적

Comsat 카드 기준 Qwen3-Embedding-8B의 열세는 MIRACL `0.0181`, MrTidy `0.0066`,
MLDR `0.0147`이다. 그런데 decontaminated 1M curriculum의 해당 train-family는
4,146행, 0.41%뿐이다. 평가 query를 추가하지 않고 이 공개 train-family의 gradient
비중과 long-document exposure만 짧게 높이는 별도 specialist를 만든다.

## Source artifact

[`korean-embedding-performance-v1-sionic-retrieval-train-family-4146@c847cbc`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-performance-v1-sionic-retrieval-train-family-4146/tree/c847cbcbf8a72a69b6c817bd448c42ede8aa76d9)은
decontaminated 1M의 aligned train/provenance에서 다음 `source_id`만 parent order 그대로
추출한다.

| source | rows |
|---|---:|
| MIRACL Korean train-family | 700 |
| MrTidy Korean train | 1,200 |
| MLDR Korean train-family | 2,246 |
| 합계 | **4,146** |

- train SHA: `6837367935ea56912375fe6a476360eb7dd0efcc0100459901e92a44029b7c60`
- provenance SHA: `9d97802b378b6c2d3bd15824db2ab3a680315f9ee6fb95492b16778c093d015e`
- row SHA mismatch / provenance index mismatch: `0 / 0`
- negatives: 모든 row 7개
- query body p50/p95: `26 / 98` characters
- positive p50/p95: `1,775 / 13,233` characters
- negative p50/p95: `1,975 / 13,621` characters

## Contamination gate

Sionic 9 + 공식 Korean 6의 text-only blocklist로 query/positive/negative 전체를
검사했다.

- critical query/evaluation-text unique match: **0**
- shared retrieval corpus unique match: `13,973`
- status: `pass_with_retrieval_corpus_exposure`

공식 train-family와 shared corpus를 의도적으로 사용하므로 이 specialist의 관련 점수는
clean zero-shot이 아니다. evaluation query/qrel은 mining, validation, checkpoint
selection에 사용하지 않는다.

## Mining과 curriculum

1M winner가 있으면 그 모델, 없으면 pinned Qwen base로 4,146 query/positive를 최대
2,048 tokens에서 encode한다. FAISS candidate pool 24에서 own positive를 제외하고
`s_neg < .95*s_pos`인 후보 7개를 score-rank quantile로 뽑는다.

source-homogeneous length bucket을 만든 뒤 같은 수의 decontaminated 1M general batch와
50:50으로 섞는다. mining drop이 없으면 final curriculum은 약 8,288행이다. 파생
train/provenance/quality/overlap/mining audit은
`LLM-OS-Models/korean-embedding-sionic-retrieval-family-quantile-hn7-replay-v1`에
자동 공개한다.

## 학습

| 항목 | 값 |
|---|---|
| Base | 1M general winner; 없으면 pinned Qwen3-Embedding-8B |
| Tuner | LoRA r64/alpha128/all-linear |
| Loss | InfoNCE tau .02, in-batch + explicit HN7, false-negative mask |
| Max length | **2,048** |
| Primary batch | 2 × accumulation 32 = effective 64 |
| OOM fallback | 1 × accumulation 64 |
| LR | 5e-6, cosine, warmup 5% |
| Steps | final rows / 64, 약 129 |
| Selection | exact-overlap-zero 512-row HN validation loss |

2,048-token 실제 backward에서 FA2가 불안정하면 SDPA/fallback으로 전환한다. 완료 후
safe-merge parity, Sionic 9 전체, 공식 Korean v1, clean 법률/강건성 결과와 model/data
revision을 공개한다.

실행기는 [`run_sionic_retrieval_adaptation_queue.sh`](../scripts/run_sionic_retrieval_adaptation_queue.sh)이다.
