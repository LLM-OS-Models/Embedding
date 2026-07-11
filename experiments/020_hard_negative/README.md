# 020 — Hard-negative mining과 false-negative 제거

목표는 쉬운 random negative 때문에 첫 step부터 loss가 0이 되는 문제를 해결하면서, 실제 정답일 가능성이 높은 top-1 문서를 무조건 negative로 넣지 않는 것입니다. 이 실험은 **학습 전용 corpus**에서만 후보를 만들며 benchmark/dev/test의 query, corpus, qrel은 입력할 수 없습니다.

## 논문에서 가져온 결정

- [NV-Retriever](https://arxiv.org/abs/2407.15831)와 [Llama-Embed-Nemotron](https://arxiv.org/abs/2511.07025)의 positive-aware 규칙을 따라 `s_neg < α × s_pos`인 후보만 남깁니다. 기본 `α=.95`는 출발점이며 `.90/.95/.98`을 실제로 비교합니다.
- [F2LLM-v2](https://arxiv.org/abs/2603.19223)의 pool-then-sample 설정을 참고해 query마다 dense 후보 pool 24개를 보존하고 학습 JSONL에는 4개를 넣습니다. 7개 설정도 별도 비교합니다.
- top-hard 하나만 반복하지 않고 이후 단계에서 BM25, base Qwen, 현재 checkpoint, teacher score-quantile 후보를 섞습니다. 현재 구현은 그중 재현 가능한 **base-Qwen exhaustive dense miner**입니다.

이 규칙은 false negative를 완전히 판별하는 장치가 아닙니다. `.95`보다 높은 후보는 제거 대상으로만 보고, 수동 audit와 reranker/teacher 판정 후 positive 승격 여부를 따로 결정합니다.

## 구현

[`mine_dense_hard_negatives.py`](../../scripts/mine_dense_hard_negatives.py)는 strict ms-swift JSONL의 query와 모든 positive document를 revision이 고정된 `Qwen/Qwen3-Embedding-8B`로 L2 정규화해 임베딩합니다. 모든 query-document cosine을 blockwise로 계산하므로 전체 `N×N` score matrix를 저장하지 않습니다.

- exact exhaustive search: query block 64 × corpus block 2,048
- embedding은 CPU `float32` memmap에 저장하고 계산 block만 GPU로 이동
- 10K–50K row에서 RAM/VRAM이 corpus 크기에 비례해 폭증하지 않음
- own positive와 NFKC+whitespace 기준 exact duplicate 제거
- query와 text가 같은 document도 기본 제거
- strict positive-relative filter: `candidate_score < .95 * positive_score`
- score 동률은 normalized-document SHA-256 오름차순으로 고정
- 후보 pool 24개와 cosine, positive score, threshold, 선택 origin을 text 없는 audit JSONL에 보존
- 원래 row의 negative도 `--include-source-negatives`일 때 동일한 filter를 통과한 경우 경쟁시킬 수 있음
- 입력 manifest의 `release_eligible`을 그대로 상속하며 manifest가 없으면 무조건 `false`
- 최종 output은 공통 [`validate_embedding_jsonl.py`](../../scripts/validate_embedding_jsonl.py)를 통과한 뒤에만 원자적으로 배치

정확 탐색의 연산량은 `O(N²D)`입니다. 50K를 넘는 corpus는 ANN으로 100–200개를 먼저 찾은 뒤 exact rescoring하는 별도 단계로 확장하되, 두 프로토콜의 결과를 같은 표로 섞지 않습니다.

## 실행

현재 256-row smoke 입력은 miner 저장·manifest 경로를 검증하기 위한 것입니다.

```bash
experiments/020_hard_negative/mine_smoke.sh
```

모델을 실행하지 않고 입력 hash, row 수, corpus 중복, 예상 dot-product 수만 확인하려면:

```bash
DRY_RUN=1 experiments/020_hard_negative/mine_smoke.sh
```

기본 산출물:

- `train.hn-qwen3-r095-n4.jsonl`: positive 1 + explicit negative 4의 strict 학습 파일
- `*.audit.jsonl`: 문서 원문 없이 row별 pool/선택 score와 hash
- `*.manifest.json`: model/revision/prompt, block 크기, 분포 통계, 입력·출력 hash, release 상태

query는 JSONL `messages[0].content`에 이미 `Instruct: ...\nQuery: ...`가 들어 있으므로 SentenceTransformers의 implicit prompt를 끄고 **저장된 문자열을 한 번만** 인코딩합니다. document prompt는 비어 있습니다.

## 데이터 선택과 공개 조건

현재 `nlpai-lab/ko-triplet-v1.0` smoke split은 한국어 query-positive-negative 형태라 miner와 trainer 연결을 빠르게 검증하기 좋지만 명시적 라이선스가 없습니다. 따라서 여기서 얻은 adapter와 mined JSONL은 private pipeline artifact이며 공개 후보에 쓰지 않습니다.

첫 10K private pilot은 `data/processed/ko_triplet_pilot_10k`의 train 10,000 / validation 512를 서로 분리해 각각 mining한 뒤 [`train_pilot_lora_r64.sh`](train_pilot_lora_r64.sh)로 실행합니다. 이 run의 목적은 hard-negative에서 loss가 실제로 학습 신호를 가지는지와 LoRA r64의 품질/VRAM을 보는 것이며, 데이터 권리 문제 때문에 성능이 좋아도 public release candidate로 승격하지 않습니다.

성능 후보에는 권리 검토를 통과한 일반 검색·공공/정부·법률·금융공시·보건·장문 데이터를 domain-balanced하게 넣습니다. 동일 domain의 positive corpus에서 찾은 dense 후보는 random negative보다 어렵고 실제 배포 검색 오류와 가깝기 때문입니다. target benchmark 문서의 URL/hash/MinHash/semantic-near-duplicate blocklist를 적용한 뒤에만 mining합니다.

## 비교표

| 축 | 설정 |
|---|---|
| miner | BM25 / base Qwen dense / current checkpoint / teacher-reranker |
| relative filter | `.90` / `.95` / `.98` |
| explicit HN | 1 / 4 / 7 |
| pool | 24 / 100 / 200 |
| sampling | top-hard / score quantile / domain-balanced |
| 검증 | clean NDCG@10, regression, training margin, false-negative audit 200건 |

성공 기준은 단순 train loss 감소가 아니라 clean retrieval 개선, 공식 MTEB 회귀 방지, false-negative audit 통과를 동시에 만족하는 것입니다.

## 10K pilot 고정 설정

| 항목 | 값 |
|---|---|
| Update | all-linear LoRA r64 / alpha128 / dropout .05 |
| Objective | InfoNCE `tau=.02`, in-batch + explicit HN 4, fake-negative mask |
| Batch | micro 16 × accumulation 4 = effective 64 queries; true in-batch pool은 16 |
| Length | query/document max 512, dynamic padding |
| Schedule | 160 optimizer steps, LR `2e-5`, warmup 5%, cosine |
| Checkpoint | eval/save 40 steps, 최대 3개 |
| Selection | mined validation loss/margin; public benchmark로 checkpoint 선택 금지 |

gradient accumulation의 다른 microbatch는 서로 in-batch negative가 되지 않는다. H100 메모리가 허용하면 accumulation을 줄이고 microbatch를 키우는 ablation이 우선이다.
