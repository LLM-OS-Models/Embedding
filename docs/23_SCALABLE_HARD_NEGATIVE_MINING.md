# 250K–1M scalable hard-negative mining

10K에서는 모든 query×corpus dot product를 계산하는
`scripts/mine_dense_hard_negatives.py`가 정확하고 충분히 빠르다. 250K–1M에서 같은
O(N²) 경로는 부적합하므로 `scripts/mine_faiss_hard_negatives.py`가 FAISS IVFFlat으로
후보를 찾고, 검색된 후보에 대해서는 float32 dot을 다시 정확히 계산한다.

## 보장과 비보장

- query와 unique positive corpus embedding은 L2 normalize된 float32 memmap이다.
- input hash/model revision/max length/attention/dtype namespace가 정확히 같을 때만
  중단된 embedding을 재사용한다.
- IVF index도 embedding namespace, dimension, corpus count, nlist/nprobe, FAISS
  version이 같을 때만 재사용한다.
- own positive, query와 exact 같은 document, `s_neg >= .95*s_pos` 후보를 제외한다.
- 선택된 후보 score는 ANN이 반환한 근사 score가 아니라 원 float32 vector dot이다.
- exact score 내림차순 top-24 pool에서 양 끝을 포함한 7개 score-rank quantile을
  결정론적으로 선택한다. `top_k`와 hash sample은 ablation으로만 유지한다.
- 그러나 IVF가 진짜 top hard negative를 놓칠 수 있으므로 candidate recall은
  approximate다. 최종 데이터에는 reranker/teacher 검증이 여전히 필요하다.

FAISS CPU 1.14.3과 NumPy 1.26.4는 `requirements/mteb-extras.txt`에 고정했다.

## 법률 250K target-adapted mining

법률 원문은 LawIRKo/AutoRAG legal corpus와 겹칠 수 있으므로 clean assertion 대신
`--allow-target-adapted`를 명시한다.

```bash
PYTHONPATH=scripts .venv-mteb/bin/python scripts/mine_faiss_hard_negatives.py \
  --input outputs/data/legal-performance-v1/train.bootstrap.jsonl \
  --output outputs/data/legal-performance-v1/train.faiss-r095-n7.jsonl \
  --audit-output outputs/data/legal-performance-v1/train.faiss-r095-n7.audit.jsonl \
  --manifest-output outputs/data/legal-performance-v1/train.faiss-r095-n7.manifest.json \
  --work-dir outputs/data/legal-performance-v1/faiss-work-qwen3-base \
  --model Qwen/Qwen3-Embedding-8B \
  --revision 1d8ad4ca9b3dd8059ad90a75d4983776a23d44af \
  --encode-batch-size 128 \
  --candidate-pool-size 24 \
  --search-k 256 \
  --num-negatives 7 \
  --selection-strategy score_rank_quantiles \
  --positive-relative-ratio .95 \
  --nlist 512 \
  --nprobe 32 \
  --training-points 50000 \
  --keep-work-dir \
  --allow-target-adapted
```

miner가 false-negative 가능성을 완전히 제거했다는 뜻은 아니다. 다음으로 audit의
input/output index를 이용해 provenance를 투영한다.

```bash
.venv-train/bin/python scripts/project_mined_provenance.py \
  --input-provenance outputs/data/legal-performance-v1/provenance.jsonl \
  --mining-audit outputs/data/legal-performance-v1/train.faiss-r095-n7.audit.jsonl \
  --output outputs/data/legal-performance-v1/provenance.faiss-r095-n7.jsonl \
  --manifest-output outputs/data/legal-performance-v1/provenance.faiss-r095-n7.manifest.json
```

그 뒤 16-row source-homogeneous batch를 만든다.

```bash
.venv-train/bin/python scripts/build_homogeneous_batches.py \
  --train outputs/data/legal-performance-v1/train.faiss-r095-n7.jsonl \
  --provenance outputs/data/legal-performance-v1/provenance.faiss-r095-n7.jsonl \
  --output outputs/data/legal-performance-v1/train.faiss-r095-n7.homogeneous-b16.jsonl \
  --provenance-output outputs/data/legal-performance-v1/provenance.faiss-r095-n7.homogeneous-b16.jsonl \
  --manifest-output outputs/data/legal-performance-v1/faiss-r095-n7.homogeneous-b16.manifest.json \
  --batch-size 16
```

법률만 한 epoch 더 돌리면 일반 검색·STS·다국어 표현이 회귀할 수 있다. 실제 campaign은
위 법률 batch 250K를 primary 25%로 두고, `performance_1m` homogeneous batch에서
750K를 replay 75%로 뽑아 두 번째 1M curriculum을 만든다. scale stage의 current-student
mined homogeneous artifact가 있으면 그것을 우선하고, 없을 때만 공개 원본 order를 쓴다.
row를 섞지 않고
완전한 16-row source-homogeneous batch 단위로만 전역 shuffle한다.

```bash
.venv-train/bin/python scripts/build_replay_curriculum.py \
  --primary-train outputs/data/legal-performance-v1/train.faiss-r095-n7.homogeneous-b16.jsonl \
  --primary-provenance outputs/data/legal-performance-v1/provenance.faiss-r095-n7.homogeneous-b16.jsonl \
  --primary-rows 250000 \
  --replay-train outputs/data/performance-v1/performance-1m/train.homogeneous-b16.jsonl \
  --replay-provenance outputs/data/performance-v1/performance-1m/provenance.homogeneous-b16.jsonl \
  --replay-rows 750000 \
  --output outputs/data/legal-performance-v1/train.faiss-r095-n7.legal25-replay75.jsonl \
  --provenance-output outputs/data/legal-performance-v1/provenance.faiss-r095-n7.legal25-replay75.jsonl \
  --manifest-output outputs/data/legal-performance-v1/faiss-r095-n7.legal25-replay75.manifest.json \
  --batch-size 16 --seed 42 \
  --adaptation-label target-adapted-legal25-general75
```

가능하면 miner와 학습 모두 직전 1M merged checkpoint에서 이어간다. 1M merge가 없을
때만 pinned Qwen base로 돌아가며 이 fallback은 queue log에 남긴다. continual stage는
LR `1e-5`, legal/general 비율 `25/75`, 한 curriculum pass를 사용한다. model card에는
법률 250K와 general 1M 두 dataset, target-adapted 표기, curriculum manifest를 함께
싣는다.

## 1M 적용

1M 전체를 처음부터 refresh하기 전에 source별 loss-active rate를 10K–50K sample로
측정한다. ko-triplet처럼 쉬운 비율이 높은 source를 우선 refresh하고 F2의 24-negative
row는 teacher score가 양호하면 보존한다. 현재 1M 실행값은 4096-d k-means의 CPU
병목을 피하기 위해 `nlist 1024`, training points 50K, `nprobe 32`, `search_k 256`,
threads 64로 고정했다. 더 큰 `nlist 4096`은 recall 측정 없이 기본값으로 쓰지 않는다.
ANN parameter는 성능을 보고 암묵적으로 바꾸지 않고 manifest에 기록한다.

## Clean과 target-adapted 분리

`--assert-no-benchmark-data`와 `--allow-target-adapted`는 정확히 하나만 선택해야 한다.
법률 성능 모델은 후자를 사용하고 LawIRKo/AutoRAG 점수를 target-adapted로 공개한다.
clean 모델은 benchmark blocklist exact/near-duplicate를 먼저 통과시킨 입력만 전자를
사용한다.
