# 030 — Teacher distillation

## 현재 고정 teacher

- `Qwen/Qwen3-Reranker-8B`
- exact revision: `77d193c791ed757ca307ee72715aa132723da912`
- license: Apache-2.0
- load contract: repository-local snapshot only, `trust_remote_code=False`, Hub token 미사용

실행 시작 시 Hub/Git credential 환경 변수를 제거하고 offline/telemetry 차단을 강제한다.
production backend는 5개 LFS blob의 실제 16GB content SHA-256도 blob identity와 대조한
후에만 모델을 로드한다.

[`cache_qwen3_reranker_scores.py`](../../scripts/cache_qwen3_reranker_scores.py)는
이 model/revision을 CLI에서 바꿀 수 없게 고정한다. 공식 모델 카드와 동일하게 다음
prompt를 사용한다.

1. system: Document가 Instruct와 Query 요구를 만족하는지 `yes`/`no`로만 판정
2. user body: `<Instruct>`, `<Query>`, `<Document>`
3. assistant suffix: 빈 `<think>...</think>` 뒤의 다음 token 위치
4. tokenizer의 단일 `no`와 `yes` token logit만 선택
5. `softmax([raw_no_logit, raw_yes_logit])[yes]`를 `[0,1]` 연속 score로 저장

기본 영어 instruction은 공식 카드의
`Given a web search query, retrieve relevant passages that answer the query`다. 다른 task
instruction은 허용하지만 전체 run에서 하나만 쓰고 instruction 및 SHA를 모든 score row와
manifest에 고정한다.

## 입력 계약

한 JSONL row가 하나의 generated query와 positive를 강제 삽입한 candidate 집합이다.
top-level/document의 알 수 없는 field, JSON duplicate key, 빈 text, non-finite retriever
score, 중복 `generated_id`, query 안의 중복 `candidate_id`는 즉시 실패한다.

```json
{
  "generated_id": "gsq-...",
  "query": "환불 규정을 찾아줘",
  "positive": {
    "candidate_id": "positive-source-id",
    "text": "구매 후 7일 이내에는 ...",
    "retriever_score": 0.91
  },
  "candidates": [
    {"candidate_id": "candidate-id", "text": "다른 문서", "retriever_score": 0.88}
  ]
}
```

기본 상한은 positive 포함 201 documents다. candidate 순서는 upstream hybrid miner의
결정론적 순서를 그대로 보존하며 positive는 출력 첫 document가 된다.

## 출력과 provenance

canonical `scores.jsonl`의 각 row는 `docs/19` compile이 소비할 수 있는
`score_field=reranker_score`와 다음 정보를 가진다.

```json
{
  "generated_id": "gsq-...",
  "query_sha256": "...",
  "score_field": "reranker_score",
  "scorer": {
    "model": "Qwen/Qwen3-Reranker-8B",
    "revision": "77d193c791ed757ca307ee72715aa132723da912",
    "backend": "pinned-local-qwen3-reranker",
    "instruction": "...",
    "instruction_sha256": "...",
    "prompt_template_sha256": "...",
    "token_no": "no",
    "token_no_id": 2152,
    "token_yes": "yes",
    "token_yes_id": 9693,
    "score_semantics": "normalized yes-token probability in [0,1]"
  },
  "documents": [
    {
      "candidate_id": "positive-source-id",
      "role": "positive",
      "text_sha256": "...",
      "raw_no_logit": -1.25,
      "raw_yes_logit": 3.5,
      "reranker_score": 0.991422514586288
    }
  ]
}
```

`no=2152`, `yes=9693`은 위 exact revision의 repository-local tokenizer에서 둘 다 단일이며
서로 다른 token임을 CPU로 검증한 값이다. query/document 원문, 로컬
cache/input/output path, 환경 변수와 인증 token은 score/state/manifest에 기록하지 않는다.
`--model-batch-size`는 한 query의 최대 201개 document를 실제 microbatch로 나누므로 전체
candidate 묶음을 한 번에 8B model에 넣지 않는다.

출력 디렉터리 구조는 다음과 같다.

- `shards/part-NNNNNN.jsonl`: 같은 filesystem에서 fsync 후 atomic rename된 score shard
- `state.json`: input SHA/rows/bytes, scorer/runtime fingerprint, 연속 shard SHA와 resume 위치
- `scores.jsonl`: 완료 shard를 순서대로 atomic concatenate한 canonical cache
- `manifest.json`: input/output SHA와 row count, scorer/prompt/tokenization/runtime provenance

runtime provenance에는 config/tokenizer/index metadata SHA와 5개 local LFS weight blob의
content-SHA identity, Python/PyTorch/Transformers/CUDA/cuDNN/GPU, dtype와 실제 attention
implementation도 포함한다. 따라서 같은 revision 이름 아래 local bytes/runtime가 바뀐
resume은 거부한다.

resume 때 state가 선언한 모든 shard의 SHA, size, row range, generated/candidate ID,
query/document SHA, raw-logit normalization을 원 input과 다시 대조한다. shard rename과 state
갱신 사이에 중단된 경우 다음 번호의 완성 shard만 전수 검증 후 채택한다. partial temp
shard는 재사용하지 않는다. 입력이 실행 중 바뀌면 최종 manifest를 만들지 않는다.

## 실행

먼저 모델을 로드하거나 파일을 쓰지 않는 CPU preflight를 수행한다.

```bash
python scripts/cache_qwen3_reranker_scores.py \
  --input data/processed/teacher/requests.jsonl \
  --output-dir outputs/teacher/qwen3-reranker-8b-scores \
  --dry-run
```

GPU가 다른 학습에 사용되지 않는 시간에만 실제 pinned 8B scorer를 실행한다.

```bash
python scripts/cache_qwen3_reranker_scores.py \
  --input data/processed/teacher/requests.jsonl \
  --output-dir outputs/teacher/qwen3-reranker-8b-scores \
  --device cuda \
  --dtype bfloat16 \
  --attention-implementation sdpa \
  --max-length 8192 \
  --model-batch-size 8 \
  --shard-size 64
```

중단 후 같은 명령을 다시 실행하면 verified shard 다음 row부터 이어간다. 모델을 다시
로드하지 않고 완성 artifact만 독립 검증할 수도 있다.

```bash
python scripts/cache_qwen3_reranker_scores.py \
  --input data/processed/teacher/requests.jsonl \
  --output-dir outputs/teacher/qwen3-reranker-8b-scores \
  --verify-only
```

CPU test용 `--backend mock`은 추가로 `--allow-mock-output`을 명시해야 한다. mock은
`mock_reranker_score` field와 `admissible_for_training=false`를 강제하므로 teacher
학습 cache로 간주하지 않는다.

```bash
python -m unittest -v tests.test_cache_qwen3_reranker_scores
```

실제 8B load 없이 공식 prompt 형식, local-only exact revision, no
`trust_remote_code`, deterministic mock, atomic resume/orphan recovery, strict schema와
duplicate/non-finite/tamper/path/token 차단을 검사한다.

## 후속 후보 teacher

- stronger public embedding ensemble
- domain-specific cross-encoder

비교:

- teacher를 filter로만 사용
- hard label InfoNCE
- candidate distribution KL/soft labels
- base Qwen representation replay

## 구현된 data-selection 계약

[`grounded_synthetic_query_factory.py`](../../scripts/grounded_synthetic_query_factory.py)는
위 production `scores.jsonl` 같은 연속 score 파일을 받아 positive threshold, positive-relative
false-negative filter, text dedup을 적용한다. 기본 `score_rank_quantiles`는 score-sorted
top-24 pool의 양 끝을 포함한 7개 rank quantile을 결정론적으로 선택한다. `top_k`와
`hash_sample_from_top_pool`은 ablation으로 유지한다. 각 row audit에는 선택 index,
최소/최대 teacher score, scorer model/revision을 저장한다.

## 구현된 listwise KD와 자동 A/B

[`mine_faiss_hard_negatives.py`](../../scripts/mine_faiss_hard_negatives.py)는 기존 HN7을
만드는 동일 current-student embedding/FAISS run에서 seed 고정 query subset과 넓은 top-200
pool을 teacher request 계약으로 추가 출력할 수 있다. 이 넓은 pool에는 student `.95`
false-negative filter를 미리 적용하지 않는다. positive보다 더 높은 student 후보도 teacher가
판정해야 하기 때문이다.

[`compile_reranker_kd_dataset.py`](../../scripts/compile_reranker_kd_dataset.py)는 scorer의
완성 state/shard/output/manifest를 다시 전수 검증하고 `admissible_for_training=true`인 pinned
production cache만 받는다. positive `.5`, positive-relative `.95`, absolute margin `.02`를
통과한 후보에서 양 끝을 포함한 rank-quantile 15개를 선택하고 strict ms-swift row에
`teacher_scores=[positive, negative...]`를 정렬 보존한다. score 수, finite `[0,1]`, positive
우위, query/document ID/hash가 하나라도 어긋나면 학습 전에 실패한다.

[`listwise_distillation.py`](../../scripts/listwise_distillation.py)와
[`listwise_kd_plugin.py`](listwise_kd_plugin.py)는 ms-swift submodule을 수정하지 않는 공식
`--external_plugins` 확장이다. 기본 objective는 다음과 같다.

```text
L = 0.3 * L_InfoNCE(in-batch + explicit HN)
  + 0.7 * KL(softmax(logit(p_teacher) / T_teacher)
             || softmax(cos(q,d) / T_student))
```

기본값은 `T_teacher=1.0`, `T_student=0.02`다. normalized yes probability를 다시 logit으로
옮겨 candidate distribution을 만들며, `MarginMSE`는 별도 ablation이다. 선택적
stop-gradient document queue는 4096개까지 유지하고 현재 positive와 cosine `.02` 이내인
queue 문서를 denominator에서 mask한다. teacher score가 없는 validation은 순수 hard
InfoNCE로 평가하므로 기존 512-row validation을 그대로 쓸 수 있다.

[`run_reranker_kd_ablation_queue.sh`](../../scripts/run_reranker_kd_ablation_queue.sh)는 1M
merged base, filter-only, listwise-KL, listwise-KL+queue4096을 같은 query/token budget으로
학습·병합한다. public Sionic/MTEB는 보지 않고 Grade-I legal과 robustness에서 base 포함
winner를 고른 뒤 그 selection만 target/legal/combined queue에 전달한다. 기본 10K는
throughput과 clean delta를 확인하는 pilot이며 이득이 확인되면 50K, 100K로 확장한다.
compile이 끝난 exact train/audit/request/score-cache/manifest는 학습과 병렬로 검증한 뒤
`LLM-OS-Models2/korean-embedding-qwen3-reranker-kd-pilot-v1` private dataset에 최대 3회
재시도 업로드한다. token은 `.env`에서 upload subprocess memory로만 전달하며 command line,
log, repo URL에는 넣지 않는다.

현재 구현/시험은 끝났지만 실제 reranker score cache와 KD 성능 결과는 아직 없다. 결과가
생기기 전에는 KD가 baseline보다 낫다고 주장하지 않는다.
