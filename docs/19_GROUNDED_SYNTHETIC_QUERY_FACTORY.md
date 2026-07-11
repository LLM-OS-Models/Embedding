# 근거 기반 한국어 합성 질의 데이터 팩토리

기준일: **2026-07-11 (Asia/Seoul)**  
설정: [`configs/synthetic_query_factory_v1.json`](../configs/synthetic_query_factory_v1.json)  
구현: [`scripts/grounded_synthetic_query_factory.py`](../scripts/grounded_synthetic_query_factory.py)  
스모크: [`tests/test_grounded_synthetic_query_factory.py`](../tests/test_grounded_synthetic_query_factory.py)

## 결론

Legalize-KR 같은 원문 구조 pair를 그대로 학습하는 것만으로는 `법명 + 조문 -> 조문` 검색에는 강해져도 실제 사용자가 쓰는 질문 분포를 충분히 만들지 못한다. 이 팩토리는 source-native `query/positive`에서 **다섯 질의 스타일**을 만들고, 답과 인용 근거가 positive 원문에 실제로 존재하는지를 기계적으로 검증한 뒤, Qwen3 reranker 같은 teacher 점수로 false negative를 걷어내고 strict ms-swift InfoNCE JSONL을 만든다.

핵심 출력은 다음 두 파일을 분리한다.

- 학습 JSONL: `messages`, `positive_messages`, `negative_messages` 세 필드만 갖는다. 현재 저장소의 strict validator와 ms-swift가 그대로 읽는다.
- audit JSONL: source repository/revision/path, 생성 모델, 답, exact evidence, citation, positive score, 각 negative의 점수와 provenance를 같은 line 순서로 보존한다.

manifest에는 설정·모든 입력·두 출력의 SHA-256, generator/scorer identity, drop counter, `.95 × positive` threshold와 pool/sample 수를 기록한다. 따라서 학습 파일에 임의 metadata를 끼워 넣지 않으면서 생성 경로를 역추적할 수 있다.

이것은 “LLM이 썼으니 정답”이라고 가정하는 파이프라인이 아니다. 다음 세 신호를 서로 분리한다.

1. **문자열 grounding**: answer가 evidence 안의 exact normalized span이고 evidence가 positive 안의 exact normalized span인지 결정적으로 검사한다.
2. **검색 relevance**: teacher/reranker 점수로 positive 품질과 negative 난도를 판단한다. 이것은 human label이 아니다.
3. **평가 청결성**: 별도 benchmark blocklist/decontamination manifest가 책임진다. 이 팩토리를 통과했다는 사실은 clean zero-shot을 뜻하지 않는다.

## 왜 이 형태인가: 논문 방법을 코드 규칙으로 옮긴 부분

아래 숫자는 각 공개 논문·공식 구현의 scale과 방법이다. 이 팩토리가 그 비공개 원 데이터를 재현한다는 뜻은 아니다.

| 계열 | 공개된 학습 수량/방법 | 이 팩토리에 옮긴 규칙 |
|---|---|---|
| [Qwen3 Embedding](https://arxiv.org/abs/2506.05176) | Qwen3-32B로 약 **150M** multi-task/multilingual weak pair를 생성했고 cosine `> .7`인 약 **12M** synthetic pair를 고품질 단계에 재사용. explicit HN과 false-negative mask를 포함한 improved InfoNCE | 강한 local generator, task/query-style 조건, exact grounding, generator identity와 원 source provenance 보존 |
| [F2LLM-v2](https://arxiv.org/abs/2603.19223) | Qwen3-8B foundation에서 Stage 1 **27M**, Stage 2 **18M**. hard-negative pool **24**에서 **7**개 표집, source-homogeneous batch, dual CE, `tau=.05` | top-24 후보를 만든 뒤 seeded hash로 7개를 표집. style/source 정보는 homogeneous batching sidecar로 사용 가능 |
| [Llama-Embed-Nemotron-8B](https://arxiv.org/abs/2511.07025) | 논문 기준 **16.1M** pair(7.7M public + 8.4M synthetic), Stage 1 11.8M/Stage 2 4.3M. `s_neg < .95 × s_pos` positive-aware mining, HN 1→4 | positive와 동일한 teacher scale에서 `.95` relative cutoff, exact positive/query duplicate 제외, threshold와 제외 사유 감사 |
| [KaLM-Embedding-V2](https://arxiv.org/abs/2506.20923) | **470M** weak pair → **6M** supervised → 같은 6M teacher KD. Qwen3-Embedding-8B teacher, `0.3 CL + 0.7 KL`, rank 50–100 HN 7 | teacher score를 버리지 않고 audit에 보존하여 후속 KL/listwise distillation에 재사용; scorer revision 명시 |

Qwen 원 논문의 150M 생성 corpus와 전체 launcher는 공개되지 않았다. 따라서 현재 구현은 공개 사실에 기반한 **독립적인 continued-tuning data factory**다. 특히 Qwen/F2의 정확한 private prompt를 복원했다고 주장하지 않는다.

## 입력 계약

입력은 `prepare_legal_embedding_data.py`가 내는 다음 source-native JSONL과 바로 호환된다.

```json
{
  "id": "legal-v1-...",
  "query": "민법 제1조 (법원)",
  "positive": "# 민법\n\n##### 제1조 (법원)\n...",
  "pair_type": "source_title_and_article_heading_to_article",
  "label_origin": "source_document_structure_not_manual_relevance_judgment",
  "provenance": {
    "repository": "legalize-kr/legalize-kr",
    "revision": "40-character commit",
    "path": "kr/...md",
    "source_document_sha256": "...",
    "section_heading": "제1조 (법원)"
  }
}
```

필수 필드 외 metadata가 있어도 읽지만, 학습 근거로 보존되는 핵심은 위 필드다. ID 중복, 빈 문자열, 너무 짧은 positive는 즉시 실패한다. Legalize-KR extractor가 이미 source revision과 문서 SHA를 보존하므로 합성 단계에서 원문 lineage가 끊기지 않는다.

성능 우선 track에서도 다음 입력은 금지한다.

- Sionic 9 또는 MTEB Korean held-out 평가 query/qrel/answer
- 평가 corpus passage에서 평가 query를 다시 생성한 행
- benchmark blocklist가 exact/near duplicate로 표시한 행

법률 원문처럼 target-like corpus를 학습하는 것은 허용할 수 있지만 `LawIRKo-target-adapted`처럼 노출 사실을 결과표에 표시해야 한다.

## 생성하는 질의 스타일

설정의 기본 style은 다음 다섯 가지다.

| style | 목적 | 기대 효과 |
|---|---|---|
| `natural_question` | 일반 사용자 완전 문장 | AutoRAG, QA retrieval 표현 |
| `keyword_search` | 짧은 명사·검색어 질의 | web search와 title/section retrieval |
| `scenario_question` | 원문 조건을 생활형으로 질문 | lexical mismatch에 대한 강건성 |
| `citation_lookup` | 법명·조문·판례 쟁점 단서 포함 | LawIRKo·법률 정밀 검색 |
| `paraphrased_fact_lookup` | 한 사실/요건을 다른 표현으로 질문 | semantic retrieval과 질문 다양성 |

각 candidate/style 조합은 한 request를 만든다. `downstream_sampling_weight`는 최종 mix sampling을 위한 권장 비율이며 `prepare`가 같은 request를 여러 번 복제한다는 뜻은 아니다. 여러 variant가 필요하면 candidate ID/style/variant가 구분되도록 별도 shard를 만들고 중복 검사를 유지한다.

생성 모델은 반드시 다음 JSON 객체 하나를 반환한다.

```json
{
  "query": "민원 결과는 접수 후 언제까지 알려줘야 하나요?",
  "answer": "14일 이내",
  "evidence_quote": "신청서를 접수한 날부터 14일 이내에 처리 결과를 신청인에게 통지하여야 한다",
  "citation": {
    "source_candidate_id": "legal-v1-...",
    "locator": "kr/...md"
  }
}
```

`source_candidate_id`와 `locator`는 request가 제공한 allowlist에 정확히 일치해야 한다. locator 후보는 `path#section`, source path, source URL, section heading에서 결정적으로 만든다. 임의의 날짜나 일반 metadata 값은 citation으로 허용하지 않는다.

## 1단계: deterministic request 생성

```bash
python scripts/grounded_synthetic_query_factory.py \
  --config configs/synthetic_query_factory_v1.json \
  prepare \
  --input data/processed/legal/source_candidates.jsonl \
  --requests data/processed/legal/synthetic/requests-00000.jsonl \
  --shard-count 64 \
  --shard-index 0
```

request ID는 factory ID, source candidate ID, style ID, positive SHA-256에서 생성한다. endpoint seed도 request ID에서 만들기 때문에 같은 입력·설정이면 request 파일이 byte-identical하다. source candidate는 ID hash로 shard되며 파일 순서와 무관하게 같은 shard에 간다.

빠른 pilot은 스타일과 row 수를 제한한다.

```bash
python scripts/grounded_synthetic_query_factory.py prepare \
  --input data/processed/legal/source_candidates.jsonl \
  --requests /tmp/legal-requests.jsonl \
  --style natural_question \
  --style citation_lookup \
  --max-candidates 10000
```

## 2-A단계: local OpenAI-compatible/vLLM endpoint

config 기본 generator는 Qwen3 논문의 generator 계열과 맞춘 `Qwen/Qwen3-32B`다. 모델은 config 또는 `prepare --model`로 교체할 수 있다. 예시 서버는 다음과 같다.

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vllm/bin/vllm serve Qwen/Qwen3-32B \
  --served-model-name Qwen/Qwen3-32B \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.94
```

서버가 준비된 뒤 request를 병렬 resolve한다.

```bash
python scripts/grounded_synthetic_query_factory.py generate \
  --requests /tmp/legal-requests.jsonl \
  --mode endpoint \
  --endpoint-base-url http://127.0.0.1:8000/v1 \
  --concurrency 64 \
  --validated /tmp/legal-valid.jsonl \
  --rejected /tmp/legal-rejected.jsonl
```

API key는 기본적으로 필요하지 않다. endpoint가 인증을 요구할 때만 config에 이름이 적힌 `SYNTHETIC_QUERY_API_KEY` 환경변수를 읽고, 값은 request·audit·manifest 어느 파일에도 쓰지 않는다. concurrency는 서버의 continuous batching을 채우기 위한 HTTP in-flight 수이지 GPU batch size가 아니다.

## 2-B단계: 오프라인/수동/에이전트 생성 응답

사람이나 별도 agent가 검수한 응답, vLLM batch job 결과, 다른 inference engine 결과도 같은 validator를 통과시킬 수 있다. offline JSONL은 request ID와 응답을 연결한다.

```json
{"request_id":"gsq-request-...","response":{"query":"...","answer":"...","evidence_quote":"...","citation":{"source_candidate_id":"legal-v1-...","locator":"kr/...md"}}}
```

OpenAI chat completion 전체 response를 `response`에 넣어도 `choices[0].message.content`를 해석한다.

```bash
python scripts/grounded_synthetic_query_factory.py generate \
  --requests /tmp/legal-requests.jsonl \
  --mode offline \
  --responses /tmp/offline-responses.jsonl \
  --validated /tmp/legal-valid.jsonl \
  --rejected /tmp/legal-rejected.jsonl
```

검증기는 다음을 결정적으로 거부한다.

- 필드가 빠지거나 설명/두 번째 JSON이 붙은 응답
- query/answer/evidence 길이 범위 위반, 한국어 query에 Hangul 없음
- answer가 evidence 내부 exact normalized span이 아님
- evidence가 positive 내부 exact normalized span이 아님
- source ID/locator가 request allowlist와 불일치
- source candidate 하나에서 normalized query가 중복
- query 대부분이 positive의 한 연속 구절을 그대로 복사

여기서 exact normalized는 Unicode NFKC와 whitespace folding 뒤의 연속 문자열을 뜻한다. 의미만 비슷한 paraphrase evidence는 통과하지 않는다. 반면 answer가 원문에 존재한다는 사실만으로 query relevance가 증명되지는 않으므로 다음 teacher 단계가 필수다.

## 3단계: teacher/reranker score 계약

factory는 전체 corpus를 rerank하지 않는다. 먼저 Qwen3-Embedding-8B/BM25/hybrid retriever로 각 생성 query당 top 100–200 candidate ID를 얻고, `Qwen/Qwen3-Reranker-8B` 같은 teacher로 **positive를 포함한 같은 후보 집합**을 점수화한다. 점수 JSONL은 다음 형식이다.

```json
{
  "generated_id": "gsq-...",
  "score_field": "reranker_score",
  "scorer": {
    "model": "Qwen/Qwen3-Reranker-8B",
    "revision": "pinned-40-character-commit",
    "prompt": "Given a Korean query, judge whether the document answers it",
    "score_semantics": "normalized yes-token probability in [0,1]"
  },
  "documents": [
    {"candidate_id": "positive-source-id", "retriever_score": 0.91, "reranker_score": 0.98},
    {"candidate_id": "other-source-id", "retriever_score": 0.88, "reranker_score": 0.74}
  ]
}
```

`score_field`를 생략하면 모든 document에 공통으로 존재하는 필드를 `reranker_score → teacher_score → retriever_score` 순서로 고른다. 비교되는 score는 한 query 안에서 같은 model/normalization으로 계산한 `[0,1]` 값이어야 한다. 서로 다른 model의 raw cosine, logit, yes probability를 한 relative threshold에 섞지 않는다.

권장 candidate 순서는 다음과 같다.

1. dense top 100과 BM25 top 100의 합집합
2. source positive 강제 삽입
3. exact/near duplicate와 같은 판례/같은 조문의 관련 section group 표시
4. reranker continuous score 계산
5. score JSONL 생성

같은 법의 인접 조문, 같은 판례의 판결요지/판시사항은 실제 관련 문서일 가능성이 높다. exact text filter만으로 충분하지 않으므로 upstream group ID 또는 reranker로 false negative를 줄여야 한다. v1 factory는 document ID/text exact 조건과 positive-relative score를 강제하고, 관계형 group exclude는 후속 candidate producer가 수행해 score file에서 제외한다.

## 4단계: positive-aware hard negative와 strict JSONL compile

```bash
python scripts/grounded_synthetic_query_factory.py compile \
  --candidates data/processed/legal/source_candidates.jsonl \
  --validated /tmp/legal-valid.jsonl \
  --scores /tmp/legal-teacher-scores.jsonl \
  --output data/processed/legal/synthetic/train.jsonl \
  --audit data/processed/legal/synthetic/audit.jsonl \
  --manifest data/processed/legal/synthetic/manifest.json \
  --work-dir data/cache/legal-synthetic-index
```

candidate source가 수백만 건/수 GB여도 전부 RAM에 올리지 않는다. source ID와 positive/provenance를 SQLite에 byte-stable하게 index하고, score row별 필요한 문서만 읽는다. `--candidate-index`로 index 파일을 고정하면 input SHA가 같은 다음 run에서 재사용한다. teacher score는 임시 SQLite index로 join한 뒤 삭제한다.

기본 선택 순서는 다음과 같다.

1. source positive가 score 목록에 정확히 한 번 있고 원문 SHA가 candidate source와 같은지 확인
2. positive score `< .5`인 생성 query drop
3. 자기 자신, positive와 같은 text, query와 같은 text, 서로 text가 같은 negative 중복 제외
4. `s_neg <= min(.95 × s_pos, s_pos - absolute_margin)`만 유지
5. negative score `< .05` 제외
6. 남은 후보를 score 내림차순으로 정렬하고 top 24 pool 생성
7. `(seed, generated_id, candidate_id)` SHA 순서로 pool에서 7개 표집
8. strict training row exact dedup

top 7만 쓰지 않고 top-24의 score-rank quantile 7개를 균등하게 고르는 이유는 F2식
난도 pool을 유지하면서 한두 개 near-positive가 모든 epoch/행을 지배하는 것을 줄이고,
reranker teacher의 high/mid/low relevance 분포를 보존하기 위해서다. pool을 score
내림차순·candidate ID 오름차순으로 고정한 뒤 양 끝을 포함하는 rank anchor를 정수
산술로 계산하므로 실행마다 동일하다. 완전한 top-k와 기존 hash sample ablation은 config의
`selection_strategy`를 각각 `top_k`, `hash_sample_from_top_pool`로 바꿔 비교한다.

출력 training row는 정확히 다음 구조다.

```json
{
  "messages": [{"role": "user", "content": "새 질의"}],
  "positive_messages": [[{"role": "user", "content": "근거 positive"}]],
  "negative_messages": [
    [{"role": "user", "content": "teacher-filtered negative 1"}],
    [{"role": "user", "content": "teacher-filtered negative 2"}]
  ]
}
```

## 5단계: 독립 검증

```bash
python scripts/grounded_synthetic_query_factory.py verify \
  --output data/processed/legal/synthetic/train.jsonl \
  --audit data/processed/legal/synthetic/audit.jsonl \
  --manifest data/processed/legal/synthetic/manifest.json

python scripts/validate_embedding_jsonl.py \
  data/processed/legal/synthetic/train.jsonl
```

`verify`는 manifest hash, strict schema, row dedup, positive/negative 충돌, audit line 정렬, query/positive/negative SHA를 다시 검사한다. 두 파일 중 하나의 한 글자만 바뀌어도 실패한다.

repository fixture는 네 단계를 GPU 없이 실행하고 같은 compile을 두 번 수행해 train/audit SHA가 같은지도 확인한다.

```bash
python -m unittest -v tests.test_grounded_synthetic_query_factory
```

## 실제 성능 실험 우선순위

Sionic 9 → 공식 MTEB Korean v1 → 종합 leaderboard 순으로 고를 때 합성 데이터 ablation은 한 번에 변수를 하나만 바꾼다.

| run | source rows | styles | teacher/HN | 목적 |
|---|---:|---|---|---|
| `SQ-10K-A` | Legalize candidate 10K | natural + citation | Qwen3 reranker, 24→7, `.95` | end-to-end pipeline/법률 gain 확인 |
| `SQ-50K-B` | legal 25K + general Korean 25K | 5 styles, weight 3:2:2:2:1 | 같은 teacher | AutoRAG/LawIRKo gain과 일반 성능 손실 동시 측정 |
| `SQ-200K-C` | legal 50K + QA/web/wiki 150K | 5 styles | hybrid candidate 200→24→7 | Sionic 9 전체 평균용 첫 meaningful run |
| `SQ-1M-D` | source-balanced 1M | style/source cap | iterative remine, score/KD cache | LoRA/DoRA/partial/full FT 비교용 |

Legalize-KR 조문형 후보 275만여 건을 처음부터 모두 생성하지 않는다. 10K에서 acceptance, positive score, eligible-negative 수, style별 benchmark delta를 보고 source/기관/문서 길이 strata를 확대한다. 가장 높은 점수만 남기면 query가 쉬워지는 selection bias가 생기므로 teacher-positive score와 negative hardness 구간별 표본을 모두 남긴다.

학습 때 audit의 `style`, `repository`, `pair_type`으로 homogeneous/source-balanced batch를 만들 수 있다. strict train JSONL에는 metadata가 없으므로 batch sampler가 필요하다면 audit line number를 join하거나 source별 strict shard를 별도로 생성한다.

## 알려진 한계와 실패 조건

- exact evidence는 hallucination을 크게 줄이지만 query와 evidence의 의미적 관계를 완전히 증명하지 않는다.
- teacher가 틀리면 false negative와 false positive가 남는다. scorer revision/prompt/score semantics를 반드시 pin한다.
- `[0,1]` relative threshold는 normalized relevance score용이다. raw logit/cosine에는 해당 scale에 맞춘 margin ablation이 필요하다.
- related legal documents는 score가 positive보다 낮아도 relevant일 수 있다. 같은 source family/group filter를 upstream에 추가해야 한다.
- generator temperature 0과 seed는 request 재현성을 높이지만 GPU kernel/server version까지 bit-identical하다고 보장하지 않는다. raw response SHA가 그 차이를 드러낸다.
- endpoint response를 얻은 뒤 config를 바꾸면 기존 validated file을 그대로 재사용하지 않는다. manifest의 config/request SHA가 다른 run은 별도 artifact다.
- source-native Legalize-KR 학습은 LawIRKo와 target similarity가 높다. 성능 점수 공개는 가능하지만 clean zero-shot과 target-adapted score를 같은 칸에 섞지 않는다.

## provenance 최소 보존 항목

공개 또는 private model card에 최소 다음을 적는다.

- candidate repository, pinned revision, source extractor/config SHA
- generator model/revision 또는 endpoint model identity, request config SHA
- 생성 요청/응답/validated file SHA와 style별 acceptance/drop 수
- retriever/miner, reranker model/revision/prompt, score normalization
- positive threshold, `.95` ratio, pool 24/sample 7, seed
- strict train/audit/manifest SHA와 row 수
- benchmark train/target-like exposure, exact/near decontamination manifest
- 학습 objective, temperature, HN 수, batch 구성, checkpoint selection 기준

이 정보가 없으면 점수가 올라도 어떤 데이터와 false-negative 정책이 효과를 냈는지 재현할 수 없다.
