# Korean legal/public source-held-out retrieval v1

> **2026-07-17 정정:** v1은 source-document SHA overlap은 0이지만, 서로 다른 원문
> 문서에 중복 수록된 동일 법률 text 때문에 legal 250K와 query/positive exact text가
> 같은 98행이 있었다. v1 수치와 공개 artifact는 역사 기록으로 유지하되 최종 모델
> 선택에는 쓰지 않는다. 현재 canonical board는 모든 선언 학습 JSONL 역할의 normalized
> exact text까지 제외한
> [`v2 text-strict 계약`](35_VALIDATION_LEAKAGE_CORRECTION_2026-07-17.md)이다.

기준: **2026-07-12**  
설정: [`configs/legal_source_holdout_v1.json`](../configs/legal_source_holdout_v1.json)  
생성·검증기: [`scripts/build_legal_source_holdout.py`](../scripts/build_legal_source_holdout.py)  
단위 테스트: [`tests/test_legal_source_holdout.py`](../tests/test_legal_source_holdout.py)

## 결론

clean comprehensive 보드의 첫 자체 Korean legal/public retrieval set은 **same-repository source-document-held-out, 독립성 I 등급**으로 정의한다. Z 등급이 아니다. query와 positive는 학습에 사용한 것과 같은 네 Legalize-KR repository와 같은 source-native 구조에서 나오기 때문이다. 다만 선택된 `source_document_sha256` 전체가 training provenance에 없고, 선택된 query/positive의 normalized SHA-256이 Sionic 9 및 공식 MTEB Korean 평가 blocklist에 없다는 것을 생성기와 독립 verifier가 강제한다.

초기 250K candidate만으로는 모든 61,750 source document가 training에 노출돼 blocked됐지만,
같은 pinned repository의 deterministic file-hash shards 12–15를 추가 추출해 문제를
해소했다. 최종 candidate 242,675행에서 177,004행이 training document exclusion 뒤
남았고, source document당 최대 한 pair와 source balance를 적용해 10,000행을 선택했다.

독립 verifier 결과는 training candidate/document overlap `0/0`, benchmark
query/positive overlap `0/0`, unique query/positive/source document
`10,000/10,000/10,000`, `verified:true`다. artifact는
[`LLM-OS-Models/korean-legal-source-heldout-retrieval-v1@ee1300f`](https://huggingface.co/datasets/LLM-OS-Models/korean-legal-source-heldout-retrieval-v1/tree/ee1300f04ea03d66bb51e23bbbda34376fece3f0)에 공개했다.

## 독립성 등급을 I로만 부르는 이유

| 항목 | 이 artifact의 상태 | 해석 |
|---|---|---|
| repository | 학습과 동일한 Legalize-KR 4종 | unseen-source가 아님 |
| source schema | 법명+조문→조문, 판시사항→판결요지 | 학습과 동일한 pair 구조 |
| source document | SHA-256 전체 holdout | 같은 원문 문서의 다른 section 누수 차단 |
| source candidate | training provenance ID holdout | 학습 pair 직접 재사용 차단 |
| benchmark exact text | query/corpus/evaluation hash holdout | Sionic 9/공식 Korean exact 평가 text 차단 |
| near/semantic duplicate | v1에서 보장하지 않음 | clean zero-shot/Z 주장 금지 |
| relevance label | 원문 구조가 제공한 1:1 관계 | 독립 human judgment가 아님 |

따라서 허용되는 표기는 `same-repository whole-source-document-held-out (I)`다. `unseen-source`, `clean zero-shot`, `Z`라고 표기하면 안 된다.

## 실제 입력 감사

현재 performance bootstrap의 source별 training exposure는 다음과 같다.

| repository | candidate/training rows | training unique source documents |
|---|---:|---:|
| `legalize-kr/admrule-kr` | 50,000 | 2,727 |
| `legalize-kr/legalize-kr` | 50,000 | 1,499 |
| `legalize-kr/ordinance-kr` | 100,000 | 7,524 |
| `legalize-kr/precedent-kr` | 50,000 | 50,000 |
| **합계** | **250,000** | **61,750** |

네 candidate 파일은 bootstrap training에 전량 사용되었다. 특히 판례는 한 문서에서 한 issue→holding pair가 나오는 구조라 50,000행이 50,000문서와 같다. 조문형 source는 한 문서에서 여러 조문 pair가 나오지만, 문서 SHA 단위로 제외하므로 현재 candidate pool의 모든 행이 함께 제외된다.

이 결과는 “원천 문서가 부족하다”는 뜻이 아니다. pinned source inventory에는 법령 5,725문서, 행정규칙 20,390문서, 판례 124,116문서, 자치법규 162,350문서가 있다. 문제는 현재 candidate artifact가 각 extractor의 앞부분을 cap한 뒤 그 결과를 전량 training에 썼다는 것이다. 나머지 문서를 별도 candidate snapshot으로 추출하면 I 등급 10K를 만들 수 있다.

## 입력과 exclusion evidence

기본 입력은 다음과 같다.

```text
candidate pairs     outputs/data/legal-performance-v1/candidates/*.jsonl
training evidence   outputs/data/legal-performance-v1/provenance.jsonl
benchmark hashes    outputs/decontamination/benchmark_blocklist/**/
                    {query_text,corpus_text,evaluation_text}.sha256.gz
normalization       configs/decontamination_policy.json
```

training provenance에서는 다음 두 집합을 만든다.

1. 모든 `source_candidate_id`
2. 모든 `provenance.source_document_sha256`

candidate의 document SHA가 두 번째 집합에 있으면 candidate ID가 다르더라도 문서 전체를 제외한다. candidate ID가 training에 있는데 document SHA가 달라졌다면 source drift 또는 provenance corruption으로 보고 즉시 실패한다.

benchmark exact exclusion은 blocklist 생성기와 똑같은 normalization을 사용한다.

```text
Unicode NFKC
zero-width U+200B/U+200C/U+200D/U+2060/U+FEFF 제거
CRLF/CR -> LF
whitespace collapse
casefold 없음
SHA-256(UTF-8)
```

각 candidate의 query 전체와 positive 전체를 따로 hash하여 모든 `query_text`, `corpus_text`, `evaluation_text` hash file과 비교한다. 이는 exact normalized match 차단이다. title/body 일부 overlap, paraphrase, semantic near-duplicate까지 차단했다는 뜻은 아니다.

## deterministic source-balanced selection

기본 target은 10,000이며 네 repository에서 가능한 한 동일한 수를 round-robin으로 뽑는다. 각 source 내부 rank는 다음 material의 SHA-256이다.

```text
seed \0 repository \0 source_candidate_id \0 normalized_query_sha256 \0 normalized_positive_sha256
```

기본 seed는 `20260712`다. 한 source가 먼저 고갈되면 남은 source에 round-robin을 계속하지만, 네 required repository 중 하나라도 후보가 전혀 없으면 artifact를 만들지 않는다. 선택 과정은 다음을 추가로 강제한다.

- normalized query exact dedup
- normalized positive exact dedup
- 한 source document당 최대 1 pair
- query/positive benchmark hash exclusion
- source_candidate/document training exclusion

wall-clock timestamp는 deterministic manifest에서 의도적으로 제외한다. source date는 원 provenance의 `선고일자 → 시행일자 → 공포일자 → 발령일자` 우선순위로 row마다 보존하고, repository revision은 pinned 40-character Git SHA 그대로 저장한다.

## 출력 형식

완성된 artifact에는 다섯 파일이 있다.

### `queries.jsonl`

```json
{
  "_id": "legal-i-q-...",
  "text": "검색 질의",
  "metadata": {
    "source_candidate_id": "legal-v1-...",
    "source_document_sha256": "...",
    "repository": "legalize-kr/legalize-kr",
    "revision": "40-char SHA",
    "source_date": {"field": "시행일자", "value": "2025-01-01"},
    "pair_type": "source_title_and_article_heading_to_article",
    "selection_reason": "...",
    "independence_grade": "I",
    "independence_label": "same-repository source-document-held-out"
  }
}
```

### `corpus.jsonl`

`_id`, `title`, `text`, `metadata`를 갖는다. corpus는 선택된 10K positive passage 전체이고 각 query는 이 10K corpus를 상대로 검색한다.

### `qrels.jsonl`

```json
{"query-id":"legal-i-q-...","corpus-id":"legal-i-d-...","score":1}
```

v1은 source-native 구조가 연결한 positive 하나만 qrel로 둔다. corpus의 다른 조문이 실제로 관련될 수 있으므로 이 relevance set은 exhaustive human judgment가 아니다. NDCG@10/Recall@10을 해석할 때 이 한계를 명시한다.

### `provenance.jsonl`

query/corpus ID, source candidate/document SHA, emitted/normalized text SHA, full source provenance, revision/date, pair type, label origin, selection reason을 같은 row 순서로 저장한다.

### `manifest.json`

입력 파일 SHA/row 수, training ID/document 집합 크기, 모든 benchmark hash file SHA/record 수, source별 후보·제외·선택 수, 출력 SHA, 누수 assertion, I-not-Z claim을 저장한다.

## build와 독립 verify

초기 capped training candidate만 넣으면 약한 row split을 만들지 않고 정확히
`blocked_insufficient_source_document_heldout_candidates`로 종료한다. 실제 완료 artifact는
file-hash shards 12–15의 16개 candidate JSONL을 `--candidate`로 전달해 생성했다.

```bash
python scripts/build_legal_source_holdout.py build \
  $(find outputs/data/legal-holdout-candidates-v1 -name '*.jsonl' -printf '--candidate %p ') \
  --output-dir outputs/evaluation/legal-source-heldout-i-v1-shards12-15 \
  --target-size 10000
```

완료 selection은 다음과 같다.

```text
status: complete
candidate rows: 242,675
eligible before benchmark exact exclusion: 177,004
selected rows/source documents: 10,000 / 10,000
admrule / statute / ordinance / precedent: 2,998 / 1,006 / 2,998 / 2,998
```

독립 verifier를 실행했다.

```bash
python scripts/build_legal_source_holdout.py verify \
  --output-dir outputs/evaluation/legal-source-heldout-i-v1-shards12-15
```

verifier는 출력 SHA뿐 아니라 training provenance와 benchmark gzip을 다시 읽고 다음을 재계산한다.

- query/corpus/qrels/provenance row 수와 ID 관계
- qrel 1:1 대응 및 score
- training candidate ID overlap 0
- training source document SHA overlap 0
- benchmark query hash overlap 0
- benchmark positive hash overlap 0
- query/positive exact dedup
- full provenance의 repository/revision/document/date 일치
- 네 repository가 training repository와 동일하다는 I 등급 조건

## 실제 10K를 만든 입력 작업

training provenance를 수정하거나 일부를 지우지 않고, 같은 pinned revision에서 training에
아직 나오지 않은 문서를 추가 추출했다.

1. 네 source별 전체 또는 deterministic shard candidate를 새 JSONL로 추출한다.
2. 기존 250K candidate와 새 candidate를 builder의 `--candidate` 반복 인자로 함께 준다.
3. builder가 61,750 training document SHA를 전부 제외하게 둔다.
4. 10K 선택 후 benchmark exact exclusion과 verifier를 통과시킨다.
5. 생성된 파일의 SHA를 clean comprehensive board protocol에 pin한다.

예시는 다음과 같다.

```bash
python scripts/build_legal_source_holdout.py build \
  --candidate outputs/data/legal-performance-v1/candidates/legalize_kr_statutes.jsonl \
  --candidate outputs/data/legal-performance-v1/candidates/legalize_kr_administrative_rules.jsonl \
  --candidate outputs/data/legal-performance-v1/candidates/legalize_kr_precedents.jsonl \
  --candidate outputs/data/legal-performance-v1/candidates/legalize_kr_ordinances.jsonl \
  --candidate outputs/data/legal-heldout-candidates-v1/additional-statutes.jsonl \
  --candidate outputs/data/legal-heldout-candidates-v1/additional-admrules.jsonl \
  --candidate outputs/data/legal-heldout-candidates-v1/additional-precedents.jsonl \
  --candidate outputs/data/legal-heldout-candidates-v1/additional-ordinances.jsonl \
  --output-dir outputs/evaluation/legal-source-heldout-i-v1 \
  --target-size 10000
```

추가 candidate snapshot은 repository revision, source document SHA, extractor config SHA와 file SHA를 별도 manifest로 pin해야 한다. 동일 문서에서 조문을 바꾸어 holdout 수를 채우는 방식은 금지한다.

## Exact 평가와 대화형 noise paired test

[`evaluate_legal_source_holdout.py`](../scripts/evaluate_legal_source_holdout.py)는 고정 query
instruction, source-native positive text, L2-normalized float32 dot product, TF32 off, corpus
ID 오름차순 tie-break로 10K×10K positive rank를 계산한다. NDCG@10, Recall@10/100,
MRR@10, 평균/중앙 rank와 10K per-query rank를 저장한다. local merged artifact는 실제
model shard SHA에서 만든 `model-<sha12>` revision만 허용하며 embedding cache도 이
revision과 dataset manifest SHA에 묶인다.

[`evaluate_conversational_noise_robustness.py`](../scripts/evaluate_conversational_noise_robustness.py)는
동일 corpus 뒤에 의미 없는 system/assistant/filler 문서를 0/1/5% 추가하고 query prompt
on/off를 교차한다. clean evaluator의 prompted-query/corpus cache를 exact hit로 재사용하며,
추가 인코딩은 raw query 10K와 noise 최대 500개뿐이다. 각 condition에서 다음을 저장한다.

- positive NDCG@10/Recall@10과 mean/median rank
- 같은 prompt의 noise 0% 대비 NDCG 유지율
- 가장 높은 noise 문서의 rank와 intrusion@1/5/10
- 6개 condition의 10K per-query positive/noise rank

최종 후보 공개 단계는 clean summary와 robustness summary의 model weight revision,
I-not-Z 독립성, prompt-on/noise-0 NDCG의 exact 일치를 검증한다. summary와 per-query
rank는 모델 repository `evaluation/`에 동봉하고 README clean 표에도 자동 반영한다.

## 테스트

GPU와 외부 network 없이 실행한다.

```bash
python -m unittest -v tests.test_legal_source_holdout
```

fixture는 네 repository에서 각각 3개 문서를 제공한다. repository별 첫 문서를 training provenance에 넣고, 두 번째 행 중 하나는 benchmark query hash, 다른 하나는 benchmark positive hash로 막는다. target 4를 만들면 source별 정확히 1행이 선택되어야 하며 build를 두 번 실행한 다섯 출력 SHA가 모두 같아야 한다. 별도 failure fixture는 모든 source document를 training에 넣고, 약한 row split을 생성하지 않은 채 blocked manifest만 쓰는지 확인한다.
