# 200K Trainer validation 누수 정정과 clean 선택 계약

기준: **2026-07-17 (Asia/Seoul)**
상태: active Qwen 200K는 중단하지 않고 완주·전 checkpoint 보존, 이후 run은 교체 완료

## 결론

기존 `ko_triplet_pilot_10k` validation 512행은 Sionic 9/공식 Korean benchmark
blocklist와는 겹치지 않지만, 현재 199,904행 200K curriculum 안에는 query-positive pair
512개가 전부 들어 있다. 따라서 active Qwen run의 `eval_loss`는 checkpoint 파일이
완성됐고 수치가 finite인지 확인하는 운영 신호로만 사용한다. 일반화 성능이나 best
checkpoint를 고르는 신호로는 사용하지 않는다.

active 학습을 재시작하지 않는 이유는 다음과 같다.

- 문제는 public benchmark 정답을 학습한 것이 아니라, 학습용 ko-triplet 일부를
  validation으로도 다시 사용한 내부 split 오류다.
- 512행은 이미 200K 학습 curriculum의 합법적인 구성원이다. gradient 자체를 무효화할
  이유는 없지만, 그 행의 loss를 held-out 성능이라고 부를 수는 없다.
- watcher가 250-step 간격 adapter를 Trainer의 `save_total_limit`과 무관한 검증 archive에
  보존한다. 완주 후 모든 checkpoint를 독립 10K clean board로 다시 평가하면 오염된
  loss를 모델 선택에서 완전히 제거할 수 있다.

## 재현된 누수 범위

Unicode NFC 및 whitespace 정규화로 active ordered 200K와 기존 validation의 모든 역할을
직접 비교했다.

| 감사 항목 | 결과 |
|---|---:|
| validation query | 512 |
| validation positive | 512 |
| validation negative unique | 471 |
| train query ↔ validation query unique overlap | **512** |
| train positive ↔ validation positive unique overlap | **512** |
| train any document ↔ validation positive unique overlap | **512** |
| exact query-positive pair overlap | **512 / 512** |

그러므로 이미 업로드된 step 250/500의 same-step loss는 adapter 무결성·완료 증거로는
유효하지만 clean 성능 순위가 아니다. 과거 50K/200K Qwen run도 같은 legacy validation을
사용한 경우 내부 best 하나만 보지 않고 남아 있는 모든 무결성 검증 checkpoint를 clean
board에 올린다.

## 새 Trainer validation 계약

첫 v1 10K 감사에서는 whole source-document SHA overlap은 0이었지만, 서로 다른 법률
원문 문서에 동일 조문이 중복 수록돼 legal 250K와 query/positive exact text가 같은
98행을 추가로 발견했다. 이 98행을 원 242,675 candidate pool의 다른 행으로 대체해
text-strict v2 10K를 다시 만들었다. 최종 원본은
`outputs/evaluation/legal-source-heldout-i-v2-text-strict`, 파생 artifact는
`outputs/data/validation/legal-source-heldout-i-v2-text-strict-512`에 둔다.

v2 build는 7개 선언 training JSONL의 모든 역할을 후보 text hash와 대조해 고유
교집합 248개를 찾았다. 이들은 모두 legal 250K에서 나왔고 200K general, 1M general,
retrieval-family, SQuAD, health, AutoRAG에서는 0이었다. stable selection 과정에서
`training_exact_query_hash` 106개를 건너뛴 뒤에도 4개 source 균형
`2,998 / 1,006 / 2,998 / 2,998`, 10,000 고유 query/positive/document를 유지했다.

독립 verifier 결과는 training candidate/document/query/positive 및 benchmark
query/positive overlap이 모두 0이다. manifest SHA-256은
`5455459ee9474430e0ba9f61be84d7a0a577f8f1a1f73f8981aefb6ef61a216e`다. private
artifact는
[`LLM-OS-Models2/korean-legal-source-heldout-retrieval-v2-text-strict@ce9d3bb5`](https://huggingface.co/datasets/LLM-OS-Models2/korean-legal-source-heldout-retrieval-v2-text-strict/tree/ce9d3bb57ca4dc5144753f6d0f8b4a2256851e97)에
업로드했고, remote visibility·allowlist·5개 manifest/data SHA를 다시 내려받아 일치시켰다.

## 후보 snapshot 보존

재시작 뒤 후보 242,675행을 다시 추출하지 않아도 v2 선택을 재현할 수 있도록, pinned
Legalize-KR revision에서 만든 shard `12, 13, 14, 15 / 16`의 JSONL 16개와 추출
manifest 16개를 private snapshot으로 보존했다. 이 snapshot은 clean 평가 결과가 아니라
v2 builder의 중간 입력 증거이며, 독립성·training-text·benchmark exclusion은 최종 v2
builder와 manifest가 별도로 강제한다.

- repository:
  [`LLM-OS-Models2/korean-legal-holdout-candidates-v1-shards12-15@18cbfef7`](https://huggingface.co/datasets/LLM-OS-Models2/korean-legal-holdout-candidates-v1-shards12-15/tree/18cbfef7162fe07470d5377e198062301698ef33)
- commit: `18cbfef7162fe07470d5377e198062301698ef33`
- candidate rows/files: `242,675 / 16`
- extractor manifests: `16`
- snapshot manifest SHA-256:
  `dca58ecb1f89a50e901097988bf15f11b1922b9640841eccf0e01dc1fd07485c`
- 검증: private visibility, exact remote allowlist, 32개 evidence content SHA 모두 일치

로컬 후보와 v2 reference manifest를 다시 묶어 검증·게시하는 명령은 다음과 같다. 토큰은
`.env`에서 프로세스 안으로만 읽으며 로그나 repository에 쓰지 않는다.

```bash
python scripts/publish_legal_candidate_snapshot.py \
  --candidate-dir outputs/data/legal-holdout-candidates-v1 \
  --reference-manifest outputs/evaluation/legal-source-heldout-i-v2-text-strict/manifest.json \
  --repo-id LLM-OS-Models2/korean-legal-holdout-candidates-v1-shards12-15 \
  --hf-token-file .env --upload
```

- 네 Legalize-KR repository에서 정확히 128행씩, 총 512행
- query/positive마다 서로 다른 whole source document
- 같은 repository의 IDF-weighted word/character-bigram lexical hard negative 4개
- stable SHA-256 selection/tie-break와 seed `20260717`
- source 10K가 이미 보장한 training source-document overlap 0
- source 10K가 이미 보장한 Sionic 9/공식 Korean query·positive exact overlap 0
- 200K, 1M, legal 250K, retrieval-family, SQuAD, health, AutoRAG 학습 JSONL의
  query-full/query-body/positive/negative 전 역할과 정규화 exact overlap 0을 생성 시 강제
- legal 250K provenance의 `source_document_sha256`와도 교집합 0을 재확인
- strict ms-swift JSONL, 행별 provenance, 모든 입력·출력 SHA-256 manifest

이 split도 같은 repository/schema에서 왔으므로 독립성 등급은 **I**이며 Z가 아니다.
또한 source-native 1:1 positive 외의 exhaustive relevance judgment가 없기 때문에 512
InfoNCE loss를 최종 retrieval metric처럼 해석하지 않는다.

생성과 독립 검증은 같은 모든 training input을 명시해서 실행한다.

```bash
python scripts/build_clean_training_validation.py build \
  --source-dir outputs/evaluation/legal-source-heldout-i-v2-text-strict \
  --output-dir outputs/data/validation/legal-source-heldout-i-v2-text-strict-512 \
  --target-size 512 --negative-count 4 --seed 20260717 \
  --training-data outputs/data/performance-v1/ablation-200k/train.homogeneous-b16.jsonl \
  --training-data outputs/data/performance-v1/performance-1m/train.homogeneous-b16.jsonl \
  --training-data outputs/data/legal-performance-v1/data/train.jsonl \
  --training-data outputs/data/performance-v1/sionic-retrieval-train-family-4146/data/train.jsonl \
  --training-data outputs/data/performance-v1/sionic-squad-train-60k/data/train.jsonl \
  --training-data outputs/data/performance-v1/sionic-health-multilingual-100k/data/train.jsonl \
  --training-data outputs/data/performance-v1/sionic-autorag-domain-100k/data/train.jsonl \
  --training-provenance outputs/data/legal-performance-v1/metadata/provenance.jsonl
```

`verify`도 동일 인자를 받아 출력 SHA, strict schema, 행별 provenance, 선언된 training
input SHA와 모든 텍스트/문서 교집합을 처음부터 다시 계산한다. 입력이 바뀌거나 출력이 한
byte라도 손상되면 실패한다.

최종 512는 query/positive/negative training-text overlap과 source-document provenance
overlap이 모두 0이며 validation/provenance SHA-256은 각각
`e95ccdf34d6de00292f84f22bfb28ae95eb0bcd9ed8cbb2120216f89701b2703` /
`a30cafb5491de73c991d15914556297a0df08e155d4b517658a8cbfadff5c517`, manifest는
`1a108fe8b5c7c29a842773c11012acb04d12cda939321c8f782bec966eed6aa4`다. private artifact는
[`LLM-OS-Models2/korean-embedding-legal-validation-v2-text-strict-512@8fdd1cad`](https://huggingface.co/datasets/LLM-OS-Models2/korean-embedding-legal-validation-v2-text-strict-512/tree/8fdd1cad0007a9bfadf328d1702dcf6973c3c03d)에
업로드하고 remote visibility·allowlist·세 파일 SHA를 재검증했다.

## 모델 선택과 공개 평가

```text
active legacy Qwen 200K
  -> 모든 250-step adapter archive
  -> checkpoint별 safe merge
  -> 독립 Grade-I legal 10K + robustness
  -> clean-first winner

future Comsat/capacity/1M/KD/specialists
  -> 새 독립 512 validation으로 train-time monitoring
  -> single-best와 last-available-5 평균 모두 독립 10K 평가
  -> clean-first winner
  -> public Sionic 9/공식 Korean 6 final-once
```

public leaderboard 결과는 checkpoint 선택, negative mining, mixture coefficient 선택에
들어가지 않는다. 512는 train-time 신호이고 10K가 model-selection board다. 두 역할을
다시 섞지 않는다.
