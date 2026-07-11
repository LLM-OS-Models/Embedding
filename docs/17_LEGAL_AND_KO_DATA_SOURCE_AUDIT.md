# Legalize-KR · LLM-Ko-Datasets 원본 감사와 임베딩 데이터 설계

기준일: **2026-07-11 (Asia/Seoul)**  
설정: [`configs/legal_data_sources_v1.json`](../configs/legal_data_sources_v1.json)  
전처리기: [`scripts/prepare_legal_embedding_data.py`](../scripts/prepare_legal_embedding_data.py)

## 결론

주어진 네 Legalize-KR 저장소는 실제로 쓸 만하다. 현재 HEAD에 법령·행정규칙·판례·자치법규 **312,581문서, 3,510,755,300 bytes(3.51 GB, 3.27 GiB)**가 있고, 조문 제목 또는 판례의 판시사항/판결요지처럼 원문 자체가 제공하는 대응 관계가 있다. 세 조문형 저장소에서 확인한 조문 heading만 **2,760,902개**, 판시사항과 판결요지 heading을 모두 가진 판례는 **66,273건**이다.

전처리기를 각 source에 독립적으로 전체 dry-run한 결과, 최소 positive 64자와 exact pair 중복 제거를 통과한 후보는 **2,756,363 rows**다. provenance를 매 row에 보존하는 논리 JSONL 크기는 총 5,614,940,344 bytes다. 실제 mix는 이를 전량 복사하지 않고 source·기관·지역별로 sampling한다.

가장 직접적인 효과는 `LawIRKo`다. 반대로 이 효과는 오염 위험과 같은 이유에서 발생한다. MTEB의 LawIRKo 정의상 corpus가 공식 법령의 개별 조문이고 query가 법명과 조문 식별자에서 파생된다. 따라서 `법명 + 조문 제목 -> 해당 조문`을 학습하고 얻은 LawIRKo 점수는 **target-like supervised adaptation**으로 표시해야 하며 clean zero-shot 점수로 부르면 안 된다.

`gyunggyung/LLM-Ko-Datasets`는 데이터셋이 아니라 외부 데이터 링크 카탈로그다. 현재 revision은 107개의 고유 Hugging Face dataset URL을 가리키지만 학습 example을 한 건도 담지 않는다. 저장소의 Apache-2.0은 카탈로그 자체에 적용될 뿐 링크된 데이터의 라이선스·스키마·수량을 보증하거나 재허가하지 않는다.

성능 우선 트랙에서는 Legalize-KR을 쓸 수 있다. 다만 Sionic 9, 공식 Korean, 종합 점수를 동시에 보존하려면 법률 데이터만 전량 학습하지 말고 전체 contrastive mix의 약 15–25%로 시작해 일반 한국어·다국어 replay와 함께 비교한다. LawIRKo를 노리는 별도 target-adapted checkpoint와 세 보드를 종합하는 checkpoint를 분리하는 편이 결과 해석과 모델 선택에 유리하다.

## 조사 방법과 재현성

다섯 저장소를 2026-07-11에 `main`의 원격 HEAD에서 depth-1로 clone한 뒤 다음을 직접 확인했다.

1. `git ls-remote ... HEAD`와 local `git rev-parse HEAD`
2. README와 실제 `LICENSE` 파일 존재 여부
3. `.git`을 제외한 실제 data glob의 문서 수와 byte 합
4. frontmatter field, Markdown heading, `parsing-failed` 수
5. source-native 구조만으로 만들 수 있는 pair와 평가 중복 위험

Legalize-KR README는 파이프라인 변경 시 전체 history를 force-push할 수 있다고 경고한다. 따라서 branch 이름만 기록하면 재현할 수 없다. 아래 40자리 commit, 파일별 SHA-256, 생성 manifest를 함께 보존해야 한다. 판례 저장소는 선고일을 commit date로 사용하므로 HEAD commit의 1999년 날짜가 저장소가 1999년에 마지막 갱신됐다는 뜻은 아니다.

## 정확한 snapshot inventory

| source | pinned HEAD | data schema | documents | bytes | 구조적 pair 원천 |
|---|---|---:|---:|---:|---:|
| [`legalize-kr/legalize-kr`](https://github.com/legalize-kr/legalize-kr) | `db3cd760c14042ee04fd9166e1bdbb662fc999bc` | `kr/**/*.md` | 5,725 | 277,323,687 | 조문 205,957개 |
| [`legalize-kr/admrule-kr`](https://github.com/legalize-kr/admrule-kr) | `64a5a272909ab5bc077b0ad9519ef31de8febb46` | `**/본문.md` | 20,390 | 294,676,303 | 조문 293,532개 |
| [`legalize-kr/precedent-kr`](https://github.com/legalize-kr/precedent-kr) | `40cd00e54df19d98562abb170c8ff51fd6fe2c2e` | `*/*/*.md` | 124,116 | 1,340,941,601 | 판시+요지 heading 66,273건 |
| [`legalize-kr/ordinance-kr`](https://github.com/legalize-kr/ordinance-kr) | `6443e5dd5833d863219064cd362111f516430bec` | `**/본문.md` | 162,350 | 1,597,813,709 | 조문 2,261,413개 |
| [`gyunggyung/LLM-Ko-Datasets`](https://github.com/gyunggyung/LLM-Ko-Datasets) | `53f7b4a4431c11a1031a56c107b5b12b245471ea` | README link catalog | 0 examples | README 49,051 | 직접 pair 없음 |

`LLM-Ko-Datasets`의 README는 551줄이며 고유 Hugging Face dataset 링크가 107개다. 표에 적힌 upstream 규모는 카탈로그 작성자의 요약값이고 이 감사의 실제 row count 측정값이 아니다.

각 HEAD가 저장한 commit timestamp는 법령 `2026-07-02T12:00:00+09:00`, 행정규칙 `2026-07-10T12:00:00+09:00`, 판례 `1999-04-23T12:00:00+09:00`, 자치법규 `2026-07-09T12:00:00+09:00`, 카탈로그 `2026-01-20T23:14:28+09:00`다. 앞서 설명했듯 Legalize-KR은 법적 사건 날짜를 Git 날짜로 사용하므로 이 timestamp를 GitHub 갱신 시각으로 해석하면 안 된다.

## 1. 법령: `legalize-kr/legalize-kr`

### 형식

각 문서는 YAML frontmatter와 Markdown 본문으로 구성된다.

```text
---
제목: 민법
법령MST: ...
법령ID: ...
법령구분: 법률
공포일자: ...
시행일자: ...
상태: 시행
출처: https://www.law.go.kr/...
---
# 민법
##### 제1조 (법원)
...
```

실측값은 다음과 같다.

- data 문서 5,725개
- `##### 제N조...`가 있는 문서 5,664개
- 조문 heading 205,957개
- metadata `법령구분` 상위값: 대통령령 2,017, 법률 1,747, 대법원규칙 209
- `parsing-failed` 표시는 없음

### 라이선스 표기의 정확한 범위

README는 원문을 국가법령정보센터 OpenAPI에서 받았고 “법령 원문: 공공저작물”, “저장소 구조, 메타데이터: MIT”라고 선언한다. 이 revision에는 최상위 `LICENSE` 파일이 없다. 따라서 문서화할 때는 **README가 그렇게 선언함**이라고 표현해야 하며, 이것을 별도의 법률 검토 결과로 바꾸어 쓰지 않는다.

### 임베딩 용도

가장 보수적인 source-native pair는 다음이다.

```text
query candidate = <법령 제목> <조문 번호와 조문 제목>
positive        = <법령 제목> + <해당 조문 원문>
```

이는 query를 자연어 질문으로 꾸며낸 것이 아니라 원문에 이미 있는 두 anchor를 이어 붙인 것이다. `목적`, `정의`처럼 여러 법에 반복되는 제목은 법명까지 query에 반드시 포함해야 한다.

## 2. 행정규칙: `legalize-kr/admrule-kr`

### 형식과 규모

경로가 기관 계층/행정규칙 종류/규칙명/`본문.md`를 표현하고, frontmatter에 다음 필드가 있다.

- `행정규칙ID`, `행정규칙일련번호`, `행정규칙명`, `행정규칙종류`
- `상위기관명`, `소관부처명`, `기관경로`
- `발령번호`, `발령일자`, `시행일자`, `현행여부`
- `본문출처`, `출처`

실측 20,390문서의 종류는 고시 9,790, 훈령 6,287, 예규 3,784, 공고 202, 지침 162, 국무총리훈령 101, 대통령훈령 52, 기타 12다.

- 조문형 문서 15,062개
- 조문 heading 293,532개
- `본문출처: parsing-failed` 674개, 약 3.31%

`parsing-failed` 문서는 첨부파일만 있거나 API 본문이 없을 수 있으므로 기본 extractor가 제외한다. 라이선스 표기는 법령 저장소와 같은 방식으로 README가 원문 공공저작물, 구조·metadata MIT라고 선언하며 최상위 `LICENSE` 파일은 없다.

## 3. 판례: `legalize-kr/precedent-kr`

### 형식과 규모

경로는 `{사건종류}/{법원등급}/{법원명}_{선고일자}_{사건번호}.md`이고 frontmatter는 `판례일련번호`, `사건번호`, `사건명`, `법원명`, `법원등급`, `사건종류`, `선고일자`, `출처`를 담는다. 본문은 보통 다음 section을 가진다.

```text
# 사건명
## 판시사항
...
## 판결요지
...
## 참조조문
...
## 판례내용
...
```

실측값:

- 전체 124,116문서
- 대법원 68,341, 하급심 55,774, 미분류 1
- `판시사항` heading 76,590문서
- `판결요지` heading 68,889문서
- 두 heading 모두 존재 66,273문서
- `판례내용` heading 124,114문서

### 임베딩 용도

`판시사항 -> 판결요지`는 이 다섯 저장소 중 가장 의미가 명확한 자연 pair다. 판시사항은 법적 쟁점을, 판결요지는 그 판례의 판단을 같은 원자료가 명시적으로 연결한다. 이 pair를 먼저 사용하고, 사건명만 query로 삼거나 전체 판결문을 무조건 positive로 붙이는 약한 pair는 별도 ablation으로 둔다.

전체 판결문은 길고 당사자·절차 문구가 반복되므로 즉시 하나의 positive로 쓰지 않는다. 후속 단계에서 이유/주문/참조조문을 분리하고, 판시사항 또는 판결요지가 실제로 뒷받침되는 chunk를 teacher reranker로 확인해야 한다.

README의 라이선스 선언은 “판례 원문: 공공저작물”, “저장소 구조, 메타데이터: MIT”이고 최상위 `LICENSE` 파일은 없다.

## 4. 자치법규: `legalize-kr/ordinance-kr`

### 형식과 규모

경로는 광역/기초 또는 본청·교육청/종류/법규명/`본문.md`이며 frontmatter에 다음이 있다.

- `자치법규ID`, `자치법규일련번호`, `자치법규명`, `자치법규종류`
- `지자체기관명`, nested `지자체구분`
- `공포일자`, `시행일자`, `제개정구분`, `자치법규분야`, `담당부서`
- `본문출처`, `출처`, 첨부파일 metadata

실측값:

- 162,350문서, 네 legal source 중 가장 큼
- 조문형 문서 162,046개
- 조문 heading 2,261,413개
- 종류가 채워진 문서: 조례 133,243, 규칙 27,508, 의회규칙 1,101, 훈령 478, 예규 13, 고시 6; 종류 결측 1개
- `본문출처: parsing-failed` 220개, 약 0.14%

법명/조문 pair는 법령과 동일하게 만들 수 있다. 지자체와 부서명이 같은 이름의 조례를 구분하는 데 중요하므로 provenance에는 전체 path와 `지자체기관명`을 보존한다. 다만 226만 조문을 모두 같은 비율로 넣으면 법률 문체가 전체 모델을 지배하므로 municipality·법규종류·분야별 cap이 필요하다.

README의 라이선스 선언은 “자치법규 원문: 공공저작물”, “저장소 구조, 메타데이터: MIT”이고 최상위 `LICENSE` 파일은 없다.

## 5. `gyunggyung/LLM-Ko-Datasets`

### 무엇이고 무엇이 아닌가

이 저장소에는 학습 corpus, Parquet, JSONL이 없다. README가 pre-training, mid-training, SFT, DPO/RLHF, 평가 데이터 링크를 정리한다. 따라서 다음을 하면 안 된다.

- README에 `MIT`, `Apache`, `CC BY`라고 적힌 문구만 보고 upstream 데이터를 자동 수집
- 카탈로그의 Apache-2.0을 링크된 107개 데이터에 적용
- README에 적힌 `1.28M`, `12M`, `280GB+` 등을 pin된 실제 row count로 기록
- 평가용 section의 KMMLU·CSAT-QA 등을 일반 학습 source로 자동 편입

각 upstream 후보는 별도로 revision, config/split, row count, schema, dataset card license, 원천 데이터 조건을 감사해야 한다.

### embedding 관점의 우선 후보

아래 숫자와 라이선스는 **카탈로그 README의 주장**이며 아직 upstream 검증값이 아니다.

| 후보 | catalog 표기 | embedding 사용법 | 다음 gate |
|---|---|---|---|
| `eliceai/korean-webtext-edu` | 128만 docs, MIT | high-quality passage pool; teacher query 생성 | exact revision/schema/license, benchmark near-dedup |
| `opendatalab/WanJuan-Korean` | 280GB+, CC BY 4.0 | domain-diverse Korean passage pool | Korean subset 품질·중복·PII·실제 조건 |
| Korean Wikipedia variants | 약 100–500MB, CC BY-SA | title/section -> passage, synthetic QA | MIRACL/MrTyDi/SQuADKor corpus blocklist |
| `nayohan/aihub-en-ko-translation-12m` | 12M, license `-` | cross-lingual positive pair 후보 | AI Hub 원천별 조건과 번역 품질 |
| StackExchange dump | 52.7GB, CC BY-SA | question -> accepted/high-score answer | 언어 filter, attribution, answer selection |
| 한국어 SFT/대화 모음 | 0.1M–1.44M급 다수 | user turn -> assistant answer의 약한 retrieval pair | helpfulness가 relevance를 보장하지 않으므로 teacher filter |
| Yi-Sang/KOREAson | 5.79M prompts + 3.7M traces | reasoning query/evidence 후보 | upstream provenance/license와 답변 grounding |

일반 웹·교과서·CoT text는 그 자체로 contrastive pair가 아니다. CPT text로 쓰거나, passage를 고른 뒤 query를 생성하고 teacher가 positive와 hard negative를 다시 판정해야 한다. 반면 QA, FAQ, title/body, parallel translation처럼 source가 대응을 명시한 구조는 pair 후보가 될 수 있다.

## Sionic 9와 공식 Korean benchmark 오염 위험

### LawIRKo: critical

MTEB `LawIRKo` task code는 corpus가 공식 법령의 개별 조문이고 query가 법률 제목과 조문 식별자의 pair에서 만들어졌다고 명시한다. 데이터 revision은 `on-and-on/lawgov_ir-ko@bd5361e486ef4be7052c506adfdf0610d04abbfe`다.

- [MTEB task source](https://github.com/embeddings-benchmark/mteb/blob/193e3f66d2deac678065a43354c9c4efc57f507d/mteb/tasks/retrieval/kor/law_ir_ko.py)
- [LawIRKo dataset](https://huggingface.co/datasets/on-and-on/lawgov_ir-ko)

따라서 Legalize-KR의 조문 pair와 benchmark의 corpus/query seed가 exact 또는 near duplicate일 가능성이 높다. 성능 우선 checkpoint에서 사용 가능하더라도 결과표에는 `LawIRKo-target-adapted`를 명시한다. clean checkpoint에서는 pinned query와 corpus를 모두 exact hash + MinHash/embedding near-dedup 대상으로 삼는다.

### AutoRAG: high

MTEB task metadata는 AutoRAG를 finance, public, healthcare, legal, commerce의 공개 문서·질문·답변으로 설명하며 dataset revision은 `yjoonjang/markers_bm@fd7df84ac089bbec763b1c6bb1b56e985df5cc5c`다. 법률·공공 slice가 네 저장소와 겹치거나 매우 유사할 수 있다.

- [MTEB AutoRAG task source](https://github.com/embeddings-benchmark/mteb/blob/193e3f66d2deac678065a43354c9c4efc57f507d/mteb/tasks/retrieval/kor/auto_rag_retrieval.py)
- [AutoRAG paper](https://arxiv.org/abs/2410.20878)

### 나머지 Sionic 9와 공식 Korean

법률 pair는 MIRACL/MrTyDi/SQuADKor 같은 Wikipedia retrieval, PublicHealthQA, Belebele, STS, topic classification에 직접 맞지 않는다. 법률 데이터 비율이 지나치면 이 점수들이 하락할 수 있다. 공식 Korean의 MIRACL retrieval/reranking과 Ko-StrategyQA도 일반 지식·질문 표현 replay가 필요하다.

권장 표기는 다음과 같다.

| checkpoint | legal data | benchmark block | 허용되는 주장 |
|---|---|---|---|
| `performance-target-adapted` | 사용, 필요하면 target-like pair 포함 | test row 직접 복사는 금지하되 target corpus exposure 공개 | Sionic 9 성능 최적화, trained-on-domain/target-like |
| `balanced-performance` | 전체 mix 15–25%에서 sweep | query/qrel exact 차단, corpus exposure 기록 | 세 leaderboard 종합 선택용 |
| `clean-release` | exact/near decontamination 뒤 사용 | query·corpus·qrel 파생 text 모두 차단 | clean holdout/zero-shot에 가까운 주장 |

## source-native candidate 생성기

전처리기는 외부 package 없이 다음을 수행한다.

- config의 exact Git revision 검증
- path를 byte-stable 순서로 정렬하고 Unicode NFC 정규화
- `parsing-failed` 문서 제외
- source-native title/heading/article 또는 판시사항/판결요지만 사용
- negative와 LLM query는 생성하지 않음
- exact query/positive 중복 제거
- repo, revision, 상대경로, source URL, 식별 metadata, 원문 SHA-256 보존
- content-derived stable ID와 sorted-key JSONL
- file-level deterministic sharding 및 atomic output

### 전체 extraction dry-run

아래 네 실행은 source마다 dedup set을 새로 만든 독립 dry-run이다. 여러 source를 한 번에 실행하면 source 간 exact duplicate까지 제거되므로 합계보다 조금 작아질 수 있다.

| source | files processed | emitted rows | exact duplicates skipped | logical JSONL bytes | logical output SHA-256 |
|---|---:|---:|---:|---:|---|
| 법령 | 5,725 | 191,283 | 19 | 436,087,091 | `382f1232…25e6` |
| 행정규칙 | 20,390 | 289,472 | 34 | 636,170,860 | `2f211493…c5b6` |
| 판례 | 124,116 | 64,814 | 890 | 173,331,613 | `e24be4ca…477c` |
| 자치법규 | 162,350 | 2,210,794 | 1,207 | 4,369,350,780 | `ef3fbff2…4a2e` |
| 독립 실행 합계 | 312,581 | **2,756,363** | 2,150 | **5,614,940,344** | source별 hash 사용 |

full SHA-256과 parameters는 config의 각 `extraction_dry_run` object에 기록했다. `logical JSONL bytes`는 dry-run 중 실제로 serialization·hashing한 byte 수이며 disk에 dataset을 썼다는 뜻은 아니다.

샘플 실행:

```bash
scripts/prepare_legal_embedding_data.py \
  --source-root data/raw/legal_source_audit \
  --verify-inventory \
  --max-files-per-source 3 \
  --output data/processed/legal_candidates/sample.jsonl \
  --manifest data/processed/legal_candidates/sample.manifest.json
```

구현 검증에서는 source별 3문서, 총 12문서에서 249 rows를 생성했고 출력 SHA-256은 `92bc8681f6a0984968217d43a2caedca4599d2681254664b496d33a813545742`였다. 이는 파이프라인 재현성 확인이며 모델 품질 결과가 아니다.

전체를 source별 shard로 준비하는 예:

```bash
scripts/prepare_legal_embedding_data.py \
  --source-root data/raw/legal_source_audit \
  --shard-count 8 \
  --shard-index 0 \
  --max-query-chars 4096 \
  --max-positive-chars 20000 \
  --output data/processed/legal_candidates/part-00000.jsonl \
  --manifest data/processed/legal_candidates/part-00000.manifest.json
```

`--shard-index`만 0–7로 바꾼다. shard는 file 단위라 한 법령의 조문들이 다른 shard로 갈라지지 않는다. raw candidate는 tokenizer 길이를 모르는 상태이므로 기본값에서 text를 자르지 않는다. 실제 학습 pack에서는 Qwen tokenizer 기준 길이를 다시 측정해야 한다.

## 다음 학습 단계

### 1. sampling

처음부터 282만 후보를 모두 같은 비율로 쓰지 않는다. 첫 balanced run의 시작점은 다음과 같다.

- 법령 조문 150k–205k
- 행정규칙 100k–200k, 기관/종류 cap
- 자치법규 200k–350k, 지자체/종류/분야 cap
- 판시사항→판결요지 50k–65k
- 전체 legal 약 0.5M–0.8M
- 일반 한국어·다국어 retrieval/STS/QA 2M 이상 replay

이 비율은 결론이 아니라 sweep 시작점이다. `legal 10/20/30%` 세 checkpoint를 같은 step/example budget으로 비교하고, Sionic 9 평균뿐 아니라 공식 Korean과 종합 평균의 Pareto frontier로 고른다.

### 2. query enrichment

source-native pair로 1차 학습한 뒤 teacher가 다음 query를 만들 수 있다.

- 법명·조문명을 그대로 묻는 lookup query
- 조문의 요건·예외·기한·금액을 묻는 semantic query
- 조문을 적용해야 하는 짧은 상황형 query
- 판례의 법적 쟁점 paraphrase

생성 text는 곧바로 positive label로 쓰지 않는다. teacher reranker가 source article의 relevance를 확인하고, answerability·citation span·중복·benchmark similarity를 저장한다. 사람이 수동 작성한 소량의 gold query도 동일한 schema와 검수 gate를 거쳐야 한다. 높은 지능으로 만든 데이터라도 자동으로 정답이 되는 것은 아니며, held-out retrieval 개선으로 가치가 확인돼야 한다.

### 3. hard negatives

법률에서 유용한 negative는 무작위 타 도메인 passage가 아니다.

- 같은 법령의 인접 조문
- 다른 법령의 동일 heading (`목적`, `정의`, `벌칙`)
- 제목이나 기관명이 비슷한 시행령/시행규칙
- 같은 사건종류·쟁점 어휘를 가지지만 결론이 다른 판례

false negative가 매우 많으므로 positive-aware threshold와 reranker 검증을 사용한다. 관련 상·하위 법령이나 동일 판례의 다른 관련 section은 negative에서 제외한다.

### 4. 평가와 공개

- LawIRKo, AutoRAG 법률 slice, 전체 Sionic 9를 매 checkpoint 측정
- 공식 Korean과 multilingual/comprehensive replay 회귀를 함께 측정
- model card에 source revision, row 수, sampling weight, query 생성기, teacher, decontamination mode 공개
- `performance-target-adapted`와 `clean-release` 모델을 이름부터 분리

이 데이터는 Sionic 9의 한두 task를 직접 끌어올릴 강한 재료다. 세 leaderboard 전체를 이기게 하는 핵심은 legal source의 양 자체가 아니라, 그 비율을 제한하면서 source-native pair → 검증된 synthetic query → hard-negative curriculum으로 바꾸고 일반 retrieval replay를 보존하는 것이다.
