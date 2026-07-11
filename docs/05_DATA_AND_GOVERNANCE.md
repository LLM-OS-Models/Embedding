# Data and governance

## 목표 데이터 단위

raw 문서만 모아서는 retrieval model을 직접 가르치지 못합니다. 최종 학습 행은 다음 정보를 가집니다.

```json
{
  "query": "...",
  "positive": ["..."],
  "negative": ["..."],
  "source": "...",
  "source_revision": "...",
  "license": "...",
  "document_sha256": "...",
  "generator": "...",
  "prompt_version": "...",
  "split": "train"
}
```

## 목표 mix

| 비중 | 영역 |
|---:|---|
| 30% | 백과·일반 factoid |
| 18% | 금융·공공·상거래 PDF/OCR |
| 15% | long-document retrieval |
| 12% | 법령·판례·행정해석·생활법령 |
| 10% | 건강·공중보건 |
| 10% | multi-hop evidence retrieval |
| 5% | 짧은 query, 오탈자, 구어체, paraphrase |

추가로 10~15%의 multilingual/general replay를 batch 또는 distillation constraint로 섞어 원 Qwen 능력 붕괴를 막습니다. 비율 합계는 실험 mix에 따라 재정규화합니다.

## 후보 문서원

- Korean Wikipedia/Wikisource: attribution/share-alike manifest 필요
- 공공누리 제0/1유형 자료: 문서별 유형 보존
- 법제처 법령·판례·법령해석·생활법령 API: 각 API 이용범위 확인
- OpenDART 구조화 데이터: 원문 제출기업 권리는 별도 확인
- CDC 미국 연방정부 작성 public-domain 문서: third-party content 제외
- PMC Open Access: 논문별 CC license whitelist

주의:

- `ko-triplet-v1.0`은 744,862개의 유용한 pilot triplet이지만 dataset card에 명시적 license가 없고 AIHub, KorQuAD, KLUE, NIKL 등 서로 다른 upstream을 합칩니다. pipeline smoke에는 사용할 수 있으나 release 후보 데이터로 자동 승인하지 않습니다.
- AIHub는 dataset별 예외와 재배포 제한을 확인해야 합니다.
- PublicHealthQA는 CC BY-NC-SA, KorQuAD는 변경/재배포 조건을 별도로 검토해야 합니다.

## Synthetic query

문서마다 2~3종을 생성합니다.

- 실제 검색창형 짧은 query
- 자연스러운 완전한 질문
- 도메인형: 요건, 예외, 수치, 연도, 비교
- multi-hop decomposition query

generator는 query뿐 아니라 answer/evidence span을 출력합니다. verifier가 다음을 검사합니다.

- positive 문서만으로 답할 수 있는가
- query가 positive 문구를 과도하게 복사하지 않았는가
- 다른 candidate도 정답이면 negative가 아니라 multi-positive인가
- 언어, 길이, 중복, 개인정보, license 조건

## Negative policy

query당 저장 후보:

- BM25 2개
- base Qwen dense 2개
- current checkpoint dense 2개
- same-entity/sibling 1~2개
- 숫자·날짜·부정·관할만 다른 adversarial 1~2개

top-1 dense candidate를 무조건 negative로 쓰지 않습니다. Nemotron/NV-Retriever 방식처럼 positive similarity 대비 너무 높은 후보는 false-negative 위험으로 제거하거나 secondary positive로 승격합니다.

## Target benchmark blocklist

아래 9종의 query, qrel, positive 문서와 corpus fingerprint를 먼저 고정합니다.

- MIRACL ko
- MrTyDi ko
- MultiLongDocRetrieval ko
- AutoRAGRetrieval
- Ko-StrategyQA
- PublicHealthQA ko
- Belebele kor-kor
- SQuADKorV1Retrieval
- LawIRKo

검사는 exact normalized hash, URL/title/article ID, 5-gram MinHash, embedding near-duplicate 순서로 수행합니다. benchmark 문서에서 query를 생성한 후 삭제하는 것은 늦습니다. 생성 전 corpus 단계에서 차단합니다.

## 평가 분리

- `dev-clean`: 공개 benchmark와 겹치지 않는 내부 개발셋
- `blind-temporal`: 2026년 이후/새 출처의 사람이 작성한 1K~2K query
- `public-9`: 마지막에만 확인하는 Sionic 비교
- `MTEB-regression`: 원 Qwen의 다국어/STS/classification 능력 보존 확인
