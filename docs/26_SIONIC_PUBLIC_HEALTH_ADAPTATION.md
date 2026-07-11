# Sionic PublicHealthQA multilingual health adaptation

기준일: 2026-07-12 (Asia/Seoul)

## 목적과 한계

Sionic 9에서 Comsat이 Qwen보다 `PublicHealthQA` 약 `+0.0150`이고, F2LLM-v2-8B는
Comsat보다도 공개 표 기준 약 `+0.0461` 높다. F2는 45M-example multilingual full
fine-tuning에서 의료 QA·instruction·flashcard source를 폭넓게 사용했다. 이 신호를
작게 격리해 Qwen3-Embedding-8B의 한국어 cross-lingual health retrieval에 전달하는
실험이다.

이 100K는 한국어 PublicHealthQA의 공식 train split이 아니다. 공개 evaluation dataset은
test 77개뿐이므로 직접 사용하지 않는다. 영어 중심 의료 데이터가 한국어 test에 도움이
될지는 Qwen의 multilingual alignment에 의존하며, build만으로 개선을 주장하지 않는다.

## 고정 데이터 100K

모든 파일은 `codefuse-ai/F2LLM-v2@d520b8ad02c86d5e5611441c6196ff65d8888927`
에서 읽는다. Parquet footer의 실제 row 수와 output cap을 분리했다.

| Source file | Available | Used | 역할 |
|---|---:|---:|---|
| `pubmedqa.parquet` | 60,227 | 25,000 | 논문 근거 기반 biomedical QA |
| `healthcaremagic.parquet` | 78,626 | 25,000 | 환자 질의·의료 답변 표현 |
| `medical_instruction.parquet` | 75,268 | 20,000 | 의료 instruction 다양성 |
| `medical_flashcards.parquet` | 33,183 | 15,000 | 짧은 개념·정의 retrieval |
| `medmcqa.parquet` | 16,526 | 10,000 | 임상·의학 지식 선택지 hard negatives |
| `medqa_en.parquet` | 3,560 | 3,000 | 의학 시험 QA |
| `webmedqa.parquet` | 27,122 | 2,000 | 중국어 의료 cross-lingual replay |
| 합계 | 294,512 | **100,000** |  |

각 row는 이미 query, positive passage, 최대 24개 explicit negative를 갖는다. builder는
중복·빈 값·길이·언어 조건을 검사하고 seed 42로 최대 7개 negative를 선택한다. 영어와
중국어 source이므로 Hangul 필수 조건은 끄되, instruction prefix를 제외한 실제 query
body 최소 4자 조건은 유지한다.

## 오염 gate

F2 collection card는 2026년 6월 기준 MKQA와 SIB200 train split이 MTEB 평가에 쓰인다고
경고한다. 두 파일은 프로젝트 전역 blocklist에서 이미 제외했다. 이 health shard는 그
파일을 사용하지 않는다.

완성 artifact는 Sionic 9 + 공식 Korean 6의 15-task text-only blocklist와 전수 비교한다.

- training query는 full instruction 포함 hash와 `Query:` 뒤 body hash를 모두 검사
- positive와 7개 negative도 검사
- benchmark `query_text` 또는 non-retrieval `evaluation_text` match가 하나라도 있으면
  build는 학습·업로드 queue에서 탈락
- retrieval corpus-only match는 원문 없이 hash/task/role/source count를 공개하고
  target-adapted로 표시

## 학습 실험

첫 candidate는 1M general winner에서 이어 학습한다.

| 설정 | 값 |
|---|---|
| Primary/replay | health 50% / general 50% |
| 총 rows | 최대 200K, complete homogeneous batch만 사용 |
| Hard negatives | current-student FAISS top-256 → ratio `.95` → pool24 quantile7 |
| Tuner | LoRA r64 우선 |
| LR | `5e-6` |
| Effective batch | 64 |
| Max length | 512, right truncation |
| Selection | held-out InfoNCE + Sionic 9 macro + official Korean + clean regression |

PublicHealthQA는 query가 77개뿐이라 한두 query rank 변화가 점수에 크게 반영된다. 따라서
그 task 하나만 보고 mixture나 checkpoint를 반복 선택하지 않는다. 100K health-only가
일반 retrieval을 훼손하면 25:75 replay로 비율을 낮추고, 그래도 macro gain이 없으면
general winner를 유지한다.

## 공개 계약

- dataset: `performance/non-commercial`, `release_eligible: false`
- model: `health-domain-adapted`; PublicHealthQA train-family exposed라고 쓰지는 않음
- exact model/data revision, task별 9개 NDCG@10, 공식 6-task score를 동봉
- 평가 test query/qrel/corpus를 synthesis, mining corpus, distillation에 사용하지 않음
- 실제 audit와 score가 끝나기 전에는 Comsat 우위나 SOTA를 주장하지 않음

## 완성 artifact

- rows: 100,000; negative 7개/row
- train SHA:
  `6f9715bb130e1d58bac74f13d4b6d1996840bf45b1569ab281a92f632ac15302`
- provenance SHA:
  `cc9e41b7d4c7442ea7f78a4071ed9d94bb439e9374297ab54216b062d67054db`
- row SHA/provenance mismatch: 0/0
- query/evaluation-text critical overlap: 0
- retrieval-corpus overlap: 114 unique hashes; PublicHealthQA overlap 0
- public dataset:
  `LLM-OS-Models/korean-embedding-performance-v1-sionic-health-100k@5fc4bb817f6970a710be53376f35e0225201d2e2`
- local reports:
  `reports/sionic-health-multilingual-100k-training-data-audit.json`,
  `reports/sionic-health-multilingual-100k-benchmark-overlap-audit.json`

장기 campaign은 1M general winner가 나온 뒤 current-student mining, 50:50 replay,
Sionic 9/공식 Korean/clean 평가, 파생 dataset과 merged model 공개를 자동 수행한다.
