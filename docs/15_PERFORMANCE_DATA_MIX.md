# 성능 우선 데이터 믹스 v1

조사·고정일: 2026-07-11. 실행 설정은 [`configs/performance_data_mix_v1.json`](../configs/performance_data_mix_v1.json), 변환기는 [`scripts/build_performance_mix.py`](../scripts/build_performance_mix.py)다.

## 결론부터

적합한 데이터를 못 찾은 상태가 아니다. **지금 바로 만들 수 있는 50K 성능 파일럿과 200K/744K/1M 확장안까지 핀으로 고정했다.** 다만 `nlpai-lab/ko-triplet-v1.0`의 기존 hard negative 1개만 반복하는 것은 이미 288-row smoke에서 loss가 즉시 0에 가까워졌으므로 충분하지 않다. 성능을 만드는 핵심은 단순한 “고품질 한국어 문서”의 양이 아니라 다음 관계가 들어 있는 학습 row다.

```text
(task instruction, 실제 검색 query, 관련 positive passage,
 같은 주제이지만 답이 아닌 hard negatives 4~7개)
```

따라서 v1은 다음을 결합한다.

1. `ko-triplet`의 넓은 한국어 QA/문서 범위
2. F2LLM-v2가 공개한 query-positive-24 hard-negative 한국어 데이터
3. 공식 benchmark의 **train split만** 사용하는 분류·STS·추론 retrieval 데이터
4. 큰 단계에서만 넣는 KaLM 영어·중국어·다국어 replay
5. 다음 데이터 버전에 넣을 법령·판례·행정규칙·자치법규 구조화 pair

이 트랙은 사용자의 우선순위에 맞춘 **private/non-commercial 성능 트랙**이다. 데이터 카드 상단 license만으로 공개 재배포 가능성을 주장하지 않으며, evaluation dev/test/validation query·qrel·candidate 유출은 성능 트랙에서도 금지한다. 나중에 권리 안전 데이터로 재학습하거나 teacher distillation하는 트랙과 분리한다.

## “고품질 한국어”만 있으면 되는가

아니다. raw 문서만 모으면 next-token CPT에는 쓸 수 있지만, embedding 검색 경계를 직접 가르치지는 못한다. 같은 문서에서도 아래처럼 목적에 맞는 특화 전처리가 필요하다.

| 필요한 단위 | 좋은 예 | 나쁜 예 | 이유 |
|---|---|---|---|
| query | 자연 질문, 키워드 검색, 구어체, 법조문 인용, 오탈자 query | 문서 첫 문장을 그대로 복사 | 실제 query/document 표현 차이를 학습 |
| positive | query의 모든 필수 조건을 만족하는 근거 passage | source 문서를 검증 없이 자동 정답 처리 | source가 실제 답이 아닐 수 있음 |
| hard negative | 같은 사건·법률·질병·상품이나 핵심 조건 하나가 다른 문서 | 완전히 다른 주제의 임의 문서만 사용 | 쉬운 negative는 loss가 즉시 0이 됨 |
| long positive | 답 근거가 앞/중간/뒤에 고르게 놓인 1K~8K+ 문서 | 짧은 문장을 padding해 장문처럼 만듦 | MLDR·실제 RAG의 evidence 위치 일반화 |
| task instruction | web search, semantic similarity, classification, evidence retrieval을 구분 | 모든 row에 같은 모호한 문구 | Qwen3의 instruction-aware embedding 계약 유지 |
| provenance | source/revision/split/task exposure/row hash | 합친 뒤 출처 삭제 | zero-shot·권리·오염 주장을 검증 가능하게 함 |

최소 전처리는 NFKC와 공백 정리, 정확 중복 제거, 길이·언어 필터다. 법률의 `제N조`, 표·목록·항 번호, 판례의 `판시사항/판결요지/이유`, OCR 흔적은 검색 단서가 되므로 무조건 평문으로 뭉개지 않는다. query와 문서는 별도 길이 bucket으로 묶고, 서로 다른 source를 같은 in-batch negative로 무작정 섞지 않는 homogeneous batching을 우선한다.

## 실제 공개 소스 감사

### 바로 실행하는 소스

| 소스·고정 revision | 확인한 live schema·규모 | v1에서 쓰는 이유 | 주의점 |
|---|---|---|---|
| [`nlpai-lab/ko-triplet-v1.0@1f5d72d`](https://huggingface.co/datasets/nlpai-lab/ko-triplet-v1.0/tree/1f5d72d21ae8309b5221a588b13930b423385bff) | train 744,862; `query`, `document`, `hard_negative` | 큰 한국어 core를 가장 빨리 확보 | 명시 license 없음, 구성·decontamination 불명, negative 1개가 현재 Qwen에 너무 쉬울 수 있음 |
| [`codefuse-ai/F2LLM-v2@d520b8a`](https://huggingface.co/datasets/codefuse-ai/F2LLM-v2/tree/d520b8ad02c86d5e5611441c6196ff65d8888927) | 기본 row는 `query`, `passage`, `negative_1..24`, 일부 `lang`; 공개 composite 60.1M/157 sources | F2가 실제 사용한 24-candidate hard-negative 구조와 다양한 한국어 query 유형을 그대로 활용 | collection card는 Apache-2.0이나 upstream별 조건은 별도; 정확 miner/decontamination은 미공개 |
| [`KaLM-Embedding/KaLM-embedding-finetuning-data@e9443ab`](https://huggingface.co/datasets/KaLM-Embedding/KaLM-embedding-finetuning-data/tree/e9443ab6f5d4dc29c79cea03834e932428ed6ab1) | `query`, `pos: list[string]`, `neg: list[string]`, 보통 negative 7; MIT 표기 | 큰 단계에서 원 Qwen의 다국어/STS 표현을 보존하는 replay | 주력 한국어 source가 아니며 여러 MTEB train family를 포함. upstream 조건과 zero-shot 노출을 별도 표기 |

요청에 적힌 `tencent/KaLM-embedding-finetuning-data`라는 공개 dataset ID는 2026-07-11 현재 존재하지 않는다. KaLM 카드와 Hub 검색에서 확인되는 canonical 공개 repo는 `KaLM-Embedding/KaLM-embedding-finetuning-data`이며 이를 SHA로 고정했다.

F2에서 선택한 실제 파일과 parquet metadata는 다음과 같다. 모두 revision `d520b8ad...`다.

| 파일 | rows | 주 용도 | target benchmark train 노출 |
|---|---:|---|---|
| `webfaq_kor.parquet` | 89,271 | web QA/retrieval | 알려진 직접 노출 없음 |
| `mqa_ko.parquet` | 137,035 | 다양한 Korean QA | 알려진 직접 노출 없음 |
| `koalpaca.parquet` | 21,126 | instruction/QA | 알려진 직접 노출 없음 |
| `koalpaca_realqa.parquet` | 17,599 | 자연스러운 QA | 알려진 직접 노출 없음 |
| `komagpie.parquet` | 428,780 | 대규모 합성·instruction retrieval | 알려진 직접 노출 없음 |
| `miracl_ko.parquet` | 753 | Wikipedia evidence retrieval | MIRACL train-family 노출 |
| `mrtidy_korean.parquet` | 1,294 | 한국어 dense retrieval | MrTidy train 노출 |
| `mldr_ko.parquet` | 2,252 | 실제 장문 retrieval | MLDR train-family 노출 |
| `pawsx_ko.parquet` | 32,109 | paraphrase/semantic boundary | PAWS-X task-family 노출 |
| `paracrawl_ko-en.parquet` | 245,966 | 한국어→영어 cross-lingual | 알려진 직접 Korean-v1/Sionic9 노출 없음 |
| `paracrawl_en-ko.parquet` | 245,974 | 영어→한국어 cross-lingual | 알려진 직접 Korean-v1/Sionic9 노출 없음 |

F2 공식 변환 코드는 source별 최대 80K cap, positive 1개, candidate 24개, step당 negative 7개 표집을 사용한다. 우리 변환기도 기본적으로 24개 중 최대 7개를 row hash 기반으로 결정론적으로 표집한다. 다만 F2 파일 하나가 parquet row group 하나인 경우가 많아 작은 random subset을 만들더라도 해당 원본 파일 전체를 받아야 할 수 있다. 50K 준비의 네트워크·디스크 병목은 약 50K row 자체가 아니라 이 큰 원본 shard다.

### 감사했지만 v1 실행 mix에는 아직 넣지 않은 Nemotron

[`nvidia/embed-nemotron-dataset-v1@f457c3e`](https://huggingface.co/datasets/nvidia/embed-nemotron-dataset-v1/tree/f457c3e2da4af3b9dd2818685d411b26298d7cbb)는 14개 공개 subset, 3,662,695 queries, 9,118,599 documents를 선언한다. 28개 Hub config가 query/corpus로 나뉜다.

```text
query config:  question_id, question, corpus_id, pos_doc[].id, neg_doc[].id
corpus config: id, text
```

일부 source는 저작권 때문에 text 대신 ID만 재배포되어 NVIDIA data-preparation 코드로 원 source를 다시 받아야 한다. 한국어에 직접 도움이 되는 공개 부분은 주로 MIRACL이고 이는 이미 target train 노출이 있는 반면, 나머지는 영어 중심 replay다. 그래서 50K/200K의 빠른 한국어 성능 실험에는 넣지 않고, 다국어 regression이 실제로 발생할 때 per-config join을 구현해 추가한다. dataset card는 research/development 용도와 source별 governing terms를 요구하며, synthetic classification subset은 생성에 쓴 Llama license 영향도 명시한다.

## benchmark train split 사용과 zero-shot 표기

평가 dataset과 같은 **task의 공식 train split**은 성능을 올리는 합법적인 supervised signal이지만 더 이상 그 task에서 zero-shot 모델은 아니다. 아래를 모델 카드와 결과 JSON에 자동 기록한다.

| task | 평가 split | 사용 가능한 train 근거 | v1 결정 |
|---|---|---|---|
| KLUE-TC | validation | `klue/klue` YNAT train 45,678 | train 1K/5K/10K/15K 사용, 노출 표시 |
| KLUE-STS | validation | KLUE STS train 11,668; score≥3.5 positive 4,533 | 최대 4K 사용, 노출 표시 |
| KorSTS | test | KorSTS train 5,691; score≥3.5 positive 2,036 | 최대 1.5K 사용, 노출 표시 |
| Ko-StrategyQA | dev | train qrels 4,377, unique train query 2,242, corpus 9,251 | train qrel ID만 출력; shared query store는 train ID로 필터하고 dev qrels는 로드하지 않음 |
| MIRACL retrieval/reranking | dev | pinned MTEB evaluation repo에는 dev만 있음; F2가 제공한 별도 `miracl_ko` train-family 753 | F2 file만 사용, MTEB dev repo 전면 제외 |
| MrTidy Korean | test | raw train query 1,295; F2 processed 1,294 | F2 train file 최대 1.2K 사용 |
| MLDR Korean | dev/test | pinned MTEB repo에는 dev/test만 있음; F2 processed training-family 2,252 | F2 file만 사용, pinned dev/test 전면 제외 |
| AutoRAG | test | 평가 repo에 usable train split 없음 | 전체 평가 repo 제외 |
| PublicHealthQA Korean | test | test만 있음 | 전체 평가 repo 제외 |
| Belebele Korean | test | test만 있음 | 전체 평가 repo 제외 |
| SQuADKorV1 Retrieval | test | 현재 MTEB 변환 repo는 test만 있음 | 전체 평가 repo 제외 |
| LawIRKo | MTEB test | underlying storage가 `train`으로 보이지만 바로 그 내용이 평가 corpus/query/qrel | repo의 **모든 split** 제외 |

따라서 `pilot_50k`는 공식 Korean-v1의 KLUE-TC, KLUE-STS, KorSTS, Ko-StrategyQA, MIRACL 계열에 supervised exposure가 있다. MIRACL은 retrieval과 reranking 두 task가 같은 family이므로 Korean-v1 여섯 task 모두에 직·간접 train-family 노출이 있는 in-domain 성능 모델로 표기한다. Sionic9에서는 MIRACL, MrTidy, MLDR, Ko-StrategyQA 4/9 task family가 노출되고, AutoRAG·PublicHealthQA·Belebele·SQuADKorV1·LawIRKo 5/9는 evaluation-row 기준 zero-shot을 유지한다.

이것은 점수를 무효화하지 않지만 “zero-shot SOTA”라고 부르면 안 된다. leaderboard에는 raw score와 함께 `trained_on_tasks`를 내보낸다. F2 카드가 경고한 `mkqa_*.parquet`와 `sib200.parquet`도 현재 MTEB가 train split을 평가에 사용하므로 전부 block했다.

## 50K → 200K → 744K → 1M 계획

정확한 source cap은 JSON 설정에 있으며 합계는 변환기가 실행 전에 검증한다.

| phase | rows | 역할 | 중요한 구성 |
|---|---:|---|---|
| `pilot_50k` | 50,000 | LoRA rank/loss/negative/length ablation | ko-triplet 25,147 + F2 Korean 21,400 + benchmark train 3,453 |
| `ablation_200k` | 200,000 | LoRA vs partial/full FT 첫 품질 비교 | ko-triplet 100,254, F2 hard-negative·cross-lingual 확대, official train task signals |
| `ko_core_744k` | 744,000 | 큰 Korean core | ko-triplet 465,254, F2 231,146, benchmark train 17,600, KaLM replay 30K |
| `performance_1m` | 1,000,000 | private performance candidate | ko-triplet 600,254, F2 Korean/cross-lingual 351,146, benchmark train 22,600, KaLM replay 26K |

`744K`라는 이름은 `ko-triplet` 744,862개를 모두 그대로 쓰겠다는 뜻이 아니다. 쉬운 negative가 많은 한 source를 전량 복제하기보다 F2의 24-candidate rows, STS/classification, long retrieval, cross-lingual replay를 섞는다. 각 큰 phase는 작은 phase의 결정론적 prefix를 포함하므로 scale curve를 비교하기 쉽다.

### 첫 학습 뒤 반드시 다시 만드는 부분

v1 파일은 공개 source가 주는 negative를 우선 사용해 빨리 시작한다. 그러나 다음 candidate refresh 없이는 최종 SOTA 후보가 아니다.

1. BM25, base Qwen3-Embedding-8B, 현재 student의 top candidates를 합친다.
2. positive와 candidate를 같은 teacher로 점수화한다.
3. `s_neg < α·s_pos`, `α ∈ {0.90, 0.95, 0.98}`를 dev-clean에서 비교한다.
4. false negative는 secondary positive로 승격하거나 제거한다.
5. query당 24개 pool을 저장하고 step마다 4~7개를 표집한다.
6. top-hard만 쓰지 않고 teacher score quantile을 고르게 포함해 distillation 후보를 만든다.
7. 같은 source·length bucket으로 batch를 구성한다.

`ko-triplet`에서 loss가 다시 즉시 0이면 더 많은 epoch나 full FT로 해결하지 않는다. 그 row는 이미 해결된 쉬운 sample이므로 current-student mining으로 교체한다.

## 법률·판례 데이터: 매우 적합하지만 raw 문서 그대로는 부족

새로 제시된 Legalize-KR 네 저장소는 LawIRKo와 실제 RAG를 겨냥한 가장 가치 있는 후보군이다. 모두 commit SHA로 catalog에 고정했다.

| 저장소·revision | 확인한 구조·최소 규모 | 만들 pair |
|---|---|---|
| [`legalize-kr/legalize-kr@db3cd76`](https://github.com/legalize-kr/legalize-kr/tree/db3cd760c14042ee04fd9166e1bdbb662fc999bc) | 법령 Markdown 5,725; YAML + `제N조` 구조 | 법령명/조문명/인용 query → 해당 조문; 같은 법의 이웃 조문 negative |
| [`legalize-kr/admrule-kr@64a5a27`](https://github.com/legalize-kr/admrule-kr/tree/64a5a272909ab5bc077b0ad9519ef31de8febb46) | 행정규칙 `본문.md` 20,390; 기관·규칙 종류·조문 metadata | 기관/업무/규칙 query → 조문; 같은 기관·유사 제목 규칙 hard negative |
| [`legalize-kr/precedent-kr@40cd00e`](https://github.com/legalize-kr/precedent-kr/tree/40cd00e54df19d98562abb170c8ff51fd6fe2c2e) | GitHub tree가 잘린 상태에서도 판례 Markdown ≥62,217; `판시사항/판결요지/본문` | 쟁점 → 판결요지/이유, 법조문 인용 → 관련 판례, 같은 법률의 다른 결론 negative |
| [`legalize-kr/ordinance-kr@6443e5d`](https://github.com/legalize-kr/ordinance-kr/tree/6443e5dd5833d863219064cd362111f516430bec) | tree가 잘린 상태에서도 자치법규 `본문.md` ≥22,959; 지역·기관 metadata | 지역+정책 query → 해당 조례; 제목이 같은 타 지역 조례 hard negative |

추천 변환은 다음과 같다.

- 문서를 `법령/장/조/항` 및 `판시사항/판결요지/이유` 단위로 구조 보존 chunking한다.
- query를 제목 복사 하나로 끝내지 않고 자연 질문, 법률가식 인용, 키워드, 관할·시점 조건을 각각 만든다.
- positive에는 답 근거가 있는 조문/문단과 상위 제목을 함께 넣는다.
- hard negative는 동일 법률의 인접 조문, 동일 키워드의 다른 법률, 같은 조례명의 다른 지자체, 같은 법조문을 인용하지만 결론이 다른 판례에서 뽑는다.
- 장문 row는 full law/case에 evidence 위치를 앞·중간·뒤로 균형화한다.
- 시행일·선고일을 보존해 과거/현행 규정이 섞인 temporal negative를 만든다.

가장 중요한 제한은 LawIRKo다. Legalize-KR의 공식 법령 원문은 LawIRKo corpus와 동일하거나 거의 동일할 수 있다. **LawIRKo query/qrel뿐 아니라 평가 corpus의 exact hash, normalized hash, MinHash near-duplicate를 먼저 block한 뒤** 남은 법령에서만 query를 생성한다. 평가 snapshot과 같은 조문으로 synthetic query를 만들면 이름만 train split일 뿐 사실상 test leakage다.

[`gyunggyung/LLM-Ko-Datasets@53f7b4a`](https://github.com/gyunggyung/LLM-Ko-Datasets/tree/53f7b4a4431c11a1031a56c107b5b12b245471ea)는 실제 payload가 아니라 dataset 링크와 설명이 있는 README catalog다. discovery에는 유용하지만 그 repo 자체를 학습 source로 세지 않는다. 링크된 각 HF dataset을 별도 revision/schema/license/task-overlap으로 다시 감사한 후 mix에 넣는다.

## 변환기 사용법

데이터를 받지 않고 네 phase와 오염 정책을 먼저 검증한다.

```bash
python scripts/build_performance_mix.py --list
python scripts/build_performance_mix.py --phase pilot_50k --dry-run
```

50K를 실제 생성한다. 공개 source들이므로 HF token 없이도 되지만 `.env`를 shell에 안전하게 load하면 rate limit이 완화된다. token 값은 출력하지 않는다.

```bash
.venv-train/bin/python scripts/build_performance_mix.py \
  --phase pilot_50k \
  --output-dir outputs/data/performance-v1/pilot-50k

.venv-train/bin/python scripts/validate_embedding_jsonl.py \
  outputs/data/performance-v1/pilot-50k/train.jsonl
```

변환기는 다음을 보장한다.

- 모든 Hub source의 40-character revision pin 검증
- phase cap 합계 검증
- known evaluation split/repo/file blocklist 검증
- 결정론적 source별 shuffle와 negative 표집
- Hangul/길이/중복/positive=negative 필터
- strict ms-swift JSONL에는 세 필드만 기록
- 별도 `provenance.jsonl`에 source, revision, split, `trained_on_tasks`, row hash 기록
- `manifest.json`에 source별 examined/rejected/accepted와 최종 file SHA-256 기록
- 기본 `private`, `research-noncommercial`, `release_eligible:false`

작은 converter smoke는 실제로 완료했다.

| smoke | 결과 |
|---|---|
| ko-triplet adapter | 2/2 생성, strict validator 통과 |
| F2 24-negative adapter | 2/2 생성, validator 통과 |
| KLUE STS adapter | 2/2 생성, low-score negative pool 변환 통과 |
| KLUE classification adapter | 2/2 생성, label negatives 변환 통과 |
| Ko-StrategyQA qrels/query/corpus join | 2/2 생성, train qrel ID만 출력하고 dev qrels를 로드하지 않은 채 통과 |
| KaLM list adapter | 2/2 생성, 7-negative schema 변환 통과 |

### 실제 50K build

2026-07-11 KST에 `pilot_50k` 전체를 생성하고 strict validator로 다시 읽었다.

| Artifact | Rows | SHA-256 |
|---|---:|---|
| `train.jsonl` | 50,000 | `b46a7be9842ab27e9dfd85e9831080d94410e5b38d956682072068ee7f18258a` |
| `provenance.jsonl` | 50,000 | `e8ccca33bb9ec73700895ab2ac17ae57e20875170be1ed0f7a3dbc20b11e6031` |

두 파일과 manifest의 로컬 크기는 약 349MiB다. output 자체는 Git에 넣지 않고
source별 accepted/examined/rejected 수와 hash를 manifest에 보존한다. `train.jsonl`은
strict ms-swift schema 검증을 통과했다. 이 build 완료는 데이터 품질이나 모델 성능
결과가 아니며, 후속 10K/50K 학습과 retrieval 평가로 판단한다.

### 실제 200K/1M build와 exact training order

200K와 1M도 생성·검증·공개를 완료했다. source shortcut을 줄이기 위한 단일-source
microbatch는 유지하되, 같은 source 안에서 max text-character length proxy로 정렬한 뒤
16-row batch를 만들고 batch 순서만 결정론적으로 섞었다. 200K에서 random source-homogeneous
order의 padded proxy는 `160,181,088`, length-bucketed order는 `85,258,880`으로
46.77% 감소했다. 실제 row proxy 합 대비 잔여 padding은 0.31%다.

| Phase | Ordered rows | Ordered train SHA-256 | Public dataset revision |
|---|---:|---|---|
| 50K active run (source-homogeneous, random within source) | 49,904 | `39078ebbbea895b2fbb0fa701367cb839a60c4d9460246b39753d783c7a5d717` | [`29c23fc`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-performance-v1-pilot-50k/tree/29c23fcdc7b34279a060ee765448e7ecadd1563e) |
| 200K | 199,904 | `59b08c0691caaa02e7520e9c98cf31f890679a82262ead789c1a7614f8baf285` | [`2872bfd`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-performance-v1-ablation-200k/tree/2872bfd02fe65cabf37cc29c08b66865bc3e58a4) |
| 1M | 999,936 | `436dc7486578f6f077bef9f4479bc0d98310d855306bd2aad0c0d40fffbf2c00` | [`fa9cfee`](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-performance-v1-performance-1m/tree/fa9cfee59356e173827ea33339c06d8d8b388acc) |

세 repository에 base `train.jsonl`/provenance뿐 아니라 exact
`train.homogeneous-b16-length-bucketed.jsonl`, 대응 provenance, manifest까지 공개했다.
따라서 모델 카드의 training SHA에서 실제 입력 순서를 그대로 복원할 수 있다.

## 현재 병목과 학습 결정

현재 병목은 “데이터를 못 찾음”이 아니다.

1. F2의 큰 single-row-group parquet를 받아야 해서 50K sampling 전 다운로드가 수 GB가 될 수 있다.
2. `ko-triplet` negative가 current Qwen에 너무 쉬운 비율을 50K에서 측정하고 re-mining해야 한다.
3. 공개 benchmark train을 의도적으로 썼으므로 zero-shot과 in-domain score를 분리 보고해야 한다.
4. 법률 raw corpus를 pair/triplet으로 만드는 query generator, LawIR corpus block, miner가 다음 데이터 factory 작업이다.
5. 1M을 먼저 만들어도 bad/easy rows가 늘면 성능이 오르지 않는다. 50K의 clean dev delta와 loss-active rate를 통과한 source만 확장한다.

튜너는 50K에서 LoRA r32/r64로 data/loss/negative를 먼저 찾는다. 같은 200K, 같은 steps/tokens에서 LoRA와 partial/full FT를 비교한다. F2가 Qwen-chat에서 full FT로 성공한 것은 강한 선례지만, 우리는 이미 embedding-pretrained인 Qwen3-Embedding-8B를 시작점으로 하므로 작은 한국어 mix에서 바로 full FT하면 다국어·STS geometry를 더 쉽게 잃을 수 있다. full FT 승격 조건은 train loss가 아니라 다음 네 가지다.

- Sionic9 clean subset와 우리 clean/temporal dev 개선
- 공식 Korean-v1의 exposed/unexposed task별 개선
- multilingual/STSB regression 허용치 통과
- VRAM/GPU-hour 대비 LoRA보다 유의미한 NDCG·STS 이득

## 일차 출처

- [F2LLM-v2 paper](https://arxiv.org/abs/2603.19223), [official training code](https://github.com/codefuse-ai/CodeFuse-Embeddings/tree/1c5291549b9cee9eeab1cd9de6a67be4d0295da0/F2LLM), [pinned data](https://huggingface.co/datasets/codefuse-ai/F2LLM-v2/tree/d520b8ad02c86d5e5611441c6196ff65d8888927)
- [KaLM-Embedding-V2 paper](https://arxiv.org/abs/2506.20923), [pinned finetuning data](https://huggingface.co/datasets/KaLM-Embedding/KaLM-embedding-finetuning-data/tree/e9443ab6f5d4dc29c79cea03834e932428ed6ab1)
- [Llama-Embed-Nemotron paper](https://arxiv.org/abs/2511.07025), [pinned released dataset](https://huggingface.co/datasets/nvidia/embed-nemotron-dataset-v1/tree/f457c3e2da4af3b9dd2818685d411b26298d7cbb)
- [pinned MTEB implementation](https://github.com/embeddings-benchmark/mteb/tree/193e3f66d2deac678065a43354c9c4efc57f507d)
- [Legalize-KR law](https://github.com/legalize-kr/legalize-kr/tree/db3cd760c14042ee04fd9166e1bdbb662fc999bc), [administrative rules](https://github.com/legalize-kr/admrule-kr/tree/64a5a272909ab5bc077b0ad9519ef31de8febb46), [precedents](https://github.com/legalize-kr/precedent-kr/tree/40cd00e54df19d98562abb170c8ff51fd6fe2c2e), [ordinances](https://github.com/legalize-kr/ordinance-kr/tree/6443e5dd5833d863219064cd362111f516430bec)
