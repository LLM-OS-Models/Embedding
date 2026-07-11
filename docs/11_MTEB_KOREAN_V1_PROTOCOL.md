# Official MTEB Korean v1 재현 프로토콜

기준 시점은 2026-07-11, MTEB `2.18.0`, git
`193e3f66d2deac678065a43354c9c4efc57f507d`입니다. 이 문서는 Sionic이
모델 카드에서 구성한 검색 전용 9종 평균과 공식 `MTEB(kor, v1)`을 섞지
않기 위한 실행 명세입니다.

## 정확한 6개 태스크

| Task | Type | Split | HF subset | Main score | Dataset revision | Instruction-aware loader instruction |
|---|---|---|---|---|---|---|
| KLUE-TC | Classification | validation | default | accuracy | `349481ec…` (`klue/klue`, `ynat`) | fallback: `Classify user passages.` |
| MIRACLReranking | Reranking | dev | ko | NDCG@10 | `d11a14c7…` | question → Wikipedia passage |
| MIRACLRetrieval | Retrieval | dev | ko | NDCG@10 | `9c09abc1…` | question → Wikipedia passage |
| Ko-StrategyQA | Retrieval | dev | default | NDCG@10 | `d243889a…` | fallback: `Retrieve text based on user query.` |
| KLUE-STS | STS | validation | default | cosine Spearman | `349481ec…` (`klue/klue`, `sts`) | fallback: `Retrieve semantically similar text.` |
| KorSTS | STS | test | default | cosine Spearman | `016f35f9…` | fallback: `Retrieve semantically similar text.` |

`KLUE-TC`는 기본 MTEB 분류 평가를 사용합니다. label별 train example 8개로
logistic regression을 학습하고 seed 42의 10개 실험 평균 accuracy를 냅니다.
`KLUE-TC.v2`, `MIRACLRetrievalHardNegatives`, 또는 다른 한국어 태스크로
교체하면 공식 Korean v1 결과가 아닙니다.

## Comsat에서 실제 적용되는 prompt

표의 `fallback`은 task metadata의 prompt가 비어 있을 때
`InstructSentenceTransformerModel` 같은 instruction-aware loader가 사용하는
AbsTask 기본값입니다. MIRACL 두 태스크는 fallback 대신 metadata에 적힌
question → Wikipedia instruction을 사용합니다.

MTEB의 task prompt와 모델 prompt는 별개입니다. Comsat은 아직 MTEB model
registry에 등록되어 있지 않으므로 `mteb run`은 일반
`SentenceTransformerEncoderWrapper`로 모델을 읽습니다. 모델 revision
`a5cc22b651c1b2e51cdd8bf671774ae93584f0ab`에 저장된 prompt는 다음과
같습니다.

```text
query:    Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:
document: <empty>
default_prompt_name: null
```

따라서 Retrieval/Reranking에는 query만 위 prefix를 받고 문서에는 prefix가
없습니다. STS와 Classification은 symmetric encode이며 `prompt_type`이 없기
때문에 prefix가 없습니다. MIRACL task metadata에도 자체 prompt가 있지만,
일반 SentenceTransformer wrapper는 Comsat에 내장된 query prompt를 사용합니다.
이 동작은 모델 제작자가 공개한 사용법 및 Sionic 9종 평가와도 일치합니다.

Qwen3-Embedding처럼 MTEB registry에 별도 loader가 있는 모델은 task metadata
instruction을 쓰므로 모델 간 prompt 문자열이 반드시 같지는 않습니다. 공식
MTEB는 모델별 권장 instruction을 허용합니다. 모든 모델에 동일 prompt를 강제하는
Sionic 9종 비교는 별도 실험입니다.

## 실행

GPU를 쓰지 않고 설치된 코드와 명세만 확인합니다.

```bash
PYTHONPATH=third_party/mteb .venv-mteb/bin/python \
  scripts/evaluate_mteb_korean_v1.py --list-only
```

Comsat 전체 6종을 실행합니다. 작업은 task별로 재개할 수 있고 이미 저장된
result는 `only-missing` 정책으로 재사용합니다.

```bash
PYTHONPATH=third_party/mteb .venv-mteb/bin/python \
  scripts/evaluate_mteb_korean_v1.py \
  --model sionic-ai/comsat-embed-ko-8b-preview \
  --revision a5cc22b651c1b2e51cdd8bf671774ae93584f0ab \
  --batch-size 2
```

짧은 태스크부터 부분 실행하려면 다음처럼 반복합니다.

```bash
PYTHONPATH=third_party/mteb .venv-mteb/bin/python \
  scripts/evaluate_mteb_korean_v1.py \
  --task KLUE-TC --task KLUE-STS --task KorSTS --task Ko-StrategyQA
```

그 다음 `MIRACLReranking`, 마지막으로 corpus 1.49M 규모의
`MIRACLRetrieval`을 실행합니다. 출력은
`outputs/evaluation/mteb_korean_v1/<model>/<revision>/`에 저장됩니다.

## 집계와 표기

- `Mean(Task)`: 6개 main score의 동일 가중 평균
- `Mean(TaskType)`: Classification, Reranking, Retrieval, STS 네 type 평균의
  동일 가중 평균
- per-type Retrieval과 STS: 각각 두 task의 동일 가중 평균
- `Rank(Borda)`: 해당 6개를 모두 제출한 leaderboard 모델들과 태스크별 순위를
  비교해야 하므로 Comsat 단독 실행만으로 정할 수 없음
- JSON 원점수는 0–1이며 leaderboard 표는 보통 100을 곱해 표시

Comsat 결과가 공식 MTEB results repository에 merge되기 전에는 README에서
`official leaderboard submitted score`가 아니라 `locally reproduced on the
official protocol`로 표기해야 합니다.

전체 6-task summary가 생기면 live backend에 local row를 가상으로 삽입해 Borda 위치를 계산합니다. 이 명령은 official row 137개의 기존 rank를 먼저 137/137 재현하지 못하면 중단하며 leaderboard를 수정하거나 결과를 제출하지 않습니다.

```bash
.venv-mteb/bin/python scripts/compare_local_mteb_korean.py \
  --summary outputs/evaluation/mteb_korean_v1/sionic-ai__comsat-embed-ko-8b-preview/a5cc22b651c1b2e51cdd8bf671774ae93584f0ab/summary.json \
  --output outputs/evaluation/mteb_korean_v1/comsat-live-comparison.json
```

README에는 `Borda if inserted (live snapshot)`과 날짜를 쓰고, 공식 제출 rank처럼 표현하지 않습니다.
