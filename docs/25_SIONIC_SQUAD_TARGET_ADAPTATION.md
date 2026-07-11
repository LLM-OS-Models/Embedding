# Sionic SQuADKorV1 train-family target adaptation

기준일: 2026-07-12 (Asia/Seoul)

## 결론

Sionic 9종에서 아직 직접 train-family 신호가 없던 `SQuADKorV1`을 보강하기 위해
KorQuAD v1.0의 원본 **train 60K만** 별도 curriculum으로 만든다. 평가에 쓰이는 원본
validation/MTEB test는 차단한다. 이 실험은 합법적인 supervised target adaptation이지만
SQuADKorV1 zero-shot 결과가 아니므로 모든 결과·모델 카드에 노출을 표시한다.

60K만 단독 과학습하지 않는다. 현재 general winner의 embedding으로 hard negative를
다시 채굴하고, general replay와 섞어 낮은 learning rate로 짧게 적응시킨 뒤 Sionic 9,
공식 Korean v1, clean holdout을 모두 비교한다.

## 원본과 경계

| 항목 | 고정값 |
|---|---|
| Repository | `KorQuAD/squad_kor_v1` |
| Revision | `01aad23853355e5f4f6317eeaaa8186811424834` |
| 사용 config/split | `squad_kor_v1/train` |
| 원본 train | 60,407 rows |
| 고유 context | 9,606 |
| loader 사용 금지 | 원본 validation 5,774, `yjoonjang/squad_kor_v1` test 전체 |
| 학습 노출 표시 | `trained_on_tasks=["SQuADKorV1"]` |

원본 train의 60,407개에서 normalized `(question, context)` 중복 153개를 제거하고
정확히 60,000개를 사용했다. 21개의 query 문자열 중복은 서로 다른 answer context가
달린 경우이며 pair hash는 겹치지 않는다. positive context 중복 50,394개는 하나의
Wikipedia 문맥에 평균 약 6.3개 질문이 달린 KorQuAD 구조다.

Hub metadata는 라이선스를 `CC-BY-ND-4.0`으로 표시하고 dataset card 본문은
`CC BY-ND 2.0 KR`도 적는다. 그래서 performance/non-commercial 트랙에는 쓰되
rights-clean 트랙으로 자동 승격하지 않는다.

평가 repository를 loader 입력으로 읽지 않았다는 사실만으로 corpus 비중복이 보장되지는
않는다. KorQuAD, MIRACL, MrTidy가 Wikipedia 문맥을 공유하기 때문이다. 완성 후 15-task
text-only blocklist 전수 감사에서 query/evaluation-text match는 0이었지만 고유 6,426개
context가 retrieval 평가 corpus와 exact match했다(MIRACL 6,376, MrTidy 5,863,
SQuADKorV1 3; task 간 중복 가능). 따라서 이 shard는 clean 데이터가 아니라
`target-adapted/shared-corpus-exposed` 데이터다.

## 변환 계약

```text
query    = question
positive = answer를 포함하는 context 전체
bootstrap negatives = 다른 고유 context 7개
instruction = Given a Korean question, retrieve the passage that contains the answer
```

초기 7개 negative는 결정론적이고 정답 context와 exact match하지 않지만 의미적으로
어렵다는 보장은 없다. 최종 target-adaptation 입력은 다음 순서로 다시 만든다.

1. 현재 general winner로 60K query와 9,606 unique context를 last-token pooling/L2
   normalization으로 임베딩한다.
2. FAISS IVFFlat으로 query당 top-256을 가져온다.
3. own positive와 exact query-document match를 제외한다.
4. `s_neg < 0.95 * s_pos`를 만족하는 상위 24개만 candidate pool로 둔다.
5. top-7만 쓰지 않고 score-rank quantile 7개를 골라 지나치게 비슷한 false negative와
   쉬운 negative 사이의 난이도를 분산한다.
6. mining manifest에 model weights SHA, input SHA, FAISS 설정, selected score 분포를
   저장한다.

## 전수 품질 감사

| 검사 | 결과 |
|---|---:|
| 출력 rows | 60,000 |
| row SHA mismatch | 0 |
| provenance index mismatch | 0 |
| negatives/row | 7 (60,000/60,000) |
| query body 4자 미만 | 0 |
| natural-question heuristic | 59,254 (98.76%) |
| query p50 / p95 | 32 / 57 chars |
| positive p50 / p95 | 462 / 862 chars |
| positive max | 10,012 chars |
| benchmark query/evaluation-text exact match | **0** |
| benchmark retrieval-corpus unique exact match | 6,426 |
| corpus-match positive / bootstrap-negative occurrences | 40,079 / 282,029 |

정확한 입력과 감사 파일:

- train SHA: `5def1584d2e9b62cbedb3428cc49b1e7eeed674c48ec7e514f40ec54b6a63e07`
- provenance SHA: `e26d81fc3ca5a957c36353c522d280606de0195986c2ea784b8101df45646ea5`
- local audit: `reports/sionic-squad-train-60k-training-data-audit.json`
- blocklist audit: `reports/sionic-squad-train-60k-benchmark-overlap-audit.json`
- public dataset revision:
  `LLM-OS-Models/korean-embedding-performance-v1-sionic-squad-train-60k@8fbc6d6d5c93c3493456079d930921ac90ec6801`

## 학습·선택 실험

사전 등록한 첫 비교는 다음과 같다.

| Candidate | 초기 checkpoint | target/replay | 목적 |
|---|---|---:|---|
| General winner | 1M 또는 그 전 단계 실제 winner | 0/100 | 회귀 기준 |
| SQuAD adapter A | General winner | 50/50, 총 120K | 빠른 supervised gain 확인 |
| SQuAD adapter B | General winner | 25/75, 총 120K | 일반 성능 보존 우선 |

두 adaptation을 동시에 전부 오래 돌리지 않는다. A를 먼저 실행해 SQuADKorV1뿐 아니라
Sionic 9 macro와 clean score가 개선되는지 본다. macro gain이 없거나 다른 task 회귀가
크면 B로 낮춘다. 기본 optimizer는 LoRA r64, BF16, InfoNCE, hard negatives 7,
effective batch 64, learning rate `5e-6`이며 512-token cap에서 시작한다. 8B full FT는
동일 token budget의 LoRA/last-4 probe가 포화된 뒤에만 승격한다.

## 승격 조건

- Sionic 9개 전체 동일 protocol로 측정하고 평균을 공개한다.
- SQuADKorV1의 train-family exposure를 row와 model card에 표시한다.
- 공식 MTEB Korean v1 6개를 exact Qwen query-instruction contract로 측정한다.
- clean 법률/source-heldout 및 noise robustness가 general winner보다 크게 회귀하지 않는다.
- validation InfoNCE loss나 SQuADKorV1 한 task만으로 winner를 고르지 않는다.
- evaluation query/qrel/corpus를 hard-negative mining corpus로 사용하지 않는다.

## 재현

```bash
.venv-train/bin/python scripts/build_performance_mix.py \
  --phase sionic_squad_train_60k \
  --output-dir outputs/data/performance-v1/sionic-squad-train-60k

.venv-train/bin/python scripts/audit_embedding_training_data.py \
  --train outputs/data/performance-v1/sionic-squad-train-60k/train.jsonl \
  --provenance outputs/data/performance-v1/sionic-squad-train-60k/provenance.jsonl \
  --output reports/sionic-squad-train-60k-training-data-audit.json \
  --expected-batch-size 0
```

dataset build 완료는 모델 성능 결과가 아니다. 실제 score는 장기 evaluation queue가 끝난
후 README의 세 보드와 모델 카드에만 채운다.
