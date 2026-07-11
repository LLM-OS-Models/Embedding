---
language:
- ko
license: other
task_categories:
- sentence-similarity
- feature-extraction
pretty_name: Korean Embedding Sionic SQuAD Train 60K
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# Korean Embedding — Sionic SQuAD train-family 60K

KorQuAD v1.0의 **원본 train split만** 질문→정답 문맥 retrieval 형식으로 변환한
60,000-row target-adaptation 데이터다. Sionic retrieval 9종 중
`SQuADKorV1`의 train-family 신호를 명시적으로 보강한다.

## 사용 조건과 점수 공개 방식

`release_eligible: false`인 performance/non-commercial 실험용 composite다. 이
저장소의 통합 라이선스는 `other`이며 upstream 권리를 재허가하지 않는다. Hub metadata는
KorQuAD source를 `CC-BY-ND-4.0`으로 표시하고, upstream dataset card 본문은
`CC BY-ND 2.0 KR`도 명시한다. 사용자는 원 source 조건을 직접 확인해야 한다.

이 데이터로 학습한 모델은 `SQuADKorV1 train-family exposed`로 공개해야 하며 그
task에서 zero-shot이라고 주장하면 안 된다. MTEB 변환 평가 repository
`yjoonjang/squad_kor_v1`의 test row와 원본 KorQuAD validation split은 읽거나
사용하지 않았다. Sionic 9의 MIRACL 등 다른 task 평가 데이터도 포함하지 않는다.

## 고정 source와 변환

| 항목 | 값 |
|---|---|
| Source | `KorQuAD/squad_kor_v1` |
| Revision | `01aad23853355e5f4f6317eeaaa8186811424834` |
| Config/split | `squad_kor_v1/train` |
| 원본 rows | 60,407 |
| 출력 rows | 60,000 |
| 고유 원본 문맥 | 9,606 |
| 제거한 중복 question/context pair | 153 |
| Query | `question` |
| Positive | answer를 포함하는 `context` |
| Bootstrap negatives | 다른 문맥 7개, deterministic |

bootstrap negative는 데이터 형식·초기 학습을 위한 후보일 뿐 최종 hard negative라고
주장하지 않는다. 최종 후보 학습 전에는 현재 student로 corpus를 재임베딩하고
positive-relative false-negative filter와 score-rank quantile sampling을 적용한다.

## 스키마와 사용법

```json
{
  "messages": [{"role": "user", "content": "Instruct: ...\nQuery: 질문"}],
  "positive_messages": [[{"role": "user", "content": "정답 문맥"}]],
  "negative_messages": [
    [{"role": "user", "content": "다른 문맥 1"}],
    [{"role": "user", "content": "다른 문맥 2"}]
  ]
}
```

```python
from datasets import load_dataset

ds = load_dataset(
    "LLM-OS-Models/korean-embedding-performance-v1-sionic-squad-train-60k",
    split="train",
)
print(ds[0]["messages"][0]["content"])
```

## 품질 감사

- 60,000/60,000 row hash 일치, provenance index mismatch 0
- 모든 row가 negative 7개 보유
- query body 4자 미만 0
- query style heuristic: natural question 59,254 (98.76%)
- query length p50/p95: 32/57 characters
- positive length p50/p95: 462/862 characters
- positive 중복 50,394는 같은 문맥에 평균 약 6.3개의 서로 다른 질문이 달린 원본
  KorQuAD 구조이며 오류가 아니다.

`metadata/training_data_quality_audit.json`에 전체 통계와 입력 SHA를 저장한다.

## 무결성과 재현

- seed: `42`
- `data/train.jsonl` SHA-256:
  `5def1584d2e9b62cbedb3428cc49b1e7eeed674c48ec7e514f40ec54b6a63e07`
- `metadata/provenance.jsonl` SHA-256:
  `e26d81fc3ca5a957c36353c522d280606de0195986c2ea784b8101df45646ea5`

```bash
.venv-train/bin/python scripts/build_performance_mix.py \
  --phase sionic_squad_train_60k \
  --output-dir outputs/data/performance-v1/sionic-squad-train-60k
```

데이터 build나 validation loss만으로 성능 향상을 주장하지 않는다. Sionic 9 전체,
공식 MTEB Korean v1, clean 종합 회귀 평가를 완료한 모델만 승격한다.
