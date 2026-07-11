---
language:
- ko
license: other
task_categories:
- sentence-similarity
- feature-extraction
pretty_name: Korean Legal Retrieval Source-Native 250K
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# Korean Legal Retrieval Source-Native 250K

Legalize-KR의 법령·행정규칙·판례·자치법규 구조에서 query/positive 관계를 추출한
250,000-row 한국어 retrieval dataset이다. `release_eligible: false`인
target-adapted 연구·비상업 성능 shard이며 통합 라이선스는 `other`다.

## 구성과 고정 revision

| Source | Revision | Rows | 구조 관계 |
|---|---|---:|---|
| `legalize-kr/legalize-kr` | `db3cd760c14042ee04fd9166e1bdbb662fc999bc` | 50,000 | 법령명+조문 → 조문 본문 |
| `legalize-kr/admrule-kr` | `64a5a272909ab5bc077b0ad9519ef31de8febb46` | 50,000 | 규칙명+조문 → 조문 본문 |
| `legalize-kr/precedent-kr` | `40cd00e54df19d98562abb170c8ff51fd6fe2c2e` | 50,000 | 판시사항 → 판결요지 |
| `legalize-kr/ordinance-kr` | `6443e5dd5833d863219064cd362111f516430bec` | 100,000 | 자치법규명+조문 → 조문 본문 |

query와 positive는 LLM이 relevance를 추측한 결과가 아니라 source 문서가 명시한
제목/조문 및 판시사항/판결요지 구조다. 단, 구조 관계가 모든 실제 사용자 query를
대표한다는 뜻은 아니다.

## 중요한 경고: bootstrap negative

`negative_messages`의 한 문서는 seed 42로 같은 source에서 결정론적으로 고른
**bootstrap negative**다. 최종 hard negative가 아니다. 학습 전 다음 단계를 권장한다.

1. base Qwen3-Embedding-8B와 current student로 positive corpus top-24를 mining한다.
2. Qwen3-Reranker-8B 같은 동일 teacher로 positive/negative를 함께 점수화한다.
3. `s_neg < 0.95 × s_pos`의 positive-relative filter로 false negative를 제거한다.
4. 같은 법/기관/지역/쟁점의 어려운 후보를 포함하되 실제 secondary positive는
   negative로 쓰지 않는다.
5. query당 4–7개를 저장하고 source-homogeneous batch로 학습한다.

## Benchmark 노출

법률·공공 원문은 LawIRKo와 AutoRAG legal/public slice의 corpus와 같거나 매우 유사할
수 있다. 이 데이터를 쓴 LawIRKo/AutoRAG 결과는 `target-adapted`로 표시해야 하며
clean zero-shot이라고 주장하면 안 된다. Sionic 9 또는 MIRACL 등 평가 query/qrel을
직접 사용하지 않았지만 corpus/domain overlap 가능성은 남는다. clean release에서는
pinned evaluation corpus의 exact normalized hash와 MinHash near-duplicate를 차단한다.

## 스키마

```json
{
  "messages": [{"role": "user", "content": "Instruct: ...\\nQuery: 법령명 제1조"}],
  "positive_messages": [[{"role": "user", "content": "# 법령명 ..."}]],
  "negative_messages": [[{"role": "user", "content": "다른 조문 ..."}]]
}
```

```python
from datasets import load_dataset

ds = load_dataset(
    "LLM-OS-Models/korean-legal-retrieval-source-native-250k",
    split="train",
)
```

`metadata/provenance.jsonl`은 source candidate ID, repository, revision, path, source
document SHA-256, section heading, bootstrap negative ID를 같은 row index로 보존한다.

## 무결성과 재현

- rows: `250,000`
- seed: `42`
- `data/train.jsonl` SHA-256:
  `1d81364bed3b4dab83a6979ef0874dd39bddb108830d35a43be7fd417d134c90`
- `metadata/provenance.jsonl` SHA-256:
  `a1b3cda735df2e112832ebfbd8e07f3ec7d889ba875f17ff2f51cb9133a9de3e`
- extractor config: `configs/legal_data_sources_v1.json`
- extractor: `scripts/prepare_legal_embedding_data.py`
- compiler: `scripts/compile_source_native_pairs.py`
- build command: `scripts/build_legal_performance_shards.sh`

## 사용 조건과 제한

Legalize-KR repositories는 정부 공공저작물 원문과 repository 구조에 대한 설명을
제공하지만, 이 카드는 독립적인 법률 판단이나 upstream 재허가가 아니다. 각 source의
README와 원천 조건을 확인해야 한다. 개인정보·유해 콘텐츠 전수 검수와 benchmark
near-duplicate 제거가 아직 완료되지 않았다. 이 버전은 성능 연구용이며 상업 사용을
보증하지 않는다.
