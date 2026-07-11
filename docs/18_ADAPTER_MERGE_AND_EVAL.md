# Adapter merge and evaluation

## 목적

ms-swift가 만든 LoRA 체크포인트는 그 자체로 완전한 SentenceTransformers 모델이 아니다. PEFT delta를 base weight에 합치는 것만으로도 충분하지 않다. Qwen3-Embedding-8B의 아래 추론 계약이 함께 보존되어야 같은 모델을 측정하게 된다.

| 항목 | 고정값 |
|---|---|
| backbone artifact | `Qwen3ForCausalLM` |
| embedding | 마지막 non-padding token의 4,096차원 hidden state |
| padding | left |
| 후처리 | 행별 L2 Normalize |
| similarity | cosine, 즉 정규화 후 inner product |
| query prompt | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:` |
| document prompt | 빈 문자열 |
| default prompt | 없음; MTEB가 query/document 역할을 지정 |

`scripts/merge_embedding_adapter.py`는 다음을 하나의 실패 원자적 작업으로 수행한다.

1. `adapter_config.json`과 adapter weight를 검사하고 SHA-256을 기록한다.
2. 고정 base revision에 LoRA를 활성화한다.
3. 고정된 한국어·영어 query/document 6개를 last-token + L2 Normalize로 임베딩한다.
4. PEFT `safe_merge=True`로 delta를 합친 뒤 동일 입력을 다시 임베딩한다.
5. 대응 행 cosine과 element/pairwise score 오차가 임계값 안인지 검사한다.
6. merged weights와 tokenizer를 저장한다.
7. `modules.json`, `1_Pooling/config.json`, `2_Normalize`, `config_sentence_transformers.json`을 복원한다.
8. prompt, pooler, normalization, architecture, token, left-padding 계약을 다시 읽어 strict equality로 검증한다.
9. 모든 검사를 통과한 임시 디렉터리만 최종 경로로 atomic rename한다.

따라서 실패한 병합물이 정상 모델 경로에 남지 않는다. 기존 출력 경로를 덮어쓰지도 않는다.

## 병합 명령

현재 GPU가 평가 또는 학습 중이면 CPU 병합을 사용한다. 8B BF16 weight를 저장할 충분한 RAM과 디스크가 필요하다.

```bash
ADAPTER=outputs/<run>/<version>/checkpoint-<step>
MERGED=artifacts/models/qwen3-embedding-8b-ko-<run>-merged

.venv-train/bin/python scripts/merge_embedding_adapter.py \
  --adapter "$ADAPTER" \
  --output-dir "$MERGED" \
  --base-model Qwen/Qwen3-Embedding-8B \
  --base-revision 1d8ad4ca9b3dd8059ad90a75d4983776a23d44af \
  --device cpu \
  --dtype bfloat16 \
  --local-files-only
```

GPU가 비었을 때는 `--device cuda`로 바꿀 수 있다. H100 한 장에서는 병합 전후 모델을 동시에 두지 않고 같은 객체를 순차 변환하므로 메모리 낭비를 줄인다. `--device auto`는 Accelerate dispatch가 꼭 필요한 환경을 위한 선택지다. 재현 가능한 공개 artifact에는 기본 `bfloat16`을 유지한다.

기본 acceptance threshold는 대응 embedding cosine `>= 0.999`, 최대 원소 절대 오차 `<= 0.05`, probe 전체 pairwise cosine score 최대 변화량 `<= 0.01`이다. BF16에서 LoRA branch를 따로 계산하는 경우와 delta를 weight에 먼저 반올림해 합치는 경우의 연산 순서가 달라 bitwise equality는 요구하지 않는다. `merge_report.json`에 실제 최소/평균 cosine과 두 종류의 최대 오차를 모두 남긴다.

`--local-files-only`는 base snapshot이 이미 cache에 있을 때만 사용한다. 새 머신에서는 이 옵션을 빼야 한다. 출력 경로가 이미 있으면 명시적으로 실패하므로, 재시도할 때 새 경로를 사용하거나 실패 원인을 확인한 뒤 기존 경로를 직접 정리한다.

## 빠른 CPU 테스트

아래 테스트는 GPU와 실제 weight를 사용하지 않는다. 잘못된 mean pooling, prompt/padding drift, parity 계산과 shape 오류를 검출한다.

```bash
.venv-train/bin/python -m unittest -v tests/test_merge_embedding_adapter.py
```

한 단계 더 강한 end-to-end smoke는 임시 디렉터리에 hidden size 16, 1-layer Qwen3와 실제 non-zero LoRA를 만들고 CLI 병합·수치 parity·atomic 저장까지 실행한다. 역시 GPU와 8B weight가 필요 없다.

```bash
.venv-train/bin/python tests/smoke_merge_tiny_qwen.py
```

병합 성공 후 실제 SentenceTransformers 로딩 smoke는 benchmark와 같은 환경에서 한다.

```bash
MERGED=artifacts/models/qwen3-embedding-8b-ko-<run>-merged

.venv-mteb/bin/python - <<'PY'
import numpy as np
from sentence_transformers import SentenceTransformer

path = "artifacts/models/qwen3-embedding-8b-ko-<run>-merged"
model = SentenceTransformer(
    path,
    device="cuda",
    model_kwargs={"attn_implementation": "flash_attention_2"},
    tokenizer_kwargs={"padding_side": "left"},
)
model.max_seq_length = 8192
q = model.encode(
    ["대한민국의 수도는 어디인가?"],
    prompt_name="query",
    normalize_embeddings=True,
)
d = model.encode(
    ["대한민국의 수도는 서울특별시이다."],
    prompt_name="document",
    normalize_embeddings=True,
)
assert q.shape == d.shape == (1, 4096)
assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=5e-3)
assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=5e-3)
print({"shape": q.shape, "cosine": float((q @ d.T)[0, 0])})
PY
```

## Sionic 9종 / AutoRAG

평가 스크립트는 local path를 그대로 받는다. Sionic 9종은 고정 prompt, full corpus, 각 task `NDCG@10`, 9개 단순 평균 프로토콜이다.

먼저 가장 빠른 gate인 AutoRAG만 실행한다.

```bash
MERGED=artifacts/models/qwen3-embedding-8b-ko-<run>-merged

PYTHONPATH=third_party/mteb .venv-mteb/bin/python scripts/evaluate_sionic9.py \
  --model "$MERGED" \
  --task AutoRAG \
  --batch-size 192 \
  --max-length 8192 \
  --attn-implementation flash_attention_2
```

AutoRAG가 base와 Comsat 대비 유망하면 같은 출력 cache를 이어서 전체 9종을 실행한다.

```bash
PYTHONPATH=third_party/mteb .venv-mteb/bin/python scripts/evaluate_sionic9.py \
  --model "$MERGED" \
  --batch-size 192 \
  --max-length 8192 \
  --attn-implementation flash_attention_2
```

`--overwrite`를 넣지 않으면 완료된 task cache를 재사용한다. 경로가 다른 병합 모델은 별도 cache가 되므로 모델별 결과가 섞이지 않는다.

## 공식 MTEB Korean v1

local directory에는 Hub commit이 없으므로 immutable run label을 `--revision`에 명시한다.
evaluator는 `merge_report.json`의 실제 model weight SHA-256 앞 12자리로 이 값을
canonicalize한다. Qwen3 파생 모델은 generic `query` prompt가 아니라 pinned MTEB의
Qwen3 task별 instruction을 적용해야 한다.

```bash
MERGED=artifacts/models/qwen3-embedding-8b-ko-<run>-merged
LOCAL_REVISION=model-<model-weights-sha256-first-12>

PYTHONPATH=third_party/mteb .venv-mteb/bin/python scripts/evaluate_mteb_korean_v1.py \
  --model "$MERGED" \
  --revision "$LOCAL_REVISION" \
  --qwen3-instruction-loader \
  --max-length 8192 \
  --batch-size 192 \
  --attn-implementation flash_attention_2
```

공식 Korean v1은 6개 task와 task-type 평균을 함께 계산한다. Sionic 9종 평균과 공식 `Mean(Task)`/`Mean(TaskType)`를 한 숫자로 합치지 않는다.
Sionic 9종은 모델 간 동일한 fixed web-search prompt를 쓰는 반면, 공식 Korean v1은
MTEB가 허용하는 Qwen3의 task metadata/fallback instruction과 passage 무지시문 계약을
쓴다. 두 결과는 각자의 protocol 안에서만 비교한다.

## 해석과 승격 조건

병합 parity 통과는 LoRA 성능이 좋다는 뜻이 아니다. 이것은 adapter-on-base와 merged artifact가 사실상 같은 encoder라는 배포 검증이다. 모델 승격은 별도로 다음 순서를 따른다.

1. merge parity와 SentenceTransformers load smoke 통과
2. private mined dev에서 base 대비 개선
3. AutoRAG가 Qwen baseline을 이기고 Comsat에 접근 또는 우위
4. Sionic 9종 전체 평균 및 각 task 회귀 확인
5. 공식 Korean v1 6종과 Borda 비교
6. 종합 suite 확인 후 immutable merged artifact 업로드

vLLM Korean evaluator는 현재 Comsat/Qwen baseline revision만 받도록 의도적으로 고정되어 있다. local merged path를 허용했다는 검증 없이 그 스크립트에 억지로 넣으면 model-card contract 검사가 무력화된다. 먼저 위 SentenceTransformers 경로로 정답 parity를 확정하고, local-vLLM backend를 추가할 때 동일 6문장 및 benchmark subset에서 결과 parity를 별도 gate로 두어야 한다.
