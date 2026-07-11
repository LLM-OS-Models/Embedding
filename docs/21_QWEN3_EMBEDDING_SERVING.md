# Qwen3 임베딩 서빙

조사·고정일: 2026-07-11. 대상은 `Qwen/Qwen3-Embedding-8B` 및 이 프로젝트에서
병합해 공개할 동일 architecture 모델이다.

## 결론

온라인 다중 사용자 API는 **vLLM pooling server 또는 Hugging Face TEI**, 단일
offline 대량 평가·index build는 이 H100에서 실제로 더 빨랐던
**SentenceTransformers + FlashAttention 2**를 기본으로 한다. vLLM은 생성 모델처럼
decode/PagedAttention 이득을 받지 않는 pooling workload에서는 항상 더 빠른 것이
아니다. vLLM 공식 문서도 pooling 지원을 편의 기능으로 설명하며 Transformers나
SentenceTransformers보다 성능이 개선된다고 보장하지 않는다.

| 상황 | 권장 경로 | 이유 |
|---|---|---|
| REST/OpenAI 호환, 동시 요청이 많음 | vLLM `--runner pooling` | continuous batching, `/v1/embeddings`, 운영 API |
| HF 중심 embedding 전용 service | TEI | Qwen3/H100 지원, 자동 batching, 작은 운영 surface |
| MTEB·고정 corpus 일괄 처리 | SentenceTransformers + FA2 | 현재 장비의 측정 throughput이 가장 높음 |
| 라이브 검색 | embed service + ANN DB + optional reranker | corpus vector는 미리 계산하고 query만 실시간 처리 |

## Qwen 입력 계약

Qwen 공식 예제는 query에 한 문장 task instruction을 붙이고 document에는 붙이지
않는다. 모델은 last-token pooling, L2 normalization을 사용하므로 normalized vector의
dot product가 cosine similarity와 같다.

```text
Instruct: Given a Korean web search query, retrieve relevant passages that answer the query
Query: 대한민국의 수도는 어디인가?
```

instruction 유무와 문구는 점수에 영향을 준다. 학습·평가·서빙에서 같은 query
template을 사용하고 document를 query template으로 감싸지 않는다.

## vLLM OpenAI-compatible server

최신 vLLM에서는 `task="embed"`의 명시적 후속 표현으로 `--runner pooling`을 쓴다.
Qwen 공식 repository의 offline 예제는 `LLM(..., task="embed")`와 `LLM.embed()`을
제공하고, vLLM 공식 serving API는 `/v1/embeddings`를 제공한다.

```bash
MODEL_ID=Qwen/Qwen3-Embedding-8B \
SERVED_MODEL_NAME=qwen3-embedding-8b \
MAX_MODEL_LEN=8192 \
MAX_NUM_BATCHED_TOKENS=65536 \
MAX_NUM_SEQS=512 \
DTYPE=bfloat16 \
scripts/serve_vllm_embedding.sh
```

공개 model card가 FP32 strict-parity merge/evaluation을 명시하면 재현 실행도
`DTYPE=float32`로 맞춘다. BF16으로 내리는 것은 운영상 가능한 별도 최적화지만,
FP32와 같은 leaderboard 점수라고 간주하지 않고 task별 회귀를 다시 측정한다.

8B에서 무조건 32K context를 열면 scheduler/KV·activation 여유가 줄어 짧은 요청의
처리량이 악화될 수 있다. 실제 traffic의 p99 길이에 맞춰 2K/4K/8K/16K/32K를
측정한다. `--gpu-memory-utilization`은 vLLM의 선점 비율이고 CUDA graph memory는
별도일 수 있으므로 OOM이면 이 값과 `max-num-batched-tokens`, `max-model-len` 순으로
낮춘다.

```bash
python scripts/embedding_api_smoke.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3-embedding-8b
```

일반 OpenAI client도 그대로 쓸 수 있다.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
query = (
    "Instruct: Given a Korean web search query, retrieve relevant passages "
    "that answer the query\nQuery: 대한민국의 수도는 어디인가?"
)
response = client.embeddings.create(
    model="qwen3-embedding-8b",
    input=[query, "대한민국의 수도는 서울특별시이다."],
    encoding_format="float",
)
vectors = [item.embedding for item in response.data]
score = sum(a * b for a, b in zip(vectors[0], vectors[1]))
```

MRL로 차원을 줄일 때는 모델이 해당 dimension으로 학습됐는지 확인하고 offline
parity를 먼저 측정한다. 저장된 corpus vector와 query vector의 dimension·prompt·model
revision이 하나라도 다르면 index를 섞지 않는다.

## TEI

TEI 1.9 문서는 Qwen3-Embedding-8B와 H100을 지원 모델·하드웨어로 명시한다.

```bash
model=Qwen/Qwen3-Embedding-8B
volume=$PWD/tei-data
docker run --gpus all -p 8080:80 -v "$volume:/data" \
  ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 \
  --model-id "$model"
```

```bash
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"text-embeddings-inference","input":["문장 1","문장 2"]}'
```

TEI도 여러 input을 한 request로 받을 수 있고 자동 batching을 제공한다. 최종 선택은
동일 model revision, prompt, length distribution, concurrency에서 vLLM과 TEI를
부하 테스트해 p50/p95 latency, input tokens/s, requests/s, VRAM으로 결정한다.

## 실제 측정과 주의

이 저장소의 H100 80GB 단일 평가에서는 Comsat/Qwen 계열 고정 대형 corpus에서:

- SentenceTransformers + FA2: 대략 260–390 documents/s
- vLLM offline pooling, 65K batched tokens: 대략 200 documents/s
- vLLM 131K tokens/1024 sequences/95% memory: 약 75.85GB를 선점한 뒤 추가
  activation 6GB OOM

이는 vLLM API가 나쁘다는 뜻이 아니라 **이 offline homogeneous benchmark**에
continuous scheduler 이득이 없었다는 뜻이다. serving 선택과 MTEB batch encoder 선택을
같은 결정으로 취급하지 않는다.

## production 구조

1. document를 같은 revision/prompt/max-length/dimension으로 미리 embed한다.
2. FAISS, Qdrant, Milvus, pgvector 같은 ANN index에 normalized vector와 document ID를
   저장한다.
3. query에 task instruction을 붙여 embedding service로 보낸다.
4. dot product top-K를 검색한다.
5. 품질이 중요하면 Qwen3-Reranker로 top-20~100만 재정렬한다.
6. model revision이 바뀌면 shadow index를 새로 만든 뒤 atomic alias switch한다.

운영 관측에는 queue time, tokenize time, input tokens/s, batch token utilization,
length histogram, truncation count, CUDA OOM/restart, vector norm, known-pair score canary를
포함한다. text나 token을 무제한 로그로 남기지 않는다.

## 근거

- [Qwen3-Embedding 공식 repository와 vLLM/SentenceTransformers 예제](https://github.com/QwenLM/Qwen3-Embedding)
- [vLLM pooling/embedding model 공식 문서](https://docs.vllm.ai/en/latest/models/pooling_models/embed/)
- [vLLM OpenAI embedding client 예제](https://docs.vllm.ai/en/stable/examples/pooling/embed/)
- [vLLM pooling 성능 비보장 주의](https://docs.vllm.ai/en/v0.17.0/models/pooling_models/)
- [Hugging Face TEI Qwen3 quick tour](https://huggingface.co/docs/text-embeddings-inference/quick_tour)
- [TEI 지원 모델·하드웨어](https://huggingface.co/docs/text-embeddings-inference/supported_models)
