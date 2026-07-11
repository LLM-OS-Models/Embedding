# vLLM offline baseline

`scripts/evaluate_mteb_korean_v1_vllm.py`는 공식 Korean MTEB v1을 한 장의 H100에서
평가하기 위한 별도 backend다. SentenceTransformers 결과와 섞이지 않도록 기본 cache는
`outputs/evaluation/mteb_korean_v1_vllm/` 아래에 저장한다.

Qwen과 Comsat의 revision, 8,192-token truncation, query/document prompt를 고정한다. 실행 전
각 모델의 `modules.json`, pooling config, SentenceTransformers config를 확인하여 last-token
pooling, Normalize 모듈(L2), cosine, query-only instruction이 그대로인지 검증한다. vLLM의
`pooler_config`는 지정하지 않는다. 모델의 SentenceTransformers 설정이 우선하도록 두는 것이
기존 backend와 맞는다.

별도 환경을 사용한다. vLLM 0.24.0은 PyTorch 2.11.0을 요구하므로 기존 학습/평가 환경에
설치하면 안 된다.

```bash
/home/ubuntu/.venvs/uv/bin/uv venv --python /usr/bin/python3 --seed .venv-vllm
/home/ubuntu/.venvs/uv/bin/uv pip install --python .venv-vllm/bin/python \
  --torch-backend=auto 'vllm==0.24.0' 'mteb==2.18.0'
```

GPU를 초기화하지 않는 검증과 MIRACL 실행 예시는 다음과 같다.

```bash
.venv-mteb/bin/python scripts/evaluate_mteb_korean_v1_vllm.py \
  --model sionic-ai/comsat-embed-ko-8b-preview --list-only

.venv-vllm/bin/python scripts/evaluate_mteb_korean_v1_vllm.py \
  --model sionic-ai/comsat-embed-ko-8b-preview \
  --task MIRACLRetrieval
```

기본 engine 시작점은 BF16, `max_num_batched_tokens=65536`, `max_num_seqs=512`, VRAM
상한 90%다. 전체 MIRACL 전에 동일 문장에 대한 SentenceTransformers/vLLM 임베딩 cosine과
AutoRAG NDCG@10을 비교하여 수치 parity를 확인한다. adapter-only PEFT checkpoint는 이
경로에 직접 넣지 않고 먼저 merge한다.

## 2026-07-11 H100 실측

Comsat `Ko-StrategyQA`에서 65K-token 설정은 정상 완료했지만 기존 evaluator보다
빠르지 않았다.

| Backend | NDCG@10 | Task elapsed | 관측 corpus 처리량 |
|---|---:|---:|---:|
| SentenceTransformers + FA2 | `0.84016` | 기존 pinned run | 약 260–350 docs/s |
| vLLM 0.24, 65K tokens / 512 seqs / VRAM 90% | `0.83830` | `79.07s` | 약 200 docs/s, 44K input tok/s |

backend score 차이는 `-0.00186`이다. 빠른 screening에는 쓸 수 있지만 공식 표의
SentenceTransformers 결과와 같은 셀에 섞지 않는다. `131072` batched tokens,
`1024` seqs, VRAM 95%는 75.85GiB 사용 상태에서 추가 6GiB activation을 요청하며
OOM이 났다. 따라서 현재 H100/Comsat/한국어 문서 길이에서는 65K vLLM보다
FA2 large-batch가 더 빠르고 정확한 기본 경로다. vLLM은 다른 모델·짧은 입력에서
다시 throughput gate를 통과할 때만 전체 corpus에 사용한다.
