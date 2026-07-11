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
