# 상위 임베딩 모델 로컬 평가 매트릭스

기준일: 2026-07-11. 실행 metadata의 단일 source는
[`configs/models_to_evaluate.json`](../configs/models_to_evaluate.json), 순차 실행기는
[`scripts/run_top_model_sionic_queue.sh`](../scripts/run_top_model_sionic_queue.sh)다.

## 원칙

세 종류의 숫자를 분리한다.

1. 모델 카드에 적힌 Sionic 9종 값은 reference-only다.
2. MTEB live backend 값은 official registered prompt/revision의 reference-only다.
3. `outputs/evaluation`에 raw result JSON이 생긴 값만 local measurement다.

Sionic 9종은 모든 모델에 같은 Qwen/Comsat web-search query prompt, document prompt
없음, max 8,192 이하, NDCG@10 macro average를 적용한다. 공식 Korean v1은 모델별
registered instruction과 task type을 쓰므로 이 표와 평균을 섞지 않는다.

## 고정 모델

| 순서 | 모델 | Sionic revision | Pooling/attention | Max | 시작 batch | remote code |
|---:|---|---|---|---:|---:|---|
| 1 | `sionic-ai/comsat-embed-ko-8b-preview` | `a5cc22b651c1b2e51cdd8bf671774ae93584f0ab` | last/causal | 8192 | 192 | no |
| 2 | `Qwen/Qwen3-Embedding-8B` | `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` | last/causal | 8192 | 192 | no |
| 3 | `codefuse-ai/F2LLM-v2-8B` | `e5725783762d69b4f8ba7e09a8872ce19a7a5ec3` | last/causal | 8192 | 192 | no |
| 4 | `SamilPwC-AXNode-GenAI/PwC-Embedding_expr` | `6c5196980c685db45b58f67bd3be2f79d794351e` | mean/bidirectional | 512 | 512 | no |
| 5 | `microsoft/harrier-oss-v1-27b` | `0c0fc62f6d8af9e8604cb818c412301b103a0093` | last/causal hybrid | 8192 | 16 | no |
| 6 | `tencent/KaLM-Embedding-Gemma3-12B-2511` | `98c19ba34197906fbc93f6f1ef79402ca3a33956` | last/causal hybrid | 8192 | 48 | yes |
| 7 | `nvidia/llama-embed-nemotron-8b` | `aa3b43a495a9b280d1bdb716da37c54bb495d630` | mean/bidirectional | 8192 | 64 | yes |

2026-07-17 익명 복구 뒤 repository-local cache를 hard offline으로 고정해 다섯 추가 모델의
exact revision `AutoConfig`와 tokenizer를 다시 로드했다. 모두 통과했고 고유 Hub blob 크기는
F2 14.11GiB, PwC 2.10GiB, Harrier 50.34GiB, KaLM 21.95GiB, Nemotron 14.00GiB다. 따라서
후속 GPU queue는 model snapshot이나 remote-code/tokenizer 누락 때문에 네트워크로 fallback하지
않는다.

각 model은 시작 batch에서 1까지 절반씩 낮추며 OOM fallback한다. 완료 task는 MTEB
ResultCache와 exact float32 embedding cache로 보존하므로 낮은 batch 재시도는 이미 끝난
encode를 반복하지 않는다. 모델 하나가 실패해도 다음
모델은 실행하지만, 하나라도 완결되지 않으면 전체 queue는 nonzero로 끝나 불완전한
비교를 성공으로 오인하지 않는다. 공개 모델과 공개 평가 데이터만 사용하므로 `.env`를
읽지 않고, 저장된 Hub credential의 implicit 전송도 차단한 익명 read-only 경로다.

## 왜 Qwen/F2의 revision이 둘인가

Hub current revision과 MTEB registry result revision이 다를 수 있다. Qwen과 F2의
감사에서는 config, SentenceTransformers contract, pooling, modules, weight-index hash가
동일했지만, local Sionic run은 current model snapshot을 고정하고 official Korean
reference는 registry revision을 그대로 기록한다. Nemotron은 두 snapshot의 weight
index는 같아도 current에 SentenceTransformers files와 bidirectional flag가 추가돼
있어 임의 동등성을 주장하지 않는다.

## vLLM admission gate

vLLM은 model card가 지원한다고 적었다는 이유만으로 local official evaluator에 넣지
않는다. 같은 revision/prompt에서 Ko-StrategyQA score 절대 차이 `<= 0.0001`, FA2 대비
속도 `>= 1.1x`를 모두 통과해야 한다. 현재 Comsat 실측은 FA2 `0.84016`, vLLM
`0.83830`, vLLM throughput도 더 느려 탈락했다. 따라서 offline MTEB는 FA2를 쓰고,
vLLM은 동시 요청 serving 경로로 유지한다.

## 실행

우리 모델의 post-training 평가를 우선한다.

```bash
WAIT_PID=<post-training-queue-pid> \
  bash scripts/run_top_model_sionic_queue.sh
```

단일 모델/단일 task 검증은 다음처럼 한다.

```bash
PYTHONPATH=third_party/mteb .venv-mteb/bin/python scripts/evaluate_sionic9.py \
  --model Qwen/Qwen3-Embedding-8B \
  --revision 1d8ad4ca9b3dd8059ad90a75d4983776a23d44af \
  --task AutoRAG \
  --batch-size 192 \
  --max-length 8192 \
  --attn-implementation flash_attention_2
```

## 해석

Harrier/KaLM/Nemotron이 multilingual leaderboard에서 높다고 Sionic 한국어 9종에서
자동으로 Comsat을 이기는 것은 아니다. model size, 학습 task exposure, prompt contract,
max length가 다르다. 이 queue의 목적은 같은 retrieval prompt에서 실제 local raw
result를 확보하는 것이지 official MTEB 값을 대체하는 것이 아니다.
