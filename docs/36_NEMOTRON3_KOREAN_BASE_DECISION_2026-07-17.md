# Nemotron-3 한국어 base 결정과 중단 복구

기준: **2026-07-17 (Asia/Seoul)**

## 목표

최단 기간에 Sionic Korean retrieval 9종 macro NDCG@10 `0.7930`을 넘는 단일 모델을
만든다. `nvidia/Nemotron-3-Embed-8B-BF16`이 이미 목표를 넘으면 불필요한 장기 학습을
먼저 하지 않고, clean holdout 회귀를 확인한 뒤 그대로 기준 모델 또는 추가학습 base로
사용한다.

## 고정 모델 계약

| 항목 | 값 |
|---|---|
| model | `nvidia/Nemotron-3-Embed-8B-BF16` |
| revision | `2b29550c4ab0646bb6bb47032dda54ea11f6dfe2` |
| license | `OpenMDW-1.1` |
| backbone | bidirectional `Ministral3Model`, 약 8B |
| pooling / dimension | masked mean / 4,096 |
| native prompt | query `query: `, document `passage: ` |
| context | model card 32K; Sionic 동등 평가는 8,192 |
| local runtime | H100 80GB, BF16, FlashAttention-2, batch 64 |

공식 카드가 밝힌 34개 평가 언어에는 한국어가 포함된다. 다만 global RTEB 1위만으로
한국어 Sionic 9 우위를 주장하지 않고 아래 pinned local run으로 직접 확인한다.

## 현재 실측과 진행 상태

- 한국어 문장 smoke: 정답 문서 cosine `0.7070`, 명백한 오답 `0.3828`.
- SQuADKorV1 고정 Sionic protocol: **0.92032**.
- 공개 reference: Qwen3-Embedding-8B `0.9063`, Comsat `0.9168`.
- Sionic 9 전체: 실행 중. MIRACL 1,486,752문서를 50K chunk별 float32 cache로 보존한다.
- Qwen 200K 학습은 Nemotron-3 판단이 끝날 때까지 재개하지 않는다.

SQuADKorV1 한 task의 우위는 base 교체 근거로 충분하지 않다. 최종 판단은 Sionic 9
전체와 benchmark 선택에 쓰지 않는 legal v2 10K 및 finance/knowledge 1.9K selector를
함께 본다. Nemotron-3 카드가 MIRACL과 MLDR를 training source로 명시하므로 해당 두
public score는 `upstream train-family exposure`로 공개하고 clean selector보다 낮은
증거 등급으로 취급한다.

## 중단 후 재개

모델과 평가 데이터는 NFS에 고정했고, 대형 retrieval embedding은 chunk 단위 atomic
cache에 저장된다. 프로세스가 죽으면 다음 명령을 그대로 다시 실행한다. 완료 chunk는
검증 후 재사용하고 미완료 chunk만 계산한다.

```bash
MODEL_PATH="$PWD/.cache/huggingface/hub/models--nvidia--Nemotron-3-Embed-8B-BF16/snapshots/2b29550c4ab0646bb6bb47032dda54ea11f6dfe2"

env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN \
  -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE -u HF_DATASETS_OFFLINE \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONPATH="$PWD/third_party/mteb" \
  .venv-mteb/bin/python scripts/evaluate_sionic9.py \
  --model "$MODEL_PATH" \
  --revision 2b29550c4ab0646bb6bb47032dda54ea11f6dfe2 \
  --batch-size 64 --max-length 8192 \
  --attn-implementation flash_attention_2 \
  --output-dir outputs/evaluation/sionic9-nemotron3-full-fixed-prompt \
  --embedding-cache-dir outputs/embedding-cache/sionic9-nemotron3/full-fixed-prompt
```

완료 증거는 다음 파일이다.

```text
outputs/evaluation/sionic9-nemotron3-full-fixed-prompt/
  .../summary.json
outputs/embedding-cache/sionic9-nemotron3/full-fixed-prompt/
  <sha256-prefix>/<sha256>.npy
  <sha256-prefix>/<sha256>.json
```

## 최단 승리 의사결정

1. Nemotron-3 raw macro가 `> 0.7930`이고 clean selector가 Qwen/Comsat 대비 guard 안이면
   Nemotron-3를 즉시 성능 기준 모델로 채택한다.
2. raw macro는 이기지만 특정 한국어 target이 약하면 전체 200K 재학습보다 그 task의
   공개 train-family와 general replay를 섞은 짧은 LoRA/partial adaptation을 먼저 한다.
3. clean selector가 유의하게 나쁘면 Nemotron-3는 teacher/miner로만 쓰고, 보존된 Qwen
   `checkpoint-1750`에서 3,123 step까지 exact resume한다.
4. Qwen resume는 원 run의 legacy validation을 그대로 사용해 optimizer/scheduler/RNG
   계약을 유지한다. 그 loss로 checkpoint를 고르지 않고 모든 adapter를 legal v2 10K와
   fixed multidomain selector로 다시 선택한다.

Qwen 원 run의 exact validation은
`data/processed/ko_triplet_pilot_10k/validation.hn-qwen3-r095-n4.jsonl`
(`sha256=f121f7eb3011ee2bfd796cb7622efd4b6f8f8ad80d09525cf083eeb18c7a9ede`)이다.
`checkpoint-1750/args.json`과 실제 파일이 일치하며 optimizer, scheduler, RNG state도 모두
남아 있다. 이 파일 대신 새 legal-v2 512를 넣으면 resume contract가 달라지므로 재개하지
않는다.

## 공개 정책

- 재배포가 허용된 학습 데이터와 derived dataset은 기본 public이다.
- dataset card에 upstream repository/revision, license, row별 source/provenance, 변환,
  dedup과 benchmark-overlap 감사를 기록한다.
- 재배포 권리가 없거나 불명확한 원문은 공개 artifact에 포함하지 않는다.
- 모델과 adapter/checkpoint도 public을 기본으로 하며 base revision, license inheritance,
  data exposure와 평가 상태를 카드에 명시한다.
- 기존 200K/1M performance manifest는 missing/custom/noncommercial source와 공개
  재배포 blocker를 선언한다. 이 artifact는 새 public repo로 복제하지 않으며, 공개 가능한
  모델 학습에는 rights-safe source로 다시 만든 curriculum만 사용한다.
- public uploader는 요청 visibility를 repo 생성 전후와 immutable commit에서 확인하고,
  전체 file allowlist와 LFS SHA/size 및 metadata SHA가 맞아야 완료로 기록한다.
- 고정 model-selection holdout은 공개하면 selector 역할이 훼손되므로 작은 private
  artifact로 유지한다. 학습 데이터 저장 공간 절감 정책의 예외다.
