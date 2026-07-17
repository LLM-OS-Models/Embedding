# 종합 최고 모델 선택과 평가 계약

기준일: **2026-07-17 (Asia/Seoul)**

> **2026-07-17 상태 정정:** 아래 `v1-20260715-184610` run과 watcher는 container
> 재시작으로 종료되었고 local checkpoint가 남지 않았다. 학습 계약 자체는 재현을 위해
> 보존한다. exact remote data/model을 복원하고 2026-07-17 11:46 KST에 같은 clean-lineage
> run을 처음부터 다시 시작했다. 새 성능 우선 순서는
> [2026-07-17 frontier plan](34_PERFORMANCE_FIRST_FRONTIER_PLAN_2026-07-17.md)에 고정한다.

## 결론

목표는 Sionic retrieval 9종 한 보드에서만 높은 모델이 아니다. **한국어 retrieval을
중심으로 broad text 이해, 다국어, 긴 문맥과 evidence 위치, 대화·OCR noise 강건성,
처리량·메모리·index 비용까지 함께 우수한 실용적 종합 최고 embedding model**을 만드는
것이다.

현재 이 목표를 만족한다고 판정된 우리 모델은 **0개**다. 실행 중인 200K LoRA r64는
첫 유효 performance candidate가 될 수 있는 학습 run이지, 아직 merge·clean selection·공개
평가를 통과한 성능 후보가 아니다. 따라서 지금은 다음 어느 것도 주장하지 않는다.

- Comsat 또는 Sionic 9 평균을 이겼다.
- 공식 MTEB Korean v1에서 상위 모델을 이겼다.
- clean Korean zero-shot 최고다.
- 한국어·다국어를 아우르는 종합 최고 모델이다.

## 무엇을 “최고”라고 부르는가

서로 다른 metric을 하나의 임의 가중 평균으로 합치지 않는다. 먼저 각 축의 필수 gate와
회귀를 확인하고, 모든 필수 축을 완주한 모델끼리 Pareto 관점으로 비교한다.

| 축 | 판단 대상 | 현재 사용 근거 | 현재 상태 |
|---|---|---|---|
| Korean retrieval | 법률·공공 clean holdout, 최종 Sionic retrieval 9종 | Grade-I 법률 10K, Sionic 고정 protocol | baseline/우리 후보 clean 결과 대기 |
| Broad text | retrieval 외 classification·pair·bitext·multilabel | text-only 7-task/414-subset diagnostic | protocol 구현, 점수 없음 |
| Multilingual | 한국어↔영어 검색과 한국어 포함 병렬문장 | XPQA 3 subsets, FLORES 406 subsets | diagnostic 점수 없음 |
| Long/context | 512/2K/4K/8K, evidence head/middle/tail | 별도 paired long-context 설계 | 실행 결과 없음 |
| Noise robustness | prompt on/off, filler/system artifact, 0/1/5% noise | Grade-I 법률 companion evaluator | baseline/우리 후보 결과 대기 |
| Efficiency | docs/s, p50/p95, peak VRAM, 차원·index bytes | 동일 hardware/runtime 측정 | 최종 후보 측정 대기 |

아주 작은 차이를 목표 함수로 쫓지 않는다. 현재 구현된 clean selection에서 NDCG@10
절대 차이 `0.002` 이하는 실질적 near-tie로 취급한다. near-tie 안에서는 최악 조건
robustness와 noise intrusion을 보고 결정한다. 공개 보드에서도 반올림 오차 수준의 차이를
“압도적 우위”로 표현하지 않고 raw score와 불확실성을 함께 공개한다.

## 현재 candidate 원장

| run | 상태 | 종합 성능 판정에 쓸 수 없는 이유 |
|---|---|---|
| 288-row LoRA r32 smoke | pipeline-only | 저장·재로딩 검증이며 retrieval 성능 실험이 아님 |
| 10K exhaustive-HN LoRA r64 | diagnostic | BF16 direct-fold merge parity가 strict gate에 미달했고 종합 평가가 없음 |
| 50K LoRA r64 | 실격 | trainer order에서 공개 평가 query exact hash 4개가 발견됨 |
| 200K LoRA r64 | active 재학습 | 2026-07-17 새 run은 아직 checkpoint·clean selection 결과가 없음 |

따라서 private checkpoint가 생성되더라도 그것은 검증할 **후보 artifact**일 뿐 valid
performance model count를 늘리지 않는다. 같은 이유로 train/eval loss, positive margin,
초기 finite gradient와 GPU utilization은 성능 우위의 증거가 아니다.

## 200K r64 exact 재학습 계약

run ID는 `qwen3-embedding-8b-ko-performance200k-lora-r64`다. 소실된 역사 run의 trainer
version은 `v1-20260715-184610`, 현재 exact 재학습 version은
`v0-20260717-114605`이며 2026-07-17 11:46 KST에 시작했다.

| 항목 | 고정값 |
|---|---|
| base | `Qwen/Qwen3-Embedding-8B@1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` |
| ordered train | 199,904 rows, source-homogeneous length buckets |
| train content SHA-256 | `8e2731ab25299ff558af675f067b253a6ce4375a850aa925acfe3b3117505e3c` |
| training manifest SHA-256 | `eeed4fcdab4eb3eecf62f6bde483451f8be12cf2aa54bff21f64f234cfbcf280` |
| validation | independently audited 512 rows |
| backend | `.venv-train-fa2`, BF16, SDPA, gradient checkpointing |
| tuner | LoRA r64, alpha 128, dropout 0.05, all-linear |
| objective | InfoNCE, in-batch negatives + row별 explicit HN 4 |
| actual microbatch | 16 rows; source-homogeneous batch와 정확히 일치 |
| accumulation | 4; optimizer-step당 64 rows, 단 microbatch 사이 in-batch negative 공유는 없음 |
| sequence | max 512, `truncation_strategy=right`, lazy tokenize |
| order | `dataset_shuffle=false`, `train_dataloader_shuffle=false`, `strict=true` |
| schedule | LR `1e-5`, cosine, warmup ratio 0.05, seed/data seed 42 |
| duration | 3,123 optimizer steps |
| validation/checkpoint | 250 steps마다 eval/save, `save_total_limit=3`, eval batch 4 |

재시작 후 FA2는 exact homogeneous-order 5+5-step 비교에서 SDPA 대비 약 3.73%만 빨랐고
사전 고정한 1.05× admission 기준을 넘지 못했다. 작은 차이를 승격 근거로 삼지 않아
SDPA를 선택했다. 새 admission report SHA-256은
`c409291a95017716925275ec3068db19ba00d734750ab77a0e38c2b1f432ec11`다.
초기 step의 finite loss/gradient와 H100 사용률은 runtime 건강성만 증명한다.

## Private checkpoint watcher의 역할

watcher는 2026-07-17 새 run과 함께 `LLM-OS-Models2` private repository를 대상으로
실행 중이다. step 250 이후 같은 간격의 completed checkpoint만 검사한다. directory step과 같은 step의 finite
`eval_loss`, 전체 BF16/F16/F32 safetensors의 finite value, LoRA A/B 구조와 안정된 file
fingerprint를 통과해야 한다.

private repository에는 checkpoint마다 다음 세 파일만 atomic commit한다.

```text
adapter_model.safetensors
adapter_config.json
candidate_manifest.json
```

optimizer, scheduler, RNG, trainer state, training args, log, raw/processed data와 local path는
업로드하지 않는다. watcher lineage에는 위 training manifest와 backend admission report
SHA가 들어간다. private 업로드 성공은 보존·복구 가능성을 뜻할 뿐 성능 승격이나 public
release를 뜻하지 않는다.

200K selection-only 경계에서는 clean winner 전체 모델도 private로 올리고 remote manifest를
재다운로드해 exact commit report를 남긴다. 이후 1M/KD/target/legal LoRA watcher는 local base의
weight SHA와 이 report의 private repo/commit을 대조한 뒤에만 시작한다. 따라서 adapter만 남고
그 adapter가 의존한 continual base를 잃는 복구 공백도 허용하지 않는다. 학습은 명시적 offline,
token-free이고 uploader/watcher만 ignored mode-0600 `.env`를 메모리에서 읽는다.

전체 모델 private backup은 source model directory를 직접 업로드하지 않는다. 같은 NFS의
일회성 격리 staging에 safetensors를 hardlink하고, SentenceTransformers/Qwen 로딩에 필요한
고정 allowlist metadata와 clean-selection evidence만 복사한다. JSON/JSONL evidence의 host
절대경로와 인식 가능한 credential은 staging에서 정규화하고 `train.log`, optimizer/trainer
state, 임의 파일 또는 symlink가 하나라도 있으면 fail closed한다. 원본 모델 directory에는
README/evaluation/manifest를 쓰지 않는다. tokenizer vocabulary/merges와 weight index는
byte-exact로 보존하고 manifest가 자신을 제외한 모든 staged file의 SHA-256/size를 결속한다.
성공 report는 `private:true`,
`remote_manifest_exact:true`, `remote_file_set_exact:true`를 모두 가져야 하며, 모든 model
safetensors의 Hub LFS SHA-256/size와 모든 metadata download SHA-256을 exact commit에서 다시
검증한다. 이 세 조건이 없는 report는 다음 continual base lineage로 사용할 수 없다.

## Clean-first 선택 정책

### 1. 후보 자격

후보는 run-level disqualification이 없어야 하고, 선택된 checkpoint의 adapter/full weights가
finite해야 하며, merge/package parity와 immutable model-weight SHA evidence를 통과해야 한다.
50K처럼 benchmark query overlap이 확인된 run은 loss가 낮아도 자동 제외한다.

### 2. Grade I clean retrieval

현재 primary selection set은 10,000-query Korean legal/public
same-repository whole-source-document-held-out다. 독립성 등급은 **I이며 Z가 아니다**.

- training candidate ID와 source-document SHA overlap은 0이다.
- Sionic 9/공식 Korean blocklist의 normalized query/positive exact hash overlap은 0이다.
- 그러나 학습과 같은 Legalize-KR 4개 repository와 같은 source-native pair 구조다.
- MinHash·dense semantic near-duplicate 차단과 독립 human relevance judgment를 보장하지
  않는다.

따라서 `unseen-source`, `clean zero-shot`, `Grade Z`라고 부르지 않는다. 허용되는 표현은
`same-repository source-document-held-out (Grade I)`다.

### 3. 실질적 near-tie

selector `clean-first-grade-i-near-tie-robustness-v1`은 public score를 입력으로 받지 않고
다음 순서로 한 후보를 고른다.

1. Grade-I clean NDCG@10 최고점에서 `0.002` 이내 후보를 near-tie로 묶는다.
2. 여기에 포함된 후보의 6개 prompt/noise 조건 중 최저 NDCG@10을 비교하고, 최고값에서
   `0.002` 이내를 다시 near-tie로 묶는다.
3. 최대 synthetic-noise intrusion@10이 최저값에서 `0.001` 이내인 후보를 남긴다.
4. 그래도 동률이면 clean NDCG@10, model ID, revision 순으로 결정론적으로 고른다.

즉 `0.0001` 같은 차이만 반복 최적화하지 않는다. 동시에 epsilon은 “모든 품질이 같다”는
뜻이 아니므로 worst-domain, long-context, multilingual, efficiency의 큰 회귀는 별도
필수 gate에서 차단한다.

### 4. Public final-once

Sionic 9와 공식 MTEB Korean v1 결과는 checkpoint selector에 들어가지 않는다. clean과
robustness로 local winner를 먼저 고른 뒤 그 **한 모델**에만 `final-selected` namespace로
Sionic 9 전체와 공식 Korean 6-task를 실행한다. runtime OOM에 따른 사전 고정 batch fallback은
허용하지만 public 점수를 보고 checkpoint나 hyperparameter를 되돌려 고르지 않는다.

이후 같은 winner에 text-only comprehensive diagnostic을 실행한다. 세 결과가 모두 있어야
private best-candidate package의 evaluation evidence가 완성된다. public weight 전환은 별도로
데이터 license/provenance와 release gate를 통과해야 한다.

## 새 text-only comprehensive diagnostic

고정 protocol은
[`configs/comprehensive_text_v1_protocol.json`](../configs/comprehensive_text_v1_protocol.json),
runner는
[`scripts/evaluate_comprehensive_text_v1.py`](../scripts/evaluate_comprehensive_text_v1.py)다.
MTEB `2.18.0`과 checkout `193e3f66d2deac678065a43354c9c4efc57f507d`, exact dataset
revision, task metadata, subset order와 repository-local snapshot을 model load 전에
검증한다. 네트워크와 credential은 runner에서 제거한다.

| task | type / split | 선택 subset | contamination/claim |
|---|---|---:|---|
| XPQARetrieval | Retrieval / test | 3 | medium, regression diagnostic only |
| FloresBitextMining | BitextMining / devtest | 406 | medium, regression diagnostic only |
| KorSarcasmClassification.v2 | Classification / test | 1 | medium, regression diagnostic only |
| KorHateClassification.v2 | Classification / test | 1 | medium, regression diagnostic only |
| KorFin | Classification / train | 1 | high, training-split diagnostic only |
| KorHateSpeechMLClassification | MultilabelClassification / test | 1 | medium, regression diagnostic only |
| KorNLI | PairClassification / test | 1 | high, public benchmark-family diagnostic only |
| **합계** | **7 tasks** | **414** | **clean selection score가 아님** |

FLORES 406 subsets가 단순 subset 평균을 지배하지 않도록 task score는 선택 subset main
score의 비가중 평균, 종합 summary는 7개 completed task score의 비가중 평균과 type별
평균을 따로 기록한다. 이 값도 서로 다른 clean/public board와 다시 평균내지 않는다.

### 명시적 제외

- `K-HATERS`: MTEB 2.18.0에 지원되는 registered task가 없어 별도 custom protocol 검토
  전까지 제외한다.
- SDS KoPub VDR T2IT와 KoViDoRe v2의 cybersecurity/economic/energy/HR 5개 asset:
  image/visual-document retrieval이므로 text-only runner에서 제외한다.

이 제외는 모델이 해당 능력이 없다는 결과가 아니다. 아직 맞는 modality evaluator를
실행하지 않았다는 뜻이다.

### 공개 benchmark contamination 주의

7-task/414-subset suite는 공개 dataset의 test/devtest를 사용하고, KorFin은 train split,
KorNLI는 공개 benchmark family다. 원 모델이나 upstream data가 이를 보았는지 완전히
증명할 수 없으므로 contamination grade가 medium/high이고 **diagnostic regression board**로만
사용한다. 이 점수로 `clean`, `zero-shot`, `Grade Z` 또는 독립 일반화 우위를 주장하지
않는다.

Sionic 9와 공식 Korean 6도 공개 benchmark다. target train-family를 쓴 모델 결과는
task별 `target-adapted/in-domain` exposure를 함께 공개하고 clean 결과와 같은 표제 아래
섞지 않는다. exact blocklist overlap 0은 exact text leakage 차단 근거이지 semantic
contamination이 절대 없다는 증명은 아니다.

## 고정 Qwen3 reranker teacher scorer

후속 false-negative filter와 score distillation teacher는
`Qwen/Qwen3-Reranker-8B@77d193c791ed757ca307ee72715aa132723da912`다. scorer는 official
`<Instruct>/<Query>/<Document>` chat contract와 마지막 위치의 `no=2152`, `yes=9693`
단일 token raw logits를 사용하고, 두 logit의 softmax `P(yes)`를 저장한다.

production scorer의 실제 model microbatch 기본값은 **8 documents**다. query 하나의
positive 포함 최대 201 documents를 `8, 8, ...`로 나누어 8B model에 넣으며, output row와
state에 model batch, prompt, token ID, raw logits, normalized probability, input/output SHA와
row count를 고정한다.

local snapshot의 5개 LFS weight content SHA-256은 다음과 같다.

```text
model-00001-of-00005.safetensors  22cdfea4a13b7b3e866573800eeeb638fc38962940adf631d06dc03befed047a
model-00002-of-00005.safetensors  d2163b74137e35b4614bd2aa5bf27bcb07de4ca61c6962495feb968385eb0df8
model-00003-of-00005.safetensors  a5038caa78c817e8acce6806104869675938a33fd4e60ed038e9931d390d6989
model-00004-of-00005.safetensors  247f85538c5996d4c296291b0e4004f618c9b17ca8cdc25d1fc726567eb15803
model-00005-of-00005.safetensors  8ba41b93c2e4ec8339ad16b000bc977fde196aeac054956cbfc8c0186ee6d4cf
```

production load는 약 16GB 실제 file bytes를 위 content identity와 전부 대조한 뒤에만
진행한다. `local_files_only=true`, `trust_remote_code=false`이며 inherited Hub/Git
credential과 telemetry를 제거한다. score cache는 atomic/resumable shard와 manifest를
사용하고 query/document 원문, token, local path를 출력하지 않는다.

이 scorer가 준비됐다는 사실도 모델 성능 개선은 아니다. 200K clean-first 결과가 나온 뒤
같은 data/token budget으로 filter-only, distribution KL, MarginMSE를 제한적으로 비교하고,
clean·broad·multilingual 회귀가 실제로 좋아질 때만 다음 scale에 채택한다.

## 다음 판정 순서

1. exact 200K data와 환경을 복원하고 3,123-step 학습·private checkpoint 보존을 처음부터 다시 실행한다.
2. finite/parity/evidence gate를 통과한 checkpoint만 merge/package한다.
3. 모든 eligible local candidate를 Grade-I clean 10K와 paired noise 6조건으로 평가한다.
4. `0.002` clean near-tie 정책으로 local winner 하나를 고른다.
5. winner에만 Sionic 9, 공식 Korean 6, comprehensive text 7/414를 final-once 실행한다.
6. long-context·multilingual·efficiency 회귀와 license/provenance를 함께 확인한다.
7. 모든 필수 gate가 통과될 때만 “종합 최고” 또는 public release를 검토한다.

어느 단계에서도 실패하면 결과를 숨기거나 public score로 이전 checkpoint를 다시 고르지
않는다. valid candidate가 계속 0이면 그 사실을 유지하고, 실패 축을 개선하는 다음
controlled ablation으로 넘어간다.
