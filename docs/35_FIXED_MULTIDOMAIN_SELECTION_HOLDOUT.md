# 고정 비공개 다영역 모델 선택 보드

기준일: **2026-07-17 (Asia/Seoul)**

## 결론

법률 Grade-I 10K 하나만으로 최종 한국어 embedding 모델을 고르면 legal specialist에
구조적으로 유리하다. 이를 막기 위해 finance 900개와 knowledge 1,000개, 총 1,900개
질의의 고정 retrieval 보드 `multidomain-selection-heldout-v1`을 만들었다. 이 보드는
**selection-only**, **never training**, **not a public benchmark**다.

비공개 원격 복구본은
[`LLM-OS-Models2/korean-embedding-multidomain-selection-heldout-v1@d261e1e3`](https://huggingface.co/datasets/LLM-OS-Models2/korean-embedding-multidomain-selection-heldout-v1/tree/d261e1e3ff64e13828e73017fe2c312aae575709)에
있다. 게시기가 private visibility, immutable 40-hex commit, 전체 파일 allowlist와 모든
원격 payload SHA/size를 다시 검증했다.

## Pinned source와 split

| domain | source | revision / split | selected |
|---|---|---|---:|
| finance | [`BCCard/BCAI-Finance-Kor-Embedding-Triplet`](https://huggingface.co/datasets/BCCard/BCAI-Finance-Kor-Embedding-Triplet/tree/f63d59969dba9916bd34c86c82112331890b11da) | `f63d59969dba9916bd34c86c82112331890b11da`, validation | 900 |
| knowledge | [`etri-lirs/KoTSQA-v.2.0`](https://huggingface.co/datasets/etri-lirs/KoTSQA-v.2.0/tree/ff9349df469a765b4561959e36ef1b3f377765cd) | `ff9349df469a765b4561959e36ef1b3f377765cd`, sealed test→internal selection | 1,000 |

source row는 immutable source ID의 SHA-256 순으로 정렬한 뒤 eligibility filter를 통과한
앞쪽 고정 개수만 선택한다. 행 순서나 Python hash seed에 의존하지 않는다. KoTSQA는
정규화한 정답 문자열이 passage 안에 실제 존재할 때만 binary positive qrel을 만든다.
정답 passage가 없는 625개, 학습/benchmark/중복 filter에 걸린 knowledge 537개와 finance
50개를 제외했다.

## 오염 차단 계약

현재와 이후 campaign이 사용하는 다음 모든 선언 training role의 query, semantic query
body, positive, negative normalized SHA-256을 하나의 고정 exclusion index로 만들었다.

- general 200K와 1M
- retrieval-family, SQuAD, health, AutoRAG specialist source
- legal 250K source-native curriculum
- BCAI finance train과 KoTSQA train 원 split

총 11,452,537 text occurrence를 검사했다. 별도로 Sionic 9와 공식 Korean 6의 pinned
benchmark blocklist 10,326,747 hash occurrence도 차단했다. 최종 assertion은 다음과 같다.

- selected query와 선언 training text exact overlap: **0**
- knowledge query/corpus와 선언 training text exact overlap: **0**
- selected query/corpus와 공개 benchmark blocklist overlap: **0**
- public benchmark score used for selection: **false**

finance query는 exact-held-out이지만 positive/negative corpus 중 training text occurrence가
1,373건이다. 따라서 finance는 **target-dev**이지 clean zero-shot이 아니다. knowledge도
exact-text-held-out일 뿐 source-document Grade I 또는 unseen-source Grade Z가 아니다.
법률 독립성은 별도의 Grade-I text-strict 10K가 담당한다.

## 산출물 identity

local root는 `outputs/evaluation/multidomain-selection-heldout-v1`이다.

| file | rows | SHA-256 |
|---|---:|---|
| `queries.jsonl` | 1,900 | `a122ec581e6992109742cf2cabd98e7989fa4f9442eecd5e270baf6117ac6e7f` |
| `corpus.jsonl` | 4,795 | `b7e6c4d54cb53d449ce6af6b39abb194e350940d72325740b91dd60bb619cff9` |
| `qrels.jsonl` | 2,941 | `4c40b3f6dcc285664b7bb3aae3a75e95bef6d609fc70f16f278fa0e95ed5a986` |
| `provenance.jsonl` | 1,900 | `3f69b210ed722889e3e567a9f9702180c2a2b657bdaab7b6937b24e01fd508fa` |
| `manifest.json` | — | `86fea553c6652388b1f67160c0e2e6b7626acf8929f86c1a2708156bd89b3c46` |

manifest의 모든 source/training/blocklist path는 workspace-relative라 host path를 유출하지
않으며 source bytes, revision, split, exclusion 수와 emitted SHA를 함께 결속한다.

## 점수와 선택 순서

[`evaluate_multidomain_selection.py`](../scripts/evaluate_multidomain_selection.py)는 다음
canonical 계약을 강제한다.

- BF16 model load, FlashAttention 2, max length 8,192, 4,096-d normalized embedding
- query에만 고정 Qwen3 Korean web-search instruction 적용
- domain별 corpus에서 exact float32 normalized dot product
- CUDA TF32 off, score tie는 corpus ID 오름차순
- standard binary multi-positive NDCG@10, Recall@10, MRR@10, Recall@100
- finance와 knowledge NDCG@10의 비가중 domain macro
- 각 후보의 summary와 1,900개 per-query relevant rank SHA를 model weight revision에 결속

selector `clean-guard-multidomain-near-tie-robustness-v2`는 공개 점수를 받지 않고 다음
lexicographic guard를 사용한다.

1. 법률 Grade-I NDCG@10 최고에서 `0.005` 이내만 남긴다.
2. 그 안에서 다영역 domain-macro NDCG@10 최고에서 `0.002` 이내만 남긴다.
3. paired noise 6조건의 worst-condition NDCG@10 최고에서 `0.002` 이내만 남긴다.
4. maximum noise intrusion@10 최저에서 `0.001` 이내만 남긴다.
5. 다영역 macro, 법률 clean, model ID와 immutable revision 순으로 결정론적으로 고른다.

따라서 legal을 크게 희생해 broad 점수를 올린 모델도, legal에만 과적합해 finance/knowledge가
무너진 모델도 대표 모델이 되지 않는다. Sionic 9, 공식 Korean 6과 comprehensive public
diagnostic은 이 선택이 끝난 정확히 한 winner에만 final-once로 실행한다.

## 재현과 게시

```bash
.venv-mteb/bin/python scripts/build_multidomain_selection_holdout.py --verify-only

.venv-mteb/bin/python scripts/evaluate_multidomain_selection.py \
  --model artifacts/models/<candidate> \
  --revision model-<weights-sha-prefix>
```

데이터 복구본 게시기는
[`publish_multidomain_selection_dataset.py`](../scripts/publish_multidomain_selection_dataset.py)다.
고정 `LLM-OS-Models2` repo 외 target, public visibility, 비정규 source path, 불완전 provenance,
원격 파일/해시 불일치를 모두 거부한다. 최종 모델 게시기는 선택 summary와 ranks를 다시
검증하고 모델 카드·evaluation manifest에 함께 넣는다.
