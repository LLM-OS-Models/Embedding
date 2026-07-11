# Sionic Comsat benchmark audit

대상: [`sionic-ai/comsat-embed-ko-8b-preview`](https://huggingface.co/sionic-ai/comsat-embed-ko-8b-preview)

## 카드가 실제로 밝힌 것

- base: `Qwen/Qwen3-Embedding-8B`
- `1M+ Korean examples`
- 4096-d, last-token pooling, L2 normalization, cosine
- query에 고정 instruction, document에는 prefix 없음
- 카드 사용 예시 max length 8,192
- license: CC BY-NC 4.0

카드가 밝히지 않은 것:

- unique document 수와 전체 token 수
- pair/triplet/multi-positive 구조
- 데이터 출처와 license manifest
- loss, temperature, effective batch, learning rate, epoch
- hard-negative miner와 false-negative 처리
- eval query/corpus dedup/decontamination
- checkpoint 선택 기준
- 정확한 MTEB version/commit, MLDR split, raw result JSON
- bootstrap confidence interval 또는 유의성 검정

따라서 `1M 문서`나 `1M token CPT`라고 해석할 수 없습니다.

## 점수 차이

| Task | Comsat | Qwen3-8B | Delta |
|---|---:|---:|---:|
| MIRACL | .6964 | .6783 | +.0181 |
| MrTyDi | .6253 | .6187 | +.0066 |
| MLDR | .5183 | .5036 | +.0147 |
| AutoRAG | .8518 | .8276 | +.0242 |
| Ko-StrategyQA | .8394 | .8363 | +.0031 |
| PublicHealthQA | .8871 | .8721 | +.0150 |
| Belebele | .9853 | .9828 | +.0025 |
| SQuADKorV1 | .9168 | .9063 | +.0105 |
| LawIRKo | .8164 | .8171 | -.0007 |
| **Macro avg** | **.7930** | **.7825** | **+.0105** |

이는 절대 NDCG 1.05 point, 상대 약 1.34% 상승입니다.

## 정확한 주장 범위

방어 가능한 표현은 다음입니다.

> 모델 카드 작성자가 비교한 14개 모델 중, 작성자가 선택한 공개 한국어 retrieval 9종의 macro NDCG@10에서 1위.

다음 표현은 근거가 부족합니다.

- 전체 MTEB multilingual SOTA
- 모든 한국어 embedding task의 SOTA
- unseen Korean retrieval generalization SOTA
- contamination-free zero-shot SOTA

공식 `MTEB(kor, v1)`은 이 9종과 다른 구성입니다. Comsat 표는 custom slice입니다.

## contamination 판단

평가 query/qrel/corpus가 모두 공개되어 있고, 모델은 평가 데이터 공개 후 만들어졌습니다. 학습 출처와 decontamination을 공개하지 않았으므로 contamination을 배제할 수 없습니다. 반대로 LawIRKo에서 미세하게 하락하고 개선이 여러 task에 분산되므로, 공개 정보만으로 누수를 단정할 증거도 없습니다.

결론은 **contamination unknown**입니다.

## 재현성 위험

- PublicHealthQA와 AutoRAG처럼 query가 매우 적은 dataset을 동일 가중합니다.
- Belebele는 .98대로 포화되어 모델 차이가 거의 없습니다.
- MLDR는 split과 truncation 설정이 중요하지만 카드에 명시되지 않았습니다.
- model card 자체 표이며 공식 results repo의 raw run 제출이 확인되지 않습니다.

## 이길 수 있는가

같은 9종 숫자는 충분히 공략 가능합니다. 특히 AutoRAG, MIRACL, MLDR, PublicHealth, SQuAD 계열에서 1~2 point씩 얻고 나머지를 보존하면 됩니다. 다만 공개 corpus에서 synthetic query를 만들거나 qrel을 학습하면 숫자는 쉬워져도 연구 가치는 사라집니다.
